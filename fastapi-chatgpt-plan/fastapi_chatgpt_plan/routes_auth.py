"""Auth-facing routes: page, session bootstrap, OAuth, logout."""

import contextlib
import secrets
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import registrations
from .errors import ApiError, AuthError
from .guards import (
    get_session_id,
    get_state,
    require_session,
    verify_csrf,
    verify_origin,
)
from .oauth import IDENTITY_SCOPES, PLAN_SCOPE, PLAN_SCOPES
from .sessions import SESSION_COOKIE, SESSION_TTL_SECONDS, Connection

router = APIRouter()

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

CALLBACK_MESSAGES = {
    "expired_sign_in": "The sign-in attempt expired. Start again.",
    "invalid_state": "The sign-in response could not be verified.",
    "sign_in_declined": "Sign-in was declined in the browser.",
    "account_mismatch": (
        "The signed-in account does not match this registration."
    ),
    "plan_permission_required": "The account did not grant any access.",
}


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def set_session_cookie(response, session) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session.id,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(
        TEMPLATE_PATH.read_text(), headers={"Cache-Control": "no-store"}
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/api/session")
async def session_info(request: Request):
    state = get_state(request)
    session = await state.get_session(get_session_id(request))
    created = False
    if session is None:
        session = await state.create_session()
        created = True
    connection = state.connection
    status = "anonymous"
    user = None
    if session.authenticated and connection is not None:
        status = (
            "reauthorization_required"
            if connection.needs_reauth
            else "authenticated"
        )
        user = {"email": connection.email, "name": connection.name}
    settings = request.app.state.settings
    payload = {
        "status": status,
        "user": user,
        "plan": {
            "enabled": settings.chatgpt_plan_enabled,
            "permitted": bool(
                connection and connection.plan_permitted and session.authenticated
            ),
        },
        "csrf": session.csrf,
    }
    response = JSONResponse(payload, headers={"Cache-Control": "no-store"})
    if created:
        set_session_cookie(response, session)
    return response


@router.post("/auth/login")
async def login(request: Request):
    session = await require_session(request)
    await verify_origin(request)
    await verify_csrf(request, session)

    app_state = request.app.state
    settings = app_state.settings
    state = get_state(request)
    registration = app_state.registration
    connection = state.connection

    client_id = registration.get("client_id")
    prior = None
    if client_id:
        prior = {
            "id_token": connection.id_token if connection else None,
            "email": (connection.email if connection else None)
            or registration.get("email"),
        }
    tx = await state.begin_transaction(
        session.id,
        verifier=secrets.token_urlsafe(64),
        redirect_uri=settings.redirect_uri,
        client_id=client_id,
    )
    scope = PLAN_SCOPES if settings.chatgpt_plan_enabled else IDENTITY_SCOPES
    url = await app_state.oauth.begin(
        tx,
        host_id=registration["host_id"],
        scope=scope,
        client_id=client_id,
        prior=prior,
    )
    return RedirectResponse(url, status_code=303)


def _callback_error(exc: AuthError) -> JSONResponse:
    code = str(exc)
    message = CALLBACK_MESSAGES.get(code, f"Sign-in failed: {code}")
    return JSONResponse(error_body(code, message), status_code=400)


@router.get("/auth/callback")
async def callback(request: Request):
    pairs = parse_qsl(
        request.scope["query_string"].decode(), keep_blank_values=True
    )
    if len(pairs) != len({key for key, _ in pairs}):
        return JSONResponse(
            error_body("invalid_callback", "Duplicated callback parameters."),
            status_code=400,
        )
    query = dict(pairs)
    state = get_state(request)
    session_id = get_session_id(request)
    tx = await state.consume_transaction(query.get("state", ""), session_id)

    app_state = request.app.state
    registration = app_state.registration
    try:
        if query.get("error"):
            raise AuthError(
                "sign_in_declined" if tx is not None else "invalid_state"
            )
        if tx is None:
            raise AuthError("invalid_state")
        client_id_hint = query.get("client_id")
        if tx.client_id is None and client_id_hint:
            registrations.save_registration(
                app_state.settings.state_dir,
                {**registration, "client_id": client_id_hint},
            )
            registration["client_id"] = client_id_hint
        client_id, identity, credentials, id_token = await app_state.oauth.complete(
            tx, query, prior_subject=registration.get("subject")
        )
    except AuthError as exc:
        return _callback_error(exc)
    except ApiError as exc:
        return JSONResponse(
            error_body("upstream_error", str(exc)), status_code=502
        )

    record = {
        "client_id": client_id,
        "issuer": identity.get("iss"),
        "subject": identity["sub"],
        "email": identity.get("email"),
        "name": identity.get("name"),
    }
    registrations.save_registration(
        app_state.settings.state_dir, {**registration, **record}
    )
    registration.update(record)

    connection = Connection(
        client_id=client_id,
        issuer=identity.get("iss"),
        subject=identity["sub"],
        email=identity.get("email"),
        name=identity.get("name"),
        id_token=id_token or "",
        access_token=credentials["access_token"],
        refresh_token=credentials["refresh_token"],
        scopes=credentials["scopes"],
        expires_at=credentials["expires_at"],
        generation=0,
        plan_permitted=PLAN_SCOPE in credentials["scopes"],
    )
    await state.set_connection(connection)

    session = await state.get_session(session_id)
    rotated = await state.rotate_session(session)
    response = RedirectResponse("/", status_code=303)
    set_session_cookie(response, rotated)
    return response


@router.post("/auth/logout")
async def logout(request: Request):
    session = await require_session(request)
    await verify_origin(request)
    await verify_csrf(request, session)

    state = get_state(request)
    connection = state.connection
    remote_revocation = "not_attempted"
    if connection and connection.refresh_token:
        confirmed = False
        with contextlib.suppress(Exception):
            confirmed = await request.app.state.oauth.revoke(
                connection.client_id, connection.refresh_token
            )
        remote_revocation = "confirmed" if confirmed else "not_confirmed"
    await state.clear_connection()
    await state.destroy_session(session.id)
    await state.close_streams()
    response = JSONResponse(
        {"status": "signed_out", "remote_revocation": remote_revocation}
    )
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
