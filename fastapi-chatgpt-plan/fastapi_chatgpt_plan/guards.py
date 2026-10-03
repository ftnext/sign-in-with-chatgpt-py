"""Shared request guards: host, origin, session, CSRF, and auth checks."""

import secrets
import time
from urllib.parse import parse_qsl

from fastapi import Request

from .client import USAGE_URL
from .errors import AuthError, PublicError
from .oauth import PLAN_SCOPE, REFRESH_MARGIN_SECONDS
from .sessions import CSRF_HEADER, SESSION_COOKIE, Connection


def get_state(request: Request):
    return request.app.state.chatgpt_state


def get_session_id(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)


async def require_session(request: Request):
    """A valid browser session (anonymous or authenticated) or 401."""
    state = get_state(request)
    session = await state.get_session(get_session_id(request))
    if session is None:
        raise PublicError(
            "session_required",
            "No valid session. Load the app page and try again.",
            401,
        )
    return session


async def require_authenticated(request: Request):
    """A session that completed sign-in, with a live connection, or 401."""
    session = await require_session(request)
    state = get_state(request)
    if (
        not session.authenticated
        or state.connection is None
        or session.generation != state.connection.generation
    ):
        raise PublicError("sign_in_required", "Not signed in.", 401)
    if state.connection.needs_reauth:
        raise PublicError("reauthorization_required", "Sign in again to continue.", 401)
    return session


async def verify_origin(request: Request) -> None:
    expected = request.app.state.expected_origin
    if expected is None:
        return
    origin = request.headers.get("origin")
    if origin:
        if origin != expected:
            raise PublicError("invalid_origin", "Untrusted request origin.", 403)
        return
    referer = request.headers.get("referer")
    if referer and referer.startswith(expected + "/"):
        return
    raise PublicError("invalid_origin", "Untrusted request origin.", 403)


async def verify_csrf(request: Request, session) -> None:
    token = request.headers.get(CSRF_HEADER)
    if token is None:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/x-www-form-urlencoded"):
            body = await request.body()
            if len(body) <= 64 * 1024:
                pairs = parse_qsl(body.decode("utf-8", "replace"))
                token = dict(pairs).get("csrf_token")
    if not token or not secrets.compare_digest(token, session.csrf):
        raise PublicError("invalid_csrf_token", "Invalid CSRF token.", 403)


async def require_plan(request: Request) -> Connection:
    """Authenticated connection allowed to use ChatGPT plan inference."""
    settings = request.app.state.settings
    if not settings.chatgpt_plan_enabled:
        raise PublicError(
            "inference_disabled",
            "ChatGPT plan inference is disabled.",
            403,
        )
    await require_authenticated(request)
    connection = get_state(request).connection
    if not connection.plan_permitted or PLAN_SCOPE not in connection.scopes:
        raise PublicError(
            "plan_permission_required",
            "ChatGPT plan usage was not granted. Check usage sharing "
            f"settings: {USAGE_URL}",
            403,
        )
    return connection


async def ensure_access_token(request: Request) -> Connection:
    """Return the connection with a fresh access token, refreshing if needed.

    Concurrent refreshes serialize on refresh_lock; a connection replaced
    while waiting (sign-out or new sign-in) is never resurrected.
    """
    connection = await require_plan(request)
    state = get_state(request)
    if connection.expires_at > time.time() + REFRESH_MARGIN_SECONDS:
        return connection
    async with state.refresh_lock:
        if not state.is_current(connection):
            raise PublicError("sign_in_required", "Not signed in.", 401)
        if connection.needs_reauth:
            raise PublicError(
                "reauthorization_required", "Sign in again to continue.", 401
            )
        if connection.expires_at > time.time() + REFRESH_MARGIN_SECONDS:
            return connection
        if not connection.refresh_token:
            connection.needs_reauth = True
            raise PublicError(
                "reauthorization_required", "Sign in again to continue.", 401
            )
        oauth = request.app.state.oauth
        generation = connection.generation
        try:
            fields = await oauth.refresh(connection)
        except AuthError:
            connection.needs_reauth = True
            raise PublicError(
                "reauthorization_required", "Sign in again to continue.", 401
            )
        current = state.connection
        if current is connection and current.generation == generation:
            current.access_token = fields["access_token"]
            current.refresh_token = fields["refresh_token"]
            current.scopes = fields["scopes"]
            current.expires_at = fields["expires_at"]
            current.id_token = fields["id_token"]
        else:
            raise PublicError("sign_in_required", "Not signed in.", 401)
        if current.needs_reauth:
            raise PublicError(
                "reauthorization_required", "Sign in again to continue.", 401
            )
        return current


def check_model_cache(state, connection) -> list | None:
    cached = state.models_cache
    if not cached:
        return None
    if cached.get("client_id") != connection.client_id:
        return None
    if cached.get("subject") != connection.subject:
        return None
    models = cached.get("models")
    return models if isinstance(models, list) else None
