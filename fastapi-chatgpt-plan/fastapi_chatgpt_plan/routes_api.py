"""Plan API routes: model listing and the Responses SSE wrapper."""

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from . import client
from .errors import AuthError, PublicError
from .guards import (
    check_model_cache,
    ensure_access_token,
    get_state,
    require_authenticated,
    require_plan,
    verify_csrf,
    verify_origin,
)
from .schemas import MAX_BODY_BYTES, ResponseRequest

router = APIRouter()


def require_current(state, connection):
    if not state.is_current(connection):
        raise PublicError(
            "sign_in_required", "The connection changed. Sign in again.", 401
        )


@router.get("/api/models")
async def list_models(request: Request):
    connection = await ensure_access_token(request)
    state = get_state(request)
    http = request.app.state.http
    try:
        models = await client.fetch_models(http, connection.access_token)
    except AuthError:
        connection.needs_reauth = True
        raise PublicError("reauthorization_required", "Sign in again to continue.", 401)
    require_current(state, connection)
    state.models_cache = {
        "client_id": connection.client_id,
        "subject": connection.subject,
        "fetched_at": time.time(),
        "models": models,
    }
    return {"models": models}


async def _limited_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise PublicError(
                "request_too_large",
                f"Request body exceeds {MAX_BODY_BYTES} bytes.",
                413,
            )
    return bytes(body)


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:3]:
        loc = ".".join(str(item) for item in error["loc"])
        parts.append(f"{loc}: {error['msg']}" if loc else error["msg"])
    return "; ".join(parts) or "Invalid request."


@router.post("/api/responses")
async def responses(request: Request):
    session = await require_authenticated(request)
    await verify_origin(request)
    await verify_csrf(request, session)
    await require_plan(request)

    body = await _limited_body(request)
    try:
        data = json.loads(body)
    except ValueError:
        raise PublicError("invalid_json", "Request body is not valid JSON.", 422)
    try:
        payload = ResponseRequest.model_validate(data)
    except ValidationError as exc:
        raise PublicError("invalid_request", _validation_message(exc), 422)

    connection = await ensure_access_token(request)
    state = get_state(request)
    http = request.app.state.http

    models = check_model_cache(state, connection)
    if models is None:
        try:
            models = await client.fetch_models(http, connection.access_token)
        except AuthError:
            connection.needs_reauth = True
            raise PublicError(
                "reauthorization_required", "Sign in again to continue.", 401
            )
        require_current(state, connection)
        state.models_cache = {
            "client_id": connection.client_id,
            "subject": connection.subject,
            "fetched_at": time.time(),
            "models": models,
        }
    if payload.model not in {model["slug"] for model in models}:
        raise PublicError(
            "unknown_model",
            "The requested model is not in the available model list.",
            422,
        )

    require_current(state, connection)
    try:
        stream = await client.create_response_stream(
            http, connection.access_token, payload.upstream_payload()
        )
    except AuthError:
        connection.needs_reauth = True
        raise PublicError("reauthorization_required", "Sign in again to continue.", 401)

    events = client.ManagedStream(stream, connection.access_token)
    if not state.register_stream(events, connection):
        await events.aclose()
        require_current(state, connection)

    async def body():
        try:
            async for chunk in events:
                yield chunk
        finally:
            await events.aclose()
            state.unregister_stream(events)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )
