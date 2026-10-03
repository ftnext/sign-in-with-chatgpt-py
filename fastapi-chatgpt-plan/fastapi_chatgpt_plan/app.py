"""Application factory: settings, lifespan, state, and route assembly."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import client, registrations, routes_api, routes_auth
from .config import Settings
from .errors import ApiError, PublicError, UsageLimitError
from .oauth import OAuth
from .registrations import StorageError
from .sessions import MemoryState

logger = logging.getLogger("fastapi_chatgpt_plan")


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status
    )


def create_app(
    settings: Settings | None = None,
    *,
    http_client=None,
    expected_host: str | None = None,
    acquire_lock: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    if expected_host is None:
        expected_host = settings.loopback_host

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lock_fd = None
        if acquire_lock:
            lock_fd = registrations.acquire_process_lock(settings.state_dir)
        owned_http = http_client is None
        app.state.http = http_client or client.make_http_client(timeout=90)
        app.state.oauth = OAuth(app.state.http)
        try:
            yield
        finally:
            await app.state.chatgpt_state.close_streams()
            if owned_http:
                await app.state.http.aclose()
            if lock_fd is not None:
                import os

                os.close(lock_fd)

    app = FastAPI(title="fastapi-chatgpt-plan", lifespan=lifespan)
    if http_client is not None:
        app.state.http = http_client
        app.state.oauth = OAuth(http_client)
    app.state.settings = settings
    app.state.chatgpt_state = MemoryState()
    app.state.registration = registrations.load_registration(
        settings.state_dir
    ) or {}
    app.state.registration["host_id"] = registrations.ensure_host_id(
        settings.state_dir
    )
    app.state.expected_host = expected_host
    app.state.expected_origin = f"http://{expected_host}"

    @app.middleware("http")
    async def host_guard(request: Request, call_next):
        expected = app.state.expected_host
        if expected and request.headers.get("host") != expected:
            return _error("invalid_host", "Unexpected Host header.", 403)
        return await call_next(request)

    @app.exception_handler(PublicError)
    async def public_error_handler(request: Request, exc: PublicError):
        return _error(exc.code, exc.message, exc.status_code)

    @app.exception_handler(UsageLimitError)
    async def usage_limit_handler(request: Request, exc: UsageLimitError):
        return _error("usage_limit", str(exc), 429)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return _error("upstream_error", str(exc), 502)

    @app.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError):
        return _error("storage_error", str(exc), 500)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return _error("invalid_request", "Invalid request.", 422)

    app.include_router(routes_auth.router)
    app.include_router(routes_api.router)
    return app
