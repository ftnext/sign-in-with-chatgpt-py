"""Async HTTP client pieces: model listing and streaming Responses calls."""

import asyncio
import contextlib
import json

import httpx

from .errors import (
    ApiError,
    AuthError,
    UsageLimitError,
    describe_status_error,
    redact,
    status_error_code,
)
from .oauth import RESOURCE

USAGE_URL = "https://chatgpt.com/settings/usage"
LIMIT_CODES = frozenset(
    {
        "subscription_sharing_usage_limit_exceeded",
        "subscription_sharing_usage_unavailable",
    }
)
TERMINAL_EVENTS = frozenset(
    {"response.completed", "response.failed", "response.incomplete", "error"}
)


def usage_limit_error() -> UsageLimitError:
    return UsageLimitError(
        "ChatGPT plan usage is unavailable or exhausted. "
        f"Check your usage sharing settings: {USAGE_URL}"
    )


def make_http_client(timeout=30) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


async def fetch_models(http: httpx.AsyncClient, token: str) -> list:
    """GET /v1/models keeping entries visible in the account's list."""
    try:
        response = await http.get(
            RESOURCE + "/models", headers={"Authorization": "Bearer " + token}
        )
    except httpx.HTTPError as exc:
        raise ApiError(redact(f"Could not reach the API: {exc}", (token,))) from exc
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if status_error_code(exc) in LIMIT_CODES:
            raise usage_limit_error() from exc
        if response.status_code == 401:
            raise AuthError("reauthorization_required") from exc
        raise ApiError(
            "Could not fetch models: " + describe_status_error(exc, secret=token)
        ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise ApiError("Unexpected model list response from the API") from exc
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ApiError("Unexpected model list response from the API")
    return [
        {
            "slug": model["slug"],
            "display_name": model.get("display_name") or model["slug"],
        }
        for model in models
        if isinstance(model, dict)
        and model.get("visibility") == "list"
        and isinstance(model.get("slug"), str)
    ]


async def create_response_stream(http, token, payload):
    """Open the upstream Responses stream; rejects surface as errors here."""
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

    client = AsyncOpenAI(
        api_key=token,
        base_url=RESOURCE,
        http_client=http,
        max_retries=0,
    )
    try:
        return await client.responses.create(**payload)
    except APIStatusError as exc:
        if status_error_code(exc) in LIMIT_CODES:
            raise usage_limit_error() from exc
        if exc.status_code == 401:
            raise AuthError("reauthorization_required") from exc
        raise ApiError(
            "The API rejected the request: " + describe_status_error(exc, secret=token)
        ) from exc
    except APIConnectionError as exc:
        raise ApiError(redact(f"Could not reach the API: {exc}", (token,))) from exc
    except OpenAIError as exc:
        raise ApiError(redact(f"API request failed: {exc}", (token,))) from exc


def _sse(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


def sse_error_event(code: str, message: str) -> str:
    return _sse("error", {"type": "error", "code": code, "message": message})


class ManagedStream:
    """Cancel an outstanding read before closing its async generator."""

    def __init__(self, stream, token):
        self.upstream = stream
        self.events = sse_events(stream, token, close_upstream=False)
        self.pending = None
        self.closed = False
        self.close_lock = asyncio.Lock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        pending = asyncio.create_task(anext(self.events))
        self.pending = pending
        try:
            result = await pending
            if self.closed:
                raise StopAsyncIteration
            return result
        except asyncio.CancelledError:
            if self.closed:
                raise StopAsyncIteration
            raise
        finally:
            if self.pending is pending:
                self.pending = None

    async def aclose(self):
        async with self.close_lock:
            if self.closed:
                return
            self.closed = True
            if self.pending is not None:
                self.pending.cancel()
                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                    await self.pending
            await self.events.aclose()
            # Also covers a stream opened before the response iterator ever starts.
            await self.upstream.close()


def _safe_payload(value, token):
    if isinstance(value, dict):
        return {key: _safe_payload(item, token) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_payload(item, token) for item in value]
    return redact(value, (token,)) if isinstance(value, str) else value


async def sse_events(stream, token, *, close_upstream=True):
    """Serialize upstream stream events as SSE without buffering.

    Preserve upstream terminal events; only locally detected failures use
    app ``error`` events. An EOF without a terminal event is not success.
    """
    from openai import OpenAIError

    terminal = False
    try:
        async for event in stream:
            event_type = event.type
            terminal = event_type in TERMINAL_EVENTS
            yield _sse(event_type, _safe_payload(event.model_dump(mode="json"), token))
            if terminal:
                return
        if not terminal:
            yield sse_error_event(
                "stream_incomplete",
                "Response stream ended without a terminal event.",
            )
    except (OpenAIError, httpx.HTTPError) as exc:
        yield sse_error_event(
            "stream_interrupted",
            redact(f"Response stream was interrupted: {exc}", (token,)),
        )
    finally:
        if close_upstream:
            await stream.close()
