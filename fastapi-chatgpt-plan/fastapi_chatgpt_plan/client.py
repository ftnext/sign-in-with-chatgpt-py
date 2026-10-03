"""Async HTTP client pieces: model listing and streaming Responses calls."""

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
FAILURE_EVENTS = frozenset({"response.failed", "response.incomplete", "error"})


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
    data = response.json()
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
            "The API rejected the request: "
            + describe_status_error(exc, secret=token)
        ) from exc
    except APIConnectionError as exc:
        raise ApiError(redact(f"Could not reach the API: {exc}", (token,))) from exc
    except OpenAIError as exc:
        raise ApiError(redact(f"API request failed: {exc}", (token,))) from exc


def _event_error(event):
    response = getattr(event, "response", None)
    error = getattr(response, "error", None) if response is not None else None
    code = getattr(error, "code", None) or getattr(event, "code", None)
    message = getattr(error, "message", None) or getattr(event, "message", None)
    return code, message


def _sse(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


def sse_error_event(code: str, message: str) -> str:
    return _sse("error", {"type": "error", "code": code, "message": message})


async def sse_events(stream, token):
    """Serialize upstream stream events as SSE without buffering.

    Terminal failures become an app ``error`` event; a stream that ends
    without any terminal event is never reported as success.
    """
    from openai import OpenAIError

    terminal = False
    try:
        async for event in stream:
            event_type = event.type
            if event_type in FAILURE_EVENTS:
                terminal = True
                code, message = _event_error(event)
                if code in LIMIT_CODES:
                    yield sse_error_event(
                        "usage_limit",
                        "ChatGPT plan usage is unavailable or exhausted. "
                        f"Check your usage sharing settings: {USAGE_URL}",
                    )
                else:
                    yield sse_error_event(
                        "upstream_failed",
                        redact(
                            f"Response stream ended as {event_type}"
                            + (f" ({code}: {message})" if code or message else ""),
                            (token,),
                        ),
                    )
                return
            if event_type == "response.completed":
                terminal = True
            yield _sse(event_type, event.model_dump(mode="json"))
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
        await stream.close()
