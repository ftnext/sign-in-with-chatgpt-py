"""HTTP client pieces: model listing and streaming Responses API calls."""

import httpx

from .errors import (
    ApiError,
    AuthError,
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


def make_http_client(timeout=30) -> httpx.Client:
    """Factory kept in one place so tests can inject a mock transport."""
    return httpx.Client(timeout=timeout)


def usage_limit_error() -> ApiError:
    return ApiError(
        "ChatGPT plan usage is unavailable or exhausted. "
        f"Check your usage sharing settings: {USAGE_URL}"
    )


def fetch_models(http: httpx.Client, token: str) -> list:
    """GET /v1/models and keep entries visible in the account's list."""
    try:
        response = http.get(
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
        model
        for model in models
        if isinstance(model, dict) and model.get("visibility") == "list"
    ]


def _event_error(event):
    """Pull (code, message) from a failed/incomplete/error stream event."""
    response = getattr(event, "response", None)
    error = getattr(response, "error", None) if response is not None else None
    code = getattr(error, "code", None) or getattr(event, "code", None)
    message = getattr(error, "message", None) or getattr(event, "message", None)
    return code, message


def stream_response(
    http,
    token,
    slug,
    input_items,
    instructions=None,
    reasoning_effort=None,
    outcome=None,
):
    """Yield output text deltas from a streaming Responses API call.

    ``outcome`` is an optional dict that receives ``usage`` on completion.
    Raises ApiError on HTTP rejection and on failed/incomplete/truncated
    streams; partial output is never treated as success.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=token,  # OAuth token sent as Bearer, not a Platform API key.
        base_url=RESOURCE,
        http_client=http,
        max_retries=0,
    )
    kwargs = {
        "model": slug,
        "input": input_items,
        "store": False,
        "stream": True,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    def generate():
        from openai import APIConnectionError, APIStatusError, OpenAIError

        completed = False
        try:
            stream = client.responses.create(**kwargs)
        except APIStatusError as exc:
            if status_error_code(exc) in LIMIT_CODES:
                raise usage_limit_error() from exc
            raise ApiError(
                "The API rejected the request: "
                + describe_status_error(exc, secret=token)
            ) from exc
        except APIConnectionError as exc:
            raise ApiError(redact(f"Could not reach the API: {exc}", (token,))) from exc
        except OpenAIError as exc:
            raise ApiError(redact(f"API request failed: {exc}", (token,))) from exc
        try:
            for event in stream:
                event_type = event.type
                if event_type == "response.output_text.delta":
                    yield event.delta
                elif event_type == "response.completed":
                    completed = True
                    usage = getattr(event.response, "usage", None)
                    if outcome is not None and usage is not None:
                        outcome["usage"] = {
                            "input": getattr(usage, "input_tokens", None),
                            "output": getattr(usage, "output_tokens", None),
                        }
                    break
                elif event_type in {
                    "response.failed",
                    "response.incomplete",
                    "error",
                }:
                    code, message = _event_error(event)
                    if code in LIMIT_CODES:
                        raise usage_limit_error()
                    detail = f" ({code}: {message})" if code or message else ""
                    raise ApiError(
                        redact(
                            f"Response stream ended as {event_type}{detail}", (token,)
                        )
                    )
        except ApiError:
            raise
        except (OpenAIError, httpx.HTTPError) as exc:
            raise ApiError("Response stream was interrupted mid-response") from exc
        finally:
            stream.close()
        if not completed:
            raise ApiError("Response stream ended without a completed event")

    return generate()
