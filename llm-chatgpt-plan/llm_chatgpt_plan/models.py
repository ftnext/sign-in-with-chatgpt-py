"""llm.Model implementation backed by the ChatGPT plan Responses API."""

import click
import httpx
import llm
from pydantic import Field

from . import client, messages, storage
from .client import ApiError
from .oauth import AuthError, OAuth
from .storage import StorageError

MODEL_ID_PREFIX = "chatgpt-plan/"
NOTICE = "Using ChatGPT plan · " + client.USAGE_URL

AUTH_GUIDANCE = {
    "sign_in_required": "Not signed in. Run: llm chatgpt-plan login",
    "reauthorization_required": "Sign-in required again. Run: llm chatgpt-plan login",
    "plan_permission_required": (
        "ChatGPT plan usage was not granted. Check your usage sharing "
        "settings: " + client.USAGE_URL
    ),
}


class ChatGPTPlan(llm.Model):
    can_stream = True

    def __init__(self, slug, display_name=None):
        self.model_id = MODEL_ID_PREFIX + slug
        self.slug = slug
        self.display_name = display_name

    class Options(llm.Options):
        reasoning_effort: str | None = Field(
            default=None,
            description=(
                "Reasoning effort passed to reasoning.effort, e.g. none, "
                "minimal, low, medium, high, xhigh, max. Which values a "
                "model accepts depends on the model."
            ),
        )

    def execute(self, prompt, stream, response, conversation):
        self._validate_request(prompt, stream, conversation)
        input_items, instructions = messages.responses_input(prompt)
        token = None
        try:
            with client.make_http_client(timeout=90) as http:
                try:
                    with storage.locked_store() as store:
                        self._check_connection_state(store)
                        token = OAuth(http).ensure_access_token_locked(store)
                except AuthError as exc:
                    raise llm.ModelError(_auth_message(exc)) from exc
                click.echo(NOTICE, err=True)
                outcome = {}
                yield from client.stream_response(
                    http,
                    token,
                    self.slug,
                    input_items,
                    instructions=instructions,
                    reasoning_effort=prompt.options.reasoning_effort,
                    outcome=outcome,
                )
                usage = outcome.get("usage") or {}
                if any(value is not None for value in usage.values()):
                    response.set_usage(**usage)
        except ApiError as exc:
            raise llm.ModelError(str(exc)) from exc
        except StorageError as exc:
            raise llm.ModelError(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise llm.ModelError(
                "HTTP error: " + client.describe_status_error(exc, secret=token)
            ) from exc
        except httpx.HTTPError as exc:
            raise llm.ModelError(f"Could not reach the API: {exc}") from exc

    @staticmethod
    def _validate_request(prompt, stream, conversation):
        if not stream:
            raise llm.ModelError(
                "chatgpt-plan models only support streaming; remove --no-stream"
            )
        if prompt.attachments:
            raise llm.ModelError("chatgpt-plan does not support attachments")
        if prompt.schema is not None:
            raise llm.ModelError(
                "chatgpt-plan does not support schemas or structured output"
            )
        if prompt.tools:
            raise llm.ModelError("chatgpt-plan does not support tools")
        if prompt.tool_results:
            raise llm.ModelError("chatgpt-plan does not support tool results")

    def _check_connection_state(self, store):
        try:
            credentials = store.load_credentials()
            cached = store.load_models()
        except StorageError as exc:
            raise llm.ModelError(
                f"Could not read chatgpt-plan state: {exc}. Run "
                "llm chatgpt-plan login to sign in again."
            ) from exc
        if not credentials or not credentials.get("access_token"):
            raise llm.ModelError("Not signed in. Run: llm chatgpt-plan login")
        models = storage.cached_models_for(credentials, cached)
        if models is None:
            raise llm.ModelError(
                "The cached model list is missing or belongs to another "
                "connection. Run: llm chatgpt-plan models --refresh"
            )
        if self.slug not in {model.get("slug") for model in models}:
            raise llm.ModelError(
                f"'{self.slug}' is not in the cached model list. Run: "
                "llm chatgpt-plan models --refresh"
            )


def _auth_message(exc: AuthError) -> str:
    code = str(exc)
    return AUTH_GUIDANCE.get(code, f"Authentication failed: {code}")
