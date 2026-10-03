"""llm plugin: use a ChatGPT plan via Sign in with ChatGPT OAuth."""

import llm

from .commands import chatgpt_plan
from .models import ChatGPTPlan
from .oauth import PLAN_SCOPE
from .storage import cached_models_for, read_credentials, read_models


@llm.hookimpl
def register_commands(cli):
    cli.add_command(chatgpt_plan)


@llm.hookimpl
def register_models(register, model_aliases):
    """Register models from the saved list only - never touches the network
    and never creates or updates credential state."""
    try:
        credentials = read_credentials()
        if not credentials or not credentials.get("access_token"):
            return
        if PLAN_SCOPE not in credentials.get("scopes", []):
            return
        models = cached_models_for(credentials, read_models())
        if not models:
            return
        for model in models:
            slug = model.get("slug")
            if isinstance(slug, str) and slug:
                register(ChatGPTPlan(slug, display_name=model.get("display_name")))
    except Exception:  # noqa: BLE001 - registration must never break llm
        # A broken or partial state directory must not break llm itself.
        # `llm chatgpt-plan models` explains how to recover.
        return
