# sign-in-with-chatgpt-py

Python tools that use Sign in with ChatGPT OAuth and the user's ChatGPT plan.
Each package lives in its own directory and has its own version and release tag.

| Directory | Purpose |
| --- | --- |
| [llm-chatgpt-plan](llm-chatgpt-plan/README.md) | An [LLM](https://llm.datasette.io/) plugin for single streaming prompts using ChatGPT plan usage |
| [fastapi-chatgpt-plan](fastapi-chatgpt-plan/README.md) | A standalone FastAPI backend (single local process, same origin) wrapping sign-in, model listing, and Responses streaming |
| [scripts](scripts/) | Original standalone reference scripts (`auth.py` and `ask.py`, run with `uv run`) |

Install the plugin from this checkout:

```bash
llm install -e ./llm-chatgpt-plan
llm chatgpt-plan login
llm chatgpt-plan models
```

See the [package README](llm-chatgpt-plan/README.md) for usage, limitations,
credential storage, development, and release instructions.

The reference scripts keep their credentials under `scripts/.chatgpt-script`.
The plugin uses its own directory inside the LLM user directory; the FastAPI
backend keeps registration data under `~/.local/share/fastapi-chatgpt-plan`
by default (tokens stay in memory only). Credentials are not shared or
migrated between any of them.

For package development, run commands from the package directory:

```bash
cd llm-chatgpt-plan        # or: cd fastapi-chatgpt-plan
uv run pytest
uv run ruff check .
uv build
```

Run the FastAPI backend locally:

```bash
cd fastapi-chatgpt-plan
uv sync
CHATGPT_IDENTITY_CLIENT_ID=oaiapp_your_identity_client uv run fastapi-chatgpt-plan
# or, with ChatGPT plan inference enabled:
CHATGPT_PLAN_ENABLED=true uv run fastapi-chatgpt-plan
```

Then open `http://127.0.0.1:8000/` to sign in. See the
[package README](fastapi-chatgpt-plan/README.md) for configuration, the API
contract, a JavaScript streaming example, error handling, and what is
stored.

The first package release tag is `llm-chatgpt-plan-0.1.0`. Publishing the
corresponding GitHub Release triggers the package's release workflow.
