# sign-in-with-chatgpt-py

Python tools that use Sign in with ChatGPT OAuth and the user's ChatGPT plan.
Each package lives in its own directory and has its own version and release tag.

| Directory | Purpose |
| --- | --- |
| [llm-chatgpt-plan](llm-chatgpt-plan/README.md) | An [LLM](https://llm.datasette.io/) plugin for single streaming prompts using ChatGPT plan usage |
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
The plugin uses its own directory inside the LLM user directory; credentials
are not shared or migrated between them.

For plugin development, run commands from `llm-chatgpt-plan/`:

```bash
cd llm-chatgpt-plan
uv run pytest
uv run ruff check .
uv build
```

The first package release tag is `llm-chatgpt-plan-0.1.0`. Publishing the
corresponding GitHub Release triggers the package's release workflow.
