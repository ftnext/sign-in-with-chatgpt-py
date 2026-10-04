# llm-chatgpt-plan

`llm-chatgpt-plan`: an [LLM](https://llm.datasette.io/) plugin that uses your
ChatGPT plan through **Sign in with ChatGPT** OAuth.

No OpenAI API key is needed. The plugin signs in with your ChatGPT account,
stores the OAuth credentials locally, and calls the Responses API within your
plan's usage allowance.

Supported platforms: macOS and Linux (the credential lock uses `fcntl`).

## Installation

Install this plugin into the same environment as `llm`:

```bash
llm install llm-chatgpt-plan
```

Or from a checkout of this repository:

```bash
llm install -e /path/to/sign-in-with-chatgpt-py/llm-chatgpt-plan
```

## Usage

Sign in once. A browser opens for ChatGPT sign-in and plan approval:

```bash
llm chatgpt-plan login
```

No working browser on this machine? Pass `--manual` to print the sign-in
URL instead (a failed browser launch falls back to this automatically).
Complete the sign-in in any browser; if the redirect back to this machine
does not reach it, paste the full callback URL from the browser's address
bar when prompted (interactive terminals only):

```bash
llm chatgpt-plan login --manual
```

This fetches your account's model list and saves it. List the saved models
(any model `llm` sees is also registered as `chatgpt-plan/<slug>`):

```bash
llm chatgpt-plan models
llm models
```

Fetch the model list again (also refreshes the access token if needed):

```bash
llm chatgpt-plan models --refresh
```

Send a prompt — answers stream to stdout:

```bash
llm -m chatgpt-plan/gpt-5.6-luna 'Say hello'
```

A system prompt (`-s`) is sent to the Responses API as `instructions`:

```bash
llm -m chatgpt-plan/gpt-5.6-luna -s 'Answer concisely in Japanese' 'What is OAuth?'
```

Request a reasoning effort with `-o reasoning_effort` (for example `none`,
`minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Which values a model
accepts depends on the model; unsupported values return the API's error.
When omitted, no `reasoning` field is sent and the model's default applies:

```bash
llm -m chatgpt-plan/gpt-5.6-luna -o reasoning_effort low 'What is OAuth?'
```

Continue a conversation with `-c`/`--continue` or `llm chat`; earlier text
turns are resent to the API on each call (the plugin never uses
`previous_response_id`):

```bash
llm -m chatgpt-plan/gpt-5.6-luna -c 'And the same for PKCE?'
```

Sign out, revoking the session remotely when possible:

```bash
llm chatgpt-plan logout
```

## What is stored

State lives in a `chatgpt-plan` directory inside the LLM user directory —
the directory that contains the `logs.db` shown by `llm logs path`:

- `credentials.json` — issued client ID, verified subject/issuer, access and
  refresh tokens, granted scopes, expiry, and a connection generation that
  changes on sign-in replacement and sign-out (mode `0600`). After
  `logout` it keeps only the client ID, identity, and generation
- `pending.json` — a client ID issued by dynamic registration whose code
  exchange did not finish, tied to the connection generation it was issued
  for; the next sign-in reuses or discards it (mode `0600`)
- `host.json` — a persistent host ID sent during dynamic client registration
- `models.json` — the cached model list, tied to the connection's client ID
  and subject
- `runtime.lock` — serializes updates between processes
- `login.lock` — serializes sign-in attempts only, so a pending sign-in
  never blocks prompts or token refreshes

One connection is active at a time. To switch to a different account or
workspace, run `llm chatgpt-plan login --replace`, which only replaces the
saved connection after the new sign-in verifies.

Access tokens are refreshed automatically when they are about to expire, and
the rotated credentials are saved atomically. If the refresh token has expired
or was revoked, sign in again with `llm chatgpt-plan login`. For temporary
network, rate-limit, or server errors, retry later; the saved connection is kept.

## Limitations of this first version

These are rejected with a clear error instead of being silently ignored:

- Non-streaming requests (`--no-stream`): this route requires streaming
- Attachments, tools, tool results, and structured output (`--schema`)
- Conversations containing non-text turns (tool calls, tool results,
  attachments): continuation is limited to plain text turns
- Multiple saved accounts: one connection is active at a time

`llm` still records prompts and responses in its local log database as usual.
That local logging is separate from the `store=False` flag sent to the API,
which asks OpenAI not to store the request server-side.

Usage allowance errors (`subscription_sharing_usage_limit_exceeded` /
`subscription_sharing_usage_unavailable`) point to your ChatGPT usage
settings: https://chatgpt.com/settings/usage

## Disconnecting

`llm chatgpt-plan logout` revokes the refresh token through the discovery
document's revocation endpoint and removes local tokens, scopes, expiry,
and the model cache, keeping only the client ID and verified identity for
the next sign-in. When the remote revocation cannot be confirmed, the local
sign-out still completes; to disconnect fully in that case, use ChatGPT →
**Settings → Security and login → Sign in with ChatGPT**.

## Reference scripts

`../scripts/auth.py` and `../scripts/ask.py` are the original standalone reference
implementation (PEP 723 scripts, run with `uv run`). They keep their own
credential storage under `scripts/.chatgpt-script` and are **not** shared with
the plugin.

## Development

```bash
cd llm-chatgpt-plan  # from the repository root
uv run pytest    # mock-HTTP tests, no real credentials needed
uv run ruff check .
```

Tests use `httpx.MockTransport`, a test RSA key for JWKS/JWT verification, the
Click test runner, and an isolated `LLM_USER_PATH`. They never touch real
credentials.

Verified with llm 0.36, openai 3.21, httpx 0.28, PyJWT 2.15 on Python 3.14.

## Releases

This package is released independently using tags named
`llm-chatgpt-plan-<version>`. The version must match `project.version` in
`pyproject.toml`; the first release tag is `llm-chatgpt-plan-0.1.0`.

Publishing a GitHub Release for a matching tag runs the tests, verifies the
tag against the package version, builds this directory, and publishes its
distributions to PyPI using the repository's `release` environment and
Trusted Publishing. Releases for other packages are skipped. The PyPI
Trusted Publisher must be configured for this repository, `publish.yml`,
and the `release` environment before publishing.
