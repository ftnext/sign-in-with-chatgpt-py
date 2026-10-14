# chatgpt-auth

Sign in to a ChatGPT Plus/Pro/Team account once and get OpenAI access tokens
with automatic refresh — for CLIs, bots, and server jobs.

Tokens and refresh tokens are stored on the local filesystem with strict
permissions. Nothing about this package talks to OpenAI except the OAuth
flow itself.

## Install

```console
$ pip install chatgpt-auth
```

## Quick start

Sign in once from a machine with a browser:

```console
$ chatgpt-auth login
```

Then get an access token anywhere (expired tokens are refreshed and saved
automatically):

```console
$ TOKEN=$(chatgpt-auth print-access-token)
$ curl -sS https://api.openai.com/v1/models -H "Authorization: Bearer $TOKEN"
```

From Python:

```python
from chatgpt_auth import Credentials

credentials = Credentials.load()
token = credentials.get_access_token()  # refreshes + saves if needed
```

Force a refresh or inspect state:

```console
$ chatgpt-auth refresh
$ chatgpt-auth status
$ chatgpt-auth logout
```

## CLI

- `login` — run the browser sign-in (`--manual` pastes the URL/redirect
  instead; `--replace` replaces an existing connection; `--timeout`,
  `--port`, `--agent-name` tune the flow).
- `status` — show the local connection state (offline, never prints tokens).
- `print-access-token` — print the current access token (refreshing first
  if needed). This is the only command that prints a token.
- `refresh` — force a token refresh.
- `logout` — wipe local tokens and try to revoke the refresh token
  remotely. Local sign-out happens even if revocation cannot be
  confirmed; to fully sign out everywhere also revoke the session in
  ChatGPT Settings.

All commands take `--credentials PATH` to point at a credentials file.

## Credentials file resolution

`Credentials.load()` / the CLI resolve the credentials path in this order:

1. The explicit `path` argument / `--credentials` option.
2. The `CHATGPT_AUTH_CREDENTIALS` environment variable.
3. `${XDG_CONFIG_HOME:-~/.config}/chatgpt-auth/credentials.json`
   (a relative `XDG_CONFIG_HOME` is ignored per the XDG spec).

Relative paths resolve against the current working directory; `~` is
expanded. Reserved aux names (`host.json`, `pending.json`, `runtime.lock`,
`login.lock`, `models.json`) are rejected as credentials paths.

## Auto refresh and saved tokens

`get_access_token()` re-checks the store on every call and refreshes when
the token expires within ~60 seconds, saving the new token before
returning it. When a refresh's outcome is unknowable (e.g. the network
died mid-request) the connection is retired rather than risking a
double-submit of a single-use refresh token; the next call raises
`ReauthorizationRequiredError` and `login` is required again. A write
failure after a successful refresh returns no token — the object refuses
further use if even retiring the connection cannot be persisted.

## Sharing a store with llm-chatgpt-plan

Point `--credentials`/`CHATGPT_AUTH_CREDENTIALS` at the existing
`credentials.json` used by `llm-chatgpt-plan` to share the same connection,
`host.json`, `pending.json`, and both lock files — so sign-ins and token
use in both tools serialize correctly. This package never touches the
`models.json` model cache; a new connection bumps `generation`, which is
how `llm-chatgpt-plan` invalidates its cached model list. Files named
anything other than `credentials.json` get their own `.<name>-suffixed`
aux files and share nothing with other stores.

## Concurrent use

Multiple processes may hold `Credentials` for the same path. Refresh
requests collapse onto a shared `fcntl.flock` runtime lock, so two
processes do not double-submit a refresh. If another process signs out
or signs in underneath you, the next call raises
`ReauthorizationRequiredError` or `ConnectionChangedError`.

## OS support

macOS and Linux (POSIX `fcntl` locking). Windows is not supported in this
version.

## Security notes

- The credentials directory must be owned by you and not group/world
  readable; token files are written `0600`.
- Tokens, refresh tokens, authorization codes, and PKCE verifiers never
  appear in logs or exception messages.
- `logout` clears local state first so an interrupted run still leaves
  the connection dead, then revokes the refresh token; a broken
  credentials file is the limit of what local sign-out can guarantee.

## API summary

```python
import chatgpt_auth

credentials = chatgpt_auth.Credentials.load()       # local file only
token = credentials.get_access_token()              # auto refresh + save
token = credentials.refresh()                       # forced refresh
credentials = chatgpt_auth.login()                  # browser sign-in
status = chatgpt_auth.status()                      # AuthStatus (local only)
result = chatgpt_auth.logout()                      # LogoutResult
```

Exceptions (`chatgpt_auth` namespace): `SignInRequiredError`,
`ReauthorizationRequiredError`, `ConnectionChangedError`,
`PlanPermissionError`, `NetworkError`, `StorageError`, `LoginTimeoutError`,
`AuthError`, and the base `ChatGPTAuthError`.

## Development

Run from this package directory:

```console
$ uv sync
$ uv run pytest
$ uv run ruff check .
$ uv build
```

## Releases

Publish a GitHub Release tagged `chatgpt-auth-<version>` matching
`project.version` in `pyproject.toml`. The first release tag is
`chatgpt-auth-0.1.0`. The `publish-chatgpt-auth.yml` workflow verifies the
tag, tests Python 3.11–3.14, builds this package's wheel and sdist, and
publishes them to PyPI using Trusted Publishing. Releases for other
packages are skipped.
