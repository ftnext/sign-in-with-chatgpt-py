# fastapi-chatgpt-plan

`fastapi-chatgpt-plan`: a standalone FastAPI backend that uses your ChatGPT
plan through **Sign in with ChatGPT** OAuth.

No OpenAI API key is needed. The server signs in with your ChatGPT account,
keeps the OAuth tokens in memory, and proxies model listing and Responses API
calls to the same origin. It is meant to run on your own machine and serve a
frontend on the same origin (`http://127.0.0.1:<port>`).

Supported platforms: macOS and Linux (the process lock uses `fcntl`). One
ChatGPT connection at a time, one process, one worker.

## Setup

```bash
cd fastapi-chatgpt-plan
uv sync
```

## Running

Identity-only sign-in (no plan inference), using an issued **public identity
client** with `http://127.0.0.1:8000/auth/callback` registered:

```bash
CHATGPT_IDENTITY_CLIENT_ID=oaiapp_your_identity_client uv run fastapi-chatgpt-plan
```

ChatGPT plan inference enabled:

```bash
CHATGPT_PLAN_ENABLED=true uv run fastapi-chatgpt-plan
```

An unregistered installation in the default identity-only mode displays setup
instructions and returns `409 identity_registration_required` from login,
without redirecting to a dynamic registration that fails with `invalid_client`.
It never silently requests plan scopes. The
[identity public-client flow](https://developers.openai.com/siwc/website)
requires a provisioned client; this package cannot issue one. It accepts an
ID-token-only response and sends no API resource in that flow. Confidential
identity clients are not supported.

Alternatively, explicitly enable plan inference to register an OSS client and
approve plan permissions. After that registration, restarting with the default
`CHATGPT_PLAN_ENABLED=false` reuses the verified client for identity-only login,
as confirmed with a real account. A provisioned identity client and an OSS plan
client must use separate state directories; the app rejects attempts to switch
client identities within one registration.

The CLI binds `127.0.0.1` only and runs a single uvicorn worker. A second
instance using the same state directory is refused via a process lock, so
registration data stays single-writer.

Open `http://127.0.0.1:8000/` to check sign-in status, sign in, sign out, and
list models — this minimal page exists only to verify authentication until a
real frontend arrives.

## Configuration

Environment variables are read once at startup; invalid values abort startup
with a clear error.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHATGPT_PLAN_ENABLED` | `false` | Enable inference and model listing; selects the OAuth scopes |
| `CHATGPT_APP_PORT` | `8000` | Loopback port (`1024`–`65535`) |
| `CHATGPT_STATE_DIR` | `~/.local/share/fastapi-chatgpt-plan` | Where registration data is stored (`~` expanded) |
| `CHATGPT_IDENTITY_CLIENT_ID` | unset | Provisioned public identity client for a new identity-only installation |

With `CHATGPT_PLAN_ENABLED=false` the server requests only
`openid profile email` and never contacts the models/responses endpoints or
refresh flow. With `true` it additionally requests
`offline_access resource.invoke chatgpt.tokens.use.direct`. A successful
sign-in without the plan scope leaves you signed in but inference-disabled —
re-sign in and approve plan access.

## API

All state-changing POSTs require the session cookie plus an
`x-csrf-token` header carrying the `csrf` value from `GET /api/session`
(or a `csrf_token` form field). The `Origin` header must match the server
origin; same-origin browser `fetch` calls satisfy this automatically.
The `Host` header is restricted to `127.0.0.1:<port>`.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/` | Minimal sign-in check page |
| GET | `/health` | Liveness only: `{"status": "ok"}` |
| GET | `/api/session` | Status, public user info, plan flags, CSRF token |
| POST | `/auth/login` | Start browser-bound sign-in; 303 redirect to OpenAI |
| GET | `/auth/callback` | OAuth callback (redirect target; not for direct use) |
| POST | `/auth/logout` | End the local session; best-effort remote revocation |
| GET | `/api/models` | List available models |
| POST | `/api/responses` | Stream a Responses call as SSE |

`GET /api/session` bootstraps an anonymous session cookie on first contact
and returns:

```json
{
  "status": "anonymous",
  "user": null,
  "plan": {"enabled": true, "permitted": false},
  "csrf": "<opaque-csrf-token>"
}
```

`status` is `anonymous`, `authenticated`, or `reauthorization_required`.
`plan.enabled` is the server configuration; `plan.permitted` is whether the
current connection actually carries the plan scope. Neither the session ID
nor any OAuth token is ever included in responses.

`GET /api/models` returns (filtered to upstream `visibility == "list"`,
order preserved):

```json
{"models": [{"slug": "<model-slug>", "display_name": "<display name>"}]}
```

### POST /api/responses

```json
{
  "model": "<model-slug>",
  "input": [
    {"role": "user", "content": "最初の質問"},
    {"role": "assistant", "content": "前の回答"},
    {"role": "user", "content": "続きの質問"}
  ],
  "instructions": "簡潔に日本語で回答する",
  "reasoning": {"effort": "low"}
}
```

- `model` and a non-empty `input` are required. `input` is limited to
  `user`/`assistant` text messages; the full conversation history is sent
  every time (nothing is stored server-side).
- `instructions` and `reasoning.effort` are optional and omitted upstream
  when absent. Accepted efforts: `none`, `minimal`, `low`, `medium`, `high`,
  `xhigh`, `max` — which values a model accepts depends on the model.
- `stream` may only be `true`, `store` only `false` (both are optional).
  The upstream call always uses `stream: true`, `store: false`.
- Unknown fields are rejected with 422 rather than dropped. Attachments,
  tools, structured output, `previous_response_id`, and `conversation` are
  outside this contract.
- Limits: 1 MiB request body (413), 200 messages and 200,000 total text
  characters including `instructions` (422). These are transport limits, not
  a model context guarantee; nothing is truncated automatically.
- `model` must appear in the upstream model list, which is fetched on demand
  and cached in memory per connection.

### SSE responses

`POST /api/responses` answers with `Content-Type: text/event-stream` and
`Cache-Control: no-store`. Upstream Responses API events are serialized and
forwarded incrementally — the answer is never buffered:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta", ..., "delta":"回答の一部"}

event: response.completed
data: {"type":"response.completed", ...}
```

Only `response.completed` counts as success. `response.failed`,
`response.incomplete`, and upstream `error` retain their event names and JSON
structure, with credential strings redacted. Locally detected EOF or transport
failures use `event: error` with `stream_incomplete` or `stream_interrupted`.
For an upstream failure, inspect `response.error` or `response.incomplete_details`;
plan usage codes `subscription_sharing_usage_limit_exceeded` and
`subscription_sharing_usage_unavailable` point to
[ChatGPT usage settings](https://chatgpt.com/settings/usage).

**Errors before the stream starts are normal HTTP errors** (see below);
once streaming has begun, failures arrive as SSE terminal events even though
the HTTP status is 200. Logout or browser cancellation may end the connection
without a terminal event; clients must not report such an EOF as success.

### JavaScript example

`EventSource` cannot POST, so use `fetch()` and consume the body. SSE chunks
can split anywhere — including inside multi-byte UTF-8 characters and across
event boundaries — so decode incrementally and split on blank lines only:

```js
const session = await (await fetch("/api/session")).json();

const response = await fetch("/api/responses", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-csrf-token": session.csrf,
  },
  body: JSON.stringify({
    model: "<model-slug>",
    input: [{ role: "user", content: "Say hello" }],
  }),
});
if (!response.ok) {
  const { error } = await response.json();
  throw new Error(`${error.code}: ${error.message}`);
}

const reader = response.body
  .pipeThrough(new TextDecoderStream())
  .getReader();

let buffer = "";
let answer = "";
let completed = false;
try {
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += value;
  let index;
  while ((index = buffer.indexOf("\n\n")) !== -1) {
    const block = buffer.slice(0, index);
    buffer = buffer.slice(index + 2);
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (event === "error") {
      const { code, message } = JSON.parse(data);
      throw new Error(`${code}: ${message}`);
    }
    if (event === "response.failed" || event === "response.incomplete") {
      const payload = JSON.parse(data).response;
      throw new Error(JSON.stringify(payload.error || payload.incomplete_details));
    }
    if (event === "response.output_text.delta") {
      answer += JSON.parse(data).delta; // stream text incrementally
    }
    if (event === "response.completed") {
      completed = true;
      // final usage/metadata in JSON.parse(data)
    }
  }
}
if (!completed) throw new Error("Stream ended before response.completed");
} finally {
  await reader.cancel(); // also cancels upstream if parsing/rendering fails
  reader.releaseLock();
}
```

## Errors

Errors before SSE starts use a stable JSON shape:

```json
{"error": {"code": "inference_disabled", "message": "ChatGPT plan inference is disabled."}}
```

| Condition | HTTP | Code examples |
| --- | --- | --- |
| No/expired session, not signed in, re-auth needed | 401 | `session_required`, `sign_in_required`, `reauthorization_required` |
| Inference disabled or plan scope not granted | 403 | `inference_disabled`, `plan_permission_required` |
| CSRF, Origin, or Host check failed | 403 | `invalid_csrf_token`, `invalid_origin`, `invalid_host` |
| Identity registration missing or incompatible | 409 | `identity_registration_required`, `registration_mismatch`, `plan_registration_required` |
| Unsupported input/parameters, unknown model | 422 | `invalid_request`, `invalid_json`, `unknown_model` |
| Body over the size limit | 413 | `request_too_large` |
| Plan usage unavailable/exhausted (pre-stream) | 429 | `usage_limit` |
| Upstream rejection or connection failure | 502 | `upstream_error` |

Usage-limit errors point to
[ChatGPT usage settings](https://chatgpt.com/settings/usage). Mid-stream
failures are reported inside SSE, never as HTTP errors and never as fabricated
success events. Upstream failure events keep their original names and payloads;
locally detected failures use `event: error`.

## What is stored

Two separate places:

- **Memory only** — access/refresh/ID tokens, scopes and expiry, app
  sessions, CSRF state, OAuth transactions, the model cache, and in-flight
  streams. All of it vanishes on restart: you sign in again after every
  server restart. Stopping the server discards the tokens; it does not
  revoke the registration with OpenAI.
- **`CHATGPT_STATE_DIR`** — `registration.json` (schema version, persistent
  host ID, issued client ID, client kind, verified issuer/subject/email/name) and
  `runtime.lock` (the single-process lock). Directory mode `0700`, file mode
  `0600`, atomic writes, symlink-safe. No tokens, cookies, PKCE verifiers,
  codes, or conversation data are written anywhere.

Because the registration survives restarts, re-login reuses the same host ID
and issued client ID instead of registering a new client each time.

The session cookie holds an opaque random ID (HttpOnly, SameSite=Lax,
Path=/; `Secure` stays off on loopback HTTP). It rotates at sign-in
completion and expires 24 hours after the last authenticated access.

Access tokens refresh automatically before model/responses calls; concurrent
refreshes are serialized, the rotated refresh token is applied immediately,
and a refresh racing a logout or re-login cannot revive a stale connection.
If refresh fails the session becomes `reauthorization_required` — sign in
again. There is no API-key fallback and no automatic request retry.

`POST /auth/logout` stops new calls, ends in-flight streams, discards the
session and tokens, and attempts remote refresh-token revocation
(`remote_revocation` in the response is `confirmed`, `not_confirmed`, or
`not_attempted`). The registration file is kept.
Only a session authenticated for the current connection can log it out.
Local disconnection precedes remote revocation; a later sign-in is protected
from delayed logout/refresh results and cookie deletion. Pending callbacks from
an older connection generation cannot restore it. Browser disconnection also
cancels an outstanding upstream read and closes the stream.

## Development

```bash
cd fastapi-chatgpt-plan  # from the repository root
uv run pytest    # mock-HTTP tests, no real credentials needed
uv run ruff check .
uv build
```

Tests use `httpx.MockTransport` with a fake authorization server and a test
RSA key for JWKS/JWT verification; they never touch real credentials or a
real account.

Verified with fastapi 0.142, uvicorn 0.54, httpx 0.28, openai 3.24, PyJWT
2.15, pydantic 2.13 on Python 3.14.

## Releases

This package is released independently using tags named
`fastapi-chatgpt-plan-<version>`; the version must match `project.version`
in `pyproject.toml`. The first release tag is `fastapi-chatgpt-plan-0.1.0`.
