# openapi

[`openapi.yaml`](openapi.yaml) is an OpenAPI 3.1 description of the HTTP
endpoints the packages in this repository call for Sign in with ChatGPT and
ChatGPT plan inference. It is a starting point for generating clients in other
languages.

It is unofficial: it is derived from this repository's code, not from an
OpenAI-published contract, and only covers the fields this repository uses.

## What it covers

| Server | Path | Purpose |
| --- | --- | --- |
| `https://auth.openai.com` | `GET /.well-known/openid-configuration` | Discovery metadata |
| `https://auth.openai.com` | `GET /.well-known/jwks.json` | ID token signing keys |
| `https://auth.openai.com` | `GET /oauth/authorize` | Authorization URL (opened in a browser) |
| `https://auth.openai.com` | `POST /oauth/token` | `authorization_code` and `refresh_token` grants |
| `https://auth.openai.com` | `POST /oauth/revoke` | Refresh token revocation |
| `https://api.openai.com/v1` | `GET /models` | Models available to the ChatGPT plan |
| `https://api.openai.com/v1` | `POST /responses` | Streamed Responses API (SSE) |

The loopback redirect the client itself serves is described as
`components.pathItems.AuthorizationCallback`. `components.securitySchemes`
defines the OAuth 2.0 flow and scopes.

## What OpenAPI cannot express

A generated client handles the HTTP calls; a hand-written layer still has to:

- Generate the PKCE verifier (`S256` challenge, unpadded base64url),
  `state`, and `nonce`, and keep them for up to 10 minutes.
- Open the authorization URL in a browser and receive the redirect on a
  loopback listener (`http://127.0.0.1:{port}/auth/callback`), comparing
  `state` in constant time.
- Use `client_id=dynamic_agent_client` for the first sign-in, take the issued
  client ID from the callback query, and use it for every later token,
  refresh, and revocation request.
- Use endpoint URLs from discovery (the paths in the spec are the usual
  values) and reject any that are not `https://auth.openai.com`.
- Verify the ID token: RS256 via JWKS (refetch once on an unknown `kid`),
  `iss`, `aud`/`azp` equal to the client ID, `nonce`, and a non-empty `sub`
  that must not change between refreshes.
- Refresh about 60 seconds before `expires_in`, keep the old scopes when
  `scope` is omitted, and require `chatgpt.tokens.use.direct` before
  inference.
- Parse the `text/event-stream` body of `/responses` into
  `ResponseStreamEvent`s (referenced as `x-event-schema`) and treat a stream
  without `response.completed` as a failure.

Some generators ignore path-level `servers`; if yours does, set the base URL
to `https://api.openai.com/v1` for the `inference` tag.

## Validate and generate

```bash
uvx openapi-spec-validator openapi/openapi.yaml
uvx openapi-python-client generate --path openapi/openapi.yaml
```
