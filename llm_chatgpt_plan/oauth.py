"""Sign in with ChatGPT OAuth flow: authorization, validation, refresh.

Synchronous port of the reference implementation in ``scripts/auth.py``
and ``scripts/ask.py`` for a single active connection.
"""

import base64
import hashlib
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt

from .errors import (
    REAUTH_CODES,
    ApiError,
    AuthError,
    describe_status_error,
    redact,
    status_error_code,
)
from .storage import locked_store, read_credentials

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = "llm-chatgpt-plan"
REFRESH_MARGIN_SECONDS = 60
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 600


class OAuth:
    def __init__(self, http: httpx.Client):
        self.http = http
        self.discovery = None
        self.keys = None
        self.keys_at = 0

    def metadata(self):
        if self.discovery is None:
            response = self.http.get(ISSUER + "/.well-known/openid-configuration")
            response.raise_for_status()
            metadata = response.json()
            if metadata.get("issuer") != ISSUER:
                raise AuthError("invalid_issuer")
            for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
                parsed = urlsplit(metadata[key])
                if parsed.scheme != "https" or parsed.netloc != "auth.openai.com":
                    raise AuthError("invalid_discovery")
            self.discovery = metadata
        return self.discovery

    def begin(self, redirect_uri, host_id, client_id=None, prior=None):
        """Build an authorization URL and the transaction to verify it."""
        metadata = self.metadata()
        verifier = secrets.token_urlsafe(64)
        tx = {
            "state": secrets.token_urlsafe(32),
            "nonce": secrets.token_urlsafe(32),
            "verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "expires": time.time() + 600,
        }
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        params = {
            "client_id": client_id or DYNAMIC_CLIENT_ID,
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": tx["state"],
            "nonce": tx["nonce"],
            "code_challenge_method": "S256",
            "code_challenge": challenge.decode().rstrip("="),
        }
        if not client_id:
            params["agent_name_hint"] = AGENT_NAME_HINT
        elif prior:
            # Hints keep tokens out of the terminal: never log the URL.
            if prior.get("id_token"):
                params["id_token_hint"] = prior["id_token"]
            if prior.get("email"):
                params["login_hint"] = prior["email"]
        return tx, metadata["authorization_endpoint"] + "?" + urlencode(params)

    def verify(self, token, client_id, nonce=None):
        try:
            metadata = self.metadata()
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise AuthError("invalid_id_token")
            key = None
            for attempt in range(2):
                if self.keys is None or time.time() - self.keys_at > 3600 or attempt:
                    response = self.http.get(metadata["jwks_uri"])
                    response.raise_for_status()
                    self.keys = jwt.PyJWKSet.from_dict(response.json())
                    self.keys_at = time.time()
                key = next(
                    (k for k in self.keys.keys if k.key_id == header.get("kid")),
                    None,
                )
                if key is not None:
                    break
            if key is None:
                raise AuthError("invalid_id_token")
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256"],
                issuer=ISSUER,
                audience=client_id,
                leeway=5,
                options={"require": ["iss", "aud", "sub", "exp", "iat"]},
            )
            if nonce is not None and claims.get("nonce") != nonce:
                raise AuthError("invalid_nonce")
            if not isinstance(claims["sub"], str) or not claims["sub"]:
                raise AuthError("invalid_subject")
            if "azp" in claims and claims["azp"] != client_id:
                raise AuthError("invalid_audience")
            if (
                isinstance(claims["aud"], list)
                and len(claims["aud"]) > 1
                and claims.get("azp") != client_id
            ):
                raise AuthError("invalid_audience")
            return claims
        except (jwt.PyJWTError, KeyError, ValueError, TypeError) as exc:
            raise AuthError("invalid_id_token") from exc

    def token_request(self, form):
        metadata = self.metadata()
        secrets_to_hide = tuple(
            form.get(k) for k in ("refresh_token", "code", "code_verifier")
        )
        try:
            response = self.http.post(metadata["token_endpoint"], data=form)
        except httpx.HTTPError as exc:
            raise ApiError(
                redact(f"Could not reach the auth server: {exc}", secrets_to_hide)
            ) from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if (
                response.status_code in (400, 401, 403)
                and status_error_code(exc) in REAUTH_CODES
            ):
                raise AuthError("reauthorization_required") from exc
            raise ApiError(
                "Token request failed: "
                + describe_status_error(
                    exc,
                    secrets=secrets_to_hide,
                )
            ) from exc
        return response.json()

    def credentials(self, tokens, old_scopes=None):
        scope_value = tokens.get("scope")
        if scope_value is None:
            scopes = list(old_scopes or [])
        elif isinstance(scope_value, str):
            scopes = scope_value.split()
        else:
            raise AuthError("invalid_token_response")
        if tokens.get("token_type", "").lower() != "bearer":
            raise AuthError("invalid_token_response")
        if (
            not isinstance(tokens.get("access_token"), str)
            or not tokens["access_token"]
        ):
            raise AuthError("plan_permission_required")
        if (
            not isinstance(tokens.get("refresh_token"), str)
            or not tokens["refresh_token"]
        ):
            raise AuthError("invalid_token_response")
        try:
            expires = float(tokens["expires_in"])
            if not 0 < expires <= 86400:
                raise ValueError()
        except (KeyError, ValueError, TypeError) as exc:
            raise AuthError("invalid_token_response") from exc
        return {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "scopes": scopes,
            "expires_at": time.time() + expires,
        }

    def complete(self, tx, query, store, persist_registration, prior_subject=None):
        """Validate the callback, exchange the code, return the credential record.

        ``persist_registration`` keeps the issued client ID from a first-time
        dynamic registration even when the exchange later fails. It is off for
        ``login --replace`` so a failed attempt never disturbs the current
        connection.
        """
        if tx is None or tx["expires"] <= time.time():
            raise AuthError("expired_sign_in")
        if not secrets.compare_digest(query.get("state", ""), tx["state"]):
            raise AuthError("invalid_state")
        if query.get("error"):
            raise AuthError("sign_in_declined")
        client_id = query.get("client_id") or tx["client_id"]
        if not client_id or client_id == DYNAMIC_CLIENT_ID:
            raise AuthError("missing_client_id")
        if tx["client_id"] and client_id != tx["client_id"]:
            raise AuthError("client_id_mismatch")
        if not query.get("code"):
            raise AuthError("missing_code")
        if persist_registration and not tx["client_id"]:
            existing = store.load_credentials() or {}
            if existing.get("client_id") != client_id:
                # Keep only the issued ID: anything else on file cannot be
                # trusted to belong to this registration.
                store.save_credentials({"client_id": client_id})
        tokens = self.token_request(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": query["code"],
                "code_verifier": tx["verifier"],
                "redirect_uri": tx["redirect_uri"],
                "resource": RESOURCE,
            }
        )
        identity = self.verify(tokens.get("id_token", ""), client_id, tx["nonce"])
        if prior_subject and prior_subject != identity["sub"]:
            raise AuthError("account_mismatch")
        record = dict(
            client_id=client_id,
            issuer=ISSUER,
            subject=identity["sub"],
            email=identity.get("email"),
            name=identity.get("name"),
            id_token=tokens["id_token"],
            **self.credentials(tokens),
        )
        return record

    def ensure_access_token(self, store_dir):
        """Return a usable access token, refreshing under lock when needed."""
        credentials = read_credentials(store_dir) or {}
        self._check_connection(credentials)
        if credentials["expires_at"] > time.time() + REFRESH_MARGIN_SECONDS:
            return credentials["access_token"]
        with locked_store(store_dir) as store:
            return self.ensure_access_token_locked(store)

    def ensure_access_token_locked(self, store):
        """Refresh-if-needed against a store whose lock the caller holds."""
        credentials = store.load_credentials() or {}
        # Another process may have refreshed while we waited on the lock.
        self._check_connection(credentials)
        if credentials["expires_at"] <= time.time() + REFRESH_MARGIN_SECONDS:
            tokens = self.token_request(
                {
                    "grant_type": "refresh_token",
                    "client_id": credentials["client_id"],
                    "refresh_token": credentials["refresh_token"],
                    "resource": RESOURCE,
                }
            )
            if tokens.get("id_token"):
                identity = self.verify(tokens["id_token"], credentials["client_id"])
                if identity["sub"] != credentials["subject"]:
                    raise AuthError("account_mismatch")
            credentials.update(self.credentials(tokens, credentials.get("scopes")))
            if tokens.get("id_token"):
                credentials["id_token"] = tokens["id_token"]
            store.save_credentials(credentials)
        if PLAN_SCOPE not in credentials["scopes"]:
            raise AuthError("plan_permission_required")
        return credentials["access_token"]

    @staticmethod
    def _check_connection(credentials):
        if not credentials.get("access_token"):
            raise AuthError("sign_in_required")
        for key in ("client_id", "subject", "refresh_token"):
            if not credentials.get(key):
                raise AuthError("sign_in_required")
        if not isinstance(credentials.get("expires_at"), (int, float)):
            raise AuthError("sign_in_required")
        if PLAN_SCOPE not in credentials.get("scopes", []):
            raise AuthError("plan_permission_required")


class CallbackListener:
    """One browser authorization result, received on a private loopback listener."""

    def __init__(self, port=0):
        self.result = None
        self.expected_state = None
        self._received = threading.Event()
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Callback URLs contain a credential: do not log them.

            def do_GET(self):
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                valid = (
                    self.headers.get("Host") == listener.host
                    and urlsplit(self.path).path == "/auth/callback"
                    and all(len(values) == 1 for values in query.values())
                    and listener.expected_state is not None
                    and secrets.compare_digest(
                        query.get("state", [""])[0], listener.expected_state
                    )
                )
                body = (
                    "Authorization received. Return to your terminal for the result."
                    if valid
                    else "This authorization result could not be verified."
                ).encode()
                self.send_response(200 if valid else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)
                if valid and listener.result is None:
                    listener.result = {key: values[0] for key, values in query.items()}
                    listener._received.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.host = f"127.0.0.1:{self.server.server_port}"
        self.redirect_uri = f"http://{self.host}/auth/callback"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self, state):
        self.expected_state = state
        self.thread.start()

    def wait(self, timeout):
        """Wait for the authorization result; returns None on timeout."""
        if self._received.wait(timeout):
            return self.result
        return None

    def close(self):
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()


def login_flow(
    store,
    http,
    *,
    replace=False,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    port=0,
    open_browser=webbrowser.open,
):
    """Run one browser sign-in against the store's single connection.

    ``replace`` starts a fresh dynamic registration (for switching accounts);
    the previous connection is only overwritten after the new one verifies.
    Returns the verified credential record.
    """
    oauth = OAuth(http)
    existing = store.load_credentials() or {}
    client_id = None if replace else existing.get("client_id")
    prior = existing if client_id else None
    listener = CallbackListener(port)
    try:
        tx, url = oauth.begin(
            listener.redirect_uri,
            host_id=store.ensure_host_id(),
            client_id=client_id,
            prior=prior,
        )
        listener.start(tx["state"])
        if not open_browser(url):
            raise RuntimeError(
                "Could not open a browser. Set a default browser and try again."
            )
        query = listener.wait(timeout)
        if query is None:
            raise TimeoutError(
                f"Timed out waiting for sign-in after {timeout} seconds. "
                "Run llm chatgpt-plan login again."
            )
        record = oauth.complete(
            tx,
            query,
            store,
            persist_registration=not replace,
            prior_subject=prior.get("subject") if prior else None,
        )
        if PLAN_SCOPE not in record["scopes"]:
            raise AuthError("plan_permission_required")
        store.save_credentials(record)
        return record
    finally:
        listener.close()
