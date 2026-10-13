"""Sign in with ChatGPT OAuth protocol: discovery, validation, refresh.

Synchronous protocol pieces ported from llm-chatgpt-plan. This module has
no storage knowledge: callers own state, locks, and persistence.
"""

import base64
import hashlib
import secrets
import time
from urllib.parse import urlencode, urlsplit

import httpx
import jwt

from .errors import (
    REAUTH_CODES,
    SAFE_REJECT_CODES,
    AuthError,
    NetworkError,
    TokenOutcomeUnknownError,
    describe_status_error,
    redact,
    status_error_code,
)

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
DEFAULT_AGENT_NAME = "chatgpt-auth"
REFRESH_MARGIN_SECONDS = 60
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 600
HTTP_TIMEOUT_SECONDS = 30
HINT_PARAMS = ("id_token_hint", "login_hint")


def make_http_client(timeout=HTTP_TIMEOUT_SECONDS) -> httpx.Client:
    """Factory kept in one place so tests can inject a mock transport."""
    return httpx.Client(timeout=timeout)


def _is_provably_unsent(exc: httpx.HTTPError) -> bool:
    """True only when the request could not have reached the server.

    Connect/pool failures mean no bytes were sent, so the grant is
    provably unconsumed. Read/write timeouts and mid-stream errors are
    ambiguous and must be treated as unknown.
    """
    return isinstance(exc, (httpx.ConnectError, httpx.PoolTimeout))


class OAuth:
    def __init__(self, http: httpx.Client):
        self.http = http
        self.discovery = None
        self.keys = None
        self.keys_at = 0

    def metadata(self):
        if self.discovery is None:
            try:
                response = self.http.get(ISSUER + "/.well-known/openid-configuration")
            except httpx.HTTPError as exc:
                raise NetworkError(f"Could not reach the auth server: {exc}") from exc
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise NetworkError(
                    "Discovery failed: " + describe_status_error(exc)
                ) from exc
            metadata = response.json()
            if metadata.get("issuer") != ISSUER:
                raise AuthError("invalid_issuer")
            for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
                parsed = urlsplit(metadata[key])
                if parsed.scheme != "https" or parsed.netloc != "auth.openai.com":
                    raise AuthError("invalid_discovery")
            revocation = metadata.get("revocation_endpoint")
            if revocation is not None:
                parsed = urlsplit(revocation)
                if parsed.scheme != "https" or parsed.netloc != "auth.openai.com":
                    raise AuthError("invalid_discovery")
            self.discovery = metadata
        return self.discovery

    def begin(self, redirect_uri, host_id, client_id=None, prior=None, agent_name=None):
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
            params["agent_name_hint"] = agent_name or DEFAULT_AGENT_NAME
        elif prior:
            # Hints keep tokens out of the terminal: never log the URL.
            if prior.get("id_token"):
                params["id_token_hint"] = prior["id_token"]
            if prior.get("email"):
                params["login_hint"] = prior["email"]
        tx["params"] = params
        return tx, metadata["authorization_endpoint"] + "?" + urlencode(params)

    def manual_url(self, tx):
        """Authorization URL safe to display: no ID token or login hints."""
        metadata = self.metadata()
        params = {
            key: value for key, value in tx["params"].items() if key not in HINT_PARAMS
        }
        return metadata["authorization_endpoint"] + "?" + urlencode(params)

    def verify(self, token, client_id, nonce=None):
        try:
            metadata = self.metadata()
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise AuthError("invalid_id_token")
            key = None
            for attempt in range(2):
                if self.keys is None or time.time() - self.keys_at > 3600 or attempt:
                    try:
                        response = self.http.get(metadata["jwks_uri"])
                        response.raise_for_status()
                    except httpx.HTTPError as exc:
                        raise NetworkError(
                            f"Could not fetch signing keys: {exc}"
                        ) from exc
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
        """POST to the token endpoint.

        Raises NetworkError when the request provably never reached the
        server or the server rejected the grant in a way that leaves it
        usable; TokenOutcomeUnknownError when the request may have been
        processed; AuthError("reauthorization_required") for revocation.
        """
        metadata = self.metadata()
        secrets_to_hide = tuple(
            form.get(k) for k in ("refresh_token", "code", "code_verifier")
        )
        try:
            response = self.http.post(metadata["token_endpoint"], data=form)
        except httpx.HTTPError as exc:
            if _is_provably_unsent(exc):
                raise NetworkError(
                    redact(f"Could not reach the auth server: {exc}", secrets_to_hide)
                ) from exc
            raise TokenOutcomeUnknownError(
                redact(f"Token request outcome is unknown: {exc}", secrets_to_hide)
            ) from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = status_error_code(exc)
            if response.status_code in (400, 401, 403) and code in REAUTH_CODES:
                raise AuthError("reauthorization_required") from exc
            if response.status_code in (400, 401, 403) and code in SAFE_REJECT_CODES:
                raise NetworkError(
                    "Token request was rejected: "
                    + describe_status_error(exc, secrets=secrets_to_hide)
                ) from exc
            raise TokenOutcomeUnknownError(
                "Token request failed: "
                + describe_status_error(exc, secrets=secrets_to_hide)
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise TokenOutcomeUnknownError("Token response was not JSON") from exc
        if not isinstance(body, dict):
            raise TokenOutcomeUnknownError("Token response was not an object")
        return body

    def credentials(self, tokens, old_scopes=None, *, refresh_response=False):
        """Validate a token endpoint response into stored credential fields.

        ``refresh_response`` relaxes the refresh_token requirement per
        RFC 6749: a refresh response may omit it, in which case the caller
        keeps the stored token.
        """
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
        if not refresh_response and (
            not isinstance(tokens.get("refresh_token"), str)
            or not tokens["refresh_token"]
        ):
            raise AuthError("invalid_token_response")
        if (
            refresh_response
            and "refresh_token" in tokens
            and (
                not isinstance(tokens["refresh_token"], str)
                or not tokens["refresh_token"]
            )
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
            "refresh_token": tokens.get("refresh_token"),
            "scopes": scopes,
            "expires_at": time.time() + expires,
        }

    def complete(self, tx, query, persist_client_id=None, prior_subject=None):
        """Validate the callback, exchange the code, return the credential record.

        ``persist_client_id`` is a callable invoked with the issued client ID
        before the code exchange, keeping first-time dynamic registrations
        recoverable even when the exchange later fails. It stays None for
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
        if persist_client_id is not None and not tx["client_id"]:
            persist_client_id(client_id)
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

    def revoke_refresh_token(self, client_id, refresh_token, *, attempts=3, sleep=None):
        """POST the refresh token to the revocation endpoint.

        Returns True only when the server confirmed with an empty 200.
        Network failures and 5xx get a short finite retry; anything else is
        a permanent "unconfirmed" so logout can still finish locally.
        """
        sleep = sleep or time.sleep
        endpoint = self.metadata().get("revocation_endpoint")
        if not endpoint:
            return False
        delay = 0.5
        for attempt in range(attempts):
            try:
                response = self.http.post(
                    endpoint,
                    data={
                        "token": refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": client_id,
                    },
                )
            except httpx.HTTPError:
                response = None
            if response is not None:
                if response.status_code == 200:
                    return True
                if 400 <= response.status_code < 500:
                    return False
            if attempt + 1 < attempts:
                sleep(delay)
                delay = min(delay * 2, 4)
        return False
