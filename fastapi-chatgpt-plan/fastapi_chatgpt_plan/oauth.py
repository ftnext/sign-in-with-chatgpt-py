"""Sign in with ChatGPT OAuth: authorization, validation, refresh, revocation.

Async variant organized for the FastAPI backend: tokens live in memory,
only registration data is persisted by the caller.
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
    ApiError,
    AuthError,
    describe_status_error,
    redact,
    status_error_code,
)

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
IDENTITY_SCOPES = "openid profile email"
PLAN_SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = "fastapi-chatgpt-plan"
REFRESH_MARGIN_SECONDS = 60


class OAuth:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.discovery = None
        self.keys = None
        self.keys_at = 0

    async def metadata(self):
        if self.discovery is None:
            response = await self.http.get(
                ISSUER + "/.well-known/openid-configuration"
            )
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

    async def begin(
        self, tx, host_id, scope, client_id=None, prior=None
    ):
        """Build the authorization URL for a prepared transaction."""
        metadata = await self.metadata()
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(tx.verifier.encode()).digest()
        )
        params = {
            "client_id": client_id or DYNAMIC_CLIENT_ID,
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": tx.redirect_uri,
            "scope": scope,
            "resource": RESOURCE,
            "state": tx.state,
            "nonce": tx.nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge.decode().rstrip("="),
        }
        if not client_id:
            params["agent_name_hint"] = AGENT_NAME_HINT
        elif prior:
            if prior.get("id_token"):
                params["id_token_hint"] = prior["id_token"]
            if prior.get("email"):
                params["login_hint"] = prior["email"]
        return metadata["authorization_endpoint"] + "?" + urlencode(params)

    async def verify(self, token, client_id, nonce=None):
        try:
            metadata = await self.metadata()
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise AuthError("invalid_id_token")
            key = None
            for attempt in range(2):
                if self.keys is None or time.time() - self.keys_at > 3600 or attempt:
                    response = await self.http.get(metadata["jwks_uri"])
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

    async def token_request(self, form):
        metadata = await self.metadata()
        secrets_to_hide = tuple(
            form.get(k) for k in ("refresh_token", "code", "code_verifier")
        )
        try:
            response = await self.http.post(metadata["token_endpoint"], data=form)
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
                + describe_status_error(exc, secrets=secrets_to_hide)
            ) from exc
        return response.json()

    @staticmethod
    def credentials(tokens, old_scopes=None):
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
        refresh_token = tokens.get("refresh_token")
        if refresh_token is not None and not isinstance(refresh_token, str):
            raise AuthError("invalid_token_response")
        try:
            expires = float(tokens["expires_in"])
            if not 0 < expires <= 86400:
                raise ValueError()
        except (KeyError, ValueError, TypeError) as exc:
            raise AuthError("invalid_token_response") from exc
        return {
            "access_token": tokens["access_token"],
            "refresh_token": refresh_token,
            "scopes": scopes,
            "expires_at": time.time() + expires,
        }

    async def complete(self, tx, query, prior_subject=None):
        """Validate the callback, exchange the code, return token + identity."""
        if tx is None or tx.expires_at <= time.time():
            raise AuthError("expired_sign_in")
        if not secrets.compare_digest(query.get("state", ""), tx.state):
            raise AuthError("invalid_state")
        if query.get("error"):
            raise AuthError("sign_in_declined")
        client_id = query.get("client_id") or tx.client_id
        if not client_id or client_id == DYNAMIC_CLIENT_ID:
            raise AuthError("missing_client_id")
        if tx.client_id and client_id != tx.client_id:
            raise AuthError("client_id_mismatch")
        if not query.get("code"):
            raise AuthError("missing_code")
        tokens = await self.token_request(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": query["code"],
                "code_verifier": tx.verifier,
                "redirect_uri": tx.redirect_uri,
                "resource": RESOURCE,
            }
        )
        identity = await self.verify(
            tokens.get("id_token", ""), client_id, tx.nonce
        )
        if prior_subject and prior_subject != identity["sub"]:
            raise AuthError("account_mismatch")
        return client_id, identity, self.credentials(tokens), tokens.get("id_token")

    async def refresh(self, connection):
        """Run one refresh_token grant; returns the replacement token fields."""
        tokens = await self.token_request(
            {
                "grant_type": "refresh_token",
                "client_id": connection.client_id,
                "refresh_token": connection.refresh_token,
                "resource": RESOURCE,
            }
        )
        id_token = tokens.get("id_token")
        if id_token:
            identity = await self.verify(id_token, connection.client_id)
            if identity["sub"] != connection.subject:
                raise AuthError("account_mismatch")
        fields = self.credentials(tokens, connection.scopes)
        fields["id_token"] = id_token or connection.id_token
        return fields

    async def revoke(self, client_id, refresh_token):
        """Attempt renewable-session revocation; returns True when confirmed."""
        metadata = await self.metadata()
        endpoint = metadata.get("revocation_endpoint")
        if not endpoint:
            return False
        try:
            response = await self.http.post(
                endpoint,
                data={
                    "token": refresh_token,
                    "token_type_hint": "refresh_token",
                    "client_id": client_id,
                },
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200
