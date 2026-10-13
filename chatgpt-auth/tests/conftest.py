import json
import os
import time
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from chatgpt_auth import service, storage
from chatgpt_auth.oauth import ISSUER, SCOPES

ISSUED_CLIENT_ID = "issued-client-123"
SUBJECT = "user-subject-1"


@pytest.fixture
def auth_dir(tmp_path, monkeypatch):
    """An isolated storage directory; env override is cleared."""
    monkeypatch.delenv("CHATGPT_AUTH_CREDENTIALS", raising=False)
    return tmp_path / "auth"


@pytest.fixture
def creds_path(auth_dir):
    return auth_dir / "credentials.json"


@pytest.fixture
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def jwks(rsa_key):
    jwk = json.loads(RSAAlgorithm.to_jwk(rsa_key.public_key()))
    jwk.update(kid="test-key", use="sig", alg="RS256")
    return {"keys": [jwk]}


def make_id_token(
    rsa_key,
    client_id=ISSUED_CLIENT_ID,
    subject=SUBJECT,
    nonce="nonce",
    issuer=ISSUER,
    kid="test-key",
    **extra,
):
    claims = {
        "iss": issuer,
        "aud": client_id,
        "sub": subject,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "nonce": nonce,
        "email": "user@example.com",
        "name": "Test User",
    }
    claims.update(extra)
    headers = {"kid": kid} if kid else {}
    return jwt.encode(claims, rsa_key, algorithm="RS256", headers=headers)


def token_payload(rsa_key, client_id, nonce="nonce", subject=SUBJECT, scope=SCOPES):
    return {
        "access_token": "access-token-1",
        "refresh_token": "refresh-token-1",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": scope,
        "id_token": make_id_token(
            rsa_key, client_id=client_id, subject=subject, nonce=nonce
        ),
    }


DISCOVERY = {
    "issuer": ISSUER,
    "authorization_endpoint": ISSUER + "/api/accounts/authorize",
    "token_endpoint": ISSUER + "/api/accounts/oauth/token",
    "jwks_uri": ISSUER + "/api/accounts/jwks",
    "revocation_endpoint": ISSUER + "/api/accounts/oauth/revoke",
}


class FakeAuthServer:
    """httpx.MockTransport handler emulating auth.openai.com."""

    def __init__(self, rsa_key, jwks, nonce_holder=None):
        self.rsa_key = rsa_key
        self.jwks = jwks
        self.nonce_holder = nonce_holder if nonce_holder is not None else {}
        self.token_requests = []
        self.id_token_override = None
        self.token_status = 200
        self.token_error = {"error": "invalid_grant"}
        self.token_delay = 0
        self.revoke_requests = []
        self.revoke_statuses = []
        self.revoke_body = b""

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == ISSUER + "/.well-known/openid-configuration":
            return httpx.Response(200, json=DISCOVERY)
        if url == DISCOVERY["jwks_uri"]:
            return httpx.Response(200, json=self.jwks)
        if url == DISCOVERY["token_endpoint"]:
            if self.token_delay:
                time.sleep(self.token_delay)
            form = dict(parse_qsl(request.content.decode()))
            self.token_requests.append(form)
            if self.token_status != 200:
                return httpx.Response(self.token_status, json=self.token_error)
            client_id = form["client_id"]
            payload = token_payload(
                self.rsa_key,
                client_id,
                nonce=self.nonce_holder.get("nonce", "nonce"),
            )
            if self.id_token_override is not None:
                payload["id_token"] = (
                    self.id_token_override(
                        client_id, self.nonce_holder.get("nonce", "nonce")
                    )
                    if callable(self.id_token_override)
                    else self.id_token_override
                )
            return httpx.Response(200, json=payload)
        if url == DISCOVERY["revocation_endpoint"]:
            self.revoke_requests.append(dict(parse_qsl(request.content.decode())))
            if self.revoke_statuses:
                status = self.revoke_statuses.pop(0)
            else:
                status = 200
            return httpx.Response(status, content=self.revoke_body)
        return httpx.Response(404, json={"detail": "not found"})

    def client(self, **kwargs) -> httpx.Client:
        kwargs.setdefault("transport", httpx.MockTransport(self.handler))
        return httpx.Client(**kwargs)


@pytest.fixture
def auth_server(rsa_key, jwks):
    return FakeAuthServer(rsa_key, jwks)


@pytest.fixture
def http_factory(auth_server):
    return auth_server.client


def drive_browser_callback(
    url,
    client_id=ISSUED_CLIENT_ID,
    code="auth-code-1",
    state_override=None,
    error=None,
    nonce_holder=None,
):
    """Simulate the browser completing the authorize redirect."""
    params = parse_qs(urlsplit(url).query)
    if nonce_holder is not None:
        nonce_holder["nonce"] = params["nonce"][0]
    redirect_uri = params["redirect_uri"][0]
    state = state_override if state_override is not None else params["state"][0]
    query = f"state={state}"
    if error:
        query += f"&error={error}"
    else:
        query += f"&code={code}&client_id={client_id}"
    return httpx.get(f"{redirect_uri}?{query}")


@pytest.fixture
def stored_connection(creds_path):
    """A verified, connected record saved via the new store."""
    record = {
        "client_id": ISSUED_CLIENT_ID,
        "issuer": ISSUER,
        "subject": SUBJECT,
        "email": "user@example.com",
        "name": "Test User",
        "id_token": "stored-id-token",
        "access_token": "stored-access",
        "refresh_token": "stored-refresh",
        "scopes": SCOPES.split(),
        "expires_at": time.time() + 3600,
    }
    store = storage.Store(creds_path)
    store.save_credentials(record)
    return record


@pytest.fixture
def patched_http(monkeypatch, auth_server):
    """Point the service layer's default HTTP factory at the fake server."""
    monkeypatch.setattr(service, "make_http_client", auth_server.client)
    return auth_server


class FakeTTY:
    """A pipe-backed stdin stand-in that reports as a terminal."""

    def __init__(self):
        self._read_fd, self._write_fd = os.pipe()

    def isatty(self):
        return True

    def fileno(self):
        return self._read_fd

    def feed(self, text):
        os.write(self._write_fd, (text + "\n").encode())

    def close_write(self):
        os.close(self._write_fd)
        self._write_fd = -1

    def close(self):
        os.close(self._read_fd)
        if self._write_fd >= 0:
            os.close(self._write_fd)


def paste_url(authorize_url, **params):
    """Build the loopback callback URL for the attempt in authorize_url."""
    parsed = parse_qs(urlsplit(authorize_url).query)
    query = {"state": parsed["state"][0], **params}
    return parsed["redirect_uri"][0] + "?" + urlencode(query), parsed["nonce"][0]
