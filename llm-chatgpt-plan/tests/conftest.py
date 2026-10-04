import json
import time
from urllib.parse import parse_qs, parse_qsl, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from llm_chatgpt_plan import storage
from llm_chatgpt_plan.oauth import ISSUER, RESOURCE, SCOPES

ISSUED_CLIENT_ID = "issued-client-123"
SUBJECT = "user-subject-1"


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    """Isolate the llm user directory for the duration of a test."""
    directory = tmp_path / "llm-user"
    monkeypatch.setenv("LLM_USER_PATH", str(directory))
    return directory


@pytest.fixture
def state_dir(user_dir):
    return user_dir / storage.STATE_DIR_NAME


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
        self.revoke_requests = []
        self.revoke_statuses = []
        self.revoke_body = b""
        self.models = [
            {
                "slug": "gpt-5.6-luna",
                "display_name": "GPT 5.6 Luna",
                "visibility": "list",
            },
            {"slug": "gpt-5-mini", "visibility": "list"},
            {"slug": "hidden-model", "visibility": "internal"},
        ]
        self.models_requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == ISSUER + "/.well-known/openid-configuration":
            return httpx.Response(200, json=DISCOVERY)
        if url == DISCOVERY["jwks_uri"]:
            return httpx.Response(200, json=self.jwks)
        if url == DISCOVERY["token_endpoint"]:
            form = dict(parse_qsl(request.content.decode()))
            self.token_requests.append(form)
            if self.token_status != 200:
                return httpx.Response(self.token_status, json=self.token_error)
            client_id = form["client_id"]
            if self.id_token_override is not None:
                payload = dict(
                    token_payload(
                        self.rsa_key,
                        client_id,
                        nonce=self.nonce_holder.get("nonce", "nonce"),
                    )
                )
                payload["id_token"] = self.id_token_override
                return httpx.Response(200, json=payload)
            return httpx.Response(
                200,
                json=token_payload(
                    self.rsa_key,
                    client_id,
                    nonce=self.nonce_holder.get("nonce", "nonce"),
                ),
            )
        if url == RESOURCE + "/models":
            self.models_requests.append(request)
            return httpx.Response(200, json={"models": self.models})
        if url == DISCOVERY["revocation_endpoint"]:
            self.revoke_requests.append(dict(parse_qsl(request.content.decode())))
            if self.revoke_statuses:
                status = self.revoke_statuses.pop(0)
            else:
                status = 200
            return httpx.Response(status, content=self.revoke_body)
        return httpx.Response(404, json={"detail": "not found"})

    def client(self, **kwargs) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), **kwargs)


@pytest.fixture
def auth_server(rsa_key, jwks):
    return FakeAuthServer(rsa_key, jwks)


def drive_browser_callback(
    url,
    client_id=ISSUED_CLIENT_ID,
    code="auth-code-1",
    state_override=None,
    error=None,
    nonce_holder=None,
):
    """Simulate the browser completing the authorize redirect.

    Reads the state/redirect_uri out of the authorization URL and performs
    the loopback GET the real browser would make.
    """
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
def stored_connection(state_dir):
    """A verified connection plus its saved model list."""
    state_dir.mkdir(parents=True)
    store = storage.Store(state_dir)
    record = {
        "client_id": ISSUED_CLIENT_ID,
        "issuer": ISSUER,
        "subject": SUBJECT,
        "email": "user@example.com",
        "id_token": "stored-id-token",
        "access_token": "stored-access",
        "refresh_token": "stored-refresh",
        "scopes": SCOPES.split(),
        "expires_at": time.time() + 3600,
    }
    store.save_credentials(record)
    store.save_models(
        {
            "fetched_at": time.time(),
            "client_id": ISSUED_CLIENT_ID,
            "subject": SUBJECT,
            "models": [
                {"slug": "gpt-5.6-luna", "display_name": "GPT 5.6 Luna"},
                {"slug": "gpt-5-mini", "display_name": "gpt-5-mini"},
            ],
        }
    )
    return record
