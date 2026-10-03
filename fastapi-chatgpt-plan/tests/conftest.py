import json
import time
from urllib.parse import parse_qs, parse_qsl, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from fastapi_chatgpt_plan import Settings, create_app
from fastapi_chatgpt_plan.oauth import (
    ISSUER,
    PLAN_SCOPES,
    RESOURCE,
)

ISSUED_CLIENT_ID = "issued-client-123"
SUBJECT = "user-subject-1"
ORIGIN = "http://testserver"


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


def token_payload(
    rsa_key,
    client_id,
    nonce="nonce",
    subject=SUBJECT,
    scope=PLAN_SCOPES,
    refresh_token="refresh-token-1",
    access_token="access-token-1",
    expires_in=3600,
):
    payload = {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "scope": scope,
        "id_token": make_id_token(
            rsa_key, client_id=client_id, subject=subject, nonce=nonce
        ),
    }
    if refresh_token is not None:
        payload["refresh_token"] = refresh_token
    return payload


DISCOVERY = {
    "issuer": ISSUER,
    "authorization_endpoint": ISSUER + "/api/accounts/authorize",
    "token_endpoint": ISSUER + "/api/accounts/oauth/token",
    "jwks_uri": ISSUER + "/api/accounts/jwks",
    "revocation_endpoint": ISSUER + "/api/accounts/oauth/revoke",
}


class FakeServers:
    """httpx.MockTransport handler covering auth.openai.com and api.openai.com."""

    def __init__(self, rsa_key, jwks, nonce_holder=None):
        self.rsa_key = rsa_key
        self.jwks = jwks
        self.nonce_holder = nonce_holder if nonce_holder is not None else {}
        self.token_requests = []
        self.token_status = 200
        self.token_error = {"error": "invalid_grant"}
        self.scope = PLAN_SCOPES
        self.next_access_token = "access-token-1"
        self.next_refresh_token = "refresh-token-1"
        self.id_token_override = None
        self.revoke_requests = []
        self.revoke_status = 200
        self.models = [
            {
                "slug": "gpt-6.1-sol",
                "display_name": "GPT 6.1 Sol",
                "visibility": "list",
            },
            {"slug": "gpt-5-mini", "visibility": "list"},
            {"slug": "hidden-model", "visibility": "internal"},
        ]
        self.models_status = 200
        self.models_error = {"error": {"code": "bad", "message": "nope"}}
        self.models_requests = []
        self.responses_payloads = []
        self.response_events = None
        self.responses_status = 200
        self.responses_error = {"detail": "admission denied"}
        self.responses_stream = None

    def token_response(self, form):
        if self.token_status != 200:
            return httpx.Response(self.token_status, json=self.token_error)
        access = self.next_access_token
        refresh = self.next_refresh_token
        payload = token_payload(
            self.rsa_key,
            form["client_id"],
            nonce=self.nonce_holder.get("nonce", "nonce"),
            scope=self.scope,
            access_token=access,
            refresh_token=refresh,
        )
        if self.id_token_override is not None:
            payload["id_token"] = self.id_token_override
        self.next_access_token = access + "-next"
        self.next_refresh_token = refresh + "-next"
        return httpx.Response(200, json=payload)

    def sse_body(self):
        events = self.response_events or []
        return "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == ISSUER + "/.well-known/openid-configuration":
            return httpx.Response(200, json=DISCOVERY)
        if url == DISCOVERY["jwks_uri"]:
            return httpx.Response(200, json=self.jwks)
        if url == DISCOVERY["token_endpoint"]:
            form = dict(parse_qsl(request.content.decode()))
            self.token_requests.append(form)
            return self.token_response(form)
        if url == DISCOVERY["revocation_endpoint"]:
            form = dict(parse_qsl(request.content.decode()))
            self.revoke_requests.append(form)
            return httpx.Response(self.revoke_status)
        if url == RESOURCE + "/models":
            self.models_requests.append(request)
            if self.models_status != 200:
                return httpx.Response(self.models_status, json=self.models_error)
            return httpx.Response(200, json={"models": self.models})
        if url == RESOURCE + "/responses":
            self.responses_payloads.append(json.loads(request.content))
            if self.responses_status != 200:
                return httpx.Response(
                    self.responses_status, json=self.responses_error
                )
            if self.responses_stream is not None:
                return httpx.Response(
                    200,
                    stream=self.responses_stream,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                content=self.sse_body(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, json={"detail": "not found"})

    def client(self, **kwargs) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self.handler), **kwargs
        )


@pytest.fixture
def servers(rsa_key, jwks):
    return FakeServers(rsa_key, jwks)


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


def build_app(servers, state_dir, plan_enabled=True, **kwargs):
    settings = Settings(
        chatgpt_plan_enabled=plan_enabled,
        chatgpt_app_port=8000,
        chatgpt_state_dir=str(state_dir),
    )
    kwargs.setdefault("expected_host", "testserver")
    kwargs.setdefault("acquire_lock", False)
    return create_app(settings, http_client=servers.client(), **kwargs)


@pytest.fixture
def app(servers, state_dir):
    return build_app(servers, state_dir, plan_enabled=True)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def bootstrap(client):
    """GET /api/session so the browser has a session + CSRF token."""
    response = client.get("/api/session")
    assert response.status_code == 200
    return response.json()


def begin_login(client):
    info = bootstrap(client)
    response = client.post(
        "/auth/login",
        data={"csrf_token": info["csrf"]},
        headers={"origin": ORIGIN},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return response.headers["location"]


def finish_login(
    client,
    url,
    nonce_holder,
    client_id=ISSUED_CLIENT_ID,
    state=None,
    error=None,
    code="auth-code-1",
):
    params = parse_qs(urlsplit(url).query)
    nonce_holder["nonce"] = params["nonce"][0]
    query = f"state={state if state is not None else params['state'][0]}"
    if error:
        query += f"&error={error}"
    else:
        query += f"&code={code}&client_id={client_id}"
    return client.get("/auth/callback?" + query, follow_redirects=False)


def sign_in(client, servers, nonce_holder=None):
    holder = nonce_holder if nonce_holder is not None else servers.nonce_holder
    url = begin_login(client)
    response = finish_login(client, url, holder)
    assert response.status_code == 303, response.text
    assert response.headers["location"] == "/"
    return holder


def post_csrf(client, path, **kwargs):
    info = client.get("/api/session").json()
    headers = kwargs.pop("headers", {})
    headers.setdefault("origin", ORIGIN)
    headers.setdefault("x-csrf-token", info["csrf"])
    return client.post(path, headers=headers, **kwargs)


def make_connection(
    client_id=ISSUED_CLIENT_ID,
    subject=SUBJECT,
    scopes=None,
    expires_at=None,
    refresh_token="refresh-token-1",
    **extra,
):
    from fastapi_chatgpt_plan.sessions import Connection

    return Connection(
        client_id=client_id,
        issuer=ISSUER,
        subject=subject,
        email="user@example.com",
        name="Test User",
        id_token="id-token-1",
        access_token="access-token-1",
        refresh_token=refresh_token,
        scopes=list(scopes if scopes is not None else PLAN_SCOPES.split()),
        expires_at=(
            expires_at if expires_at is not None else time.time() + 3600
        ),
        generation=0,
        plan_permitted=True,
        **extra,
    )


COMPLETED_RESPONSE = {
    "id": "resp-1",
    "created_at": 1,
    "model": "gpt-6.1-sol",
    "object": "response",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "status": "completed",
    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
}


def delta_event(text, sequence=1):
    return {
        "type": "response.output_text.delta",
        "delta": text,
        "item_id": "item-1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": sequence,
    }


def completed_event(sequence=3):
    return {
        "type": "response.completed",
        "response": dict(COMPLETED_RESPONSE),
        "sequence_number": sequence,
    }


def failed_event(code="server_error", message="upstream failed", sequence=2):
    response = dict(COMPLETED_RESPONSE)
    response["status"] = "failed"
    response["error"] = {"code": code, "message": message}
    return {
        "type": "response.failed",
        "response": response,
        "sequence_number": sequence,
    }
