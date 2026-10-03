import time
from http.cookies import SimpleCookie
from types import SimpleNamespace

import httpx
import pytest
from conftest import DISCOVERY, begin_login, finish_login, post_csrf, sign_in

from fastapi_chatgpt_plan.errors import AuthError
from fastapi_chatgpt_plan.guards import MODELS_CACHE_TTL_SECONDS
from fastapi_chatgpt_plan.sessions import (
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    MemoryState,
)


async def test_new_session_purges_expired_state():
    state = MemoryState()
    abandoned = await state.create_session()
    active = await state.create_session()
    expired_tx = await state.begin_transaction(abandoned.id, "v", "uri", None)
    active_tx = await state.begin_transaction(active.id, "v", "uri", None)
    abandoned.expires_at = expired_tx.expires_at = time.time() - 1
    await state.create_session()
    assert abandoned.id not in state.sessions
    assert active.id in state.sessions
    assert expired_tx.state not in state.transactions
    assert active_tx.state in state.transactions


def test_authenticated_response_renews_cookie_and_session(client, servers):
    sign_in(client, servers)
    state = client.app.state.chatgpt_state
    session = state.sessions[client.cookies.get(SESSION_COOKIE)]
    session.expires_at = time.time() + 60
    response = client.get("/api/models")
    assert response.status_code == 200
    cookie = SimpleCookie(response.headers["set-cookie"])
    assert (
        SESSION_TTL_SECONDS - 2
        <= int(cookie[SESSION_COOKIE]["max-age"])
        <= SESSION_TTL_SECONDS
    )
    assert session.expires_at >= time.time() + SESSION_TTL_SECONDS - 2
    logout = post_csrf(client, "/auth/logout")
    assert "Max-Age=0" in logout.headers["set-cookie"]
    assert session.id not in state.sessions


@pytest.mark.parametrize("endpoint", ["discovery", "jwks"])
@pytest.mark.parametrize("failure", ["status", "transport", "json"])
def test_auth_metadata_failures_are_structured(client, servers, endpoint, failure):
    url = begin_login(client) if endpoint == "jwks" else None
    original = servers.handler
    target = (
        DISCOVERY["jwks_uri"]
        if endpoint == "jwks"
        else DISCOVERY["issuer"] + "/.well-known/openid-configuration"
    )

    def handler(request):
        if str(request.url) == target:
            if failure == "transport":
                raise httpx.ConnectError("transport failed", request=request)
            return httpx.Response(
                503 if failure == "status" else 200, content="not json"
            )
        return original(request)

    client.app.state.oauth.http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    response = (
        finish_login(client, url, servers.nonce_holder)
        if url
        else post_csrf(client, "/auth/login", follow_redirects=False)
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"
    assert client.app.state.chatgpt_state.connection is None


def test_expired_model_cache_fetches_new_models(client, servers):
    sign_in(client, servers)
    assert client.get("/api/models").status_code == 200
    state = client.app.state.chatgpt_state
    state.models_cache["fetched_at"] = time.time() - MODELS_CACHE_TTL_SECONDS - 1
    servers.models.append({"slug": "new-model", "visibility": "list"})
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "new-model", "input": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert len(servers.models_requests) == 2
    assert servers.responses_payloads[-1]["model"] == "new-model"


def test_refresh_without_rotation_keeps_token(client, servers):
    sign_in(client, servers)
    connection = client.app.state.chatgpt_state.connection
    old_token = connection.refresh_token
    original = servers.token_response

    def token_response(form):
        data = original(form).json()
        data.pop("refresh_token")
        return httpx.Response(200, json=data)

    servers.token_response = token_response
    for _ in range(2):
        connection.expires_at = time.time() - 1
        assert client.get("/api/models").status_code == 200
        assert connection.refresh_token == old_token
    refreshes = [
        r for r in servers.token_requests if r["grant_type"] == "refresh_token"
    ]
    assert [r["refresh_token"] for r in refreshes] == [old_token, old_token]


def test_malformed_models_return_structured_error(client, servers):
    sign_in(client, servers)
    original = servers.handler

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, content="not json")
        return original(request)

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = client.get("/api/models")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


async def test_overlapping_first_signins_cannot_replace_registration():
    state = MemoryState()
    sessions = [await state.create_session() for _ in range(2)]
    txs = [await state.begin_transaction(s.id, "v", "uri", None) for s in sessions]
    committed = []
    await state.finish_sign_in(
        SimpleNamespace(generation=0), txs[0], lambda: committed.append("first")
    )
    with pytest.raises(AuthError, match="invalid_state"):
        await state.finish_sign_in(
            SimpleNamespace(generation=0), txs[1], lambda: committed.append("second")
        )
    assert committed == ["first"]
