"""Additional error cases not covered by the original review regressions."""

import httpx
import pytest
from conftest import DISCOVERY, begin_login, finish_login, post_csrf, sign_in
from fastapi.testclient import TestClient

from fastapi_chatgpt_plan.sessions import SESSION_COOKIE


@pytest.mark.parametrize("endpoint", ["discovery", "jwks"])
@pytest.mark.parametrize("failure", ["transport", "json"])
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
            return httpx.Response(200, content="not json")
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


def test_stale_session_does_not_get_renewed(client, servers):
    sign_in(client, servers)
    state = client.app.state.chatgpt_state
    old_id = client.cookies.get(SESSION_COOKIE)
    sign_in(TestClient(client.app), servers)
    response = client.get(
        "/api/session", headers={"cookie": f"{SESSION_COOKIE}={old_id}"}
    )
    assert response.json()["status"] == "anonymous"
    assert "set-cookie" not in response.headers
    assert state.sessions[old_id].generation != state.connection.generation


def test_untrusted_host_does_not_renew_cookie(client, servers):
    sign_in(client, servers)
    response = client.get("/api/models", headers={"host": "untrusted.example"})
    assert response.status_code == 403
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("phase", ["login", "refresh"])
def test_malformed_token_response_is_upstream_error(client, servers, phase):
    if phase == "refresh":
        sign_in(client, servers)
        client.app.state.chatgpt_state.connection.expires_at = 0
    servers.token_response = lambda form: httpx.Response(200, content="invalid json")
    if phase == "login":
        response = finish_login(client, begin_login(client), servers.nonce_holder)
    else:
        response = client.get("/api/models")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"
