"""App-level behavior: health, session bootstrap, host/origin/CSRF guards."""

from conftest import ORIGIN, bootstrap, post_csrf


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "fastapi-chatgpt-plan" in response.text


def test_session_bootstrap_sets_cookie_and_csrf(client):
    info = bootstrap(client)
    assert info["status"] == "anonymous"
    assert info["user"] is None
    assert info["csrf"]
    cookie = client.cookies.get("fastapi_chatgpt_plan_session")
    assert cookie


def test_cookie_attributes_on_creation(client):
    client.cookies.clear()
    response = client.get("/api/session")
    header = response.headers["set-cookie"]
    assert "HttpOnly" in header
    assert "SameSite=lax" in header
    assert "Path=/" in header
    assert "Secure" not in header


def test_unknown_host_rejected(client):
    response = client.get("/health", headers={"host": "evil.example.com"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "invalid_host"


def test_login_requires_session(client):
    response = client.post(
        "/auth/login",
        data={"csrf_token": "x"},
        headers={"origin": ORIGIN},
        follow_redirects=False,
    )
    assert response.status_code == 401


def test_login_requires_csrf(client):
    bootstrap(client)
    response = client.post(
        "/auth/login", headers={"origin": ORIGIN}, follow_redirects=False
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "invalid_csrf_token"


def test_login_rejects_foreign_origin(client):
    info = bootstrap(client)
    response = client.post(
        "/auth/login",
        data={"csrf_token": info["csrf"]},
        headers={"origin": "http://evil.example.com"},
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "invalid_origin"


def test_login_rejects_missing_origin(client):
    info = bootstrap(client)
    response = client.post(
        "/auth/login",
        data={"csrf_token": info["csrf"]},
        follow_redirects=False,
    )
    assert response.status_code == 403


def test_logout_requires_csrf(client):
    bootstrap(client)
    response = client.post("/auth/logout", headers={"origin": ORIGIN})
    assert response.status_code == 403


def test_unauthenticated_api_rejected(client):
    assert client.get("/api/models").status_code in (401, 403)
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "x", "input": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "sign_in_required"
