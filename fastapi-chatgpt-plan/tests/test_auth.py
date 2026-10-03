"""OAuth flow: authorize URL, callback validation, registration persistence."""

import json
from urllib.parse import parse_qs, urlsplit

from conftest import (
    ISSUED_CLIENT_ID,
    SUBJECT,
    begin_login,
    build_app,
    finish_login,
    make_id_token,
    sign_in,
)
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from fastapi_chatgpt_plan import registrations
from fastapi_chatgpt_plan.oauth import (
    IDENTITY_SCOPES,
    ISSUER,
    PLAN_SCOPES,
    RESOURCE,
)


def _authorize_params(url):
    return parse_qs(urlsplit(url).query)


def test_first_login_authorize_url(client, servers):
    url = begin_login(client)
    parsed = urlsplit(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        ISSUER + "/api/accounts/authorize"
    )
    params = _authorize_params(url)
    assert params["client_id"] == ["dynamic_agent_client"]
    assert params["agent_name_hint"] == ["fastapi-chatgpt-plan"]
    assert params["ext_agent_host_id"][0].startswith("urn:uuid:")
    assert params["response_type"] == ["code"]
    assert params["redirect_uri"] == ["http://127.0.0.1:8000/auth/callback"]
    assert params["scope"] == [PLAN_SCOPES]
    assert params["resource"] == [RESOURCE]
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"]
    assert params["state"] and params["nonce"]
    assert "id_token_hint" not in params


def test_full_login_roundtrip(client, servers):
    url = begin_login(client)
    old_cookie = client.cookies.get("fastapi_chatgpt_plan_session")
    response = finish_login(client, url, servers.nonce_holder)
    assert response.status_code == 303
    new_cookie = client.cookies.get("fastapi_chatgpt_plan_session")
    assert new_cookie and new_cookie != old_cookie
    info = client.get("/api/session").json()
    assert info["status"] == "authenticated"
    assert info["user"]["email"] == "user@example.com"
    assert info["plan"]["permitted"] is True
    registration = registrations.load_registration(
        client.app.state.settings.state_dir
    )
    assert registration["client_id"] == ISSUED_CLIENT_ID
    assert registration["subject"] == SUBJECT
    assert registration["issuer"] == ISSUER


def test_registration_file_has_no_tokens(client, servers):
    sign_in(client, servers)
    path = client.app.state.settings.state_dir / "registration.json"
    raw = path.read_text()
    data = json.loads(raw)
    for key in ("access_token", "refresh_token", "id_token"):
        assert key not in data
    assert "access-token-1" not in raw
    assert "refresh-token-1" not in raw
    import stat

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_callback_replay_rejected(client, servers):
    url = begin_login(client)
    first = finish_login(client, url, servers.nonce_holder)
    assert first.status_code == 303
    second = finish_login(client, url, servers.nonce_holder)
    assert second.status_code == 400


def test_callback_bad_state(client, servers):
    url = begin_login(client)
    response = finish_login(client, url, {}, state="wrong-state")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_state"


def test_callback_declined(client, servers):
    url = begin_login(client)
    response = finish_login(client, url, {}, error="access_denied")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "sign_in_declined"


def test_callback_duplicate_params_rejected(client, servers):
    url = begin_login(client)
    params = parse_qs(urlsplit(url).query)
    state = params["state"][0]
    response = client.get(
        f"/auth/callback?state={state}&state={state}&code=c&client_id=x"
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_callback"


def test_callback_without_session_rejected(client, servers):
    url = begin_login(client)
    params = parse_qs(urlsplit(url).query)
    client.cookies.clear()
    response = client.get(
        "/auth/callback?state={}&code=c&client_id={}".format(
            params["state"][0], ISSUED_CLIENT_ID
        )
    )
    assert response.status_code == 400


def test_wrong_nonce_rejected(client, servers):
    url = begin_login(client)
    response = finish_login(client, url, {})
    assert response.status_code == 400


def test_bad_id_token_signature_rejected(client, servers):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    holder = {}
    url = begin_login(client)
    params = parse_qs(urlsplit(url).query)
    holder["nonce"] = params["nonce"][0]
    servers.id_token_override = make_id_token(
        other_key, nonce=holder["nonce"]
    )
    response = client.get(
        "/auth/callback?state={}&code=c&client_id={}".format(
            params["state"][0], ISSUED_CLIENT_ID
        )
    )
    assert response.status_code == 400


def test_identity_only_mode_requests_identity_scope(servers, state_dir):
    app = build_app(servers, state_dir, plan_enabled=False)
    client = TestClient(app)
    servers.scope = IDENTITY_SCOPES
    url = begin_login(client)
    params = _authorize_params(url)
    assert params["scope"] == [IDENTITY_SCOPES]
    holder = sign_in(client, servers)
    assert holder
    info = client.get("/api/session").json()
    assert info["status"] == "authenticated"
    assert info["plan"] == {"enabled": False, "permitted": False}
    assert client.get("/api/models").status_code == 403
    assert client.get("/api/models").json()["error"]["code"] == "inference_disabled"
    assert servers.models_requests == []


def test_plan_scope_missing_keeps_sign_in_but_disallows_plan(client, servers):
    servers.scope = IDENTITY_SCOPES
    sign_in(client, servers)
    info = client.get("/api/session").json()
    assert info["status"] == "authenticated"
    assert info["plan"]["permitted"] is False
    response = client.get("/api/models")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "plan_permission_required"


def test_second_login_reuses_issued_client_id(client, servers):
    sign_in(client, servers)
    url = begin_login(client)
    params = _authorize_params(url)
    assert params["client_id"] == [ISSUED_CLIENT_ID]
    assert "agent_name_hint" not in params
    assert params["id_token_hint"]
    assert params["login_hint"] == ["user@example.com"]
    sign_in(client, servers)


def test_restart_keeps_host_id_and_client_id(servers, state_dir):
    client = TestClient(build_app(servers, state_dir, plan_enabled=True))
    sign_in(client, servers)
    before = registrations.load_registration(state_dir)
    assert before["client_id"] == ISSUED_CLIENT_ID
    assert before["host_id"].startswith("urn:uuid:")

    restarted = TestClient(build_app(servers, state_dir, plan_enabled=True))
    url = begin_login(restarted)
    params = _authorize_params(url)
    assert params["client_id"] == [ISSUED_CLIENT_ID]
    assert params["ext_agent_host_id"] == [before["host_id"]]
    assert "id_token_hint" not in params
    after = registrations.load_registration(state_dir)
    assert after["host_id"] == before["host_id"]
    assert after["client_id"] == ISSUED_CLIENT_ID


def test_account_mismatch_does_not_replace_registration(client, servers):
    sign_in(client, servers)
    url = begin_login(client)
    params = parse_qs(urlsplit(url).query)
    servers.id_token_override = make_id_token(
        servers.rsa_key, subject="someone-else", nonce=params["nonce"][0]
    )
    servers.nonce_holder["nonce"] = params["nonce"][0]
    response = client.get(
        "/auth/callback?state={}&code=c2&client_id={}".format(
            params["state"][0], ISSUED_CLIENT_ID
        )
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "account_mismatch"
    registration = registrations.load_registration(
        client.app.state.settings.state_dir
    )
    assert registration["subject"] == SUBJECT


def test_client_id_mismatch_rejected(client, servers):
    sign_in(client, servers)
    url = begin_login(client)
    params = parse_qs(urlsplit(url).query)
    response = client.get(
        "/auth/callback?state={}&code=c&client_id=other-client".format(
            params["state"][0]
        )
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "client_id_mismatch"


def test_logout_clears_session_and_revokes(client, servers):
    sign_in(client, servers)
    response = client.post(
        "/auth/logout",
        headers={"origin": "http://testserver"},
        data={"csrf_token": client.get("/api/session").json()["csrf"]},
    )
    assert response.status_code == 200
    assert response.json()["remote_revocation"] == "confirmed"
    assert servers.revoke_requests
    form = servers.revoke_requests[0]
    assert form["token_type_hint"] == "refresh_token"
    assert form["client_id"] == ISSUED_CLIENT_ID
    info = client.get("/api/session").json()
    assert info["status"] == "anonymous"
    assert client.get("/api/models").status_code == 401
    registration = registrations.load_registration(
        client.app.state.settings.state_dir
    )
    assert registration["client_id"] == ISSUED_CLIENT_ID


def test_logout_revocation_unconfirmed_still_signs_out(client, servers):
    sign_in(client, servers)
    servers.revoke_status = 503
    response = client.post(
        "/auth/logout",
        headers={"origin": "http://testserver"},
        data={"csrf_token": client.get("/api/session").json()["csrf"]},
    )
    assert response.status_code == 200
    assert response.json()["remote_revocation"] == "not_confirmed"
    assert client.get("/api/session").json()["status"] == "anonymous"
