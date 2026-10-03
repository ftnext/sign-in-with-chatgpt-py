import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import ISSUED_CLIENT_ID, SUBJECT, make_id_token
from cryptography.hazmat.primitives.asymmetric import rsa

from llm_chatgpt_plan import storage
from llm_chatgpt_plan.errors import ApiError
from llm_chatgpt_plan.oauth import (
    DYNAMIC_CLIENT_ID,
    ISSUER,
    PLAN_SCOPE,
    RESOURCE,
    AuthError,
    OAuth,
)


def make_tx(
    state="state-1",
    nonce="nonce-1",
    client_id=None,
    redirect_uri="http://127.0.0.1:1/auth/callback",
):
    return {
        "state": state,
        "nonce": nonce,
        "verifier": "v" * 64,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "expires": time.time() + 600,
    }


class TestBegin:
    def test_dynamic_registration_params(self, auth_server):
        oauth = OAuth(auth_server.client())
        tx, url = oauth.begin("http://127.0.0.1:9/auth/callback", host_id="urn:uuid:h")
        params = parse_qs(urlsplit(url).query)
        assert params["client_id"] == [DYNAMIC_CLIENT_ID]
        assert params["agent_name_hint"] == ["llm-chatgpt-plan"]
        assert params["ext_agent_host_id"] == ["urn:uuid:h"]
        assert params["response_type"] == ["code"]
        assert params["code_challenge_method"] == ["S256"]
        assert params["code_challenge"][0] != tx["verifier"]
        assert params["state"] == [tx["state"]]
        assert params["nonce"] == [tx["nonce"]]
        assert params["resource"] == [RESOURCE]
        assert PLAN_SCOPE in params["scope"][0]
        assert url.startswith(ISSUER)

    def test_reauth_uses_issued_client_and_hints(self, auth_server):
        oauth = OAuth(auth_server.client())
        prior = {"id_token": "stored-id-token", "email": "user@example.com"}
        _, url = oauth.begin(
            "http://127.0.0.1:9/auth/callback",
            host_id="urn:uuid:h",
            client_id=ISSUED_CLIENT_ID,
            prior=prior,
        )
        params = parse_qs(urlsplit(url).query)
        assert params["client_id"] == [ISSUED_CLIENT_ID]
        assert "agent_name_hint" not in params
        assert params["id_token_hint"] == ["stored-id-token"]
        assert params["login_hint"] == ["user@example.com"]

    def test_bad_discovery_rejected(self, rsa_key):
        def handler(request):
            return httpx.Response(200, json={"issuer": "https://evil.example"})

        oauth = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(AuthError):
            oauth.begin("http://127.0.0.1:9/auth/callback", host_id="h")


class TestComplete:
    def query(self, **overrides):
        q = {"state": "state-1", "code": "code-1", "client_id": ISSUED_CLIENT_ID}
        q.update(overrides)
        return q

    def test_success_persists_issued_client_id(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with storage.locked_store(state_dir) as store:
            record = oauth.complete(
                make_tx(nonce="nonce"),
                self.query(),
                store,
                persist_registration=True,
            )
            assert record["client_id"] == ISSUED_CLIENT_ID
            assert record["subject"] == SUBJECT
            assert PLAN_SCOPE in record["scopes"]
            assert store.load_credentials()["client_id"] == ISSUED_CLIENT_ID

    def test_expired_tx(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with storage.locked_store(state_dir) as store:
            tx = make_tx()
            tx["expires"] = time.time() - 1
            with pytest.raises(AuthError, match="expired_sign_in"):
                oauth.complete(tx, self.query(), store, True)

    def test_wrong_state_rejected(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="invalid_state"),
        ):
            oauth.complete(make_tx(), self.query(state="other"), store, True)

    def test_declined_callback(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="sign_in_declined"),
        ):
            oauth.complete(
                make_tx(),
                self.query(error="access_denied", code=None),
                store,
                True,
            )
        # declined callbacks must not exchange the code
        assert auth_server.token_requests == []

    def test_dynamic_client_id_in_callback_rejected(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="missing_client_id"),
        ):
            oauth.complete(
                make_tx(),
                self.query(client_id=DYNAMIC_CLIENT_ID),
                store,
                True,
            )

    def test_client_id_mismatch_on_reauth(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="client_id_mismatch"),
        ):
            oauth.complete(
                make_tx(client_id=ISSUED_CLIENT_ID),
                self.query(client_id="other-client"),
                store,
                True,
            )

    def test_missing_code(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="missing_code"),
        ):
            oauth.complete(make_tx(), self.query(code=None), store, True)

    def test_id_token_checks(self, auth_server, rsa_key, state_dir):
        cases = {
            "bad nonce": make_id_token(rsa_key, nonce="other"),
            "bad issuer": make_id_token(rsa_key, issuer="https://evil.example"),
            "bad audience": make_id_token(
                rsa_key, client_id=ISSUED_CLIENT_ID, aud="someone-else"
            ),
            "expired": make_id_token(rsa_key, exp=int(time.time()) - 100),
        }
        for token in cases.values():
            auth_server.id_token_override = token
            oauth = OAuth(auth_server.client())
            with (
                storage.locked_store(state_dir) as store,
                pytest.raises(AuthError, match="invalid"),
            ):
                oauth.complete(make_tx(nonce="nonce"), self.query(), store, True)
            auth_server.token_requests.clear()

    def test_wrong_signing_key_rejected(self, auth_server, state_dir, rsa_key):
        other = rsa.generate_private_key(65537, 2048)
        auth_server.id_token_override = make_id_token(other)
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="invalid"),
        ):
            oauth.complete(make_tx(nonce="nonce"), self.query(), store, True)

    def test_subject_mismatch_on_reauth(self, auth_server, state_dir):
        auth_server.id_token_override = make_id_token(
            auth_server.rsa_key, subject="someone-else"
        )
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="account_mismatch"),
        ):
            oauth.complete(
                make_tx(client_id=ISSUED_CLIENT_ID, nonce="nonce"),
                self.query(),
                store,
                True,
                prior_subject=SUBJECT,
            )

    def test_replace_does_not_persist_client_id_before_verify(
        self, auth_server, state_dir
    ):
        auth_server.token_status = 500
        oauth = OAuth(auth_server.client())
        with storage.locked_store(state_dir) as store:
            store.save_credentials({"client_id": "old", "access_token": "x"})
            with pytest.raises(ApiError, match="HTTP 500"):
                oauth.complete(
                    make_tx(), self.query(), store, persist_registration=False
                )
            assert store.load_credentials()["client_id"] == "old"


class TestRefresh:
    def record(self, **overrides):
        record = {
            "client_id": ISSUED_CLIENT_ID,
            "subject": SUBJECT,
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "scopes": ["openid", PLAN_SCOPE],
            "expires_at": time.time() + 3600,
        }
        record.update(overrides)
        return record

    def test_fresh_token_returned_without_http(self, auth_server, state_dir):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record())
        oauth = OAuth(auth_server.client())
        token = oauth.ensure_access_token(state_dir)
        assert token == "old-access"
        assert auth_server.token_requests == []

    def test_expired_token_refreshed_and_saved(self, auth_server, state_dir):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record(expires_at=time.time() + 10))
        oauth = OAuth(auth_server.client())
        token = oauth.ensure_access_token(state_dir)
        assert token == "access-token-1"
        (form,) = auth_server.token_requests
        assert form["grant_type"] == "refresh_token"
        assert form["client_id"] == ISSUED_CLIENT_ID
        assert form["refresh_token"] == "old-refresh"
        assert form["resource"] == RESOURCE
        assert "scope" not in form
        saved = storage.read_credentials(state_dir)
        assert saved["access_token"] == "access-token-1"
        assert saved["refresh_token"] == "refresh-token-1"

    def test_refresh_failure_asks_for_login(self, auth_server, state_dir):
        auth_server.token_status = 400
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record(expires_at=time.time() - 5))
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="reauthorization_required"):
            oauth.ensure_access_token(state_dir)

    def test_no_connection(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="sign_in_required"):
            oauth.ensure_access_token(state_dir)

    def test_missing_plan_scope(self, auth_server, state_dir):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record(scopes=["openid"]))
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="plan_permission_required"):
            oauth.ensure_access_token(state_dir)
