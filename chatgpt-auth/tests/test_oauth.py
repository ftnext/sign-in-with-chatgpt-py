import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import ISSUED_CLIENT_ID, SUBJECT, make_id_token
from cryptography.hazmat.primitives.asymmetric import rsa

from chatgpt_auth.errors import (
    AuthError,
    NetworkError,
    TokenOutcomeUnknownError,
)
from chatgpt_auth.oauth import (
    DYNAMIC_CLIENT_ID,
    ISSUER,
    PLAN_SCOPE,
    RESOURCE,
    OAuth,
)

REDIRECT_URI = "http://127.0.0.1:1/auth/callback"


def make_tx(
    state="state-1",
    nonce="nonce-1",
    client_id=None,
    redirect_uri=REDIRECT_URI,
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
        tx, url = oauth.begin(REDIRECT_URI, host_id="urn:uuid:h")
        params = parse_qs(urlsplit(url).query)
        assert params["client_id"] == [DYNAMIC_CLIENT_ID]
        assert params["agent_name_hint"] == ["chatgpt-auth"]
        assert params["ext_agent_host_id"] == ["urn:uuid:h"]
        assert params["response_type"] == ["code"]
        assert params["code_challenge_method"] == ["S256"]
        assert params["code_challenge"][0] != tx["verifier"]
        assert params["state"] == [tx["state"]]
        assert params["nonce"] == [tx["nonce"]]
        assert params["resource"] == [RESOURCE]
        assert PLAN_SCOPE in params["scope"][0]
        assert url.startswith(ISSUER)

    def test_agent_name_hint_is_customizable(self, auth_server):
        oauth = OAuth(auth_server.client())
        _, url = oauth.begin(
            REDIRECT_URI, host_id="urn:uuid:h", agent_name="my-python-app"
        )
        params = parse_qs(urlsplit(url).query)
        assert params["agent_name_hint"] == ["my-python-app"]

    def test_reauth_uses_issued_client_and_hints(self, auth_server):
        oauth = OAuth(auth_server.client())
        prior = {"id_token": "stored-id-token", "email": "user@example.com"}
        _, url = oauth.begin(
            REDIRECT_URI,
            host_id="urn:uuid:h",
            client_id=ISSUED_CLIENT_ID,
            prior=prior,
            agent_name="ignored-on-reauth",
        )
        params = parse_qs(urlsplit(url).query)
        assert params["client_id"] == [ISSUED_CLIENT_ID]
        assert "agent_name_hint" not in params
        assert params["id_token_hint"] == ["stored-id-token"]
        assert params["login_hint"] == ["user@example.com"]

    def test_reauth_without_prior_sends_no_hints(self, auth_server):
        oauth = OAuth(auth_server.client())
        _, url = oauth.begin(
            REDIRECT_URI, host_id="urn:uuid:h", client_id=ISSUED_CLIENT_ID
        )
        params = parse_qs(urlsplit(url).query)
        assert "agent_name_hint" not in params
        assert "id_token_hint" not in params
        assert "login_hint" not in params

    def test_bad_discovery_rejected(self):
        def handler(request):
            return httpx.Response(200, json={"issuer": "https://evil.example"})

        oauth = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(AuthError):
            oauth.begin(REDIRECT_URI, host_id="h")

    def test_discovery_endpoint_origin_checked(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": "https://evil.example/authorize",
                    "token_endpoint": ISSUER + "/token",
                    "jwks_uri": ISSUER + "/jwks",
                },
            )

        oauth = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(AuthError, match="invalid_discovery"):
            oauth.begin(REDIRECT_URI, host_id="h")


class TestVerify:
    def test_id_token_checks(self, auth_server, rsa_key):
        oauth = OAuth(auth_server.client())
        token = make_id_token(rsa_key, nonce="n1")
        claims = oauth.verify(token, ISSUED_CLIENT_ID, nonce="n1")
        assert claims["sub"] == SUBJECT

    @pytest.mark.parametrize(
        "override",
        [
            {"issuer": "https://evil.example"},
            {"client_id": "other-client"},
            {"subject": ""},
            {"kid": "unknown-kid"},
        ],
    )
    def test_invalid_claims_rejected(self, auth_server, rsa_key, override):
        oauth = OAuth(auth_server.client())
        token = make_id_token(rsa_key, **override)
        with pytest.raises(AuthError):
            oauth.verify(token, ISSUED_CLIENT_ID)

    def test_wrong_nonce_rejected(self, auth_server, rsa_key):
        oauth = OAuth(auth_server.client())
        token = make_id_token(rsa_key, nonce="n1")
        with pytest.raises(AuthError):
            oauth.verify(token, ISSUED_CLIENT_ID, nonce="n2")

    def test_wrong_signing_key_rejected(self, auth_server):
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        oauth = OAuth(auth_server.client())
        token = make_id_token(other_key)
        with pytest.raises(AuthError):
            oauth.verify(token, ISSUED_CLIENT_ID)

    def test_azp_mismatch_rejected(self, auth_server, rsa_key):
        oauth = OAuth(auth_server.client())
        token = make_id_token(rsa_key, azp="different-client")
        with pytest.raises(AuthError):
            oauth.verify(token, ISSUED_CLIENT_ID)


class TestTokenRequest:
    def test_reauth_code_raises_auth_error(self, auth_server):
        auth_server.token_status = 400
        auth_server.token_error = {"error": "invalid_grant"}
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="reauthorization_required"):
            oauth.token_request({"grant_type": "refresh_token"})

    def test_safe_reject_keeps_network_error(self, auth_server):
        auth_server.token_status = 400
        auth_server.token_error = {"error": "invalid_client"}
        oauth = OAuth(auth_server.client())
        with pytest.raises(NetworkError, match="invalid_client"):
            oauth.token_request({"grant_type": "refresh_token"})

    def test_unknown_rejection_is_outcome_unknown(self, auth_server):
        auth_server.token_status = 400
        auth_server.token_error = {"error": "temporarily_unavailable"}
        oauth = OAuth(auth_server.client())
        with pytest.raises(TokenOutcomeUnknownError):
            oauth.token_request({"grant_type": "refresh_token"})

    def test_5xx_is_outcome_unknown(self, auth_server):
        auth_server.token_status = 500
        oauth = OAuth(auth_server.client())
        with pytest.raises(TokenOutcomeUnknownError):
            oauth.token_request({"grant_type": "refresh_token"})

    def test_connect_error_is_network_error(self):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        oauth = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(NetworkError):
            oauth.token_request({"grant_type": "refresh_token"})

    def test_read_timeout_is_outcome_unknown(self, auth_server):
        def handler(request):
            if "token" in str(request.url):
                raise httpx.ReadTimeout("slow", request=request)
            return auth_server.handler(request)

        oauth = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(TokenOutcomeUnknownError):
            oauth.token_request({"grant_type": "refresh_token"})

    def test_secret_values_never_leak_into_errors(self, auth_server):
        auth_server.token_status = 400
        auth_server.token_error = {
            "error": "bad",
            "error_description": "token refresh-secret-value was rejected",
        }
        oauth = OAuth(auth_server.client())
        with pytest.raises(TokenOutcomeUnknownError) as info:
            oauth.token_request({"refresh_token": "refresh-secret-value"})
        assert "refresh-secret-value" not in str(info.value)


class TestCredentialsParsing:
    def test_first_exchange_requires_refresh_token(self, auth_server):
        oauth = OAuth(auth_server.client())
        tokens = {
            "access_token": "a",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        with pytest.raises(AuthError, match="invalid_token_response"):
            oauth.credentials(tokens)

    def test_refresh_response_may_omit_refresh_token(self, auth_server):
        oauth = OAuth(auth_server.client())
        tokens = {
            "access_token": "a",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        parsed = oauth.credentials(tokens, refresh_response=True)
        assert parsed["access_token"] == "a"
        assert parsed["refresh_token"] is None

    def test_scope_omitted_inherits_previous(self, auth_server):
        oauth = OAuth(auth_server.client())
        tokens = {
            "access_token": "a",
            "refresh_token": "r",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        parsed = oauth.credentials(tokens, ["openid", PLAN_SCOPE])
        assert parsed["scopes"] == ["openid", PLAN_SCOPE]

    @pytest.mark.parametrize(
        "tokens",
        [
            {
                "access_token": "a",
                "refresh_token": "r",
                "token_type": "basic",
                "expires_in": 3600,
            },
            {
                "access_token": "",
                "refresh_token": "r",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            {
                "access_token": "a",
                "refresh_token": "",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            {
                "access_token": "a",
                "refresh_token": "r",
                "token_type": "Bearer",
                "expires_in": 0,
            },
            {
                "access_token": "a",
                "refresh_token": "r",
                "token_type": "Bearer",
                "expires_in": 99999,
            },
            {"access_token": "a", "refresh_token": "r", "token_type": "Bearer"},
            {
                "access_token": "a",
                "refresh_token": "r",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": ["not-a-string"],
            },
        ],
    )
    def test_invalid_responses_rejected(self, auth_server, tokens):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError):
            oauth.credentials(tokens)


class TestComplete:
    def query(self, **overrides):
        q = {"state": "state-1", "code": "code-1", "client_id": ISSUED_CLIENT_ID}
        q.update(overrides)
        return q

    def test_success_returns_record(self, auth_server):
        oauth = OAuth(auth_server.client())
        record = oauth.complete(make_tx(nonce="nonce"), self.query())
        assert record["client_id"] == ISSUED_CLIENT_ID
        assert record["subject"] == SUBJECT
        assert record["issuer"] == ISSUER
        assert PLAN_SCOPE in record["scopes"]
        assert record["access_token"] == "access-token-1"

    def test_persist_client_id_invoked_before_exchange(self, auth_server):
        persisted = []
        oauth = OAuth(auth_server.client())
        record = oauth.complete(
            make_tx(nonce="nonce"),
            self.query(),
            persist_client_id=persisted.append,
        )
        assert persisted == [ISSUED_CLIENT_ID]
        assert record["client_id"] == ISSUED_CLIENT_ID

    def test_expired_tx(self, auth_server):
        oauth = OAuth(auth_server.client())
        tx = make_tx()
        tx["expires"] = time.time() - 1
        with pytest.raises(AuthError, match="expired_sign_in"):
            oauth.complete(tx, self.query())

    def test_wrong_state_rejected(self, auth_server):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="invalid_state"):
            oauth.complete(make_tx(), self.query(state="other"))

    def test_declined_callback(self, auth_server):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="sign_in_declined"):
            oauth.complete(make_tx(), self.query(error="access_denied", code=None))

    def test_dynamic_client_id_in_callback_rejected(self, auth_server):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="missing_client_id"):
            oauth.complete(make_tx(), self.query(client_id=DYNAMIC_CLIENT_ID))

    def test_client_id_mismatch_on_reauth(self, auth_server):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="client_id_mismatch"):
            oauth.complete(
                make_tx(client_id=ISSUED_CLIENT_ID),
                self.query(client_id="different-client"),
            )

    def test_missing_code(self, auth_server):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="missing_code"):
            oauth.complete(make_tx(), self.query(code=None))

    def test_subject_mismatch_on_reauth(self, auth_server, rsa_key):
        auth_server.id_token_override = make_id_token(
            rsa_key, subject="someone-else", nonce="nonce"
        )
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="account_mismatch"):
            oauth.complete(
                make_tx(nonce="nonce"),
                self.query(),
                prior_subject=SUBJECT,
            )


class TestManualUrl:
    def test_hints_removed_from_shown_url(self, auth_server):
        oauth = OAuth(auth_server.client())
        prior = {"id_token": "stored-id-token", "email": "user@example.com"}
        tx, url = oauth.begin(
            REDIRECT_URI,
            host_id="urn:uuid:h",
            client_id=ISSUED_CLIENT_ID,
            prior=prior,
        )
        assert "id_token_hint" in url
        shown = oauth.manual_url(tx)
        params = parse_qs(urlsplit(shown).query)
        assert "id_token_hint" not in params
        assert "login_hint" not in params
        assert params["client_id"] == [ISSUED_CLIENT_ID]
        assert "stored-id-token" not in shown
        assert "user@example.com" not in shown


class TestRevoke:
    def test_confirmed_200(self, auth_server):
        oauth = OAuth(auth_server.client())
        assert oauth.revoke_refresh_token("client", "refresh-token") is True
        (form,) = auth_server.revoke_requests
        assert form["token"] == "refresh-token"
        assert form["client_id"] == "client"

    def test_retries_only_on_5xx_and_network(self, auth_server):
        auth_server.revoke_statuses = [500, 503, 200]
        oauth = OAuth(auth_server.client())
        assert oauth.revoke_refresh_token("c", "r", sleep=lambda s: None) is True
        assert len(auth_server.revoke_requests) == 3

    def test_4xx_is_permanent_unconfirmed(self, auth_server):
        auth_server.revoke_statuses = [400, 200]
        oauth = OAuth(auth_server.client())
        assert oauth.revoke_refresh_token("c", "r", sleep=lambda s: None) is False
        assert len(auth_server.revoke_requests) == 1

    def test_no_endpoint_is_unconfirmed(self):
        def handler(request):
            if "openid-configuration" in str(request.url):
                data = {
                    "issuer": ISSUER,
                    "authorization_endpoint": ISSUER + "/a",
                    "token_endpoint": ISSUER + "/t",
                    "jwks_uri": ISSUER + "/j",
                }
                return httpx.Response(200, json=data)
            return httpx.Response(404)

        oauth = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        assert oauth.revoke_refresh_token("c", "r") is False
