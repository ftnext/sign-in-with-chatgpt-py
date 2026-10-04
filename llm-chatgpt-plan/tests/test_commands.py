import threading
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from click.testing import CliRunner
from conftest import ISSUED_CLIENT_ID, SUBJECT, drive_browser_callback

from llm_chatgpt_plan import client, commands, oauth, storage
from llm_chatgpt_plan.commands import chatgpt_plan
from llm_chatgpt_plan.oauth import OAuth


@pytest.fixture
def runner():
    return CliRunner()


def mock_http(monkeypatch, server):
    monkeypatch.setattr(client, "make_http_client", lambda **kw: server.client(**kw))


def mock_browser(monkeypatch, server, **callback_kwargs):
    def fake_open(url):
        drive_browser_callback(url, nonce_holder=server.nonce_holder, **callback_kwargs)
        return True

    monkeypatch.setattr(commands.webbrowser, "open", fake_open)


class TestModelsCommand:
    def test_lists_cached_models_without_http(
        self, runner, stored_connection, monkeypatch
    ):
        def fail(**kw):
            raise AssertionError("network access during cached listing")

        monkeypatch.setattr(client, "make_http_client", fail)
        result = runner.invoke(chatgpt_plan, ["models"])
        assert result.exit_code == 0, result.output
        assert "chatgpt-plan/gpt-5.6-luna" in result.output
        assert "GPT 5.6 Luna" in result.output
        assert "chatgpt-plan/gpt-5-mini" in result.output

    def test_not_signed_in(self, runner, user_dir):
        result = runner.invoke(chatgpt_plan, ["models"])
        assert result.exit_code != 0
        assert "llm chatgpt-plan login" in result.output

    def test_no_saved_list(self, runner, state_dir):
        state_dir.mkdir(parents=True)
        storage.Store(state_dir).save_credentials(
            {
                "client_id": "c",
                "subject": "s",
                "access_token": "t",
                "scopes": ["chatgpt.tokens.use.direct"],
                "expires_at": time.time() + 3600,
            }
        )
        result = runner.invoke(chatgpt_plan, ["models"])
        assert result.exit_code != 0
        assert "models --refresh" in result.output

    def test_stale_list_not_shown(self, runner, stored_connection, state_dir):
        storage.Store(state_dir).save_models(
            {"client_id": "other", "subject": "other", "models": []}
        )
        result = runner.invoke(chatgpt_plan, ["models"])
        assert result.exit_code != 0
        assert "--refresh" in result.output

    def test_refresh_fetches_with_bearer(
        self, runner, stored_connection, auth_server, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)
        result = runner.invoke(chatgpt_plan, ["models", "--refresh"])
        assert result.exit_code == 0, result.output
        (request,) = auth_server.models_requests
        assert request.headers["authorization"] == "Bearer stored-access"
        # visibility filter applied, slug prefix shown
        assert "chatgpt-plan/gpt-5.6-luna" in result.output
        assert "hidden-model" not in result.output
        saved = storage.read_models(storage.state_dir())
        assert [m["slug"] for m in saved["models"]] == [
            "gpt-5.6-luna",
            "gpt-5-mini",
        ]
        assert saved["client_id"] == ISSUED_CLIENT_ID

    def test_refresh_refreshes_expired_token(
        self, runner, auth_server, state_dir, monkeypatch
    ):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(
                {
                    "client_id": ISSUED_CLIENT_ID,
                    "subject": SUBJECT,
                    "access_token": "old",
                    "refresh_token": "old-refresh",
                    "scopes": ["chatgpt.tokens.use.direct"],
                    "expires_at": time.time() - 100,
                }
            )
        mock_http(monkeypatch, auth_server)
        result = runner.invoke(chatgpt_plan, ["models", "--refresh"])
        assert result.exit_code == 0, result.output
        (form,) = auth_server.token_requests
        assert form["grant_type"] == "refresh_token"
        (request,) = auth_server.models_requests
        assert request.headers["authorization"] == "Bearer access-token-1"

    def test_refresh_failure_keeps_cache(
        self, runner, stored_connection, auth_server, state_dir, monkeypatch
    ):
        auth_server.models = None

        def handler(request):
            if str(request.url).endswith("/models"):
                return httpx.Response(500, json={"detail": "down"})
            return auth_server.handler(request)

        monkeypatch.setattr(
            client,
            "make_http_client",
            lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw),
        )
        before = storage.read_models(state_dir)
        result = runner.invoke(chatgpt_plan, ["models", "--refresh"])
        assert result.exit_code != 0
        assert storage.read_models(state_dir) == before


class TestLogin:
    def test_first_login_saves_connection_and_models(
        self, runner, auth_server, state_dir, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)
        mock_browser(monkeypatch, auth_server)
        result = runner.invoke(chatgpt_plan, ["login", "--timeout", "30"])
        assert result.exit_code == 0, result.output
        saved = storage.read_credentials(state_dir)
        assert saved["client_id"] == ISSUED_CLIENT_ID
        assert saved["subject"] == SUBJECT
        assert saved["access_token"] == "access-token-1"
        # token exchange used the issued client ID, not the dynamic one
        (form,) = auth_server.token_requests
        assert form["client_id"] == ISSUED_CLIENT_ID
        assert form["grant_type"] == "authorization_code"
        assert form["code_verifier"]
        # model list was fetched and saved for this connection
        models = storage.read_models(state_dir)
        assert models["client_id"] == ISSUED_CLIENT_ID
        assert {m["slug"] for m in models["models"]} == {
            "gpt-5.6-luna",
            "gpt-5-mini",
        }
        # host id persisted
        host_id = storage.Store(state_dir).load_host_id()
        assert host_id.startswith("urn:uuid:")
        # tokens never reach the terminal
        assert "access-token-1" not in result.output
        assert "refresh-token-1" not in result.output
        assert "127.0.0.1" not in result.output

    def test_reauth_reuses_issued_client_id(
        self, runner, auth_server, stored_connection, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)
        urls = []

        def fake_open(url):
            urls.append(url)
            drive_browser_callback(
                url, client_id=ISSUED_CLIENT_ID, nonce_holder=auth_server.nonce_holder
            )
            return True

        monkeypatch.setattr(commands.webbrowser, "open", fake_open)
        result = runner.invoke(chatgpt_plan, ["login"])
        assert result.exit_code == 0, result.output
        params = parse_qs(urlsplit(urls[0]).query)
        assert params["client_id"] == [ISSUED_CLIENT_ID]
        assert "agent_name_hint" not in params
        # authorization URL was never shown (it carries id_token_hint)
        assert urls[0] not in result.output

    def test_login_declined(self, runner, auth_server, state_dir, monkeypatch):
        mock_http(monkeypatch, auth_server)
        mock_browser(monkeypatch, auth_server, error="access_denied")
        result = runner.invoke(chatgpt_plan, ["login"])
        assert result.exit_code != 0
        assert storage.read_credentials(state_dir) is None

    def test_model_fetch_failure_keeps_credentials(
        self, runner, auth_server, state_dir, monkeypatch
    ):
        def handler(request):
            if str(request.url).endswith("/v1/models"):
                return httpx.Response(500, json={"detail": "down"})
            return auth_server.handler(request)

        monkeypatch.setattr(
            client,
            "make_http_client",
            lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw),
        )
        mock_browser(monkeypatch, auth_server)
        result = runner.invoke(chatgpt_plan, ["login"])
        assert result.exit_code == 0, result.output
        assert "--refresh" in result.output
        assert storage.read_credentials(state_dir)["client_id"] == (ISSUED_CLIENT_ID)
        assert storage.read_models(state_dir) is None

    def test_browser_failure_falls_back_to_manual_url(
        self, runner, auth_server, state_dir, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)
        monkeypatch.setattr(commands.webbrowser, "open", lambda url: False)
        result = runner.invoke(chatgpt_plan, ["login", "--timeout", "1"])
        assert result.exit_code != 0
        assert "Could not open a browser" in result.output
        # the hint-free sign-in URL is shown so the user can continue manually
        assert "https://auth.openai.com" in result.output
        assert "id_token_hint" not in result.output

    def test_replace_on_failure_keeps_old_connection(
        self, runner, auth_server, stored_connection, state_dir, monkeypatch
    ):
        auth_server.token_status = 500
        mock_http(monkeypatch, auth_server)
        mock_browser(monkeypatch, auth_server, client_id="new-client")
        result = runner.invoke(chatgpt_plan, ["login", "--replace"])
        assert result.exit_code != 0
        saved = storage.read_credentials(state_dir)
        assert saved["client_id"] == ISSUED_CLIENT_ID
        assert saved["access_token"] == "stored-access"
        assert "HTTP 500" in result.output
        assert "Sign-in expired or was revoked" not in result.output

    def test_manual_login_completes_via_loopback(
        self, runner, auth_server, state_dir, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)
        opened = []
        monkeypatch.setattr(
            commands.webbrowser, "open", lambda url: opened.append(url) or True
        )

        def fake_show(url, browser_failed):
            # the user opens the shown URL elsewhere; the loopback callback
            # still reaches this machine's listener
            threading.Thread(
                target=drive_browser_callback,
                args=(url,),
                kwargs={"nonce_holder": auth_server.nonce_holder},
                daemon=True,
            ).start()

        monkeypatch.setattr(commands, "_show_manual_url", fake_show)
        result = runner.invoke(chatgpt_plan, ["login", "--manual", "--timeout", "10"])
        assert result.exit_code == 0, result.output
        assert opened == []  # the browser was never touched
        saved = storage.read_credentials(state_dir)
        assert saved["client_id"] == ISSUED_CLIENT_ID
        assert saved["access_token"] == "access-token-1"


class TestLogout:
    def test_revokes_remotely_and_wipes_tokens(
        self, runner, auth_server, stored_connection, state_dir, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)
        result = runner.invoke(chatgpt_plan, ["logout"])
        assert result.exit_code == 0, result.output
        (form,) = auth_server.revoke_requests
        assert form == {
            "token": "stored-refresh",
            "token_type_hint": "refresh_token",
            "client_id": ISSUED_CLIENT_ID,
        }
        assert "revoked" in result.output
        saved = storage.read_credentials(state_dir)
        # identity and issued client id survive; tokens and cache are gone
        assert saved["client_id"] == ISSUED_CLIENT_ID
        assert saved["subject"] == SUBJECT
        assert saved["issuer"] == oauth.ISSUER
        assert saved["generation"] == 1
        for key in (
            "access_token",
            "refresh_token",
            "id_token",
            "scopes",
            "expires_at",
        ):
            assert key not in saved
        assert storage.read_models(state_dir) is None
        assert "stored-refresh" not in result.output

    def test_unconfirmed_revocation_still_completes_locally(
        self, runner, auth_server, stored_connection, state_dir, monkeypatch
    ):
        auth_server.revoke_statuses = [500, 502, 500]
        mock_http(monkeypatch, auth_server)
        monkeypatch.setattr(oauth.time, "sleep", lambda s: None)
        result = runner.invoke(chatgpt_plan, ["logout"])
        assert result.exit_code == 0, result.output
        # bounded retry, then the local sign-out completes anyway
        assert len(auth_server.revoke_requests) == 3
        assert "Signed out locally" in result.output
        assert "Sign in with ChatGPT" in result.output
        saved = storage.read_credentials(state_dir)
        assert "refresh_token" not in saved
        assert saved["client_id"] == ISSUED_CLIENT_ID

    def test_client_error_is_not_retried(
        self, runner, auth_server, stored_connection, state_dir, monkeypatch
    ):
        auth_server.revoke_statuses = [400]
        mock_http(monkeypatch, auth_server)
        result = runner.invoke(chatgpt_plan, ["logout"])
        assert result.exit_code == 0, result.output
        assert len(auth_server.revoke_requests) == 1
        assert "Signed out locally" in result.output
        assert "refresh_token" not in storage.read_credentials(state_dir)

    def test_not_signed_in_is_a_noop(self, runner, auth_server, state_dir, monkeypatch):
        mock_http(monkeypatch, auth_server)
        result = runner.invoke(chatgpt_plan, ["logout"])
        assert result.exit_code == 0, result.output
        assert "Not signed in" in result.output
        assert auth_server.revoke_requests == []

    def test_interrupted_revocation_still_signed_out_locally(
        self, runner, auth_server, stored_connection, state_dir, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)
        monkeypatch.setattr(
            OAuth,
            "revoke_refresh_token",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        result = runner.invoke(chatgpt_plan, ["logout"])
        assert result.exit_code != 0
        # the tokens were wiped before revocation, so aborting cannot
        # leave usable credentials behind
        saved = storage.read_credentials(state_dir)
        assert "refresh_token" not in saved
        assert "access_token" not in saved
        assert saved["client_id"] == ISSUED_CLIENT_ID

    def test_new_login_during_logout_is_not_wiped(
        self, runner, auth_server, stored_connection, state_dir, monkeypatch
    ):
        mock_http(monkeypatch, auth_server)

        def fake_revoke(self, client_id, refresh_token, **kwargs):
            # a new sign-in lands while the remote revocation is in flight
            with storage.locked_store(state_dir) as store:
                store.save_credentials(
                    {
                        "client_id": "new-client",
                        "subject": "new-subject",
                        "access_token": "new-access",
                        "refresh_token": "new-refresh",
                        "scopes": ["openid"],
                        "expires_at": time.time() + 3600,
                        "generation": 9,
                    }
                )
            return True

        monkeypatch.setattr(OAuth, "revoke_refresh_token", fake_revoke)
        result = runner.invoke(chatgpt_plan, ["logout"])
        assert result.exit_code == 0, result.output
        saved = storage.read_credentials(state_dir)
        assert saved["client_id"] == "new-client"
        assert saved["access_token"] == "new-access"
        assert saved["generation"] == 9
