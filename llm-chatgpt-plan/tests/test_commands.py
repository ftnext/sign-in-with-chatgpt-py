import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from click.testing import CliRunner
from conftest import ISSUED_CLIENT_ID, SUBJECT, drive_browser_callback

from llm_chatgpt_plan import client, commands, storage
from llm_chatgpt_plan.commands import chatgpt_plan


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

    def test_browser_failure(self, runner, auth_server, state_dir, monkeypatch):
        mock_http(monkeypatch, auth_server)
        monkeypatch.setattr(commands.webbrowser, "open", lambda url: False)
        result = runner.invoke(chatgpt_plan, ["login"])
        assert result.exit_code != 0
        assert "browser" in result.output.lower()

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
