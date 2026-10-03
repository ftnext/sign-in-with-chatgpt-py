"""Regression tests for redaction and service/auth failure classification."""

import time

import httpx
import pytest
from click.testing import CliRunner

from llm_chatgpt_plan import client, storage
from llm_chatgpt_plan.commands import chatgpt_plan
from llm_chatgpt_plan.errors import REAUTH_CODES, ApiError, AuthError
from llm_chatgpt_plan.oauth import OAuth


@pytest.mark.parametrize(
    "body",
    [
        {"detail": "echo stored-access"},
        {"detail": {"nested": "stored-access"}},
        {
            "error": {
                "code": "stored-access",
                "param": "stored-access",
                "message": "stored-access",
            }
        },
    ],
)
def test_models_command_redacts_all_error_fields(stored_connection, monkeypatch, body):
    monkeypatch.setattr(
        client,
        "make_http_client",
        lambda **kw: httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, json=body)
            ),
            **kw,
        ),
    )
    result = CliRunner().invoke(chatgpt_plan, ["models", "--refresh"])
    assert result.exit_code == 1
    assert "stored-access" not in result.output
    assert "[redacted]" in result.output


@pytest.mark.parametrize("code", sorted(REAUTH_CODES))
@pytest.mark.parametrize("nested", [False, True])
def test_terminal_refresh_error_requires_login(auth_server, code, nested):
    auth_server.token_status = 400
    auth_server.token_error = {"error": {"code": code} if nested else code}
    with pytest.raises(AuthError, match="reauthorization_required"):
        OAuth(auth_server.client()).token_request({"refresh_token": "dummy"})


@pytest.mark.parametrize("status", [429, 500, 503])
def test_temporary_refresh_failure_preserves_connection(
    auth_server, stored_connection, state_dir, monkeypatch, status
):
    stored_connection["expires_at"] = time.time() - 10
    with storage.locked_store(state_dir) as store:
        store.save_credentials(stored_connection)
    auth_server.token_status = status
    auth_server.token_error = {
        "error": {"code": "server_error", "message": "echo stored-refresh"}
    }
    monkeypatch.setattr(
        client, "make_http_client", lambda **kw: auth_server.client(**kw)
    )
    result = CliRunner().invoke(chatgpt_plan, ["models", "--refresh"])
    assert result.exit_code == 1
    assert f"HTTP {status}" in result.output
    assert "[redacted]" in result.output
    assert "stored-refresh" not in result.output
    assert "llm chatgpt-plan login" not in result.output
    assert storage.read_credentials(state_dir) == stored_connection
    assert auth_server.models_requests == []


def test_invalid_client_is_configuration_error(auth_server):
    auth_server.token_status = 400
    auth_server.token_error = {"error": "invalid_client"}
    with pytest.raises(ApiError, match="invalid_client"):
        OAuth(auth_server.client()).token_request({})


@pytest.mark.parametrize(
    "status,code,expected",
    [
        (401, "invalid_token", "reauthorization_required"),
        (403, "subscription_sharing_usage_limit_exceeded", "settings/usage"),
        (403, "subscription_sharing_usage_unavailable", "settings/usage"),
        (403, "chatpass_v2_scope_not_authorized", "chatpass_v2_scope_not_authorized"),
        (503, "subscription_sharing_user_unavailable", "HTTP 503"),
    ],
)
def test_model_listing_classifies_api_errors(status, code, expected):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json={"error": {"code": code}})
            )
        ) as http,
        pytest.raises((ApiError, AuthError), match=expected),
    ):
        client.fetch_models(http, "dummy")


@pytest.mark.parametrize("operation", ["models", "token"])
def test_transport_failure_is_safe_service_error(auth_server, operation):
    def fail(request):
        raise httpx.ConnectError("network error echo dummy-secret")

    with (
        httpx.Client(transport=httpx.MockTransport(fail)) as http,
        pytest.raises(ApiError) as info,
    ):
        if operation == "models":
            client.fetch_models(http, "dummy-secret")
        else:
            oauth = OAuth(http)
            oauth.discovery = {"token_endpoint": "https://auth.openai.com/token"}
            oauth.token_request({"refresh_token": "dummy-secret"})
    assert "dummy-secret" not in str(info.value)
    assert "[redacted]" in str(info.value)
    assert "reauthorization" not in str(info.value)
