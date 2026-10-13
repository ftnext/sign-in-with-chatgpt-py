import time

import pytest
from click.testing import CliRunner
from conftest import drive_browser_callback

from chatgpt_auth import storage
from chatgpt_auth.cli import main
from chatgpt_auth.oauth import SCOPES


@pytest.fixture
def runner():
    return CliRunner()


def connected_record(**overrides):
    record = {
        "client_id": "issued-client-123",
        "issuer": "https://auth.openai.com",
        "subject": "user-subject-1",
        "email": "user@example.com",
        "name": "Test User",
        "access_token": "stored-access",
        "refresh_token": "stored-refresh",
        "scopes": SCOPES.split(),
        "expires_at": time.time() + 3600,
        "generation": 1,
    }
    record.update(overrides)
    return record


def save(creds_path, record):
    storage.Store(creds_path).save_credentials(record)


def test_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("login", "status", "print-access-token", "refresh", "logout"):
        assert command in result.output


def test_module_entry_point_help():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chatgpt_auth", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "login" in result.stdout


def test_status_not_connected(runner, creds_path):
    result = runner.invoke(main, ["--credentials", str(creds_path), "status"])
    assert result.exit_code == 0
    assert "Not signed in" in result.output


def test_status_connected(runner, creds_path):
    save(creds_path, connected_record())
    result = runner.invoke(main, ["--credentials", str(creds_path), "status"])
    assert result.exit_code == 0
    assert "Signed in" in result.output
    assert "user@example.com" in result.output
    assert "stored-access" not in result.output


def test_status_corrupt_exits_1(runner, creds_path):
    creds_path.parent.mkdir(parents=True, mode=0o700)
    creds_path.write_text("{broken")
    result = runner.invoke(main, ["--credentials", str(creds_path), "status"])
    assert result.exit_code == 1


def test_print_access_token(runner, creds_path):
    save(creds_path, connected_record())
    result = runner.invoke(
        main, ["--credentials", str(creds_path), "print-access-token"]
    )
    assert result.exit_code == 0
    assert result.output == "stored-access\n"


def test_print_access_token_not_signed_in(runner, creds_path):
    result = runner.invoke(
        main, ["--credentials", str(creds_path), "print-access-token"]
    )
    assert result.exit_code == 1
    assert "stored-access" not in result.output


def test_refresh_command(runner, creds_path, auth_server, patched_http):
    save(creds_path, connected_record())
    result = runner.invoke(main, ["--credentials", str(creds_path), "refresh"])
    assert result.exit_code == 0
    assert "access-token-1" not in result.output
    assert "refreshed" in result.output.lower()
    (form,) = auth_server.token_requests
    assert form["grant_type"] == "refresh_token"


def test_logout_unconfirmed_warns_but_exits_0(
    runner, creds_path, auth_server, patched_http
):
    save(creds_path, connected_record())
    auth_server.revoke_statuses = [500, 500, 500]
    result = runner.invoke(main, ["--credentials", str(creds_path), "logout"])
    assert result.exit_code == 0
    assert "Signed out locally" in result.output
    assert "Could not confirm" in (result.output + result.stderr)
    saved = storage.read_record(creds_path)
    assert "access_token" not in saved


def test_logout_not_signed_in(runner, creds_path):
    result = runner.invoke(main, ["--credentials", str(creds_path), "logout"])
    assert result.exit_code == 0
    assert "Not signed in" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["login", "--timeout", "0"],
        ["login", "--timeout", "601"],
        ["login", "--port", "80"],
        ["login", "--port", "70000"],
    ],
)
def test_login_argument_validation_exit_2(runner, creds_path, args):
    result = runner.invoke(main, ["--credentials", str(creds_path)] + args)
    assert result.exit_code == 2


def test_login_manual_flow(runner, creds_path, auth_server, patched_http, monkeypatch):
    import chatgpt_auth.cli as cli_module

    def show(url, browser_failed):
        drive_browser_callback(
            url,
            code="cli-code",
            client_id="cli-client",
            nonce_holder=auth_server.nonce_holder,
        )

    monkeypatch.setattr(cli_module, "_show_manual_url", show)
    result = runner.invoke(
        main,
        [
            "--credentials",
            str(creds_path),
            "login",
            "--manual",
            "--timeout",
            "10",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    saved = storage.read_record(creds_path)
    assert saved["client_id"] == "cli-client"


def test_env_var_credentials_path(runner, creds_path, monkeypatch):
    save(creds_path, connected_record())
    monkeypatch.setenv("CHATGPT_AUTH_CREDENTIALS", str(creds_path))
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Signed in" in result.output
