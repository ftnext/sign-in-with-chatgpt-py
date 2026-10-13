import time

import pytest
from conftest import ISSUED_CLIENT_ID, SUBJECT

from chatgpt_auth import service, storage
from chatgpt_auth.credentials import Credentials
from chatgpt_auth.errors import (
    ConnectionChangedError,
    ReauthorizationRequiredError,
    SignInRequiredError,
    StorageError,
)
from chatgpt_auth.oauth import SCOPES


def connected_record(**overrides):
    record = {
        "client_id": ISSUED_CLIENT_ID,
        "issuer": "https://auth.openai.com",
        "subject": SUBJECT,
        "email": "user@example.com",
        "name": "Test User",
        "id_token": "stored-id-token",
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


class TestLoad:
    def test_load_binds_to_path_and_connection(self, creds_path):
        save(creds_path, connected_record())
        credentials = Credentials.load(creds_path)
        assert credentials.path == creds_path
        assert credentials.client_id == ISSUED_CLIENT_ID
        assert credentials.subject == SUBJECT
        assert credentials.email == "user@example.com"

    def test_load_missing_record(self, creds_path):
        with pytest.raises(SignInRequiredError):
            Credentials.load(creds_path)
        assert not creds_path.parent.exists()

    def test_load_retired_record(self, creds_path):
        save(creds_path, connected_record())
        with storage.locked_store(creds_path) as store:
            store.retire_connection()
        with pytest.raises(ReauthorizationRequiredError):
            Credentials.load(creds_path)

    def test_load_corrupt_record(self, creds_path):
        creds_path.parent.mkdir(parents=True, mode=0o700)
        creds_path.write_text("{broken")
        with pytest.raises(StorageError):
            Credentials.load(creds_path)

    def test_repr_hides_tokens(self, creds_path):
        save(creds_path, connected_record())
        credentials = Credentials.load(creds_path)
        text = repr(credentials)
        for secret in ("stored-access", "stored-refresh", "stored-id-token"):
            assert secret not in text


class TestGetAccessToken:
    def test_fresh_token_no_http(self, creds_path, auth_server):
        save(creds_path, connected_record())
        credentials = Credentials.load(creds_path)
        assert credentials.get_access_token() == "stored-access"
        assert auth_server.token_requests == []

    def test_refresh_saves_new_token(self, creds_path, auth_server, patched_http):
        save(creds_path, connected_record(expires_at=time.time() + 30))
        credentials = Credentials.load(creds_path)
        assert credentials.get_access_token() == "access-token-1"
        saved = storage.read_record(creds_path)
        assert saved["access_token"] == "access-token-1"
        assert credentials.expires_at == saved["expires_at"]
        assert credentials.expires_at > time.time() + 3000

    def test_generation_backfill_is_not_a_change(self, creds_path, patched_http):
        record = connected_record(expires_at=time.time() + 30)
        del record["generation"]
        save(creds_path, record)
        credentials = Credentials.load(creds_path)
        assert credentials.get_access_token() == "access-token-1"
        assert storage.read_record(creds_path)["generation"] == 1

    def test_logout_by_another_process_needs_login(self, creds_path, patched_http):
        save(creds_path, connected_record())
        credentials = Credentials.load(creds_path)
        service.sign_out(creds_path)
        with pytest.raises(ReauthorizationRequiredError):
            credentials.get_access_token()

    def test_replaced_connection_detected(self, creds_path, patched_http):
        save(creds_path, connected_record())
        credentials = Credentials.load(creds_path)
        save(
            creds_path,
            connected_record(subject=SUBJECT, generation=2),
        )
        with pytest.raises(ConnectionChangedError):
            credentials.get_access_token()

    def test_same_account_new_generation_detected(self, creds_path, patched_http):
        save(creds_path, connected_record())
        credentials = Credentials.load(creds_path)
        record = dict(connected_record())
        record["generation"] = 2
        save(creds_path, record)
        with pytest.raises(ConnectionChangedError):
            credentials.refresh()

    def test_different_account_detected(self, creds_path, patched_http):
        save(creds_path, connected_record())
        credentials = Credentials.load(creds_path)
        save(creds_path, connected_record(subject="someone-else"))
        with pytest.raises(ConnectionChangedError):
            credentials.get_access_token()

    def test_broken_object_refuses_later_use(
        self, creds_path, monkeypatch, auth_server
    ):
        import httpx

        save(creds_path, connected_record(expires_at=time.time() + 30))
        credentials = Credentials.load(creds_path)

        def retire_fails(self):
            raise storage.StorageError("disk gone")

        def handler(request):
            if "token" in str(request.url):
                return httpx.Response(500)
            return auth_server.handler(request)

        monkeypatch.setattr(storage.Store, "retire_connection", retire_fails)
        monkeypatch.setattr(
            service,
            "make_http_client",
            lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw),
        )
        with pytest.raises(StorageError):
            credentials.get_access_token()
        with pytest.raises(StorageError, match="no longer be used"):
            credentials.get_access_token()
        with pytest.raises(StorageError, match="no longer be used"):
            credentials.refresh()
