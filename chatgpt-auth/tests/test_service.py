import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import (
    ISSUED_CLIENT_ID,
    SUBJECT,
    FakeTTY,
    drive_browser_callback,
    make_id_token,
    paste_url,
)

from chatgpt_auth import service, storage
from chatgpt_auth.errors import (
    AuthError,
    LoginTimeoutError,
    NetworkError,
    PlanPermissionError,
    ReauthorizationRequiredError,
    SignInRequiredError,
    StorageError,
    TokenOutcomeUnknownError,
)
from chatgpt_auth.oauth import SCOPES
from chatgpt_auth.types import ConnectionIdentity


def sign_in(creds_path, auth_server, **kwargs):
    kwargs.setdefault("http_factory", auth_server.client)
    kwargs.setdefault("timeout", 10)
    return service.sign_in(creds_path, **kwargs)


def identity_of(record):
    return ConnectionIdentity(
        issuer=record["issuer"],
        client_id=record["client_id"],
        subject=record["subject"],
        generation=storage.normalized_generation(record),
    )


class TestSignIn:
    def test_manual_login_via_pasted_url(self, creds_path, auth_server):
        tty = FakeTTY()
        shown = []

        def show(url, browser_failed):
            shown.append(browser_failed)
            paste, nonce = paste_url(url, code="c1", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)

        try:
            record = sign_in(
                creds_path,
                auth_server,
                manual=True,
                show_url=show,
                input_stream=tty,
                open_browser=lambda url: pytest.fail("browser used"),
            )
        finally:
            tty.close()
        assert shown == [False]
        assert record["client_id"] == ISSUED_CLIENT_ID
        saved = storage.read_record(creds_path)
        assert saved["access_token"] == "access-token-1"
        assert saved["generation"] == 1
        (form,) = auth_server.token_requests
        assert form["code"] == "c1"

    def test_browser_flow(self, creds_path, auth_server):
        opened = []

        def drive(url):
            opened.append(url)
            threading.Thread(
                target=drive_browser_callback,
                args=(url,),
                kwargs={"nonce_holder": auth_server.nonce_holder},
                daemon=True,
            ).start()
            return True

        record = sign_in(creds_path, auth_server, open_browser=drive)
        assert record["client_id"] == ISSUED_CLIENT_ID
        params = parse_qs(urlsplit(opened[0]).query)
        assert params["agent_name_hint"] == ["chatgpt-auth"]

    def test_custom_agent_name(self, creds_path, auth_server):
        opened = []

        def drive(url):
            opened.append(url)
            threading.Thread(
                target=drive_browser_callback,
                args=(url,),
                kwargs={"nonce_holder": auth_server.nonce_holder},
                daemon=True,
            ).start()
            return True

        sign_in(creds_path, auth_server, open_browser=drive, agent_name="my-python-app")
        params = parse_qs(urlsplit(opened[0]).query)
        assert params["agent_name_hint"] == ["my-python-app"]

    def test_browser_failure_falls_back_to_manual(self, creds_path, auth_server):
        tty = FakeTTY()
        shown = []

        def show(url, browser_failed):
            shown.append(browser_failed)
            paste, nonce = paste_url(url, code="c2", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)

        try:
            record = sign_in(
                creds_path,
                auth_server,
                open_browser=lambda url: False,
                show_url=show,
                input_stream=tty,
            )
        finally:
            tty.close()
        assert shown == [True]
        assert record["client_id"] == ISSUED_CLIENT_ID

    def test_timeout(self, creds_path, auth_server):
        with pytest.raises(LoginTimeoutError):
            sign_in(
                creds_path,
                auth_server,
                timeout=1,
                open_browser=lambda url: False,
                input_stream=None,
            )

    def test_stale_attempt_aborted_by_generation_bump(self, creds_path, auth_server):
        tty = FakeTTY()

        def show(url, browser_failed):
            paste, nonce = paste_url(url, code="c3", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            with storage.locked_store(creds_path) as store:
                store.bump_generation()
            tty.feed(paste)

        try:
            with pytest.raises(AuthError, match="connection_changed"):
                sign_in(
                    creds_path,
                    auth_server,
                    manual=True,
                    show_url=show,
                    input_stream=tty,
                )
        finally:
            tty.close()
        saved = storage.read_record(creds_path)
        assert saved == {"generation": 1}

    def test_replace_failure_keeps_current_connection(
        self, creds_path, auth_server, stored_connection
    ):
        auth_server.token_status = 500
        tty = FakeTTY()

        def show(url, browser_failed):
            paste, nonce = paste_url(url, code="c4", client_id="new-client")
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)

        try:
            with pytest.raises(TokenOutcomeUnknownError):
                sign_in(
                    creds_path,
                    auth_server,
                    replace=True,
                    manual=True,
                    show_url=show,
                    input_stream=tty,
                )
        finally:
            tty.close()
        saved = storage.read_record(creds_path)
        assert saved["access_token"] == stored_connection["access_token"]
        assert saved["refresh_token"] == stored_connection["refresh_token"]

    def test_reauth_reuses_client_id_without_agent_hint(
        self, creds_path, auth_server, stored_connection
    ):
        opened = []

        def drive(url):
            opened.append(url)
            threading.Thread(
                target=drive_browser_callback,
                args=(url,),
                kwargs={
                    "client_id": ISSUED_CLIENT_ID,
                    "nonce_holder": auth_server.nonce_holder,
                },
                daemon=True,
            ).start()
            return True

        record = sign_in(creds_path, auth_server, open_browser=drive)
        params = parse_qs(urlsplit(opened[0]).query)
        assert params["client_id"] == [ISSUED_CLIENT_ID]
        assert "agent_name_hint" not in params
        assert record["generation"] == 2

    def test_reauth_rejects_different_subject(
        self, creds_path, auth_server, stored_connection, rsa_key
    ):
        auth_server.id_token_override = lambda cid, nonce: make_id_token(
            rsa_key, subject="someone-else", nonce=nonce
        )

        def drive(url):
            threading.Thread(
                target=drive_browser_callback,
                args=(url,),
                kwargs={
                    "client_id": ISSUED_CLIENT_ID,
                    "nonce_holder": auth_server.nonce_holder,
                },
                daemon=True,
            ).start()
            return True

        with pytest.raises(AuthError, match="account_mismatch"):
            sign_in(creds_path, auth_server, open_browser=drive)

    def test_missing_plan_scope_rejected(self, creds_path, auth_server, rsa_key):
        def token_handler(request):
            url = str(request.url)
            if "token" in url:
                payload = {
                    "access_token": "a",
                    "refresh_token": "r",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "openid profile email",
                    "id_token": make_id_token(
                        rsa_key, nonce=auth_server.nonce_holder["nonce"]
                    ),
                }
                import httpx

                return httpx.Response(200, json=payload)
            return auth_server.handler(request)

        import httpx

        factory = lambda **kw: httpx.Client(
            transport=httpx.MockTransport(token_handler), **kw
        )
        tty = FakeTTY()

        def show(url, browser_failed):
            paste, nonce = paste_url(url, code="c5", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)

        try:
            with pytest.raises(AuthError, match="plan_permission_required"):
                sign_in(
                    creds_path,
                    auth_server,
                    manual=True,
                    http_factory=factory,
                    show_url=show,
                    input_stream=tty,
                )
        finally:
            tty.close()
        assert storage.read_record(creds_path) is None or not storage.read_record(
            creds_path
        ).get("access_token")

    def test_failed_exchange_leaves_pending_for_retry(self, creds_path, auth_server):
        auth_server.token_status = 500

        def drive(url):
            drive_browser_callback(url, nonce_holder=auth_server.nonce_holder)
            return True

        with pytest.raises(TokenOutcomeUnknownError):
            sign_in(creds_path, auth_server, open_browser=drive)
        pending = storage.Store(creds_path).load_pending()
        assert pending["client_id"] == ISSUED_CLIENT_ID


class TestEnsureAccessToken:
    def test_fresh_token_returned_without_http(
        self, creds_path, stored_connection, auth_server
    ):
        identity = identity_of(stored_connection)
        token = service.ensure_access_token(creds_path, identity)
        assert token == "stored-access"
        assert auth_server.token_requests == []

    def test_expired_token_refreshed_and_saved(
        self, creds_path, stored_connection, auth_server
    ):
        store = storage.Store(creds_path)
        store.save_credentials({**stored_connection, "expires_at": time.time() + 30})
        identity = identity_of(stored_connection)
        token = service.ensure_access_token(
            creds_path, identity, http_factory=auth_server.client
        )
        assert token == "access-token-1"
        (form,) = auth_server.token_requests
        assert form["grant_type"] == "refresh_token"
        assert form["refresh_token"] == "stored-refresh"
        saved = storage.read_record(creds_path)
        assert saved["access_token"] == "access-token-1"
        assert saved["expires_at"] > time.time() + 3000

    def test_record_without_generation_normalizes(self, creds_path, auth_server):
        record = {
            "client_id": ISSUED_CLIENT_ID,
            "issuer": "https://auth.openai.com",
            "subject": SUBJECT,
            "access_token": "a",
            "refresh_token": "r",
            "scopes": SCOPES.split(),
            "expires_at": time.time() + 30,
        }
        storage.Store(creds_path).save_credentials(record)
        identity = identity_of(record)
        token = service.ensure_access_token(
            creds_path, identity, http_factory=auth_server.client
        )
        assert token == "access-token-1"
        saved = storage.read_record(creds_path)
        assert saved["generation"] == 1
        # the backfilled generation is not a connection change
        assert service.ensure_access_token(creds_path, identity) == "access-token-1"

    def test_no_connection(self, creds_path):
        identity = ConnectionIdentity("i", "c", "s", 1)
        with pytest.raises(SignInRequiredError):
            service.ensure_access_token(creds_path, identity)

    def test_missing_plan_scope(self, creds_path):
        record = {
            "client_id": ISSUED_CLIENT_ID,
            "issuer": "https://auth.openai.com",
            "subject": SUBJECT,
            "access_token": "a",
            "refresh_token": "r",
            "scopes": ["openid"],
            "expires_at": time.time() + 3600,
            "generation": 1,
        }
        storage.Store(creds_path).save_credentials(record)
        with pytest.raises(PlanPermissionError):
            service.ensure_access_token(creds_path, identity_of(record))

    def test_revocation_retires_and_asks_for_login(
        self, creds_path, stored_connection, auth_server
    ):
        storage.Store(creds_path).save_credentials(
            {
                **stored_connection,
                "expires_at": time.time() - 5,
                "generation": 1,
            }
        )
        auth_server.token_status = 400
        auth_server.token_error = {"error": "invalid_grant"}
        with pytest.raises(ReauthorizationRequiredError):
            service.ensure_access_token(
                creds_path,
                identity_of(stored_connection),
                http_factory=auth_server.client,
            )
        saved = storage.read_record(creds_path)
        assert "access_token" not in saved
        assert saved["generation"] == 2

    def test_unknown_outcome_retires_conservatively(
        self, creds_path, stored_connection, auth_server
    ):
        storage.Store(creds_path).save_credentials(
            {**stored_connection, "expires_at": time.time() - 5}
        )
        auth_server.token_status = 500
        with pytest.raises(ReauthorizationRequiredError):
            service.ensure_access_token(
                creds_path,
                identity_of(stored_connection),
                http_factory=auth_server.client,
            )
        saved = storage.read_record(creds_path)
        assert "access_token" not in saved

    def test_unsent_failure_keeps_state(
        self, creds_path, stored_connection, auth_server
    ):
        import httpx

        storage.Store(creds_path).save_credentials(
            {**stored_connection, "expires_at": time.time() - 5}
        )

        def handler(request):
            if "token" in str(request.url):
                raise httpx.ConnectError("refused", request=request)
            return auth_server.handler(request)

        factory = lambda **kw: httpx.Client(
            transport=httpx.MockTransport(handler), **kw
        )
        with pytest.raises(NetworkError):
            service.ensure_access_token(
                creds_path,
                identity_of(stored_connection),
                http_factory=factory,
            )
        saved = storage.read_record(creds_path)
        assert saved["access_token"] == "stored-access"

    def test_safe_rejection_keeps_state(
        self, creds_path, stored_connection, auth_server
    ):
        storage.Store(creds_path).save_credentials(
            {**stored_connection, "expires_at": time.time() - 5}
        )
        auth_server.token_status = 400
        auth_server.token_error = {"error": "invalid_client"}
        with pytest.raises(NetworkError):
            service.ensure_access_token(
                creds_path,
                identity_of(stored_connection),
                http_factory=auth_server.client,
            )
        assert storage.read_record(creds_path)["access_token"] == "stored-access"

    def test_scope_loss_retires(self, creds_path, stored_connection, auth_server):
        storage.Store(creds_path).save_credentials(
            {**stored_connection, "expires_at": time.time() - 5}
        )
        auth_server.id_token_override = None

        def handler(request):
            url = str(request.url)
            if "token" in url:
                import httpx

                payload = {
                    "access_token": "a2",
                    "refresh_token": "r2",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "openid profile email",
                }
                return httpx.Response(200, json=payload)
            return auth_server.handler(request)

        import httpx

        factory = lambda **kw: httpx.Client(
            transport=httpx.MockTransport(handler), **kw
        )
        with pytest.raises(ReauthorizationRequiredError):
            service.ensure_access_token(
                creds_path,
                identity_of(stored_connection),
                http_factory=factory,
            )
        assert "access_token" not in storage.read_record(creds_path)

    def test_subject_mismatch_retires(
        self, creds_path, stored_connection, auth_server, rsa_key
    ):
        storage.Store(creds_path).save_credentials(
            {**stored_connection, "expires_at": time.time() - 5}
        )
        auth_server.id_token_override = lambda cid, nonce: make_id_token(
            rsa_key, subject="someone-else", nonce=nonce
        )
        with pytest.raises(ReauthorizationRequiredError):
            service.ensure_access_token(
                creds_path,
                identity_of(stored_connection),
                http_factory=auth_server.client,
            )
        assert "access_token" not in storage.read_record(creds_path)

    def test_save_failure_returns_no_token(
        self, creds_path, stored_connection, auth_server, monkeypatch
    ):
        storage.Store(creds_path).save_credentials(
            {**stored_connection, "expires_at": time.time() - 5}
        )
        real_save = storage.Store.save_credentials
        calls = []

        def failing_save(self, record):
            calls.append(record)
            if "access-token-1" == record.get("access_token"):
                raise storage.StorageError("disk full")
            return real_save(self, record)

        monkeypatch.setattr(storage.Store, "save_credentials", failing_save)
        with pytest.raises(storage.StorageError):
            service.ensure_access_token(
                creds_path,
                identity_of(stored_connection),
                http_factory=auth_server.client,
            )
        assert "access_token" not in storage.read_record(creds_path)


class TestForceRefresh:
    def test_force_refresh_updates_and_saves(
        self, creds_path, stored_connection, auth_server
    ):
        service.force_refresh(
            creds_path,
            identity_of(stored_connection),
            http_factory=auth_server.client,
        )
        (form,) = auth_server.token_requests
        assert form["grant_type"] == "refresh_token"
        assert storage.read_record(creds_path)["access_token"] == "access-token-1"


class TestSignOut:
    def test_logout_retires_and_revokes(
        self, creds_path, stored_connection, auth_server
    ):
        result = service.sign_out(creds_path, http_factory=auth_server.client)
        assert result.local_cleared is True
        assert result.was_connected is True
        assert result.remote_revoked is True
        (form,) = auth_server.revoke_requests
        assert form["token"] == "stored-refresh"
        saved = storage.read_record(creds_path)
        assert "access_token" not in saved
        assert saved["client_id"] == ISSUED_CLIENT_ID

    def test_logout_without_connection(self, creds_path, auth_server):
        result = service.sign_out(creds_path, http_factory=auth_server.client)
        assert result.was_connected is False
        assert result.remote_revoked is None
        assert auth_server.revoke_requests == []

    def test_unconfirmed_revocation_reported(
        self, creds_path, stored_connection, auth_server
    ):
        auth_server.revoke_statuses = [500, 500, 500]
        result = service.sign_out(
            creds_path,
            http_factory=lambda **kw: auth_server.client(**kw),
        )
        assert result.local_cleared is True
        assert result.remote_revoked is False
        saved = storage.read_record(creds_path)
        assert "access_token" not in saved


class TestStatus:
    def test_not_connected(self, creds_path):
        report = service.connection_status(creds_path)
        assert report.state == "not_connected"
        assert report.connected is False

    def test_retired_shows_not_connected(self, creds_path, stored_connection):
        with storage.locked_store(creds_path) as store:
            store.retire_connection()
        report = service.connection_status(creds_path)
        assert report.state == "not_connected"

    def test_active(self, creds_path, stored_connection):
        report = service.connection_status(creds_path)
        assert report.state == "active"
        assert report.connected is True
        assert report.email == "user@example.com"
        assert report.expires_at == stored_connection["expires_at"]

    def test_needs_refresh(self, creds_path, stored_connection):
        storage.Store(creds_path).save_credentials(
            {**stored_connection, "expires_at": time.time() + 30}
        )
        report = service.connection_status(creds_path)
        assert report.state == "needs_refresh"

    def test_status_creates_nothing(self, creds_path):
        service.connection_status(creds_path)
        assert not creds_path.parent.exists()

    def test_corrupt_raises(self, creds_path):
        creds_path.parent.mkdir(parents=True, mode=0o700)
        creds_path.write_text("{broken")
        with pytest.raises(StorageError):
            service.connection_status(creds_path)


def test_interrupt_during_refresh_retires(creds_path, stored_connection, auth_server):
    """Ctrl-C mid-refresh: the possibly-consumed token is retired."""
    import httpx

    storage.Store(creds_path).save_credentials(
        {**stored_connection, "expires_at": time.time() - 5, "generation": 1}
    )

    def handler(request):
        if "token" in str(request.url):
            raise KeyboardInterrupt
        return auth_server.handler(request)

    factory = lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw)
    with pytest.raises(KeyboardInterrupt):
        service.ensure_access_token(
            creds_path,
            identity_of(stored_connection),
            http_factory=factory,
        )
    saved = storage.read_record(creds_path)
    assert "access_token" not in saved


def test_manual_login_reads_stdin_by_default(creds_path, auth_server, monkeypatch):
    """login(manual=True) without input_stream still accepts TTY paste."""
    import sys

    tty = FakeTTY()
    monkeypatch.setattr(sys, "stdin", tty)

    def show(url, browser_failed):
        paste, nonce = paste_url(url, code="c9", client_id="cli-client")
        auth_server.nonce_holder["nonce"] = nonce
        tty.feed(paste)

    try:
        record = sign_in(creds_path, auth_server, manual=True, show_url=show)
    finally:
        tty.close()
    assert record["client_id"] == "cli-client"
