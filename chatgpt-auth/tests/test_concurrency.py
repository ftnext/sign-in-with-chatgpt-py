import threading
import time

from conftest import ISSUED_CLIENT_ID, SUBJECT

from chatgpt_auth import service, storage
from chatgpt_auth.errors import ReauthorizationRequiredError, SignInRequiredError
from chatgpt_auth.oauth import SCOPES
from chatgpt_auth.types import ConnectionIdentity


def connected_record(**overrides):
    record = {
        "client_id": ISSUED_CLIENT_ID,
        "issuer": "https://auth.openai.com",
        "subject": SUBJECT,
        "email": "user@example.com",
        "access_token": "stored-access",
        "refresh_token": "stored-refresh",
        "scopes": SCOPES.split(),
        "expires_at": time.time() + 30,
        "generation": 1,
    }
    record.update(overrides)
    return record


def identity():
    return ConnectionIdentity(
        issuer="https://auth.openai.com",
        client_id=ISSUED_CLIENT_ID,
        subject=SUBJECT,
        generation=1,
    )


def test_concurrent_refreshes_converge_once(creds_path, auth_server):
    """Two needed refreshes collapse to one exchange; both get the token."""
    storage.Store(creds_path).save_credentials(connected_record())
    barrier = threading.Barrier(2)
    tokens, errors = [], []

    def get():
        try:
            barrier.wait(10)
            tokens.append(
                service.ensure_access_token(
                    creds_path, identity(), http_factory=auth_server.client
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=get) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert tokens == ["access-token-1", "access-token-1"]
    refreshes = [
        form
        for form in auth_server.token_requests
        if form["grant_type"] == "refresh_token"
    ]
    assert len(refreshes) == 1
    saved = storage.read_record(creds_path)
    assert saved["access_token"] == "access-token-1"
    assert saved["refresh_token"] == "refresh-token-1"


def test_forced_refreshes_serialize_with_fresh_token(creds_path, auth_server):
    """Forced refreshes run one at a time, each on the latest refresh token."""
    storage.Store(creds_path).save_credentials(connected_record())
    barrier = threading.Barrier(2)
    done, errors = [], []

    def refresh():
        try:
            barrier.wait(10)
            service.force_refresh(
                creds_path, identity(), http_factory=auth_server.client
            )
            done.append(True)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=refresh) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert len(done) == 2
    refreshes = [
        form["refresh_token"]
        for form in auth_server.token_requests
        if form["grant_type"] == "refresh_token"
    ]
    assert refreshes[0] == "stored-refresh"
    # the second refresh must use whatever the first one stored
    saved = storage.read_record(creds_path)
    assert refreshes[1] == saved["refresh_token"]


def test_logout_during_refresh_revives_nothing(creds_path, auth_server):
    """logout while a refresh is in flight: the connection stays retired."""
    storage.Store(creds_path).save_credentials(connected_record())
    auth_server.token_delay = 0.4
    errors = []

    def get():
        try:
            service.ensure_access_token(
                creds_path, identity(), http_factory=auth_server.client
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=get)
    thread.start()
    time.sleep(0.1)
    result = service.sign_out(creds_path, http_factory=auth_server.client)
    thread.join(30)
    assert result.was_connected is True
    saved = storage.read_record(creds_path)
    assert "access_token" not in saved
    # the in-flight refresh either failed before writing or its result was
    # wiped by the logout: never a live token afterwards
    assert not errors or isinstance(
        errors[0], (ReauthorizationRequiredError, SignInRequiredError)
    )


def test_revoke_interruption_leaves_no_local_tokens(creds_path, auth_server):
    """A stalled revoke still leaves the local connection dead."""
    storage.Store(creds_path).save_credentials(connected_record())

    def hanging_factory(**kw):
        raise RuntimeError("network gone")

    result = service.sign_out(creds_path, http_factory=hanging_factory)
    assert result.local_cleared is True
    assert result.remote_revoked is False
    saved = storage.read_record(creds_path)
    assert "access_token" not in saved
    assert saved["generation"] == 2


def test_login_and_refresh_locks_are_independent(creds_path):
    """A held login lock does not block token operations and vice versa."""
    storage.Store(creds_path).save_credentials(connected_record())
    acquired = threading.Event()
    release = threading.Event()
    errors = []

    def hold_login():
        with storage.login_lock(creds_path):
            acquired.set()
            release.wait(30)

    thread = threading.Thread(target=hold_login)
    thread.start()
    try:
        assert acquired.wait(10)
        with storage.locked_store(creds_path, wait=1) as store:
            assert store.load_credentials() is not None
    finally:
        release.set()
        thread.join(10)
    assert errors == []


def test_fresh_token_read_waits_for_lock(creds_path):
    """Even an in-expiry token is re-read under the runtime lock."""
    storage.Store(creds_path).save_credentials(
        connected_record(expires_at=time.time() + 3600)
    )
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with storage.locked_store(creds_path):
            acquired.set()
            release.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    result = []

    def get():
        result.append(service.ensure_access_token(creds_path, identity()))

    try:
        assert acquired.wait(10)
        getter = threading.Thread(target=get)
        getter.start()
        getter.join(1)
        assert getter.is_alive()
        release.set()
        getter.join(10)
        assert result == ["stored-access"]
    finally:
        release.set()
        holder.join(10)
