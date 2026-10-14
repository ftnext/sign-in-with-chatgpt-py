import json
import multiprocessing
import os
import stat
import threading
import time

import pytest

from chatgpt_auth import storage


def test_read_returns_none_when_missing(creds_path):
    assert storage.read_record(creds_path) is None
    # Read-only access must not create the directory or files
    assert not creds_path.parent.exists()


def test_store_permissions_and_atomic_write(creds_path):
    with storage.locked_store(creds_path) as store:
        store.save_credentials({"client_id": "abc"})
        assert json.loads(creds_path.read_text()) == {"client_id": "abc"}
        assert stat.S_IMODE(creds_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(creds_path.parent.stat().st_mode) == 0o700
        leftovers = [p for p in creds_path.parent.iterdir() if p.name != "runtime.lock"]
        assert sorted(p.name for p in leftovers) == ["credentials.json"]


def test_aux_files_use_canonical_names(creds_path):
    auth_dir = creds_path.parent
    with storage.locked_store(creds_path) as store:
        store.ensure_host_id()
        store.save_pending({"client_id": "c", "generation": 1})
    names = {p.name for p in auth_dir.iterdir()}
    assert {"host.json", "pending.json", "runtime.lock"} <= names


def test_aux_files_use_suffixed_names(auth_dir):
    creds_path = auth_dir / "custom.json"
    with storage.locked_store(creds_path) as store:
        store.ensure_host_id()
        store.save_pending({"client_id": "c", "generation": 1})
        store.save_credentials({"client_id": "c"})
    names = {p.name for p in auth_dir.iterdir()}
    assert {
        "custom.json.host.json",
        "custom.json.pending.json",
        "custom.json.runtime.lock",
    } <= names


def test_stores_with_custom_names_do_not_share_aux_files(auth_dir):
    for name in ("one.json", "two.json"):
        with storage.locked_store(auth_dir / name) as store:
            store.ensure_host_id()
    names = {p.name for p in auth_dir.iterdir()}
    assert "one.json.host.json" in names
    assert "two.json.host.json" in names
    assert "host.json" not in names


def test_host_id_persisted_once(creds_path):
    with storage.locked_store(creds_path) as store:
        first = store.ensure_host_id()
        second = store.ensure_host_id()
    assert first == second
    assert first.startswith("urn:uuid:")
    host_file = creds_path.parent / "host.json"
    assert stat.S_IMODE(host_file.stat().st_mode) == 0o600


def test_lock_contention_fails_fast(creds_path):
    with (
        storage.locked_store(creds_path),
        pytest.raises(storage.StorageError, match="Another process"),
        storage.locked_store(creds_path, wait=0),
    ):
        pass


def test_corrupt_credentials_raise_storage_error(creds_path):
    creds_path.parent.mkdir(parents=True, mode=0o700)
    creds_path.write_text("{not json")
    with pytest.raises(storage.StorageError):
        storage.read_record(creds_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("client_id", 123),
        ("issuer", None),
        ("subject", ["x"]),
        ("access_token", 1.5),
        ("refresh_token", {}),
        ("email", 42),
        ("scopes", "openid profile"),
        ("scopes", ["ok", 1]),
        ("expires_at", "soon"),
        ("expires_at", True),
        ("expires_at", float("inf")),
        ("expires_at", float("nan")),
        ("generation", "1"),
        ("generation", True),
    ],
)
def test_known_key_type_violations_rejected(creds_path, key, value):
    creds_path.parent.mkdir(parents=True, mode=0o700)
    creds_path.write_text(json.dumps({key: value}))
    with pytest.raises(storage.StorageError):
        storage.read_record(creds_path)


def test_unknown_extra_keys_tolerated(creds_path):
    creds_path.parent.mkdir(parents=True, mode=0o700)
    creds_path.write_text(
        json.dumps({"client_id": "c", "future_field": {"any": "thing"}})
    )
    assert storage.read_record(creds_path)["future_field"] == {"any": "thing"}


def test_symlink_credentials_rejected(creds_path):
    creds_path.parent.mkdir(parents=True, mode=0o700)
    target = creds_path.parent / "elsewhere.json"
    target.write_text("{}")
    os.symlink(target, creds_path)
    with pytest.raises(storage.StorageError):
        storage.read_record(creds_path)
    store = storage.Store(creds_path)
    with pytest.raises(storage.StorageError):
        store.save_credentials({"x": 1})


def test_symlink_directory_rejected(tmp_path):
    target = tmp_path / "outside"
    target.mkdir(mode=0o755)
    link = tmp_path / "auth"
    link.symlink_to(target, target_is_directory=True)
    creds_path = link / "credentials.json"
    with pytest.raises(storage.StorageError, match="symlink"):
        storage.read_record(creds_path)
    with pytest.raises(storage.StorageError, match="symlink"):
        storage.Store(creds_path)
    with (
        pytest.raises(storage.StorageError, match="symlink"),
        storage.locked_store(creds_path),
    ):
        pass
    assert list(target.iterdir()) == []


def test_group_readable_directory_rejected(auth_dir):
    auth_dir.mkdir(mode=0o755)
    with pytest.raises(storage.StorageError, match="group or other"):
        storage.Store(auth_dir / "credentials.json")


def test_group_writable_lock_target_directory_rejected(creds_path):
    creds_path.parent.mkdir(parents=True, mode=0o770)
    os.chmod(creds_path.parent, 0o770)
    with pytest.raises(storage.StorageError):
        storage.Store(creds_path)


def test_directory_swap_after_open_cannot_redirect_write(creds_path, tmp_path):
    auth_dir = creds_path.parent
    target = tmp_path / "outside"
    target.mkdir(mode=0o755)
    moved = auth_dir.with_name("original")
    with storage.locked_store(creds_path) as store:
        auth_dir.rename(moved)
        auth_dir.symlink_to(target, target_is_directory=True)
        store.save_credentials({"access_token": "dummy"})
        assert store.load_credentials() == {"access_token": "dummy"}
    assert (moved / "credentials.json").exists()
    assert list(target.iterdir()) == []


def _hold_runtime_lock(creds_path, acquired, release):
    with storage.locked_store(creds_path):
        acquired.set()
        release.wait(30)


def _hold_login_lock(creds_path, acquired, release):
    with storage.login_lock(creds_path):
        acquired.set()
        release.wait(30)


def test_lock_wait_acquires_after_other_process_releases(creds_path):
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_runtime_lock, args=(creds_path, acquired, release))
    proc.start()
    try:
        assert acquired.wait(10)
        started = time.monotonic()
        with (
            pytest.raises(storage.StorageError, match="Another process"),
            storage.locked_store(creds_path, wait=0.3),
        ):
            pass
        assert time.monotonic() - started >= 0.2
        release.set()
        with storage.locked_store(creds_path, wait=10) as store:
            store.save_credentials({"client_id": "after"})
        assert storage.read_record(creds_path)["client_id"] == "after"
    finally:
        release.set()
        proc.join(10)
        assert not proc.is_alive()


def test_login_lock_blocks_second_signin_but_not_runtime(creds_path):
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_login_lock, args=(creds_path, acquired, release))
    proc.start()
    try:
        assert acquired.wait(10)
        with (
            pytest.raises(storage.StorageError, match="Another sign-in"),
            storage.login_lock(creds_path, wait=0.3),
        ):
            pass
        # token operations do not take the sign-in lock
        with storage.locked_store(creds_path, wait=0) as store:
            store.save_credentials({"client_id": "c"})
    finally:
        release.set()
        proc.join(10)
        assert not proc.is_alive()


def test_lock_wait_is_interruptible(creds_path):
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with storage.locked_store(creds_path):
            acquired.set()
            release.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert acquired.wait(10)
        with (
            pytest.raises(storage.StorageError, match="Another process"),
            storage.locked_store(creds_path, wait=0.3),
        ):
            pass
    finally:
        release.set()
        thread.join(10)


def test_connection_generation_backfills_and_bumps(creds_path):
    with storage.locked_store(creds_path) as store:
        assert store.connection_generation() is None
        store.save_credentials({"client_id": "c", "access_token": "t"})
        assert store.connection_generation() == 1
        assert store.load_credentials()["generation"] == 1
        generation, credentials = store.bump_generation()
        assert generation == 2
        assert credentials["generation"] == 2


def test_retire_connection_wipes_tokens_and_keeps_identity(creds_path):
    with storage.locked_store(creds_path) as store:
        store.save_credentials(
            {
                "client_id": "c",
                "issuer": "i",
                "subject": "s",
                "email": "e",
                "access_token": "a",
                "refresh_token": "r",
                "id_token": "it",
                "scopes": ["openid"],
                "expires_at": 1.0,
            }
        )
        generation, before = store.retire_connection()
        assert generation == 1
        assert before["refresh_token"] == "r"
        assert store.load_credentials() == {
            "client_id": "c",
            "issuer": "i",
            "subject": "s",
            "email": "e",
            "generation": 1,
        }


def test_pending_registration_scoped_to_generation(creds_path):
    with storage.locked_store(creds_path) as store:
        assert store.load_pending() is None
        store.save_pending({"client_id": "c", "generation": 3})
        assert store.pending_for(4) is None
        assert store.load_pending() is None
        store.save_pending({"client_id": "c", "generation": 5})
        assert store.pending_for(5) == {"client_id": "c", "generation": 5}
        pending_file = creds_path.parent / "pending.json"
        assert stat.S_IMODE(pending_file.stat().st_mode) == 0o600


def test_read_record_never_writes(creds_path):
    creds_path.parent.mkdir(parents=True, mode=0o700)
    creds_path.write_text(json.dumps({"client_id": "c"}))
    before = {p.name for p in creds_path.parent.iterdir()}
    storage.read_record(creds_path)
    after = {p.name for p in creds_path.parent.iterdir()}
    assert before == after == {"credentials.json"}


def test_models_file_never_touched(creds_path):
    auth_dir = creds_path.parent
    auth_dir.mkdir(parents=True, mode=0o700)
    models = auth_dir / "models.json"
    models.write_text(json.dumps({"models": [{"slug": "m"}]}))
    with storage.locked_store(creds_path) as store:
        store.save_credentials({"client_id": "c"})
        store.retire_connection()
    assert json.loads(models.read_text()) == {"models": [{"slug": "m"}]}


def test_write_fsyncs_directory(creds_path, monkeypatch):
    """os.replace is followed by a directory fsync for durability."""
    import os

    calls = []
    real_fsync = os.fsync

    def spy(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", spy)
    storage.Store(creds_path).save_credentials({"client_id": "c"})
    assert len(set(calls)) >= 2
