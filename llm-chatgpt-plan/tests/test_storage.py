import json
import multiprocessing
import os
import stat
import threading
import time

import pytest

from llm_chatgpt_plan import storage


def test_state_dir_under_llm_user_dir(user_dir):
    assert storage.state_dir() == user_dir / "chatgpt-plan"


def test_read_returns_none_when_missing(state_dir):
    assert storage.read_credentials(state_dir) is None
    assert storage.read_models(state_dir) is None
    # Read-only access must not create the directory or files
    assert not state_dir.exists()


def test_store_permissions_and_atomic_write(state_dir):
    with storage.locked_store(state_dir) as store:
        store.save_credentials({"client_id": "abc"})
        path = state_dir / "credentials.json"
        assert json.loads(path.read_text()) == {"client_id": "abc"}
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
        # No temp files left behind
        leftovers = [p for p in state_dir.iterdir() if p.name != "runtime.lock"]
        assert sorted(p.name for p in leftovers) == ["credentials.json"]


def test_host_id_persisted_once(state_dir):
    with storage.locked_store(state_dir) as store:
        first = store.ensure_host_id()
        second = store.ensure_host_id()
    assert first == second
    assert first.startswith("urn:uuid:")
    assert stat.S_IMODE((state_dir / "host.json").stat().st_mode) == 0o600


def test_lock_contention_fails_fast(state_dir):
    with (
        storage.locked_store(state_dir),
        pytest.raises(storage.StorageError, match="Another process"),
        storage.locked_store(state_dir, wait=0),
    ):
        pass


def test_corrupt_credentials_raise_storage_error(state_dir):
    state_dir.mkdir(parents=True)
    (state_dir / "credentials.json").write_text("{not json")
    with pytest.raises(storage.StorageError):
        storage.read_credentials(state_dir)


def test_symlink_credentials_rejected(state_dir):
    state_dir.mkdir(parents=True)
    target = state_dir / "elsewhere.json"
    target.write_text("{}")
    os.symlink(target, state_dir / "credentials.json")
    with pytest.raises(storage.StorageError):
        storage.read_credentials(state_dir)
    store = storage.Store(state_dir)
    with pytest.raises(storage.StorageError):
        store.save_credentials({"x": 1})


def test_cached_models_requires_matching_connection():
    creds = {"client_id": "a", "subject": "s1"}
    good = {"client_id": "a", "subject": "s1", "models": [{"slug": "m"}]}
    other_client = {"client_id": "b", "subject": "s1", "models": [{"slug": "m"}]}
    other_subject = {"client_id": "a", "subject": "s2", "models": [{"slug": "m"}]}
    assert storage.cached_models_for(creds, good) == [{"slug": "m"}]
    assert storage.cached_models_for(creds, other_client) is None
    assert storage.cached_models_for(creds, other_subject) is None
    assert storage.cached_models_for(creds, None) is None
    assert storage.cached_models_for({"client_id": "a"}, good) is None


@pytest.mark.parametrize("operation", ["store", "lock", "credentials", "models"])
@pytest.mark.parametrize("dangling", [False, True])
def test_symlink_directory_rejected_without_touching_target(
    tmp_path, operation, dangling
):
    target = tmp_path / "outside"
    if not dangling:
        target.mkdir(mode=0o755)
    link = tmp_path / "chatgpt-plan"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(storage.StorageError, match="symlink"):
        if operation == "store":
            storage.Store(link)
        elif operation == "lock":
            with storage.locked_store(link):
                pass
        elif operation == "credentials":
            storage.read_credentials(link)
        else:
            storage.read_models(link)
    if dangling:
        assert not target.exists()
    else:
        assert stat.S_IMODE(target.stat().st_mode) == 0o755
        assert list(target.iterdir()) == []


def test_directory_swap_after_open_cannot_redirect_write(state_dir, tmp_path):
    target = tmp_path / "outside"
    target.mkdir(mode=0o755)
    moved = state_dir.with_name("original")
    with storage.locked_store(state_dir) as store:
        state_dir.rename(moved)
        state_dir.symlink_to(target, target_is_directory=True)
        store.save_credentials({"access_token": "dummy"})
        assert store.load_credentials() == {"access_token": "dummy"}
    assert (moved / "credentials.json").exists()
    assert list(target.iterdir()) == []
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_directory_swap_before_open_rejected(state_dir, tmp_path, monkeypatch):
    state_dir.mkdir(parents=True)
    target = tmp_path / "outside"
    target.mkdir()
    real_open = os.open

    def swap(path, flags, *args, **kwargs):
        if path == state_dir:
            state_dir.rename(state_dir.with_name("original"))
            state_dir.symlink_to(target, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    with pytest.raises(storage.StorageError), storage.locked_store(state_dir):
        pass
    assert list(target.iterdir()) == []


def _hold_runtime_lock(directory, acquired, release):
    with storage.locked_store(directory):
        acquired.set()
        release.wait(30)


def _hold_login_lock(directory, acquired, release):
    with storage.login_lock(directory):
        acquired.set()
        release.wait(30)


def test_lock_wait_acquires_after_other_process_releases(state_dir):
    ctx = multiprocessing.get_context("fork")
    acquired = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_runtime_lock, args=(state_dir, acquired, release))
    proc.start()
    try:
        assert acquired.wait(10)
        # the lock is contended: a short wait must fail, a generous one waits
        started = time.monotonic()
        with (
            pytest.raises(storage.StorageError, match="Another process"),
            storage.locked_store(state_dir, wait=0.3),
        ):
            pass
        assert time.monotonic() - started >= 0.2
        release.set()
        with storage.locked_store(state_dir, wait=10) as store:
            store.save_credentials({"client_id": "after"})
        assert storage.read_credentials(state_dir)["client_id"] == "after"
    finally:
        release.set()
        proc.join(10)
        assert not proc.is_alive()


def test_login_lock_blocks_second_signin_but_not_runtime(state_dir):
    ctx = multiprocessing.get_context("fork")
    acquired = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_login_lock, args=(state_dir, acquired, release))
    proc.start()
    try:
        assert acquired.wait(10)
        with (
            pytest.raises(storage.StorageError, match="Another sign-in"),
            storage.login_lock(state_dir, wait=0.3),
        ):
            pass
        # inference/refresh do not take the sign-in lock
        with storage.locked_store(state_dir, wait=0) as store:
            store.save_credentials({"client_id": "c"})
    finally:
        release.set()
        proc.join(10)
        assert not proc.is_alive()


def test_lock_wait_is_interruptible(state_dir):
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with storage.locked_store(state_dir):
            acquired.set()
            release.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert acquired.wait(10)
        # bounded wait expires even while the lock stays held
        with (
            pytest.raises(storage.StorageError, match="Another process"),
            storage.locked_store(state_dir, wait=0.3),
        ):
            pass
    finally:
        release.set()
        thread.join(10)


def test_connection_generation_backfills_and_bumps(state_dir):
    with storage.locked_store(state_dir) as store:
        assert store.connection_generation() is None
        store.save_credentials({"client_id": "c", "access_token": "t"})
        # missing generation is backfilled on first update, not on refresh
        assert store.connection_generation() == 1
        assert store.load_credentials()["generation"] == 1
        generation, credentials = store.bump_generation()
        assert generation == 2
        assert credentials["generation"] == 2
        assert store.load_credentials()["generation"] == 2


def test_retire_connection_wipes_tokens_and_keeps_identity(state_dir):
    with storage.locked_store(state_dir) as store:
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
        assert before["refresh_token"] == "r"  # returned for revocation
        assert store.load_credentials() == {
            "client_id": "c",
            "issuer": "i",
            "subject": "s",
            "email": "e",
            "generation": 1,
        }


def test_pending_registration_scoped_to_generation(state_dir):
    with storage.locked_store(state_dir) as store:
        assert store.load_pending() is None
        store.save_pending({"client_id": "c", "generation": 3})
        # a pending record for another generation is unrelated: drop it
        assert store.pending_for(4) is None
        assert store.load_pending() is None
        store.save_pending({"client_id": "c", "generation": 5})
        assert store.pending_for(5) == {"client_id": "c", "generation": 5}
        assert stat.S_IMODE((state_dir / "pending.json").stat().st_mode) == 0o600
