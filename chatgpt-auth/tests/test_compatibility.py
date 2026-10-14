"""Compatibility with llm-chatgpt-plan as an external oracle.

The old package's modules are loaded from the sibling source tree with a
stubbed ``llm`` dependency (only ``llm.user_dir`` is needed), so no old
runtime dependencies are installed. When the sibling source is absent the
tests skip — a skip never counts as compatibility verified.

Compatibility baseline: repository commit
5282e24165fed269c243892517541cef0994c779 (main at implementation start).
"""

import sys
import threading
import time
import types
from pathlib import Path

import pytest

from chatgpt_auth import storage
from chatgpt_auth.errors import ReauthorizationRequiredError
from chatgpt_auth.oauth import SCOPES

OLD_PACKAGE = (
    Path(__file__).resolve().parents[2] / "llm-chatgpt-plan" / "llm_chatgpt_plan"
)

pytestmark = pytest.mark.skipif(
    not OLD_PACKAGE.is_dir(),
    reason="llm-chatgpt-plan source tree not present",
)


@pytest.fixture
def old_tool(tmp_path, monkeypatch):
    """Load llm_chatgpt_plan.{errors,storage,oauth} without its package init."""
    stub_llm = types.ModuleType("llm")
    stub_llm.user_dir = lambda: tmp_path / "llm-user"
    monkeypatch.setitem(sys.modules, "llm", stub_llm)
    package = types.ModuleType("llm_chatgpt_plan")
    package.__path__ = [str(OLD_PACKAGE)]
    monkeypatch.setitem(sys.modules, "llm_chatgpt_plan", package)
    import importlib

    modules = types.SimpleNamespace()
    for name in ("errors", "storage", "oauth"):
        setattr(
            modules,
            name,
            importlib.import_module(f"llm_chatgpt_plan.{name}"),
        )
    return modules


def connected_record(**overrides):
    record = {
        "client_id": "issued-client-123",
        "issuer": "https://auth.openai.com",
        "subject": "user-subject-1",
        "email": "user@example.com",
        "access_token": "stored-access",
        "refresh_token": "stored-refresh",
        "scopes": SCOPES.split(),
        "expires_at": time.time() + 3600,
        "generation": 1,
    }
    record.update(overrides)
    return record


def test_shared_flock_excludes_both_tools(creds_path, old_tool):
    """The canonical store's runtime lock serializes old and new processes."""
    acquired = threading.Event()
    release = threading.Event()
    old_dir = creds_path.parent

    def hold():
        with old_tool.storage.locked_store(old_dir):
            acquired.set()
            release.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert acquired.wait(10)
        with (
            pytest.raises(storage.StorageError, match="Another process"),
            storage.locked_store(creds_path, wait=0.2),
        ):
            pass
    finally:
        release.set()
        thread.join(10)


def test_shared_login_lock_excludes_both_tools(creds_path, old_tool):
    acquired = threading.Event()
    release = threading.Event()
    old_dir = creds_path.parent

    def hold():
        with old_tool.storage.login_lock(old_dir):
            acquired.set()
            release.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert acquired.wait(10)
        with (
            pytest.raises(storage.StorageError, match="Another sign-in"),
            storage.login_lock(creds_path, wait=0.2),
        ):
            pass
    finally:
        release.set()
        thread.join(10)


def test_old_tool_reads_new_record(creds_path, old_tool):
    storage.Store(creds_path).save_credentials(connected_record())
    record = old_tool.storage.read_credentials(creds_path.parent)
    assert record["access_token"] == "stored-access"
    assert record["client_id"] == "issued-client-123"


def test_new_tool_reads_old_record(creds_path, old_tool):
    old_dir = creds_path.parent
    old_tool.storage.Store(old_dir).save_credentials(connected_record())
    record = storage.read_record(creds_path)
    assert record["access_token"] == "stored-access"


def test_new_credentials_detect_old_logout(creds_path, old_tool):
    from chatgpt_auth.credentials import Credentials

    storage.Store(creds_path).save_credentials(connected_record())
    credentials = Credentials.load(creds_path)
    with old_tool.storage.locked_store(creds_path.parent) as store:
        store.retire_connection()
    with pytest.raises(ReauthorizationRequiredError):
        credentials.get_access_token()


def test_old_tool_rejects_new_retired_record(creds_path, old_tool):
    """A record retired by the new tool is 'not signed in' for the old tool."""
    storage.Store(creds_path).save_credentials(connected_record())
    with storage.locked_store(creds_path) as store:
        store.retire_connection()
    import httpx

    old_oauth = old_tool.oauth.OAuth(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    )
    with pytest.raises(old_tool.errors.AuthError, match="sign_in_required"):
        old_oauth.ensure_access_token(creds_path.parent)


def test_pending_format_matches(creds_path, old_tool):
    with storage.locked_store(creds_path) as store:
        store.save_pending({"client_id": "c", "generation": 3})
    pending = old_tool.storage.Store(creds_path.parent).load_pending()
    assert pending == {"client_id": "c", "generation": 3}


def test_host_id_format_matches(creds_path, old_tool):
    with storage.locked_store(creds_path) as store:
        host_id = store.ensure_host_id()
    old_host_id = old_tool.storage.Store(creds_path.parent).load_host_id()
    assert old_host_id == host_id
    assert host_id.startswith("urn:uuid:")


def test_old_generation_backfill_matches_new_normalization(creds_path, old_tool):
    """A generation-less record refreshes to the same generation both ways."""
    record = connected_record()
    del record["generation"]
    storage.Store(creds_path).save_credentials(record)
    with old_tool.storage.locked_store(creds_path.parent) as store:
        assert store.connection_generation() == 1
    saved = storage.read_record(creds_path)
    assert storage.normalized_generation(saved) == 1


def test_old_model_cache_rejected_on_connection_mismatch(creds_path, old_tool):
    cached = {"client_id": "c1", "subject": "s1", "models": [{"slug": "m"}]}
    other = {"client_id": "c2", "subject": "s1"}
    assert old_tool.storage.cached_models_for(other, cached) is None
