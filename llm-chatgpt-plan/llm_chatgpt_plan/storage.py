"""Protected local storage for the single ChatGPT plan connection.

Files live in ``llm.user_dir() / "chatgpt-plan"``:

- ``credentials.json``: the one active connection (issued client ID, verified
  subject/issuer, tokens, scope, expiry, connection generation). After
  logout it keeps only the client ID, identity, and generation.
- ``pending.json``: an issued client ID from a dynamic registration whose
  code exchange never finished, tied to the generation it was issued for.
- ``host.json``: the persistent OAuth host ID for this installation.
- ``models.json``: cached model list plus the client ID/subject it belongs to.
- ``runtime.lock``: advisory lock serializing updates (fcntl; macOS/Linux).
- ``login.lock``: advisory lock serializing sign-in attempts only.
"""

import json
import os
import secrets
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import llm

STATE_DIR_NAME = "chatgpt-plan"
CREDENTIALS_FILE = "credentials.json"
PENDING_FILE = "pending.json"
HOST_FILE = "host.json"
MODELS_FILE = "models.json"
LOCK_FILE = "runtime.lock"
LOGIN_LOCK_FILE = "login.lock"
LOCK_WAIT_SECONDS = 5.0


class StorageError(Exception):
    """A safe, public message about local credential storage."""


def state_dir() -> Path:
    return llm.user_dir() / STATE_DIR_NAME


@contextmanager
def _open_directory(directory: Path, *, create=False):
    """Anchor operations to a directory opened without following a symlink.

    Resolve the parent only: macOS may legitimately symlink a user-path
    ancestor (such as /var). Never resolve the plugin directory itself.
    """
    directory = directory.parent.resolve() / directory.name
    fd = None
    try:
        if directory.is_symlink():
            raise StorageError("Refusing symlinked ChatGPT plan directory")
        if create:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        yield fd
    except OSError as exc:
        raise StorageError(f"Could not access ChatGPT plan directory: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _read_json(path: Path, dir_fd=None) -> dict | None:
    """Read a JSON object file. Returns None when it does not exist.

    Never creates directories or files; safe to call from model
    registration which must not write anything.
    """
    if dir_fd is None:
        if path.parent.is_symlink():
            raise StorageError("Refusing symlinked ChatGPT plan directory")
        if not path.parent.exists():
            return None
        with _open_directory(path.parent) as fd:
            return _read_json(path, dir_fd=fd)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        with os.fdopen(fd) as stream:
            data = json.load(stream)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise StorageError(f"Could not parse {path.name}: {exc}") from exc
    except OSError as exc:
        raise StorageError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise StorageError(f"Unexpected content in {path.name}")
    return data


def _write_json(path: Path, data: dict, dir_fd: int) -> None:
    """Atomically replace a JSON file with mode 0600, never via symlink."""
    import stat

    try:
        if stat.S_ISLNK(
            os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        ):
            raise StorageError(f"Refusing to write through symlink: {path.name}")
    except FileNotFoundError:
        pass
    name = path.name + "." + secrets.token_hex(16)
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=dir_fd,
    )
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    finally:
        try:
            os.unlink(name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass


def read_credentials(directory: Path | None = None) -> dict | None:
    return _read_json((directory or state_dir()) / CREDENTIALS_FILE)


def read_models(directory: Path | None = None) -> dict | None:
    return _read_json((directory or state_dir()) / MODELS_FILE)


def cached_models_for(credentials: dict, cached: dict | None) -> list | None:
    """Return the cached model list only if it belongs to this connection."""
    if not cached or not credentials.get("subject"):
        return None
    if cached.get("client_id") != credentials.get("client_id"):
        return None
    if cached.get("subject") != credentials.get("subject"):
        return None
    models = cached.get("models")
    return models if isinstance(models, list) else None


class Store:
    """Read/write access to the state directory. Writes assume the lock."""

    # Pending registration: an issued client ID kept separately from the
    # active connection so a failed exchange cannot corrupt it.
    def load_pending(self) -> dict | None:
        return self._read(PENDING_FILE)

    def save_pending(self, pending: dict) -> None:
        self._write(PENDING_FILE, pending)

    def delete_pending(self) -> None:
        self._delete(PENDING_FILE)

    def delete_models(self) -> None:
        self._delete(MODELS_FILE)

    def _delete(self, name: str) -> None:
        with self._directory_fd() as fd:
            try:
                os.unlink(name, dir_fd=fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise StorageError(f"Could not remove {name}: {exc}") from exc

    def pending_for(self, generation) -> dict | None:
        """Return the pending registration only if it belongs to this
        connection generation, deleting anything unrelated."""
        pending = self.load_pending()
        if pending is None:
            return None
        if pending.get("generation") != generation:
            self.delete_pending()
            return None
        return pending

    def connection_generation(self) -> int | None:
        """Return the connection generation, backfilling it on first update.

        The generation changes only on login replacement and logout, never
        on token refresh, so a stale sign-in attempt can be told apart from
        the live connection.
        """
        credentials = self.load_credentials() or {}
        generation = credentials.get("generation")
        if isinstance(generation, int) and not isinstance(generation, bool):
            return generation
        if credentials:
            credentials["generation"] = 1
            self.save_credentials(credentials)
            return 1
        return None

    def bump_generation(self) -> tuple[int, dict]:
        """Advance the connection generation and return (new, credentials)."""
        credentials = self.load_credentials() or {}
        generation = credentials.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool):
            generation = 0
        credentials["generation"] = generation + 1
        self.save_credentials(credentials)
        return credentials["generation"], credentials

    _KEPT_CONNECTION_KEYS = ("client_id", "issuer", "subject", "email", "name")

    def retire_connection(self) -> tuple[int, dict]:
        """Wipe tokens now and bump the generation; identity keys survive.

        Doing both under one lock means an interrupted sign-out still
        leaves the connection locally dead: a concurrent sign-in either
        aborts on the generation bump or lands a newer record afterwards.
        Returns (new generation, pre-wipe credentials).
        """
        credentials = self.load_credentials() or {}
        generation = credentials.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool):
            generation = 0
        kept = {
            key: credentials[key]
            for key in self._KEPT_CONNECTION_KEYS
            if key in credentials
        }
        kept["generation"] = generation + 1
        self.save_credentials(kept)
        return kept["generation"], credentials

    def __init__(self, directory: Path, *, dir_fd=None):
        self.directory = directory.parent.resolve() / directory.name
        self.dir_fd = dir_fd
        with self._directory_fd(create=True) as fd:
            os.fchmod(fd, 0o700)

    @contextmanager
    def _directory_fd(self, *, create=False):
        if self.dir_fd is not None:
            yield self.dir_fd
        else:
            with _open_directory(self.directory, create=create) as fd:
                yield fd

    def _read(self, name):
        with self._directory_fd() as fd:
            return _read_json(self._path(name), dir_fd=fd)

    def _write(self, name, data):
        with self._directory_fd() as fd:
            _write_json(self._path(name), data, dir_fd=fd)

    def _path(self, name: str) -> Path:
        return self.directory / name

    def load_credentials(self) -> dict | None:
        return self._read(CREDENTIALS_FILE)

    def save_credentials(self, credentials: dict) -> None:
        self._write(CREDENTIALS_FILE, credentials)

    def load_models(self) -> dict | None:
        return self._read(MODELS_FILE)

    def save_models(self, models: dict) -> None:
        self._write(MODELS_FILE, models)

    def load_host_id(self) -> str | None:
        host = self._read(HOST_FILE)
        if host:
            return host.get("host_id")
        return None

    def ensure_host_id(self) -> str:
        host_id = self.load_host_id()
        if not host_id:
            host_id = "urn:uuid:" + str(uuid.uuid4())
            self._write(HOST_FILE, {"host_id": host_id})
        return host_id


def _acquire_lock(dir_fd: int, name: str, wait: float, message: str) -> int:
    """Take an exclusive flock with a bounded wait; StorageError on expiry.

    Uses fcntl flock: macOS/Linux only, no Windows support. The deadline is
    measured on a monotonic clock and sleeps are short, so Ctrl-C stays
    interruptible. ``wait=0`` restores fail-fast behaviour.
    """
    import fcntl
    import random

    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
    deadline = time.monotonic() + wait
    delay = 0.02
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StorageError(message) from exc
                time.sleep(min(delay * (0.5 + random.random()), remaining))
                delay = min(delay * 2, 0.25)
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def locked_store(directory: Path | None = None, *, wait: float = LOCK_WAIT_SECONDS):
    """Hold the exclusive process lock while reading or updating state.

    Short contention is absorbed by waiting up to ``wait`` seconds; a lock
    still held past the deadline raises StorageError.
    """
    directory = directory or state_dir()
    with _open_directory(directory, create=True) as dir_fd:
        os.fchmod(dir_fd, 0o700)
        fd = _acquire_lock(
            dir_fd,
            LOCK_FILE,
            wait,
            "Another process is updating the ChatGPT plan connection. "
            "Try again after it finishes.",
        )
        try:
            yield Store(directory, dir_fd=dir_fd)
        finally:
            os.close(fd)


@contextmanager
def login_lock(directory: Path | None = None, *, wait: float = LOCK_WAIT_SECONDS):
    """Serialize sign-in attempts only; inference never takes this lock."""
    directory = directory or state_dir()
    with _open_directory(directory, create=True) as dir_fd:
        os.fchmod(dir_fd, 0o700)
        fd = _acquire_lock(
            dir_fd,
            LOGIN_LOCK_FILE,
            wait,
            "Another sign-in is in progress. Try again after it finishes.",
        )
        try:
            yield
        finally:
            os.close(fd)
