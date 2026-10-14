"""Protected local storage for the single ChatGPT connection.

One connection is a file set next to the credentials file: host.json,
pending.json, runtime.lock and login.lock when the credentials file is
named ``credentials.json`` (the same names llm-chatgpt-plan uses, so an
explicit path into its store shares files and locks), or
``<name>.host.json`` etc. for any other credentials file name.

models.json is never read, written, or deleted by this package.
"""

import json
import math
import os
import secrets
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import aux_path
from .errors import StorageError

LOCK_WAIT_SECONDS = 5.0

_STRING_KEYS = (
    "client_id",
    "issuer",
    "subject",
    "id_token",
    "access_token",
    "refresh_token",
)
_NULLABLE_STRING_KEYS = ("email", "name")
KNOWN_KEYS = frozenset(
    _STRING_KEYS + _NULLABLE_STRING_KEYS + ("scopes", "expires_at", "generation")
)


def validate_record(record: dict) -> dict:
    """Check the stored connection record's known fields.

    Unknown extra keys are tolerated (the record stays forward/backward
    compatible); a wrong type on a known key is corruption and rejected.
    """
    for key in _STRING_KEYS:
        if key in record and not isinstance(record[key], str):
            raise StorageError(f"Corrupt credentials: {key} must be a string")
    for key in _NULLABLE_STRING_KEYS:
        if (
            key in record
            and record[key] is not None
            and not isinstance(record[key], str)
        ):
            raise StorageError(f"Corrupt credentials: {key} must be a string")
    if "scopes" in record:
        scopes = record["scopes"]
        if not isinstance(scopes, list) or not all(
            isinstance(scope, str) for scope in scopes
        ):
            raise StorageError("Corrupt credentials: scopes must be a list")
    if "expires_at" in record:
        expires_at = record["expires_at"]
        if (
            not isinstance(expires_at, (int, float))
            or isinstance(expires_at, bool)
            or not math.isfinite(expires_at)
        ):
            raise StorageError("Corrupt credentials: invalid expires_at")
    if "generation" in record:
        generation = record["generation"]
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise StorageError("Corrupt credentials: invalid generation")
    return record


def normalized_generation(record: dict | None) -> int:
    """The generation used for connection matching.

    Records written by older tools may lack a generation; the first write
    backfills it to 1 (the old format's initial generation), so a missing
    value is treated as 1 and that first write is not a connection change.
    """
    generation = (record or {}).get("generation")
    if isinstance(generation, int) and not isinstance(generation, bool):
        return generation
    return 1


def is_connected(record: dict | None) -> bool:
    """True when the record holds a live connection (tokens present)."""
    if not record:
        return False
    return all(
        record.get(key)
        for key in ("client_id", "subject", "access_token", "refresh_token")
    ) and isinstance(record.get("expires_at"), (int, float))


@contextmanager
def _open_directory(directory: Path, *, create=False):
    """Anchor operations to a directory opened without following a symlink.

    Resolve the parent only: macOS may legitimately symlink a user-path
    ancestor (such as /var). Never resolve the state directory itself.
    An existing directory must be dedicated and protected: owned by the
    current user and closed to group/other. It is never chmod'ed. A newly
    created directory is normalized to 0700.
    """
    directory = directory.parent.resolve() / directory.name
    fd = None
    try:
        if directory.is_symlink():
            raise StorageError("Refusing symlinked credentials directory")
        created = False
        if create:
            created = not directory.exists()
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise StorageError("Credentials directory is not owned by this user")
        if created:
            os.fchmod(fd, 0o700)
        elif stat.S_IMODE(info.st_mode) & 0o077:
            raise StorageError(
                "Credentials directory must not be accessible by group or other"
            )
        yield fd
    except OSError as exc:
        raise StorageError(f"Could not access credentials directory: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _read_json(path: Path, dir_fd=None) -> dict | None:
    """Read a JSON object file. Returns None when it does not exist.

    Never creates directories or files; safe for read-only queries.
    """
    if dir_fd is None:
        if path.parent.is_symlink():
            raise StorageError("Refusing symlinked credentials directory")
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
    try:
        if stat.S_ISLNK(
            os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        ):
            raise StorageError(f"Refusing to write through symlink: {path.name}")
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise StorageError(f"Could not inspect {path.name}: {exc}") from exc
    name = path.name + "." + secrets.token_hex(16)
    try:
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
            os.fsync(dir_fd)
        finally:
            try:
                os.unlink(name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
    except OSError as exc:
        raise StorageError(f"Could not write {path.name}: {exc}") from exc


def read_record(credentials_path: Path) -> dict | None:
    """Validated credentials record, or None. Never creates anything."""
    record = _read_json(credentials_path)
    if record is None:
        return None
    return validate_record(record)


class Store:
    """Read/write access to one credentials file set. Writes assume the
    runtime lock is held."""

    def __init__(self, credentials_path: Path, *, dir_fd=None):
        directory = credentials_path.parent
        self.directory = directory.parent.resolve() / directory.name
        self.credentials_path = self.directory / credentials_path.name
        self.dir_fd = dir_fd
        with self._directory_fd(create=True):
            pass

    def _path(self, kind: str) -> Path:
        if kind == "credentials":
            return self.credentials_path
        return aux_path(self.credentials_path, kind)

    @contextmanager
    def _directory_fd(self, *, create=False):
        if self.dir_fd is not None:
            yield self.dir_fd
        else:
            with _open_directory(self.directory, create=create) as fd:
                yield fd

    def _read(self, kind):
        with self._directory_fd() as fd:
            return _read_json(self._path(kind), dir_fd=fd)

    def _write(self, kind, data):
        with self._directory_fd() as fd:
            _write_json(self._path(kind), data, dir_fd=fd)

    def _delete(self, kind):
        with self._directory_fd() as fd:
            name = self._path(kind).name
            try:
                os.unlink(name, dir_fd=fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise StorageError(f"Could not remove {name}: {exc}") from exc

    def load_credentials(self) -> dict | None:
        record = self._read("credentials") or None
        if record is None:
            return None
        return validate_record(record)

    def save_credentials(self, credentials: dict) -> None:
        self._write("credentials", credentials)

    def load_pending(self) -> dict | None:
        return self._read("pending")

    def save_pending(self, pending: dict) -> None:
        self._write("pending", pending)

    def delete_pending(self) -> None:
        self._delete("pending")

    def pending_for(self, generation) -> dict | None:
        """Pending registration only if it belongs to this generation."""
        pending = self.load_pending()
        if pending is None:
            return None
        if pending.get("generation") != generation:
            self.delete_pending()
            return None
        return pending

    def connection_generation(self) -> int | None:
        """Return the connection generation, backfilling it on first update.

        The generation changes only on login replacement, logout and
        retirement, never on token refresh, so a stale sign-in attempt can
        be told apart from the live connection.
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

    def load_host_id(self) -> str | None:
        host = self._read("host")
        if host:
            host_id = host.get("host_id")
            if isinstance(host_id, str):
                return host_id
        return None

    def ensure_host_id(self) -> str:
        host_id = self.load_host_id()
        if not host_id:
            host_id = "urn:uuid:" + str(uuid.uuid4())
            self._write("host", {"host_id": host_id})
        return host_id


def _acquire_lock(dir_fd: int, name: str, wait: float, message: str) -> int:
    """Take an exclusive flock with a bounded wait; StorageError on expiry.

    Uses fcntl flock: macOS/Linux only, no Windows support. The deadline is
    measured on a monotonic clock and sleeps are short, so Ctrl-C stays
    interruptible. ``wait=0`` restores fail-fast behaviour. Lock files are
    never replaced or deleted.
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
def locked_store(credentials_path: Path, *, wait: float = LOCK_WAIT_SECONDS):
    """Hold the exclusive runtime lock while reading or updating state.

    Short contention is absorbed by waiting up to ``wait`` seconds; a lock
    still held past the deadline raises StorageError.
    """
    path = Path(credentials_path)
    directory = path.parent
    with _open_directory(directory, create=True) as dir_fd:
        fd = _acquire_lock(
            dir_fd,
            aux_path(path, "runtime_lock").name,
            wait,
            "Another process is updating the ChatGPT connection. "
            "Try again after it finishes.",
        )
        try:
            yield Store(path, dir_fd=dir_fd)
        finally:
            os.close(fd)


@contextmanager
def login_lock(credentials_path: Path, *, wait: float = LOCK_WAIT_SECONDS):
    """Serialize sign-in attempts only; token use never takes this lock."""
    path = Path(credentials_path)
    directory = path.parent
    with _open_directory(directory, create=True) as dir_fd:
        fd = _acquire_lock(
            dir_fd,
            aux_path(path, "login_lock").name,
            wait,
            "Another sign-in is in progress. Try again after it finishes.",
        )
        try:
            yield
        finally:
            os.close(fd)
