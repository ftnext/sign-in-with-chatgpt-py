"""Protected local file storage for registration data (never OAuth tokens).

The registration file holds only non-token data: schema version, host ID,
the issued client ID, and the verified identity (issuer/subject/email/name)
needed to match later sign-ins. It lives in the configured state directory
with 0700/0600 permissions and is replaced atomically.
"""

import json
import os
import secrets
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

REGISTRATION_FILE = "registration.json"
LOCK_FILE = "runtime.lock"
SCHEMA_VERSION = 1
REGISTRATION_KEYS = frozenset(
    {"schema_version", "host_id", "client_id", "issuer", "subject", "email", "name"}
)


class StorageError(Exception):
    """A safe, public message about local registration storage."""


@contextmanager
def _open_directory(directory: Path, *, create=False):
    directory = directory.parent.resolve() / directory.name
    fd = None
    try:
        if directory.is_symlink():
            raise StorageError("Refusing symlinked state directory")
        if create:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        yield fd
    except OSError as exc:
        raise StorageError(f"Could not access state directory: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _read_json(path: Path, dir_fd=None) -> dict | None:
    if dir_fd is None:
        if path.parent.is_symlink():
            raise StorageError("Refusing symlinked state directory")
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


def load_registration(directory: Path) -> dict | None:
    """Read the registration file; returns only non-token fields."""
    data = _read_json(directory / REGISTRATION_FILE)
    if data is None:
        return None
    return {key: data[key] for key in REGISTRATION_KEYS if key in data}


def save_registration(directory: Path, registration: dict) -> None:
    """Atomically persist registration data. Token fields are never written."""
    record = {
        key: registration[key] for key in REGISTRATION_KEYS if key in registration
    }
    record["schema_version"] = SCHEMA_VERSION
    with _open_directory(directory, create=True) as dir_fd:
        os.fchmod(dir_fd, 0o700)
        _write_json(directory / REGISTRATION_FILE, record, dir_fd)


def ensure_host_id(directory: Path) -> str:
    registration = load_registration(directory) or {}
    host_id = registration.get("host_id")
    if host_id:
        return host_id
    host_id = "urn:uuid:" + str(uuid.uuid4())
    registration["host_id"] = host_id
    save_registration(directory, registration)
    return host_id


def acquire_process_lock(directory: Path):
    """Exclusive single-process lock on the state directory.

    Uses fcntl flock (macOS/Linux); the descriptor must stay open for the
    lifetime of the server. Raises StorageError when another server holds
    the lock. Returns the open file descriptor.
    """
    import fcntl

    with _open_directory(directory, create=True) as dir_fd:
        os.fchmod(dir_fd, 0o700)
        fd = os.open(
            LOCK_FILE,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=dir_fd,
        )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise StorageError(
            "Another server is already using this state directory. "
            "Stop it or choose a different CHATGPT_STATE_DIR."
        ) from exc
    return fd
