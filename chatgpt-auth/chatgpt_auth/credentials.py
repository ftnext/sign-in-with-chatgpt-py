"""The public Credentials object: a stored connection bound to its path.

The object keeps the resolved path and the connection identity captured
at load/login time. Every token operation re-reads the store and
re-verifies the connection, so a replace, logout, or retirement by
another process is detected instead of silently switching accounts. It
never holds a long-lived HTTP client; each operation creates and closes
its own. No token material is exposed through attributes or repr.
"""

from pathlib import Path

from . import service
from .config import resolve_path
from .errors import (
    ReauthorizationRequiredError,
    SignInRequiredError,
    StorageError,
)
from .storage import is_connected, normalized_generation, read_record
from .types import ConnectionIdentity


class Credentials:
    """A verified connection to one ChatGPT account."""

    def __init__(self, path, record):
        self._path = Path(path)
        self._identity = ConnectionIdentity(
            issuer=record["issuer"],
            client_id=record["client_id"],
            subject=record["subject"],
            generation=normalized_generation(record),
        )
        self._email = record.get("email")
        self._name = record.get("name")
        self._expires_at = record.get("expires_at")
        self._scopes = tuple(record.get("scopes") or ())
        self._broken = False

    @classmethod
    def _from_record(cls, path, record):
        return cls(path, record)

    @classmethod
    def load(cls, path=None):
        """Load the stored connection. No network, no writes.

        Raises SignInRequiredError when nothing is stored,
        ReauthorizationRequiredError when the stored connection was
        retired, and StorageError when the record is corrupt.
        """
        resolved = resolve_path(path)
        record = read_record(resolved)
        if record is None or not all(
            record.get(key) for key in ("issuer", "client_id", "subject")
        ):
            raise SignInRequiredError("Not signed in. Run: chatgpt-auth login")
        if not is_connected(record):
            raise ReauthorizationRequiredError(
                "The connection was signed out or revoked. Run: chatgpt-auth login"
            )
        return cls(resolved, record)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def issuer(self) -> str:
        return self._identity.issuer

    @property
    def client_id(self) -> str:
        return self._identity.client_id

    @property
    def subject(self) -> str:
        return self._identity.subject

    @property
    def generation(self) -> int:
        return self._identity.generation

    @property
    def email(self) -> str | None:
        return self._email

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def expires_at(self) -> float | None:
        return self._expires_at

    @property
    def scopes(self) -> tuple:
        return self._scopes

    def _ensure_usable(self):
        if self._broken:
            raise StorageError(
                "This Credentials object can no longer be used because a "
                "storage write failed. Restore access to the storage "
                "location and run: chatgpt-auth login"
            )

    def _sync_snapshot(self):
        record = read_record(self._path)
        if not isinstance(record, dict):
            return
        expires_at = record.get("expires_at")
        if isinstance(expires_at, (int, float)):
            self._expires_at = expires_at
        scopes = record.get("scopes")
        if isinstance(scopes, list):
            self._scopes = tuple(scopes)

    def get_access_token(self) -> str:
        """A valid access token; refreshes and saves when near expiry."""
        self._ensure_usable()
        try:
            token = service.ensure_access_token(self._path, self._identity)
        except service.UnrecoverableStorageError:
            self._broken = True
            raise
        self._sync_snapshot()
        return token

    def refresh(self) -> None:
        """Force a refresh now; the result is saved before returning."""
        self._ensure_usable()
        try:
            service.force_refresh(self._path, self._identity)
        except service.UnrecoverableStorageError:
            self._broken = True
            raise
        self._sync_snapshot()

    def __repr__(self):
        return (
            f"Credentials(client_id={self.client_id!r}, "
            f"subject={self.subject!r}, email={self.email!r})"
        )
