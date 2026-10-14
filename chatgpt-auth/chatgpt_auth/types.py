"""Non-secret result types shared by the public API and the service layer."""

from dataclasses import dataclass
from pathlib import Path

STATE_NOT_CONNECTED = "not_connected"
STATE_ACTIVE = "active"
STATE_NEEDS_REFRESH = "needs_refresh"


@dataclass(frozen=True)
class ConnectionIdentity:
    """The fields a stored record must keep matching to stay the same
    connection: issuer, client ID, subject, and normalized generation."""

    issuer: str
    client_id: str
    subject: str
    generation: int


@dataclass(frozen=True)
class AuthStatus:
    """A local-only snapshot of the stored connection.

    ``state`` is one of ``not_connected``, ``active``, or
    ``needs_refresh`` (the stored token is within the refresh margin or
    already expired). Identity fields are None when nothing is stored.
    No token fields are ever included.
    """

    state: str
    path: Path
    issuer: str | None = None
    client_id: str | None = None
    subject: str | None = None
    email: str | None = None
    name: str | None = None
    expires_at: float | None = None
    scopes: tuple = ()

    @property
    def connected(self) -> bool:
        return self.state != STATE_NOT_CONNECTED


@dataclass(frozen=True)
class LogoutResult:
    """Outcome of a logout: local removal and remote revocation reported
    separately. ``remote_revoked`` is None when there was no refresh token
    to revoke or no revocation endpoint to call."""

    local_cleared: bool
    remote_revoked: bool | None
    was_connected: bool
