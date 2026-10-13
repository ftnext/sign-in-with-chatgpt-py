"""State transitions for the single stored connection.

The service layer owns every lock acquisition: callers never manage
locks. Lock order is always login lock then runtime lock; the runtime
lock is held only for snapshot, pending save, and commit sections, never
while waiting on the browser.
"""

import sys
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from .callback import CallbackListener, wait_for_result
from .config import resolve_path
from .errors import (
    AuthError,
    ConnectionChangedError,
    LoginTimeoutError,
    NetworkError,
    PlanPermissionError,
    ReauthorizationRequiredError,
    SignInRequiredError,
    StorageError,
    TokenOutcomeUnknownError,
)
from .oauth import (
    DEFAULT_AGENT_NAME,
    DEFAULT_TIMEOUT_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    PLAN_SCOPE,
    REFRESH_MARGIN_SECONDS,
    OAuth,
    make_http_client,
)
from .storage import (
    is_connected,
    locked_store,
    login_lock,
    normalized_generation,
    read_record,
)

if TYPE_CHECKING:
    from .credentials import Credentials

from .types import (
    STATE_ACTIVE,
    STATE_NEEDS_REFRESH,
    STATE_NOT_CONNECTED,
    AuthStatus,
    ConnectionIdentity,
    LogoutResult,
)

_now = time.time


def _default_show_url(url, browser_failed):
    print(
        "Open this URL in a browser to sign in with ChatGPT:\n" + url,
        file=sys.stderr,
    )


def _identity_keys(record):
    return (record.get("issuer"), record.get("client_id"), record.get("subject"))


def match_connection(record, identity: ConnectionIdentity) -> dict:
    """Verify the stored record is still the connection we bound to.

    A retired record keeps the identity keys but loses its tokens and
    bumps the generation: it asks for re-login, not a reload. A record
    with fresh tokens under a new generation is a different connection.
    """
    if record is None:
        raise SignInRequiredError("Not signed in. Run: chatgpt-auth login")
    if not all(record.get(key) for key in ("issuer", "client_id", "subject")):
        raise SignInRequiredError("Not signed in. Run: chatgpt-auth login")
    if _identity_keys(record) != (
        identity.issuer,
        identity.client_id,
        identity.subject,
    ):
        raise ConnectionChangedError(
            "The stored connection is a different account or client. "
            "Load the credentials again."
        )
    if not is_connected(record):
        raise ReauthorizationRequiredError(
            "The connection was signed out or revoked. Run: chatgpt-auth login"
        )
    if normalized_generation(record) != identity.generation:
        raise ConnectionChangedError(
            "The connection was replaced by another sign-in. "
            "Load the credentials again."
        )
    if PLAN_SCOPE not in record.get("scopes", []):
        raise PlanPermissionError(
            "ChatGPT plan usage was not granted. Check your usage sharing "
            "settings: https://chatgpt.com/settings/usage"
        )
    return record


class UnrecoverableStorageError(StorageError):
    """The storage location could not be written even to retire the
    connection; callers mark their bound objects unusable."""


def _retire(store) -> None:
    """Retire the connection the same way the old tool does.

    Keeps the identity keys and an advanced generation, drops every token
    field and any pending registration. When even this cannot be written
    the failure is unrecoverable for this object.
    """
    try:
        store.retire_connection()
        store.delete_pending()
    except StorageError as exc:
        raise UnrecoverableStorageError(
            "Could not disable the stored connection. Restore access to the "
            "storage location, then run: chatgpt-auth login"
        ) from exc


def _refresh_locked(store, record, http_factory):
    """Exchange the refresh token and save the merged record.

    Implements the failure table: provably unsent requests and safe
    rejections keep the stored state; revocations, unknown outcomes, and
    validation failures retire the connection; a save failure after a
    successful exchange never returns the new token.
    """
    form = {
        "grant_type": "refresh_token",
        "client_id": record["client_id"],
        "refresh_token": record["refresh_token"],
        "resource": "https://api.openai.com/v1",
    }
    factory = http_factory or make_http_client
    with factory(timeout=HTTP_TIMEOUT_SECONDS) as http:
        oauth = OAuth(http)
        try:
            tokens = oauth.token_request(form)
        except NetworkError:
            raise
        except TokenOutcomeUnknownError as exc:
            _retire(store)
            raise ReauthorizationRequiredError(
                "The token update result is unknown and the connection was "
                "retired. Run: chatgpt-auth login"
            ) from exc
        except AuthError as exc:
            if str(exc) == "reauthorization_required":
                _retire(store)
                raise ReauthorizationRequiredError(
                    "The stored sign-in was revoked. Run: chatgpt-auth login"
                ) from exc
            raise
        try:
            if tokens.get("id_token"):
                identity = oauth.verify(tokens["id_token"], record["client_id"])
                if identity["sub"] != record["subject"]:
                    raise AuthError("account_mismatch")
            update = oauth.credentials(
                tokens, record.get("scopes"), refresh_response=True
            )
        except AuthError as exc:
            _retire(store)
            raise ReauthorizationRequiredError(
                "The token update could not be verified and the connection "
                "was retired. Run: chatgpt-auth login"
            ) from exc
    if PLAN_SCOPE not in update["scopes"]:
        _retire(store)
        raise ReauthorizationRequiredError(
            "The token update dropped ChatGPT plan usage and the connection "
            "was retired. Run: chatgpt-auth login"
        )
    merged = dict(record)
    merged.update(
        access_token=update["access_token"],
        scopes=update["scopes"],
        expires_at=update["expires_at"],
        refresh_token=update["refresh_token"] or record["refresh_token"],
    )
    if tokens.get("id_token"):
        merged["id_token"] = tokens["id_token"]
    if "generation" not in merged:
        merged["generation"] = 1
    try:
        store.save_credentials(merged)
    except StorageError:
        # The new token must not be returned; do not resend the old one.
        _retire(store)
        raise


def ensure_access_token(credentials_path, identity, *, http_factory=None) -> str:
    """Return a usable access token, refreshing under lock when needed."""
    path = Path(credentials_path)
    record = match_connection(read_record(path), identity)
    if record["expires_at"] > _now() + REFRESH_MARGIN_SECONDS:
        return record["access_token"]
    with locked_store(path) as store:
        record = match_connection(store.load_credentials(), identity)
        if record["expires_at"] <= _now() + REFRESH_MARGIN_SECONDS:
            _refresh_locked(store, record, http_factory)
            record = match_connection(store.load_credentials(), identity)
        return record["access_token"]


def force_refresh(credentials_path, identity, *, http_factory=None) -> None:
    """Refresh unconditionally, always with the freshest stored token."""
    path = Path(credentials_path)
    with locked_store(path) as store:
        record = match_connection(store.load_credentials(), identity)
        _refresh_locked(store, record, http_factory)


def sign_in(
    credentials_path,
    *,
    agent_name=DEFAULT_AGENT_NAME,
    replace=False,
    manual=False,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    port=0,
    http_factory=None,
    open_browser=None,
    show_url=None,
    input_stream=None,
    warn=None,
) -> dict:
    """Run one sign-in against the store's single connection.

    ``replace`` starts a fresh dynamic registration (for switching
    accounts); the previous connection is only overwritten after the new
    one verifies. ``manual`` skips the browser entirely and shows a
    hint-free sign-in URL, the same path a failed browser launch falls
    back to. The manual path accepts either the loopback callback or a
    pasted callback URL on ``input_stream``; only the first completed
    result is exchanged. Returns the verified credential record.
    """
    path = Path(credentials_path)
    factory = http_factory or make_http_client
    open_browser = webbrowser.open if open_browser is None else open_browser
    show_url = _default_show_url if show_url is None else show_url
    with login_lock(path):
        with locked_store(path) as store:
            existing = store.load_credentials() or {}
            generation = store.connection_generation()
            host_id = store.ensure_host_id()
            pending = store.pending_for(generation) or {}
        client_id = (
            None if replace else existing.get("client_id") or pending.get("client_id")
        )
        prior = (
            existing if client_id and existing.get("client_id") == client_id else None
        )
        with factory(timeout=HTTP_TIMEOUT_SECONDS) as http:
            oauth = OAuth(http)
            listener = CallbackListener(port)
            try:
                tx, url = oauth.begin(
                    listener.redirect_uri,
                    host_id=host_id,
                    client_id=client_id,
                    prior=prior,
                    agent_name=agent_name,
                )
                listener.start(tx["state"])
                deadline = time.monotonic() + timeout
                browser_opened = False
                if not manual:
                    try:
                        browser_opened = bool(open_browser(url))
                    except (OSError, RuntimeError, webbrowser.Error):
                        browser_opened = False
                if browser_opened:
                    query = listener.wait(timeout)
                else:
                    show_url(oauth.manual_url(tx), not manual)
                    query = wait_for_result(
                        listener, deadline, stream=input_stream, warn=warn
                    )
                if query is None:
                    raise LoginTimeoutError(
                        f"Timed out waiting for sign-in after {timeout} seconds. "
                        "Run chatgpt-auth login again."
                    )

                persist_client_id = None
                if not replace:

                    def persist_client_id(issued):
                        with locked_store(path) as pending_store:
                            current = pending_store.load_credentials() or {}
                            if current.get("client_id") != issued:
                                pending_store.save_pending(
                                    {"client_id": issued, "generation": generation}
                                )

                record = oauth.complete(
                    tx,
                    query,
                    persist_client_id=persist_client_id,
                    prior_subject=prior.get("subject") if prior else None,
                )
                if PLAN_SCOPE not in record["scopes"]:
                    raise AuthError("plan_permission_required")
                with locked_store(path) as commit_store:
                    if commit_store.connection_generation() != generation:
                        raise AuthError("connection_changed")
                    record["generation"] = (generation or 0) + 1
                    commit_store.save_credentials(record)
                    commit_store.delete_pending()
                return record
            finally:
                listener.close()


def connection_status(credentials_path) -> AuthStatus:
    """Local-only snapshot: no network, no refresh, no writes."""
    path = Path(credentials_path)
    record = read_record(path)
    if not is_connected(record):
        return AuthStatus(
            state=STATE_NOT_CONNECTED,
            path=path,
            issuer=(record or {}).get("issuer"),
            client_id=(record or {}).get("client_id"),
            subject=(record or {}).get("subject"),
            email=(record or {}).get("email"),
            name=(record or {}).get("name"),
        )
    state = (
        STATE_ACTIVE
        if record["expires_at"] > _now() + REFRESH_MARGIN_SECONDS
        else STATE_NEEDS_REFRESH
    )
    return AuthStatus(
        state=state,
        path=path,
        issuer=record.get("issuer"),
        client_id=record.get("client_id"),
        subject=record.get("subject"),
        email=record.get("email"),
        name=record.get("name"),
        expires_at=record.get("expires_at"),
        scopes=tuple(record.get("scopes") or ()),
    )


def sign_out(credentials_path, *, http_factory=None) -> LogoutResult:
    """Retire the local connection first, then try remote revocation.

    The local wipe commits before the network call so an interrupted or
    stalled revocation never leaves usable credentials behind; the old
    state is never restored. Revocation retries only on transport
    failures and 5xx, with a bounded backoff.
    """
    path = Path(credentials_path)
    factory = http_factory or make_http_client
    with locked_store(path) as store:
        _generation, credentials = store.retire_connection()
        store.delete_pending()
    refresh_token = credentials.get("refresh_token")
    client_id = credentials.get("client_id")
    if not refresh_token or not client_id:
        return LogoutResult(
            local_cleared=True, remote_revoked=None, was_connected=False
        )
    confirmed = False
    try:
        with factory(timeout=HTTP_TIMEOUT_SECONDS) as http:
            confirmed = OAuth(http).revoke_refresh_token(client_id, refresh_token)
    except Exception:  # noqa: BLE001
        # revocation is best-effort: local sign-out already committed
        confirmed = False
    return LogoutResult(
        local_cleared=True, remote_revoked=confirmed, was_connected=True
    )


def login(
    path=None,
    *,
    agent_name=DEFAULT_AGENT_NAME,
    replace=False,
    manual=False,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    port=0,
    **injected,
) -> "Credentials":
    """Sign in interactively and return a bound Credentials object."""
    from .credentials import Credentials

    resolved = resolve_path(path)
    record = sign_in(
        resolved,
        agent_name=agent_name,
        replace=replace,
        manual=manual,
        timeout=timeout,
        port=port,
        **injected,
    )
    return Credentials._from_record(resolved, record)


def status(path=None) -> AuthStatus:
    return connection_status(resolve_path(path))


def logout(path=None, *, http_factory=None) -> LogoutResult:
    return sign_out(resolve_path(path), http_factory=http_factory)
