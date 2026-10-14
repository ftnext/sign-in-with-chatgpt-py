"""Public, credential-safe exceptions and error helpers."""


class ChatGPTAuthError(Exception):
    """Base class for every error this package raises."""


class SignInRequiredError(ChatGPTAuthError):
    """No connection exists at the storage location."""


class ReauthorizationRequiredError(ChatGPTAuthError):
    """The connection was retired or revoked; sign in again."""


class PlanPermissionError(ChatGPTAuthError):
    """ChatGPT plan usage was not granted for this connection."""


class ConnectionChangedError(ChatGPTAuthError):
    """The stored connection changed; load the credentials again."""


class StorageError(ChatGPTAuthError):
    """A safe, public message about local credential storage."""


class NetworkError(ChatGPTAuthError):
    """A safe message describing an auth-server or transport failure."""


class LoginTimeoutError(ChatGPTAuthError):
    """The browser sign-in did not complete within the timeout."""


class AuthError(ChatGPTAuthError):
    """Safe, public authentication code, never an upstream response body."""


class TokenOutcomeUnknownError(ChatGPTAuthError):
    """Internal: the request may have reached the server; outcome unknown."""


REAUTH_CODES = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)

# OAuth error codes which confirm the grant was never consumed.
SAFE_REJECT_CODES = frozenset({"invalid_client", "invalid_request"})


def redact(value, secrets=()):
    """Redact credentials even in nested error fields or exception text."""
    text = str(value)
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    return text


def status_error_code(exc):
    try:
        body = exc.response.json()
    except (ValueError, AttributeError):
        return None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, dict):
            code = error.get("code")
            return code if isinstance(code, str) else None
    return None


def describe_status_error(exc, secret=None, secrets=()):
    """Extract useful error fields without headers or echoed credentials."""
    status = getattr(exc.response, "status_code", "unknown")
    detail = code = param = None
    try:
        body = exc.response.json()
    except (ValueError, AttributeError):
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            param = error.get("param")
            detail = error.get("message")
        elif isinstance(error, str):
            code = error
            detail = body.get("error_description")
        if detail is None:
            detail = body.get("detail")
    bits = [f"HTTP {status}"]
    if code:
        bits.append(str(code))
    if param:
        bits.append(f"param={param}")
    if detail:
        bits.append(str(detail))
    return redact(": ".join(bits), (*secrets, secret))
