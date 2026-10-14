"""chatgpt-auth: Sign in with ChatGPT for CLI and Python applications."""

from .credentials import Credentials
from .errors import (
    AuthError,
    ChatGPTAuthError,
    ConnectionChangedError,
    LoginTimeoutError,
    NetworkError,
    PlanPermissionError,
    ReauthorizationRequiredError,
    SignInRequiredError,
    StorageError,
)
from .service import login, logout, status
from .types import AuthStatus, LogoutResult

__version__ = "0.1.0"

__all__ = [
    "AuthError",
    "AuthStatus",
    "ChatGPTAuthError",
    "ConnectionChangedError",
    "Credentials",
    "LoginTimeoutError",
    "LogoutResult",
    "NetworkError",
    "PlanPermissionError",
    "ReauthorizationRequiredError",
    "SignInRequiredError",
    "StorageError",
    "login",
    "logout",
    "status",
]
