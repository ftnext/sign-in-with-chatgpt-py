"""Credentials file location: resolution order, naming rules, validation.

A connection is not a single JSON file: it is the credentials file plus
the auxiliary files (host.json, pending.json, runtime.lock, login.lock)
that live next to it. When the credentials file is named the canonical
``credentials.json``, the auxiliary files keep their canonical names —
the same names llm-chatgpt-plan uses in its state directory — so pointing
this tool at an existing store shares files and locks with the old tool.
Any other file name gets auxiliary names derived by suffix, so distinct
stores never share auxiliary files.
"""

import os
from pathlib import Path

from .errors import StorageError

ENV_VAR = "CHATGPT_AUTH_CREDENTIALS"
DEFAULT_DIR_NAME = "chatgpt-auth"
CREDENTIALS_NAME = "credentials.json"

CANONICAL_AUX = {
    "host": "host.json",
    "pending": "pending.json",
    "runtime_lock": "runtime.lock",
    "login_lock": "login.lock",
    "models": "models.json",
}
AUX_SUFFIXES = {
    "host": ".host.json",
    "pending": ".pending.json",
    "runtime_lock": ".runtime.lock",
    "login_lock": ".login.lock",
    "models": ".models.json",
}
RESERVED_NAMES = frozenset(CANONICAL_AUX.values())
RESERVED_SUFFIXES = tuple(AUX_SUFFIXES.values())


def validate_credentials_name(name: str) -> None:
    """Reject file names that would collide with a store's own files."""
    if not name or name in (".", ".."):
        raise StorageError("Credentials file name is empty")
    if name in RESERVED_NAMES or name.endswith(RESERVED_SUFFIXES):
        raise StorageError(f"Reserved credentials file name: {name}")


def aux_path(credentials_path: Path, kind: str) -> Path:
    """Auxiliary file for ``kind`` next to the given credentials file."""
    if credentials_path.name == CREDENTIALS_NAME:
        name = CANONICAL_AUX[kind]
    else:
        name = credentials_path.name + AUX_SUFFIXES[kind]
    return credentials_path.parent / name


def _anchor(path: Path, cwd: Path) -> Path:
    """Expand ~ and pin a relative path to the given working directory."""
    expanded = Path(os.path.expanduser(str(path)))
    if not expanded.is_absolute():
        expanded = cwd / expanded
    return Path(os.path.abspath(expanded))


def default_path(*, env=None, cwd=None) -> Path:
    """${XDG_CONFIG_HOME:-~/.config}/chatgpt-auth/credentials.json.

    A relative XDG_CONFIG_HOME is ignored in favor of ~/.config.
    """
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    xdg = env.get("XDG_CONFIG_HOME")
    if xdg:
        candidate = Path(os.path.expanduser(xdg))
        base = candidate if candidate.is_absolute() else Path.home() / ".config"
    else:
        base = Path.home() / ".config"
    return base / DEFAULT_DIR_NAME / CREDENTIALS_NAME


def resolve_path(explicit=None, *, env=None, cwd=None) -> Path:
    """Resolve the credentials file path in priority order:

    1. the explicit ``path`` argument (CLI --credentials / API path)
    2. the CHATGPT_AUTH_CREDENTIALS environment variable
    3. the default location
    """
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    if explicit is not None:
        path = _anchor(Path(explicit), cwd)
    elif env.get(ENV_VAR):
        path = _anchor(Path(env[ENV_VAR]), cwd)
    else:
        path = default_path(env=env, cwd=cwd)
    validate_credentials_name(path.name)
    return path
