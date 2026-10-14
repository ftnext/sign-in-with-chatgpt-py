import pytest

from chatgpt_auth import config
from chatgpt_auth.errors import StorageError


def test_explicit_path_wins(tmp_path):
    explicit = tmp_path / "explicit.json"
    env = {"CHATGPT_AUTH_CREDENTIALS": str(tmp_path / "env.json")}
    assert config.resolve_path(explicit, env=env) == explicit


def test_env_var_used_when_no_explicit(tmp_path):
    env = {"CHATGPT_AUTH_CREDENTIALS": str(tmp_path / "env" / "creds.json")}
    assert config.resolve_path(env=env) == tmp_path / "env" / "creds.json"


def test_default_under_xdg(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    assert (
        config.resolve_path(env=env)
        == tmp_path / "xdg" / "chatgpt-auth" / "credentials.json"
    )


def test_default_without_xdg():
    env = {}
    expected = config.Path.home() / ".config" / "chatgpt-auth" / "credentials.json"
    assert config.resolve_path(env=env) == expected


def test_relative_xdg_ignored(tmp_path):
    env = {"XDG_CONFIG_HOME": "relative/xdg"}
    expected = config.Path.home() / ".config" / "chatgpt-auth" / "credentials.json"
    assert config.resolve_path(env=env) == expected


def test_tilde_expanded(tmp_path):
    env = {"CHATGPT_AUTH_CREDENTIALS": "~/weird-name/auth.json"}
    path = config.resolve_path(env=env)
    assert path == config.Path.home() / "weird-name" / "auth.json"


def test_relative_path_anchored_to_cwd(tmp_path):
    path = config.resolve_path("sub/creds.json", env={}, cwd=tmp_path)
    assert path == tmp_path / "sub" / "creds.json"


def test_relative_env_path_anchored_to_cwd(tmp_path):
    env = {"CHATGPT_AUTH_CREDENTIALS": "sub/creds.json"}
    assert config.resolve_path(env=env, cwd=tmp_path) == tmp_path / "sub" / "creds.json"


@pytest.mark.parametrize(
    "name",
    [
        "host.json",
        "pending.json",
        "models.json",
        "runtime.lock",
        "login.lock",
        "extra.host.json",
        "extra.pending.json",
        "extra.runtime.lock",
        "extra.login.lock",
        "extra.models.json",
    ],
)
def test_reserved_names_rejected(tmp_path, name):
    with pytest.raises(StorageError):
        config.resolve_path(tmp_path / name, env={})


def test_canonical_aux_names(tmp_path):
    path = tmp_path / "credentials.json"
    assert config.aux_path(path, "host") == tmp_path / "host.json"
    assert config.aux_path(path, "pending") == tmp_path / "pending.json"
    assert config.aux_path(path, "runtime_lock") == tmp_path / "runtime.lock"
    assert config.aux_path(path, "login_lock") == tmp_path / "login.lock"
    assert config.aux_path(path, "models") == tmp_path / "models.json"


def test_custom_aux_names_derived_by_suffix(tmp_path):
    path = tmp_path / "custom.json"
    assert config.aux_path(path, "host") == tmp_path / "custom.json.host.json"
    assert config.aux_path(path, "pending") == tmp_path / "custom.json.pending.json"
    assert (
        config.aux_path(path, "runtime_lock") == tmp_path / "custom.json.runtime.lock"
    )
    assert config.aux_path(path, "login_lock") == tmp_path / "custom.json.login.lock"


def test_different_stores_share_no_aux_files(tmp_path):
    a = config.aux_path(tmp_path / "a.json", "host")
    b = config.aux_path(tmp_path / "b.json", "host")
    c = config.aux_path(tmp_path / "credentials.json", "host")
    assert len({a, b, c}) == 3
