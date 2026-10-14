"""The ``chatgpt-auth`` command line: login, status, token, refresh, logout."""

import sys
import time
import webbrowser
from datetime import UTC, datetime

import click

from . import service
from .credentials import Credentials
from .errors import (
    ChatGPTAuthError,
    ConnectionChangedError,
    LoginTimeoutError,
    NetworkError,
    PlanPermissionError,
    ReauthorizationRequiredError,
    SignInRequiredError,
    StorageError,
)
from .oauth import (
    DEFAULT_AGENT_NAME,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
)

USAGE_URL = "https://chatgpt.com/settings/usage"


@click.group()
@click.option(
    "--credentials",
    "credentials_path",
    metavar="PATH",
    default=None,
    help="Credentials file to use (overrides CHATGPT_AUTH_CREDENTIALS)",
)
@click.pass_context
def main(ctx, credentials_path):
    """Manage a Sign in with ChatGPT connection."""
    ctx.obj = credentials_path


def _path(ctx):
    return ctx.obj


def _auth_message(exc) -> str:
    if isinstance(exc, SignInRequiredError):
        return "Not signed in. Run: chatgpt-auth login"
    if isinstance(exc, ReauthorizationRequiredError):
        return str(exc) or "Sign-in expired or was revoked. Run: chatgpt-auth login"
    if isinstance(exc, PlanPermissionError):
        return (
            "ChatGPT plan usage was not granted. Check your usage sharing "
            f"settings: {USAGE_URL}"
        )
    if isinstance(exc, ConnectionChangedError):
        return str(exc)
    if isinstance(exc, (StorageError, NetworkError, LoginTimeoutError)):
        return str(exc)
    code = str(exc)
    guidance = {
        "sign_in_declined": "Sign-in was declined in the browser.",
        "expired_sign_in": "The sign-in attempt expired. Run login again.",
        "connection_changed": (
            "The connection changed while signing in. Run chatgpt-auth login again."
        ),
        "account_mismatch": (
            "A different ChatGPT account signed in. Run login again or use "
            "--replace to switch accounts."
        ),
    }
    return guidance.get(
        code,
        f"Sign-in failed: {code}. Run chatgpt-auth login to try again.",
    )


def _show_manual_url(url, browser_failed):
    if browser_failed:
        click.echo("Could not open a browser automatically.", err=True)
    click.echo("Open this URL to sign in with ChatGPT and approve plan usage:")
    click.echo(url)
    if sys.stdin.isatty():
        click.echo(
            "After approving, the browser is redirected to a local address. "
            "If the redirect does not reach this machine, copy the full URL "
            "from the address bar and paste it here."
        )
    else:
        click.echo(
            "Pasted URLs are not accepted in this session; complete the "
            "sign-in in a browser that can reach this machine."
        )


def _fail(exc):
    raise click.ClickException(_auth_message(exc)) from exc


@main.command()
@click.option(
    "--replace",
    is_flag=True,
    help="Replace the current connection with a new sign-in",
)
@click.option(
    "--manual",
    is_flag=True,
    help="Show the sign-in URL instead of opening a browser; the loopback "
    "callback or a pasted callback URL completes sign-in",
)
@click.option(
    "--timeout",
    type=int,
    default=DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help="Seconds to wait for browser sign-in (1-600)",
)
@click.option("--port", type=int, default=0, help="Callback port (0 = automatic)")
@click.option(
    "--agent-name",
    default=DEFAULT_AGENT_NAME,
    show_default=True,
    help="Application name sent on first-time registration",
)
@click.pass_context
def login(ctx, replace, manual, timeout, port, agent_name):
    """Sign in with ChatGPT and save the OAuth connection"""
    if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise click.UsageError("--timeout must be between 1 and 600")
    if not (port == 0 or 1024 <= port <= 65535):
        raise click.UsageError("--port must be 0 or between 1024 and 65535")
    try:
        if not manual:
            click.echo(
                "Opening a browser to sign in with ChatGPT and approve plan usage...",
                err=True,
            )
        credentials = service.login(
            _path(ctx),
            agent_name=agent_name,
            replace=replace,
            manual=manual,
            timeout=timeout,
            port=port,
            open_browser=webbrowser.open,
            show_url=_show_manual_url,
            input_stream=sys.stdin,
            warn=lambda message: click.echo(message, err=True),
        )
    except ChatGPTAuthError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        click.echo("Sign-in aborted.", err=True)
        ctx.exit(130)
    account = credentials.email or credentials.subject
    click.echo(f"Signed in and verified ChatGPT plan access for {account}.")


@main.command()
@click.pass_context
def status(ctx):
    """Show the local state of the stored connection"""
    try:
        report = service.status(_path(ctx))
    except ChatGPTAuthError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        ctx.exit(130)
    if not report.connected:
        click.echo("Not signed in.")
        if report.email:
            click.echo(f"Last account: {report.email}")
        click.echo(f"Storage: {report.path}")
        return
    click.echo(
        "Signed in." if report.state == "active" else "Signed in (refresh needed)."
    )
    if report.email:
        click.echo(
            f"Account: {report.email}" + (f" ({report.name})" if report.name else "")
        )
    elif report.subject:
        click.echo(f"Account: {report.subject}")
    expires = datetime.fromtimestamp(report.expires_at, tz=UTC)
    remaining = int(report.expires_at - time.time())
    if remaining > 0:
        click.echo(
            f"Access token expires: {expires.isoformat()} (in {remaining // 60} min)"
        )
    else:
        click.echo(f"Access token expired: {expires.isoformat()}")
    click.echo(f"Storage: {report.path}")


@main.command(name="print-access-token")
@click.pass_context
def print_access_token(ctx):
    """Print a valid access token (stdout only, nothing else)"""
    try:
        token = Credentials.load(_path(ctx)).get_access_token()
    except ChatGPTAuthError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        ctx.exit(130)
    click.echo(token)


@main.command()
@click.pass_context
def refresh(ctx):
    """Force a token refresh now; the token is not displayed"""
    try:
        Credentials.load(_path(ctx)).refresh()
    except ChatGPTAuthError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        ctx.exit(130)
    click.echo("Access token refreshed and saved.")


@main.command()
@click.pass_context
def logout(ctx):
    """Sign out: remove local tokens and try to revoke the session"""
    try:
        result = service.logout(_path(ctx))
    except ChatGPTAuthError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        ctx.exit(130)
    if not result.was_connected:
        click.echo("Not signed in.")
        return
    if result.remote_revoked:
        click.echo("Signed out. The ChatGPT session was revoked remotely.")
    else:
        click.echo("Signed out locally.")
        click.echo(
            "Could not confirm the remote session was revoked. To "
            "disconnect this app completely, remove it in ChatGPT under "
            "Settings > Security and login > Sign in with ChatGPT.",
            err=True,
        )
