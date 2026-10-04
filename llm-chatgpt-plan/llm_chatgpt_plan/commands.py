"""The ``llm chatgpt-plan`` command group: login, models, and logout."""

import sys
import time
import webbrowser

import click
import httpx

from . import client, oauth, storage
from .client import ApiError
from .models import MODEL_ID_PREFIX
from .oauth import AuthError, OAuth
from .storage import StorageError


@click.group(name="chatgpt-plan")
def chatgpt_plan():
    "Manage the Sign in with ChatGPT connection"


def _save_models(store, record, models):
    store.save_models(
        {
            "fetched_at": time.time(),
            "client_id": record.get("client_id"),
            "subject": record.get("subject"),
            "models": [
                {
                    "slug": model["slug"],
                    "display_name": model.get("display_name", model["slug"]),
                }
                for model in models
                if isinstance(model.get("slug"), str)
            ],
        }
    )


def _print_models(models):
    for model in models:
        slug = model.get("slug", "?")
        name = model.get("display_name") or slug
        click.echo(f"{MODEL_ID_PREFIX}{slug}\t{name}")


def _auth_exception(exc: AuthError) -> click.ClickException:
    code = str(exc)
    guidance = {
        "sign_in_required": "Not signed in. Run: llm chatgpt-plan login",
        "reauthorization_required": (
            "Sign-in expired or was revoked. Run: llm chatgpt-plan login"
        ),
        "plan_permission_required": (
            "ChatGPT plan usage was not granted. Check your usage sharing "
            "settings: " + client.USAGE_URL
        ),
        "sign_in_declined": "Sign-in was declined in the browser.",
        "expired_sign_in": "The sign-in attempt expired. Run login again.",
        "connection_changed": (
            "The connection changed while signing in. Run llm chatgpt-plan login again."
        ),
    }
    message = guidance.get(
        code,
        f"Sign-in failed: {code}. Run llm chatgpt-plan login to try again.",
    )
    return click.ClickException(message)


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


@chatgpt_plan.command()
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
    default=oauth.DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help="Seconds to wait for browser sign-in (1-600)",
)
@click.option("--port", type=int, default=0, help="Callback port (0 = automatic)")
def login(replace, manual, timeout, port):
    "Sign in with ChatGPT and save OAuth credentials for this llm"
    if not 1 <= timeout <= oauth.MAX_TIMEOUT_SECONDS:
        raise click.ClickException("--timeout must be between 1 and 600")
    if not (port == 0 or 1024 <= port <= 65535):
        raise click.ClickException("--port must be 0 or between 1024 and 65535")
    try:
        with (
            storage.login_lock(),
            client.make_http_client(timeout=30) as http,
        ):
            if not manual:
                click.echo(
                    "Opening a browser to sign in with ChatGPT and approve "
                    "plan usage..."
                )
            record = oauth.login_flow(
                storage.state_dir(),
                http,
                replace=replace,
                manual=manual,
                timeout=timeout,
                port=port,
                open_browser=webbrowser.open,
                show_url=_show_manual_url,
                input_stream=sys.stdin,
                warn=lambda message: click.echo(message, err=True),
            )
            click.echo("Signed in and verified ChatGPT plan access.")
            try:
                models = client.fetch_models(http, record["access_token"])
            except (AuthError, ApiError, httpx.HTTPError) as exc:
                click.echo(
                    f"Signed in, but fetching the model list failed "
                    f"({exc}). Run: llm chatgpt-plan models --refresh",
                    err=True,
                )
                return
            with storage.locked_store() as store:
                current = store.load_credentials() or {}
                if (
                    current.get("client_id"),
                    current.get("subject"),
                    current.get("generation"),
                ) != (
                    record.get("client_id"),
                    record.get("subject"),
                    record.get("generation"),
                ):
                    click.echo(
                        "Signed in, but the connection changed before the "
                        "model list could be saved. Run: "
                        "llm chatgpt-plan models --refresh",
                        err=True,
                    )
                    return
                _save_models(store, record, models)
            click.echo(f"Saved {len(models)} model(s).")
            click.echo("List them with: llm chatgpt-plan models")
            click.echo("Prompt with: llm -m " + MODEL_ID_PREFIX + "<slug> '...'")
    except AuthError as exc:
        raise _auth_exception(exc) from exc
    except ApiError as exc:
        raise click.ClickException(str(exc)) from exc
    except TimeoutError as exc:
        raise click.ClickException(str(exc)) from exc
    except StorageError as exc:
        raise click.ClickException(str(exc)) from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise click.ClickException(
            f"Could not reach the auth server or local state: {exc}"
        ) from exc
    except KeyboardInterrupt:
        raise click.ClickException("Sign-in aborted.") from None


@chatgpt_plan.command(name="models")
@click.option(
    "--refresh",
    is_flag=True,
    help="Fetch the model list again and update the saved copy",
)
def models_command(refresh):
    "Show the ChatGPT plan models saved for this connection"
    if not refresh:
        try:
            credentials = storage.read_credentials()
            cached = storage.read_models()
        except StorageError as exc:
            raise click.ClickException(
                f"Could not read chatgpt-plan state: {exc}. Run "
                "llm chatgpt-plan login to sign in again."
            ) from exc
        if not credentials or not credentials.get("access_token"):
            raise click.ClickException("Not signed in. Run: llm chatgpt-plan login")
        models = storage.cached_models_for(credentials, cached)
        if models is None:
            raise click.ClickException(
                "No model list saved for this connection. Run: "
                "llm chatgpt-plan models --refresh"
            )
        _print_models(models)
        return
    try:
        with (
            storage.locked_store() as store,
            client.make_http_client(timeout=30) as http,
        ):
            token = OAuth(http).ensure_access_token_locked(store)
            models = client.fetch_models(http, token)
            record = store.load_credentials() or {}
            _save_models(store, record, models)
        _print_models(models)
    except AuthError as exc:
        raise _auth_exception(exc) from exc
    except ApiError as exc:
        raise click.ClickException(
            f"Could not refresh the model list: {exc}. The saved list "
            "was left unchanged."
        ) from exc
    except StorageError as exc:
        raise click.ClickException(str(exc)) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise click.ClickException(
            f"Could not reach the API: {exc}. The saved list was left unchanged."
        ) from exc
    except KeyboardInterrupt:
        raise click.ClickException("Refresh aborted.") from None


@chatgpt_plan.command()
def logout():
    """Sign out: revoke the remote session and remove local tokens.

    Keeps the issued client ID and verified identity so the next login can
    reuse them; clears tokens, scopes, expiry, and the model cache.
    """
    try:
        with client.make_http_client(timeout=30) as http:
            with storage.locked_store() as store:
                # Bump the generation first so any in-flight sign-in or
                # refresh cannot resurrect this connection afterwards.
                generation, credentials = store.bump_generation()
                refresh_token = credentials.get("refresh_token")
                client_id = credentials.get("client_id")
            if not refresh_token or not client_id:
                click.echo("Not signed in.")
                return
            try:
                confirmed = OAuth(http).revoke_refresh_token(client_id, refresh_token)
            except (AuthError, httpx.HTTPError, ValueError, KeyError):
                confirmed = False
            with storage.locked_store() as store:
                current = store.load_credentials() or {}
                if current.get("generation") == generation:
                    # Only wipe if no new sign-in landed meanwhile.
                    kept = {
                        key: current[key]
                        for key in (
                            "client_id",
                            "issuer",
                            "subject",
                            "email",
                            "name",
                            "generation",
                        )
                        if key in current
                    }
                    store.save_credentials(kept)
                    store.delete_models()
        if confirmed:
            click.echo("Signed out. The ChatGPT session was revoked remotely.")
        else:
            click.echo("Signed out locally.")
            click.echo(
                "Could not confirm the remote session was revoked. To "
                "disconnect this app completely, remove it in ChatGPT under "
                "Settings > Security and login > Sign in with ChatGPT.",
                err=True,
            )
    except StorageError as exc:
        raise click.ClickException(str(exc)) from exc
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise click.ClickException(
            f"Could not reach the auth server or local state: {exc}"
        ) from exc
    except KeyboardInterrupt:
        raise click.ClickException("Sign-out aborted.") from None
