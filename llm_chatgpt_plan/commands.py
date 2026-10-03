"""The ``llm chatgpt-plan`` command group: login and models."""

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
    }
    message = guidance.get(
        code,
        f"Sign-in failed: {code}. Run llm chatgpt-plan login to try again.",
    )
    return click.ClickException(message)


@chatgpt_plan.command()
@click.option(
    "--replace",
    is_flag=True,
    help="Replace the current connection with a new sign-in",
)
@click.option(
    "--timeout",
    type=int,
    default=oauth.DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help="Seconds to wait for browser sign-in (1-600)",
)
@click.option("--port", type=int, default=0, help="Callback port (0 = automatic)")
def login(replace, timeout, port):
    "Sign in with ChatGPT and save OAuth credentials for this llm"
    if not 1 <= timeout <= oauth.MAX_TIMEOUT_SECONDS:
        raise click.ClickException("--timeout must be between 1 and 600")
    if not (port == 0 or 1024 <= port <= 65535):
        raise click.ClickException("--port must be 0 or between 1024 and 65535")
    try:
        with (
            storage.locked_store() as store,
            client.make_http_client(timeout=30) as http,
        ):
            click.echo(
                "Opening a browser to sign in with ChatGPT and approve plan usage..."
            )
            record = oauth.login_flow(
                store,
                http,
                replace=replace,
                timeout=timeout,
                port=port,
                open_browser=webbrowser.open,
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
