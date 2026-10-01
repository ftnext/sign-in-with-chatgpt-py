# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28,<1", "PyJWT[crypto]>=2.10,<3"]
# ///
"""1: Browser sign-in, validate identity, and save OAuth credentials locally.

Run: uv run scripts/auth.py
No inference request is made by this script.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt

DEFAULT_STATE = Path(__file__).resolve().parent / ".chatgpt-script"


@contextmanager
def locked_store(directory: Path):
    """Hold an exclusive process lock while reading or updating credentials."""
    import fcntl

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        directory / "runtime.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "この認証保存先は別のスクリプト／サーバが使用中です。"
            ) from exc
        yield Store(directory)
    finally:
        os.close(fd)


class Store:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.path = self.directory / "credentials.json"
        if self.path.is_symlink():
            raise ValueError("Credential file must not be a symlink")
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "host_id": "urn:uuid:" + str(uuid.uuid4()),
                "accounts": {},
                "active": None,
            }
        )
        self.save()

    def save(self):
        fd, name = tempfile.mkstemp(dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(self.data, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def account(self, client_id=None):
        return self.data["accounts"].get(client_id or self.data["active"])


ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE


class AuthError(Exception):
    """Safe, public error code, never an upstream response body."""


class OAuth:
    def __init__(self, store, http):
        self.store, self.http = store, http
        self.discovery = None
        self.keys = None
        self.keys_at = 0
        self.lock = asyncio.Lock()

    async def metadata(self):
        if self.discovery is None:
            response = await self.http.get(ISSUER + "/.well-known/openid-configuration")
            response.raise_for_status()
            metadata = response.json()
            if metadata.get("issuer") != ISSUER:
                raise AuthError("invalid_issuer")
            for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
                parsed = urlsplit(metadata[key])
                if parsed.scheme != "https" or parsed.netloc != "auth.openai.com":
                    raise AuthError("invalid_discovery")
            self.discovery = metadata
        return self.discovery

    async def start(self, redirect_uri, client_id=None):
        metadata = await self.metadata()
        account = self.store.account(client_id) if client_id else None
        if client_id and account is None:
            raise AuthError("unknown_account")
        verifier = secrets.token_urlsafe(64)
        tx = {
            "state": secrets.token_urlsafe(32),
            "nonce": secrets.token_urlsafe(32),
            "verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "expires": time.time() + 600,
        }
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        params = {
            "client_id": client_id or "dynamic_agent_client",
            "ext_agent_host_id": self.store.data["host_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": tx["state"],
            "nonce": tx["nonce"],
            "code_challenge_method": "S256",
            "code_challenge": challenge.decode().rstrip("="),
        }
        if not client_id:
            params["agent_name_hint"] = "ChatGPT Python Scripts"
        elif account:
            if account.get("id_token"):
                params["id_token_hint"] = account["id_token"]
            if account.get("email"):
                params["login_hint"] = account["email"]
        return tx, metadata["authorization_endpoint"] + "?" + urlencode(params)

    async def verify(self, token, client_id, nonce=None):
        try:
            metadata = await self.metadata()
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise AuthError("invalid_id_token")
            for attempt in range(2):
                if self.keys is None or time.time() - self.keys_at > 3600 or attempt:
                    response = await self.http.get(metadata["jwks_uri"])
                    response.raise_for_status()
                    self.keys = jwt.PyJWKSet.from_dict(response.json())
                    self.keys_at = time.time()
                key = next(
                    (k for k in self.keys.keys if k.key_id == header.get("kid")), None
                )
                if key is not None:
                    break
            if key is None:
                raise AuthError("invalid_id_token")
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256"],
                issuer=ISSUER,
                audience=client_id,
                leeway=5,
                options={"require": ["iss", "aud", "sub", "exp", "iat"]},
            )
            if nonce is not None and claims.get("nonce") != nonce:
                raise AuthError("invalid_nonce")
            if not isinstance(claims["sub"], str) or not claims["sub"]:
                raise AuthError("invalid_subject")
            if "azp" in claims and claims["azp"] != client_id:
                raise AuthError("invalid_audience")
            if (
                isinstance(claims["aud"], list)
                and len(claims["aud"]) > 1
                and claims.get("azp") != client_id
            ):
                raise AuthError("invalid_audience")
            return claims
        except (jwt.PyJWTError, KeyError, ValueError, TypeError) as exc:
            raise AuthError("invalid_id_token") from exc

    async def token_request(self, form):
        metadata = await self.metadata()
        response = await self.http.post(metadata["token_endpoint"], data=form)
        if not response.is_success:
            raise AuthError("reauthorization_required")
        return response.json()

    def credentials(self, tokens, old_scopes=None):
        scopes = tokens.get("scope", " ".join(old_scopes or [])).split()
        if tokens.get("token_type", "").lower() != "bearer":
            raise AuthError("invalid_token_response")
        if (
            not isinstance(tokens.get("access_token"), str)
            or not tokens["access_token"]
        ):
            raise AuthError("plan_permission_required")
        if (
            not isinstance(tokens.get("refresh_token"), str)
            or not tokens["refresh_token"]
        ):
            raise AuthError("invalid_token_response")
        try:
            expires = float(tokens["expires_in"])
            if not 0 < expires <= 86400:
                raise ValueError()
        except (KeyError, ValueError, TypeError) as exc:
            raise AuthError("invalid_token_response") from exc
        return {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "scopes": scopes,
            "expires_at": time.time() + expires,
        }

    async def complete(self, tx, query):
        if tx is None or tx["expires"] <= time.time():
            raise AuthError("expired_sign_in")
        if not secrets.compare_digest(query.get("state", ""), tx["state"]):
            raise AuthError("invalid_state")
        if query.get("error"):
            raise AuthError("sign_in_declined")
        client_id = query.get("client_id") or tx["client_id"]
        if not client_id or client_id == "dynamic_agent_client":
            raise AuthError("missing_client_id")
        if tx["client_id"] and client_id != tx["client_id"]:
            raise AuthError("client_id_mismatch")
        if not query.get("code"):
            raise AuthError("missing_code")
        # Retain issued registration even if exchange fails; no identity is trusted yet.
        self.store.data["accounts"].setdefault(client_id, {"client_id": client_id})
        self.store.save()
        tokens = await self.token_request(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": query["code"],
                "code_verifier": tx["verifier"],
                "redirect_uri": tx["redirect_uri"],
                "resource": RESOURCE,
            }
        )
        identity = await self.verify(tokens.get("id_token", ""), client_id, tx["nonce"])
        previous = self.store.account(client_id)
        if previous.get("subject") and previous["subject"] != identity["sub"]:
            raise AuthError("account_mismatch")
        record = dict(
            previous,
            client_id=client_id,
            issuer=ISSUER,
            subject=identity["sub"],
            email=identity.get("email"),
            name=identity.get("name"),
            id_token=tokens["id_token"],
            **self.credentials(tokens),
        )
        self.store.data["accounts"][client_id] = record
        self.store.data["active"] = client_id
        self.store.save()
        return record


class CallbackListener:
    """One browser authorization result, received on a private loopback listener."""

    def __init__(self, port=0):
        self.loop = asyncio.get_running_loop()
        self.result = self.loop.create_future()
        self.expected_state = None
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Callback URLs contain a credential: do not log them.

            def do_GET(self):
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                valid = (
                    self.headers.get("Host") == listener.host
                    and urlsplit(self.path).path == "/auth/callback"
                    and all(len(values) == 1 for values in query.values())
                    and listener.expected_state is not None
                    and secrets.compare_digest(
                        query.get("state", [""])[0], listener.expected_state
                    )
                )
                body = (
                    "認可結果を受信しました。ターミナルで処理結果を確認してください。"
                    if valid
                    else "この認可結果を確認できませんでした。"
                ).encode()
                self.send_response(200 if valid else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)
                if valid:
                    self.server.result_loop.call_soon_threadsafe(
                        listener.deliver,
                        {key: values[0] for key, values in query.items()},
                    )

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.server.result_loop = self.loop
        self.host = f"127.0.0.1:{self.server.server_port}"
        self.redirect_uri = f"http://{self.host}/auth/callback"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self, state):
        self.expected_state = state
        self.thread.start()

    def deliver(self, query):
        if not self.result.done():
            self.result.set_result(query)

    async def close(self):
        if self.thread.is_alive():
            await asyncio.to_thread(self.server.shutdown)
            await asyncio.to_thread(self.thread.join)
        self.server.server_close()


async def authenticate(args, store, http=None, open_browser=webbrowser.open):
    owned_http = http is None
    http = http or httpx.AsyncClient(timeout=30)
    listener = CallbackListener(args.port)
    try:
        oauth = OAuth(store, http)
        # Reuse the selected registration; --new starts a separate registration.
        client_id = None if args.new else args.account or store.data["active"]
        if not args.new and not client_id and len(store.data["accounts"]) == 1:
            client_id = next(iter(store.data["accounts"]))
        tx, url = await oauth.start(listener.redirect_uri, client_id)
        listener.start(tx["state"])
        print("ブラウザでChatGPTへログインし、プラン利用を許可してください。")
        if not open_browser(url):
            raise RuntimeError(
                "ブラウザを開けませんでした。既定のブラウザを設定して再実行してください。"
            )
        query = await asyncio.wait_for(listener.result, timeout=args.timeout)
        account = await oauth.complete(tx, query)
        if PLAN_SCOPE not in account["scopes"]:
            raise AuthError("plan_permission_required")
        print("認証・ChatGPTプラン利用権限の確認が完了しました。")
        print(f"保存先: {store.path}")
        print("次は ask.py を実行してください。")
    finally:
        await listener.close()
        if owned_http:
            await http.aclose()


def main():
    parser = argparse.ArgumentParser(
        description="ChatGPT認証情報を取得・検証・保存する"
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--account", help="再認証する保存済みclient ID")
    choice.add_argument(
        "--new", action="store_true", help="別アカウント／ワークスペースを登録する"
    )
    parser.add_argument(
        "--port", type=int, default=0, help="コールバックのポート。0で自動選択"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="認可を待つ秒数（最大600秒）"
    )
    args = parser.parse_args()
    if not (args.port == 0 or 1024 <= args.port <= 65535):
        parser.error("portは0または1024〜65535にしてください")
    if not 1 <= args.timeout <= 600:
        parser.error("timeoutは1〜600秒にしてください")
    try:
        with locked_store(args.state_dir) as store:
            asyncio.run(authenticate(args, store))
    except AuthError as error:
        print(f"認証失敗: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except TimeoutError:
        print("認可待ちがタイムアウトしました。再実行してください。", file=sys.stderr)
        raise SystemExit(1) from None
    except (httpx.HTTPError, OSError, ValueError):
        print(
            "認証サーバへの接続、保存先、ローカルポートを確認してください。",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("認証を中止しました。", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
