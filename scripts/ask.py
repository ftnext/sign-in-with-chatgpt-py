# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28,<1", "PyJWT[crypto]>=2.10,<3", "openai>=2,<3"]
# ///
"""2: Load OAuth credentials, refresh when needed, and use the ChatGPT plan.

Run: uv run scripts/ask.py --list-models
Then: uv run scripts/ask.py --model MODEL_SLUG '質問'
This script does not start a browser or a local HTTP server.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import jwt
from openai import APIError, APIStatusError, AsyncOpenAI

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

    async def access_token(self):
        async with self.lock:
            account = self.store.account()
            if not account or not account.get("access_token"):
                raise AuthError("sign_in_required")
            if PLAN_SCOPE not in account.get("scopes", []):
                raise AuthError("plan_permission_required")
            if account["expires_at"] <= time.time() + 60:
                tokens = await self.token_request(
                    {
                        "grant_type": "refresh_token",
                        "client_id": account["client_id"],
                        "refresh_token": account["refresh_token"],
                        "resource": RESOURCE,
                    }
                )
                if tokens.get("id_token"):
                    identity = await self.verify(
                        tokens["id_token"], account["client_id"]
                    )
                    if identity["sub"] != account["subject"]:
                        raise AuthError("account_mismatch")
                replacement = self.credentials(tokens, account["scopes"])
                account.update(replacement)
                if tokens.get("id_token"):
                    account["id_token"] = tokens["id_token"]
                self.store.save()
            if PLAN_SCOPE not in account["scopes"]:
                raise AuthError("plan_permission_required")
            return account["access_token"]


async def ask(args, store, http=None):
    owned_http = http is None
    http = http or httpx.AsyncClient(timeout=90)
    try:
        oauth = OAuth(store, http)
        if args.account:
            if store.account(args.account) is None:
                raise AuthError("unknown_account")
            # Select for this invocation without persisting a changed active account.
            previous = store.data["active"]
            store.data["active"] = args.account
        else:
            previous = store.data["active"]
        try:
            token = await oauth.access_token()
        finally:
            # Refresh may save the record; restore the persistent account selection too.
            if store.data["active"] != previous:
                store.data["active"] = previous
                store.save()
        response = await http.get(
            RESOURCE + "/models", headers={"Authorization": "Bearer " + token}
        )
        if response.status_code in (401, 403):
            raise AuthError("reauthorization_required")
        response.raise_for_status()
        models = [
            model
            for model in response.json()["models"]
            if model.get("visibility") == "list"
        ]
        if args.list_models:
            for model in models:
                print(f"{model['slug']}\t{model.get('display_name', model['slug'])}")
            return
        if args.model not in {model["slug"] for model in models}:
            raise RuntimeError(
                "指定モデルは利用可能な一覧にありません。--list-modelsで確認してください。"
            )
        print(
            "Using ChatGPT plan · https://chatgpt.com/settings/usage", file=sys.stderr
        )
        completed = False
        # api_key is the SDK argument name: this is an OAuth token, not a Platform key.
        async with AsyncOpenAI(
            api_key=token, base_url=RESOURCE, http_client=http, max_retries=0
        ) as sdk:
            stream = await sdk.responses.create(
                model=args.model,
                input=[{"role": "user", "content": args.question}],
                store=False,
                stream=True,
            )
            async with stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                    elif event.type == "response.completed":
                        completed = True
                        break
                    elif event.type in {
                        "response.failed",
                        "response.incomplete",
                        "error",
                    }:
                        error = getattr(getattr(event, "response", None), "error", None)
                        code = getattr(error, "code", None) or getattr(
                            event, "code", None
                        )
                        if code in {
                            "subscription_sharing_usage_limit_exceeded",
                            "subscription_sharing_usage_unavailable",
                        }:
                            raise RuntimeError(
                                "ChatGPT利用枠・権限をUsage設定で確認してください。"
                            )
                        raise RuntimeError("推論が失敗または未完了で終了しました。")
            if not completed:
                raise RuntimeError("回答ストリームが途中で終了しました。")
            print()
    finally:
        if owned_http:
            await http.aclose()


def main():
    parser = argparse.ArgumentParser(
        description="保存済み認証情報でChatGPTプランを利用する"
    )
    parser.add_argument("question", nargs="?")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--account", help="使用する保存済みclient ID（省略でアクティブアカウント）"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="モデルslugと表示名を一覧にする"
    )
    parser.add_argument("--model", help="--list-modelsに表示されたslug")
    args = parser.parse_args()
    if not args.list_models and (
        not args.model or not args.question or not args.question.strip()
    ):
        parser.error("--modelと質問を指定するか、--list-modelsを使ってください")
    if not (args.state_dir / "credentials.json").is_file():
        parser.error("認証情報がありません。先にauth.pyを同じ保存先で実行してください")
    try:
        with locked_store(args.state_dir) as store:
            asyncio.run(ask(args, store))
    except AuthError as error:
        print(
            f"接続を確認できません: {error}。auth.pyで再認証してください。",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except APIStatusError as error:
        print(
            f"OpenAIがリクエストを拒否しました（HTTP {error.status_code}）。",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except (APIError, httpx.HTTPError, OSError, ValueError, KeyError):
        print("推論への接続または認証情報の読み込みに失敗しました。", file=sys.stderr)
        raise SystemExit(1) from None
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("推論を中止しました。", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
