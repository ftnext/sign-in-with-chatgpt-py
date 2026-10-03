"""Regressions found during interactive account and concurrency verification."""

import asyncio
import contextlib
import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import (
    ISSUED_CLIENT_ID,
    ORIGIN,
    begin_login,
    bootstrap,
    build_app,
    completed_event,
    delta_event,
    failed_event,
    finish_login,
    post_csrf,
    sign_in,
)
from fastapi.testclient import TestClient

from fastapi_chatgpt_plan import Settings, create_app, registrations
from fastapi_chatgpt_plan.oauth import IDENTITY_SCOPES
from fastapi_chatgpt_plan.sessions import SESSION_COOKIE


def test_fresh_identity_mode_stops_before_upstream(servers, state_dir):
    client = TestClient(build_app(servers, state_dir, plan_enabled=False))
    info = bootstrap(client)
    assert info["login"]["available"] is False
    response = post_csrf(client, "/auth/login", follow_redirects=False)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "identity_registration_required"
    assert not client.app.state.chatgpt_state.transactions
    assert client.app.state.oauth.discovery is None
    assert not servers.token_requests


def test_public_identity_client_needs_only_id_token(servers, state_dir):
    app = create_app(
        Settings(
            chatgpt_state_dir=str(state_dir),
            chatgpt_identity_client_id=ISSUED_CLIENT_ID,
        ),
        http_client=servers.client(),
        expected_host="testserver",
        acquire_lock=False,
    )
    original = servers.token_response

    def identity_response(form):
        return httpx.Response(200, json={"id_token": original(form).json()["id_token"]})

    servers.token_response = identity_response
    client = TestClient(app)
    url = begin_login(client)
    params = parse_qs(urlsplit(url).query)
    assert params["scope"] == [IDENTITY_SCOPES]
    for field in ("resource", "ext_agent_host_id", "agent_name_hint"):
        assert field not in params
    response = finish_login(client, url, servers.nonce_holder)
    assert response.status_code == 303
    assert "resource" not in servers.token_requests[0]
    info = client.get("/api/session").json()
    assert info["status"] == "authenticated"
    assert info["plan"] == {"enabled": False, "permitted": False}
    assert app.state.chatgpt_state.connection.refresh_token is None
    assert client.get("/api/models").status_code == 403
    assert not servers.models_requests
    assert registrations.load_registration(state_dir)["client_kind"] == "identity"
    restarted = TestClient(
        create_app(
            Settings(chatgpt_state_dir=str(state_dir)),
            http_client=servers.client(),
            expected_host="testserver",
            acquire_lock=False,
        )
    )
    sign_in(restarted, servers)
    assert restarted.get("/api/session").json()["status"] == "authenticated"
    plan = TestClient(build_app(servers, state_dir, plan_enabled=True))
    assert post_csrf(plan, "/auth/login", follow_redirects=False).status_code == 409


@pytest.mark.parametrize("failure", ["signature", "token_exchange", "missing_code"])
def test_failed_initial_callback_never_persists_client(servers, state_dir, failure):
    client = TestClient(build_app(servers, state_dir))
    before = (state_dir / "registration.json").read_bytes()
    url = begin_login(client)
    if failure == "signature":
        servers.id_token_override = "invalid-token"
    elif failure == "token_exchange":
        servers.token_status = 400
    response = finish_login(
        client,
        url,
        servers.nonce_holder,
        code="" if failure == "missing_code" else "code",
    )
    assert response.status_code == 400
    assert (state_dir / "registration.json").read_bytes() == before
    assert "client_id" not in client.app.state.registration
    assert client.app.state.chatgpt_state.connection is None


def test_anonymous_logout_cannot_disconnect_owner(servers, state_dir):
    app = build_app(servers, state_dir)
    owner, stranger = TestClient(app), TestClient(app)
    sign_in(owner, servers)
    response = post_csrf(stranger, "/auth/logout")
    assert response.status_code == 401
    assert owner.get("/api/models").status_code == 200
    assert not servers.revoke_requests


def test_reauthorization_required_can_still_logout(client, servers):
    sign_in(client, servers)
    client.app.state.chatgpt_state.connection.needs_reauth = True
    assert post_csrf(client, "/auth/logout").status_code == 200
    assert client.app.state.chatgpt_state.connection is None


async def test_logout_detaches_before_revocation_and_preserves_new_login(
    servers, state_dir
):
    owner = TestClient(build_app(servers, state_dir))
    sign_in(owner, servers)
    app = owner.app
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_revoke(*args):
        entered.set()
        await release.wait()
        return True

    app.state.oauth.revoke = slow_revoke
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=ORIGIN,
        cookies={SESSION_COOKIE: owner.cookies.get(SESSION_COOKIE)},
    ) as client:
        info = (await client.get("/api/session")).json()
        pending = asyncio.create_task(
            client.post(
                "/auth/logout",
                headers={
                    "origin": ORIGIN,
                    "x-csrf-token": info["csrf"],
                },
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert (await client.get("/api/models")).status_code == 401
            info = (await client.get("/api/session")).json()
            start = await client.post(
                "/auth/login",
                headers={
                    "origin": ORIGIN,
                    "x-csrf-token": info["csrf"],
                },
                follow_redirects=False,
            )
            params = parse_qs(urlsplit(start.headers["location"]).query)
            servers.nonce_holder["nonce"] = params["nonce"][0]
            callback = await client.get(
                "/auth/callback",
                params={
                    "state": params["state"][0],
                    "code": "new-code",
                    "client_id": ISSUED_CLIENT_ID,
                },
                follow_redirects=False,
            )
            assert callback.status_code == 303
            new_connection = app.state.chatgpt_state.connection
            release.set()
            logout = await asyncio.wait_for(pending, 2)
            assert logout.status_code == 200
            assert "set-cookie" not in logout.headers
            assert app.state.chatgpt_state.connection is new_connection
            assert (await client.get("/api/models")).status_code == 200
        finally:
            release.set()
            await pending


async def test_logout_invalidates_inflight_callback(client, servers):
    sign_in(client, servers)
    url = begin_login(client)
    state = client.app.state.chatgpt_state
    # The callback may already have consumed its transaction before logout.
    params = parse_qs(urlsplit(url).query)
    tx = await state.consume_transaction(
        params["state"][0], client.cookies.get(SESSION_COOKIE)
    )
    session = await state.get_session(client.cookies.get(SESSION_COOKIE))
    old_connection = state.connection
    await state.logout(session)
    committed = []
    from fastapi_chatgpt_plan.errors import AuthError

    with pytest.raises(AuthError, match="invalid_state"):
        await state.finish_sign_in(old_connection, tx, lambda: committed.append(True))
    assert not committed
    assert state.connection is None


class BlockingUpstream(httpx.AsyncByteStream):
    def __init__(self):
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield ("data: " + json.dumps(delta_event("first")) + "\n\n").encode()
        self.blocked.set()
        await self.release.wait()
        yield ("data: " + json.dumps(completed_event()) + "\n\n").encode()

    async def aclose(self):
        self.closed = True


async def start_response(owner):
    """Drive ASGI directly so a blocked upstream remains observable."""
    sid = owner.cookies.get(SESSION_COOKIE)
    csrf = owner.get("/api/session").json()["csrf"]
    body = json.dumps(
        {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]}
    ).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/responses",
        "raw_path": b"/api/responses",
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 123),
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"cookie", f"{SESSION_COOKIE}={sid}".encode()),
            (b"origin", ORIGIN.encode()),
            (b"x-csrf-token", csrf.encode()),
            (b"content-type", b"application/json"),
        ],
    }
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    await incoming.put({"type": "http.request", "body": body, "more_body": False})
    task = asyncio.create_task(owner.app(scope, incoming.get, outgoing.put))
    start = await asyncio.wait_for(outgoing.get(), 2)
    assert start["status"] == 200
    first = await asyncio.wait_for(outgoing.get(), 2)
    assert b"response.output_text.delta" in first["body"]
    return task, incoming, outgoing


@pytest.mark.parametrize("stop", ["logout", "disconnect"])
async def test_blocked_upstream_closes_on_logout_or_disconnect(
    servers, state_dir, stop
):
    owner = TestClient(build_app(servers, state_dir))
    sign_in(owner, servers)
    upstream = BlockingUpstream()
    servers.responses_stream = upstream
    task, incoming, outgoing = await start_response(owner)
    try:
        await asyncio.wait_for(upstream.blocked.wait(), 2)
        if stop == "logout":
            sid = owner.cookies.get(SESSION_COOKIE)
            info = owner.get("/api/session").json()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=owner.app),
                base_url=ORIGIN,
                cookies={SESSION_COOKIE: sid},
            ) as ac:
                response = await ac.post(
                    "/auth/logout",
                    headers={
                        "origin": ORIGIN,
                        "x-csrf-token": info["csrf"],
                    },
                )
                assert response.status_code == 200
        else:
            await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 2)
        assert upstream.closed
        assert not owner.app.state.chatgpt_state.active_streams
        assert not upstream.release.is_set()
        while not outgoing.empty():
            assert b"response.completed" not in (await outgoing.get()).get("body", b"")
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("kind", ["response.failed", "response.incomplete", "error"])
def test_failure_sse_preserves_structure_and_redacts_credentials(
    servers, state_dir, kind
):
    from test_api import _sse_parse

    owner = TestClient(build_app(servers, state_dir))
    sign_in(owner, servers)
    if kind == "error":
        event = {
            "type": "error",
            "code": "server_error",
            "message": "access-token-1",
            "param": "input",
            "sequence_number": 2,
        }
    else:
        event = failed_event(message="access-token-1")
        event["type"] = kind
        event["response"]["status"] = kind.removeprefix("response.")
    servers.response_events = [event]
    response = post_csrf(
        owner,
        "/api/responses",
        json={
            "model": "gpt-6.1-sol",
            "input": [{"role": "user", "content": "hi"}],
        },
    )
    events = _sse_parse(response.text)
    assert len(events) == 1
    assert events[0][0] == kind
    assert events[0][1]["type"] == kind
    assert events[0][1]["sequence_number"] == 2
    assert "access-token-1" not in response.text
    if kind != "error":
        assert events[0][1]["response"]["error"]["code"] == "server_error"


@pytest.mark.parametrize("operation", ["models", "responses"])
async def test_logout_rejects_upstream_results_opened_for_old_generation(
    servers,
    state_dir,
    monkeypatch,
    operation,
):
    from fastapi_chatgpt_plan import client as api_client

    owner = TestClient(build_app(servers, state_dir))
    sign_in(owner, servers)
    app = owner.app
    entered, release = asyncio.Event(), asyncio.Event()
    closed = []
    if operation == "models":
        original = api_client.fetch_models

        async def delayed(*args):
            entered.set()
            await release.wait()
            return await original(*args)

        monkeypatch.setattr(api_client, "fetch_models", delayed)
    else:
        app.state.chatgpt_state.models_cache = {
            "client_id": ISSUED_CLIENT_ID,
            "subject": app.state.chatgpt_state.connection.subject,
            "models": [{"slug": "gpt-6.1-sol"}],
        }

        class OpenedStream:
            async def close(self):
                closed.append(True)

        async def delayed(*args):
            entered.set()
            await release.wait()
            return OpenedStream()

        monkeypatch.setattr(api_client, "create_response_stream", delayed)
    csrf = owner.get("/api/session").json()["csrf"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=ORIGIN,
        cookies={SESSION_COOKIE: owner.cookies.get(SESSION_COOKIE)},
    ) as ac:
        if operation == "models":
            pending = asyncio.create_task(ac.get("/api/models"))
        else:
            pending = asyncio.create_task(
                ac.post(
                    "/api/responses",
                    headers={
                        "origin": ORIGIN,
                        "x-csrf-token": csrf,
                    },
                    json={
                        "model": "gpt-6.1-sol",
                        "input": [{"role": "user", "content": "hi"}],
                    },
                )
            )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert (
                await ac.post(
                    "/auth/logout",
                    headers={
                        "origin": ORIGIN,
                        "x-csrf-token": csrf,
                    },
                )
            ).status_code == 200
            release.set()
            assert (await asyncio.wait_for(pending, 2)).status_code == 401
            assert app.state.chatgpt_state.models_cache is None
            assert not app.state.chatgpt_state.active_streams
            if operation == "responses":
                assert closed == [True]
        finally:
            release.set()
            await pending
