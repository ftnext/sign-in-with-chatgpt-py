"""Refresh, logout races, process lock, and streaming behavior."""

import asyncio
import json
import time

import httpx
import pytest
from conftest import (
    ORIGIN,
    build_app,
    completed_event,
    delta_event,
    sign_in,
)
from fastapi.testclient import TestClient

from fastapi_chatgpt_plan import registrations
from fastapi_chatgpt_plan.registrations import StorageError
from fastapi_chatgpt_plan.sessions import SESSION_COOKIE


def _signed_in(servers, state_dir):
    client = TestClient(build_app(servers, state_dir, plan_enabled=True))
    sign_in(client, servers)
    return client


def _async_client(app, sid):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=ORIGIN,
        cookies={SESSION_COOKIE: sid},
    )


def _refresh_requests(servers):
    return [
        form
        for form in servers.token_requests
        if form.get("grant_type") == "refresh_token"
    ]


async def test_refresh_runs_once_under_concurrency(servers, state_dir):
    client = _signed_in(servers, state_dir)
    app = client.app
    connection = app.state.chatgpt_state.connection
    connection.expires_at = time.time() - 1
    sid = client.cookies.get(SESSION_COOKIE)
    async with _async_client(app, sid) as ac:
        first, second = await asyncio.gather(
            ac.get("/api/models"), ac.get("/api/models")
        )
    assert first.status_code == 200 and second.status_code == 200
    assert len(_refresh_requests(servers)) == 1


async def test_refresh_rotates_and_reuses_latest_token(servers, state_dir):
    client = _signed_in(servers, state_dir)
    app = client.app
    connection = app.state.chatgpt_state.connection
    sid = client.cookies.get(SESSION_COOKIE)
    async with _async_client(app, sid) as ac:
        connection.expires_at = time.time() - 1
        first = await ac.get("/api/models")
        assert first.status_code == 200
        connection.expires_at = time.time() - 1
        second = await ac.get("/api/models")
        assert second.status_code == 200
    refreshes = _refresh_requests(servers)
    assert len(refreshes) == 2
    assert refreshes[0]["refresh_token"] == "refresh-token-1"
    assert refreshes[1]["refresh_token"] == "refresh-token-1-next"
    assert refreshes[0]["client_id"] == refreshes[1]["client_id"]
    assert "scope" not in refreshes[0]


async def test_refresh_failure_marks_reauth_required(servers, state_dir):
    client = _signed_in(servers, state_dir)
    app = client.app
    connection = app.state.chatgpt_state.connection
    connection.expires_at = time.time() - 1
    servers.token_status = 400
    response = client.get("/api/models")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "reauthorization_required"
    info = client.get("/api/session").json()
    assert info["status"] == "reauthorization_required"


async def test_logout_during_refresh_does_not_revive_connection(
    servers, state_dir
):
    client = _signed_in(servers, state_dir)
    app = client.app
    connection = app.state.chatgpt_state.connection
    connection.expires_at = time.time() - 1

    started = asyncio.Event()
    release = asyncio.Event()
    original = app.state.oauth.refresh

    async def gated_refresh(conn):
        started.set()
        await release.wait()
        return await original(conn)

    app.state.oauth.refresh = gated_refresh
    sid = client.cookies.get(SESSION_COOKIE)
    async with _async_client(app, sid) as ac:
        pending = asyncio.create_task(ac.get("/api/models"))
        await started.wait()
        info = (await ac.get("/api/session")).json()
        logout = await ac.post(
            "/auth/logout",
            headers={"origin": ORIGIN, "x-csrf-token": info["csrf"]},
        )
        assert logout.status_code == 200
        release.set()
        response = await pending
    assert response.status_code == 401
    assert app.state.chatgpt_state.connection is None


def test_process_lock_rejects_second_holder(state_dir):
    fd = registrations.acquire_process_lock(state_dir)
    try:
        with pytest.raises(StorageError):
            registrations.acquire_process_lock(state_dir)
    finally:
        import os

        os.close(fd)


def test_old_cookie_invalid_on_new_instance(servers, state_dir):
    first = _signed_in(servers, state_dir)
    old_cookie = first.cookies.get(SESSION_COOKIE)
    second = TestClient(build_app(servers, state_dir, plan_enabled=True))
    second.cookies.set(SESSION_COOKIE, old_cookie)
    info = second.get("/api/session").json()
    assert info["status"] == "anonymous"


class _GatedStream(httpx.AsyncByteStream):
    def __init__(self, chunks, gate):
        self.chunks = chunks
        self.gate = gate

    async def __aiter__(self):
        yield self.chunks[0]
        await self.gate.wait()
        for chunk in self.chunks[1:]:
            yield chunk


def _sse_block(event):
    return ("data: " + json.dumps(event) + "\n\n").encode()


async def test_first_delta_arrives_before_upstream_finishes(
    servers, state_dir
):
    client = _signed_in(servers, state_dir)
    app = client.app
    gate = asyncio.Event()
    servers.responses_stream = _GatedStream(
        [
            _sse_block(delta_event("first", 1)),
            _sse_block(delta_event("second", 2)),
            _sse_block(completed_event(3)),
        ],
        gate,
    )
    sid = client.cookies.get(SESSION_COOKIE)
    csrf = (await _api_session(app, sid))["csrf"]
    body = json.dumps(
        {
            "model": "gpt-6.1-sol",
            "input": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
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
            (b"content-length", str(len(body)).encode()),
        ],
    }
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await asyncio.Future()

    messages = asyncio.Queue()

    async def send(message):
        await messages.put(message)

    app_task = asyncio.create_task(app(scope, receive, send))
    try:
        start = await asyncio.wait_for(messages.get(), 10)
        assert start["type"] == "http.response.start"
        assert start["status"] == 200

        first = await asyncio.wait_for(messages.get(), 10)
        assert first["type"] == "http.response.body"
        assert b"response.output_text.delta" in first["body"]
        assert not gate.is_set()

        gate.set()
        rest = first["body"]
        while True:
            message = await asyncio.wait_for(messages.get(), 10)
            assert message["type"] == "http.response.body"
            rest += message["body"]
            if not message.get("more_body"):
                break
        await asyncio.wait_for(app_task, 10)
    finally:
        if not app_task.done():
            gate.set()
            app_task.cancel()
    assert b"response.completed" in rest


async def _api_session(app, sid):
    async with _async_client(app, sid) as ac:
        response = await ac.get("/api/session")
        return response.json()
