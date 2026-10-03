"""Plan API: model listing and the Responses SSE wrapper."""

import json

from conftest import (
    ORIGIN,
    build_app,
    completed_event,
    delta_event,
    failed_event,
    post_csrf,
    sign_in,
)
from fastapi.testclient import TestClient


def _signed_in_client(servers, state_dir):
    client = TestClient(build_app(servers, state_dir, plan_enabled=True))
    sign_in(client, servers)
    return client


def _sse_parse(text):
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((name, data))
    return events


def test_models_filters_visibility_and_preserves_order(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert [m["slug"] for m in models] == ["gpt-6.1-sol", "gpt-5-mini"]
    assert models[0]["display_name"] == "GPT 6.1 Sol"
    assert models[1]["display_name"] == "gpt-5-mini"
    auth = servers.models_requests[0].headers.get("authorization")
    assert auth == "Bearer access-token-1"


def test_models_disabled_returns_403(servers, state_dir):
    servers.scope = "openid profile email"
    client = TestClient(build_app(servers, state_dir, plan_enabled=False))
    sign_in(client, servers)
    response = client.get("/api/models")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "inference_disabled"
    assert servers.models_requests == []


def test_models_usage_limit_returns_429(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.models_status = 429
    servers.models_error = {
        "error": {
            "code": "subscription_sharing_usage_limit_exceeded",
            "message": "limit",
        }
    }
    response = client.get("/api/models")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "usage_limit"


def test_responses_streams_sse_events(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.response_events = [
        delta_event("Hello", 1),
        delta_event(" world", 2),
        completed_event(3),
    ]
    response = post_csrf(
        client,
        "/api/responses",
        json={
            "model": "gpt-6.1-sol",
            "input": [
                {"role": "user", "content": "最初の質問"},
                {"role": "assistant", "content": "前の回答"},
                {"role": "user", "content": "続きの質問"},
            ],
            "instructions": "簡潔に日本語で回答する",
            "reasoning": {"effort": "low"},
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    events = _sse_parse(response.text)
    assert [name for name, _ in events] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0][1]["delta"] == "Hello"
    assert events[1][1]["delta"] == " world"

    payload = servers.responses_payloads[0]
    assert payload["model"] == "gpt-6.1-sol"
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["instructions"] == "簡潔に日本語で回答する"
    assert payload["reasoning"] == {"effort": "low"}
    assert [m["role"] for m in payload["input"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert payload["input"][2]["content"] == "続きの質問"


def test_responses_omits_unset_options(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.response_events = [delta_event("x", 1), completed_event(2)]
    response = post_csrf(
        client,
        "/api/responses",
        json={
            "model": "gpt-6.1-sol",
            "input": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    payload = servers.responses_payloads[0]
    assert "instructions" not in payload
    assert "reasoning" not in payload


def test_responses_validation_rejections(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    base = {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]}

    cases = [
        ({**base, "extra_field": 1}, 422),
        ({**base, "input": []}, 422),
        ({**base, "input": [{"role": "system", "content": "x"}]}, 422),
        ({**base, "input": [{"role": "user"}]}, 422),
        ({**base, "reasoning": {"effort": "ultra"}}, 422),
        ({**base, "stream": False}, 422),
        ({**base, "store": True}, 422),
        ({**base, "previous_response_id": "r"}, 422),
        ({**base, "model": "not-in-list"}, 422),
        ({"model": "gpt-6.1-sol"}, 422),
    ]
    for body, expected in cases:
        response = post_csrf(client, "/api/responses", json=body)
        assert response.status_code == expected, (body, response.text)
        assert "error" in response.json()
    assert servers.responses_payloads == []


def test_responses_size_limits(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    big_message = {
        "model": "gpt-6.1-sol",
        "input": [{"role": "user", "content": "x" * 200_001}],
    }
    response = post_csrf(client, "/api/responses", json=big_message)
    assert response.status_code == 422

    too_many = {
        "model": "gpt-6.1-sol",
        "input": [{"role": "user", "content": "x"}] * 201,
    }
    response = post_csrf(client, "/api/responses", json=too_many)
    assert response.status_code == 422

    response = client.post(
        "/api/responses",
        headers={
            "origin": ORIGIN,
            "x-csrf-token": client.get("/api/session").json()["csrf"],
            "content-type": "application/json",
        },
        content=b"x" * (1024 * 1024 + 1),
    )
    assert response.status_code == 413
    assert servers.responses_payloads == []


def test_responses_invalid_json(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    response = client.post(
        "/api/responses",
        headers={
            "origin": ORIGIN,
            "x-csrf-token": client.get("/api/session").json()["csrf"],
            "content-type": "application/json",
        },
        content=b"{not json",
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_json"


def test_responses_requires_csrf(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    response = client.post(
        "/api/responses",
        headers={"origin": ORIGIN},
        json={"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 403


def test_responses_failed_stream_reports_error_event(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.response_events = [delta_event("partial", 1), failed_event()]
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    events = _sse_parse(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "upstream_failed"


def test_responses_usage_limit_mid_stream(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.response_events = [
        delta_event("partial", 1),
        failed_event(code="subscription_sharing_usage_limit_exceeded"),
    ]
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]},
    )
    events = _sse_parse(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "usage_limit"


def test_responses_stream_without_terminal_event(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.response_events = [delta_event("partial", 1)]
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]},
    )
    events = _sse_parse(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "stream_incomplete"


def test_responses_upstream_rejection_is_http_error(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.responses_status = 400
    servers.responses_error = {"detail": "admission denied"}
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "upstream_error"
    assert "access-token" not in json.dumps(body)


def test_responses_usage_limit_before_stream(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.responses_status = 429
    servers.responses_error = {
        "error": {
            "code": "subscription_sharing_usage_limit_exceeded",
            "message": "limit",
        }
    }
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429


def test_sse_never_leaks_tokens(servers, state_dir):
    client = _signed_in_client(servers, state_dir)
    servers.response_events = [delta_event("hi", 1), completed_event(2)]
    response = post_csrf(
        client,
        "/api/responses",
        json={"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi"}]},
    )
    assert "access-token" not in response.text
    assert "refresh-token" not in response.text
    assert "authorization" not in response.text.lower()
