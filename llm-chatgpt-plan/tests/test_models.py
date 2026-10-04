import json
import time

import httpx
import llm
import pytest
from llm.models import Prompt
from llm.parts import Message, ReasoningPart, TextPart

import llm_chatgpt_plan
from llm_chatgpt_plan import client, storage
from llm_chatgpt_plan.models import ChatGPTPlan
from llm_chatgpt_plan.oauth import RESOURCE


def registered():
    models = []
    llm_chatgpt_plan.register_models(models.append, [])
    return models


class TestRegistration:
    def test_registers_saved_models(self, stored_connection):
        models = registered()
        assert [m.model_id for m in models] == [
            "chatgpt-plan/gpt-5.6-luna",
            "chatgpt-plan/gpt-5-mini",
        ]
        assert all(m.can_stream for m in models)
        assert "ChatGPTPlan: chatgpt-plan/gpt-5.6-luna" in str(models[0])

    def test_no_models_when_not_signed_in(self, state_dir):
        assert registered() == []
        # and it must not create any state
        assert not state_dir.exists()

    def test_no_models_for_registration_stub_only(self, state_dir):
        state_dir.mkdir(parents=True)
        storage.Store(state_dir).save_credentials({"client_id": "x"})
        assert registered() == []

    def test_no_models_for_other_connection(self, stored_connection, state_dir):
        storage.Store(state_dir).save_models(
            {
                "fetched_at": time.time(),
                "client_id": "other",
                "subject": "other",
                "models": [{"slug": "stale"}],
            }
        )
        assert registered() == []

    def test_corrupt_cache_does_not_raise(self, stored_connection, state_dir):
        (state_dir / "models.json").write_text("{broken")
        assert registered() == []

    def test_register_commands_adds_group(self):
        import click

        cli = click.Group()
        llm_chatgpt_plan.register_commands(cli)
        assert "chatgpt-plan" in cli.commands
        assert set(cli.commands["chatgpt-plan"].commands) == {
            "login",
            "logout",
            "models",
        }


def delta(text, seq):
    return (
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "item_id": "i1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": seq,
            "delta": text,
            "logprobs": [],
        },
    )


def completed(seq=99, status="completed", error=None):
    return (
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": seq,
            "response": {
                "id": "resp_1",
                "created_at": 1,
                "model": "gpt-5.6-luna",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": status,
                "error": error,
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            },
        },
    )


def sse_body(events):
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    )


class DummyResponse:
    def __init__(self):
        self.usage = None

    def set_usage(self, **kwargs):
        self.usage = kwargs


def make_prompt(model, text="こんにちは", **kwargs):
    return Prompt(text, model=model, **kwargs)


@pytest.fixture
def api_capture(stored_connection, monkeypatch):
    """Serve inference over a mock transport and capture the request."""
    captured = {}

    def handler(request):
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        events = [delta("Hello", 1), delta(" world", 2), completed(3)]
        return httpx.Response(
            200,
            content=sse_body(events).encode(),
            headers={"content-type": "text/event-stream"},
        )

    def factory(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(client, "make_http_client", factory)
    return captured


def run(model, prompt, stream=True, conversation=None):
    response = DummyResponse()
    chunks = list(model.execute(prompt, stream, response, conversation))
    return chunks, response


class TestExecute:
    def test_streams_deltas_and_usage(self, api_capture, capsys):
        model = ChatGPTPlan("gpt-5.6-luna")
        chunks, response = run(model, make_prompt(model))
        assert chunks == ["Hello", " world"]
        assert response.usage == {"input": 11, "output": 7}
        err = capsys.readouterr().err
        assert "chatgpt.com/settings/usage" in err

    def test_request_shape(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        run(
            model,
            make_prompt(
                model,
                "質問",
                system="簡潔に",
                options=model.Options(reasoning_effort="low"),
            ),
        )
        body = api_capture["body"]
        assert body["model"] == "gpt-5.6-luna"  # no chatgpt-plan/ prefix
        assert body["input"] == [{"role": "user", "content": "質問"}]
        assert body["store"] is False
        assert body["stream"] is True
        assert body["instructions"] == "簡潔に"
        assert body["reasoning"] == {"effort": "low"}
        request = api_capture["request"]
        assert request.headers["authorization"] == "Bearer stored-access"
        assert str(request.url) == RESOURCE + "/responses"

    def test_reasoning_omitted_by_default(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        run(model, make_prompt(model))
        assert "reasoning" not in api_capture["body"]
        assert "instructions" not in api_capture["body"]

    def test_no_stream_rejected_before_api(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        with pytest.raises(llm.ModelError, match="streaming"):
            run(model, make_prompt(model), stream=False)
        assert "request" not in api_capture

    def test_first_turn_of_conversation_allowed(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        conversation = model.conversation()
        chunks, _ = run(model, make_prompt(model), conversation=conversation)
        assert chunks == ["Hello", " world"]

    def test_attachments_rejected(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = make_prompt(model, attachments=[llm.Attachment(content=b"x")])
        with pytest.raises(llm.ModelError, match="attachments"):
            run(model, prompt)
        assert "request" not in api_capture

    def test_schema_rejected(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = make_prompt(model, schema={"type": "object"})
        with pytest.raises(llm.ModelError, match="schema"):
            run(model, prompt)

    def test_tools_rejected(self, api_capture):
        def some_tool():
            pass

        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = make_prompt(model, tools=[some_tool])
        with pytest.raises(llm.ModelError, match="tools"):
            run(model, prompt)

    def test_tool_results_rejected(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = make_prompt(model, tool_results=[object()])
        with pytest.raises(llm.ModelError, match="tool results"):
            run(model, prompt)

    def test_slug_not_in_list_rejected(self, api_capture):
        model = ChatGPTPlan("unknown-model")
        with pytest.raises(llm.ModelError, match="models --refresh"):
            run(model, make_prompt(model))
        assert "request" not in api_capture

    def test_not_signed_in(self, user_dir):
        model = ChatGPTPlan("gpt-5.6-luna")
        with pytest.raises(llm.ModelError, match="login"):
            run(model, make_prompt(model))


class TestContinuation:
    """prompt.messages is the single source: history becomes input items."""

    def test_history_replayed_in_order(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = Prompt(
            "",
            model=model,
            messages=[
                Message(role="system", parts=[TextPart("S")]),
                Message(role="user", parts=[TextPart("u1")]),
                Message(role="assistant", parts=[TextPart("a1")]),
                Message(role="user", parts=[TextPart("u2")]),
            ],
        )
        run(model, prompt)
        body = api_capture["body"]
        assert body["input"] == [
            {"role": "user", "content": "u1"},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "a1", "annotations": []}],
            },
            {"role": "user", "content": "u2"},
        ]
        # system text is sent as instructions, never as a system input item
        assert body["instructions"] == "S"
        assert "previous_response_id" not in body
        assert "conversation" not in body
        assert body["store"] is False

    def test_loaded_conversation_replayed(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        conversation = model.conversation()
        conversation.loaded_messages = [
            {"role": "user", "content": [{"type": "text", "text": "old-q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "old-a"}]},
        ]
        response = conversation.prompt("next-q")
        run(model, response.prompt, conversation=conversation)
        body = api_capture["body"]
        roles = [item.get("role") for item in body["input"]]
        assert roles == ["user", "assistant", "user"]
        assert body["input"][0]["content"] == "old-q"
        assert body["input"][1]["content"][0]["text"] == "old-a"
        assert body["input"][1]["content"][0]["type"] == "output_text"
        assert body["input"][2]["content"] == "next-q"
        assert "instructions" not in body

    def test_conversation_system_not_duplicated(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        response = model.conversation().prompt("hi", system="sys-1")
        run(model, response.prompt)
        # llm bakes --system into the chain; it must appear exactly once
        assert api_capture["body"]["instructions"] == "sys-1"

    def test_explicit_messages_with_system_kwarg(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = Prompt(
            "",
            model=model,
            messages=[Message(role="user", parts=[TextPart("u1")])],
            system="sys-x",
        )
        run(model, prompt)
        assert api_capture["body"]["instructions"] == "sys-x"

    def test_assistant_id_and_developer_role(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = Prompt(
            "",
            model=model,
            messages=[
                {"role": "developer", "parts": [{"type": "text", "text": "D"}]},
                {"role": "user", "content": "q"},
            ],
        )
        run(model, prompt)
        body = api_capture["body"]
        assert body["instructions"] == "D"
        assert body["input"] == [{"role": "user", "content": "q"}]

    def test_redacted_empty_reasoning_is_skipped(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = Prompt(
            "",
            model=model,
            messages=[
                Message(
                    role="assistant",
                    parts=[ReasoningPart(redacted=True), TextPart("a1")],
                ),
                Message(role="user", parts=[TextPart("u2")]),
            ],
        )
        run(model, prompt)
        body = api_capture["body"]
        assert body["input"][0]["content"][0]["text"] == "a1"

    @pytest.mark.parametrize(
        "message",
        [
            Message(role="tool", parts=[TextPart("out")]),
            Message(role="assistant", parts=[ReasoningPart(text="thought")]),
            Message(
                role="user",
                parts=[
                    llm.parts.AttachmentPart(attachment=llm.Attachment(content=b"x"))
                ],
            ),
            {"role": "tool", "content": "out"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "n"}]},
            {"role": "user", "parts": [{"type": "tool_call", "name": "t"}]},
        ],
        ids=[
            "tool-role",
            "reasoning-text",
            "attachment-part",
            "dict-tool-role",
            "dict-refusal",
            "dict-tool-call",
        ],
    )
    def test_non_text_history_rejected(self, api_capture, message):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = Prompt(
            "",
            model=model,
            messages=[
                message,
                Message(role="user", parts=[TextPart("hi")]),
            ],
        )
        with pytest.raises(llm.ModelError, match="chatgpt-plan"):
            run(model, prompt)
        assert "request" not in api_capture

    def test_empty_input_rejected(self, api_capture):
        model = ChatGPTPlan("gpt-5.6-luna")
        prompt = Prompt(
            "",
            model=model,
            messages=[Message(role="system", parts=[TextPart("only-sys")])],
        )
        with pytest.raises(llm.ModelError, match="Nothing to send"):
            run(model, prompt)
        assert "request" not in api_capture


@pytest.fixture
def api_events(stored_connection, monkeypatch):
    """Mock transport returning a programmable SSE event list or status."""
    state = {"events": [], "status": 200, "error_body": {}}

    def handler(request):
        if state["status"] != 200:
            return httpx.Response(state["status"], json=state["error_body"])
        return httpx.Response(
            200,
            content=sse_body(state["events"]).encode(),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        client,
        "make_http_client",
        lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw),
    )
    return state


class TestStreamFailures:
    def model(self):
        return ChatGPTPlan("gpt-5.6-luna")

    def test_failed_event(self, api_events):
        api_events["events"] = [
            delta("partial", 1),
            (
                "response.failed",
                {
                    "type": "response.failed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "created_at": 1,
                        "model": "m",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                        "status": "failed",
                        "error": {"code": "server_error", "message": "boom"},
                    },
                },
            ),
        ]
        with pytest.raises(llm.ModelError, match="failed"):
            run(self.model(), make_prompt(self.model()))

    def test_incomplete_event(self, api_events):
        api_events["events"] = [
            delta("partial", 1),
            (
                "response.incomplete",
                {
                    "type": "response.incomplete",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "created_at": 1,
                        "model": "m",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                        "status": "incomplete",
                    },
                },
            ),
        ]
        with pytest.raises(llm.ModelError, match="incomplete"):
            run(self.model(), make_prompt(self.model()))

    def test_stream_ends_without_completed(self, api_events):
        api_events["events"] = [delta("partial", 1)]
        with pytest.raises(llm.ModelError, match="without a completed"):
            run(self.model(), make_prompt(self.model()))

    def test_http_error_detail_extracted(self, api_events):
        api_events["status"] = 400
        api_events["error_body"] = {"detail": "Input must be a list"}
        with pytest.raises(llm.ModelError, match="Input must be a list"):
            run(self.model(), make_prompt(self.model()))

    def test_usage_limit_error_guidance(self, api_events):
        api_events["status"] = 403
        api_events["error_body"] = {
            "error": {
                "code": "subscription_sharing_usage_limit_exceeded",
                "message": "limit",
            }
        }
        with pytest.raises(llm.ModelError, match="settings/usage"):
            run(self.model(), make_prompt(self.model()))

    def test_stream_mid_failure_usage_limit(self, api_events):
        api_events["events"] = [
            delta("partial", 1),
            (
                "response.failed",
                {
                    "type": "response.failed",
                    "sequence_number": 2,
                    "response": {
                        "id": "r",
                        "created_at": 1,
                        "model": "m",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                        "status": "failed",
                        "error": {
                            "code": "subscription_sharing_usage_unavailable",
                            "message": "x",
                        },
                    },
                },
            ),
        ]
        with pytest.raises(llm.ModelError, match="settings/usage"):
            run(self.model(), make_prompt(self.model()))

    def test_token_echoed_by_server_is_redacted(self, api_events):
        api_events["status"] = 401
        api_events["error_body"] = {
            "error": {
                "message": "Incorrect API key provided: stored-access",
            }
        }
        with pytest.raises(llm.ModelError) as info:
            run(self.model(), make_prompt(self.model()))
        assert "stored-access" not in str(info.value)
        assert "[redacted]" in str(info.value)

    def test_no_tokens_in_errors(self, api_events):
        api_events["status"] = 500
        api_events["error_body"] = {
            "error": {"message": "upstream exploded", "code": "server_error"}
        }
        with pytest.raises(llm.ModelError) as info:
            run(self.model(), make_prompt(self.model()))
        assert "stored-access" not in str(info.value)
        assert "stored-refresh" not in str(info.value)


@pytest.mark.parametrize(
    "event_type", ["response.failed", "response.incomplete", "error"]
)
def test_stream_errors_redact_token_in_code_and_message(api_events, event_type):
    error = {"code": "echo-stored-access", "message": "echo stored-access"}
    if event_type == "error":
        payload = {"type": event_type, "sequence_number": 1, **error, "param": None}
    else:
        _, payload = completed(status=event_type.split(".")[1], error=error)
        payload["type"] = event_type
    api_events["events"] = [(event_type, payload)]
    model = ChatGPTPlan("gpt-5.6-luna")
    with pytest.raises(llm.ModelError) as info:
        run(model, make_prompt(model))
    assert "stored-access" not in str(info.value)
    assert "[redacted]" in str(info.value)


def test_connection_replaced_before_snapshot_is_revalidated(
    stored_connection, state_dir, monkeypatch
):
    requests = []

    def factory(**kwargs):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(
                dict(
                    stored_connection,
                    client_id="B",
                    subject="B",
                    access_token="B-token",
                )
            )
            store.save_models(
                {"client_id": "B", "subject": "B", "models": [{"slug": "only-B"}]}
            )
        return httpx.Client(
            transport=httpx.MockTransport(lambda r: requests.append(r)), **kwargs
        )

    monkeypatch.setattr(client, "make_http_client", factory)
    model = ChatGPTPlan("gpt-5.6-luna")
    with pytest.raises(llm.ModelError, match="not in the cached model list"):
        run(model, make_prompt(model))
    assert requests == []


def test_inference_uses_coherent_snapshot_and_releases_lock(
    api_capture, state_dir, monkeypatch
):
    original = client.stream_response

    def replace_then_stream(http, token, slug, input_items, **kwargs):
        # A replacement after the snapshot must not change its bearer token.
        # Acquiring the lock here also proves it isn't held during streaming.
        with storage.locked_store(state_dir) as store:
            store.save_credentials(
                {"client_id": "B", "subject": "B", "access_token": "B-token"}
            )
            store.save_models(
                {"client_id": "B", "subject": "B", "models": [{"slug": "only-B"}]}
            )
        return original(http, token, slug, input_items, **kwargs)

    monkeypatch.setattr(client, "stream_response", replace_then_stream)
    model = ChatGPTPlan("gpt-5.6-luna")
    chunks, _ = run(model, make_prompt(model))
    assert chunks == ["Hello", " world"]
    assert api_capture["request"].headers["authorization"] == "Bearer stored-access"
    assert api_capture["body"]["model"] == "gpt-5.6-luna"
