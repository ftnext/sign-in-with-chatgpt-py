"""Convert llm's message chain into Responses API input items.

``prompt.messages`` is the single canonical entry point: for conversation
prompts llm pre-bakes the full ordered chain (restored or live history plus
the current turn) into it, so this module never merges sources itself.
"""

import llm
from llm.parts import ReasoningPart, TextPart

_CONTENT_TEXT_TYPES = {"text", "input_text", "output_text"}


def _part_text(part, role):
    """Text of one part, or None when it carries no content to preserve."""
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, ReasoningPart) and part.redacted and not part.text:
        # An opaque marker only: no reasoning text exists to resend.
        return None
    name = type(part).__name__
    raise llm.ModelError(
        f"chatgpt-plan cannot continue a conversation containing "
        f"{name}: only plain text turns are supported"
    )


def _dict_part_text(part, role):
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        raise llm.ModelError(
            "chatgpt-plan found an unsupported message part in the conversation history"
        )
    kind = part.get("type")
    if kind == "reasoning" and part.get("redacted") and not part.get("text"):
        return None
    if kind == "text":
        return part.get("text") or ""
    raise llm.ModelError(
        f"chatgpt-plan cannot continue a conversation containing a "
        f"{kind!r} part: only plain text turns are supported"
    )


def _content_text(item, role):
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise llm.ModelError(
            "chatgpt-plan found unsupported content in the conversation history"
        )
    kind = item.get("type")
    if kind in _CONTENT_TEXT_TYPES:
        return item.get("text") or ""
    raise llm.ModelError(
        f"chatgpt-plan cannot continue a conversation containing "
        f"{kind!r} content: only plain text turns are supported"
    )


def _message_texts(message):
    """(role, [texts]) for Message objects and dict-shaped messages."""
    if isinstance(message, dict):
        role = message.get("role")
        parts = message.get("parts")
        if parts is not None:
            return role, [_dict_part_text(part, role) for part in parts]
        content = message.get("content")
        if isinstance(content, str):
            return role, [content]
        if isinstance(content, list):
            return role, [_content_text(item, role) for item in content]
        if content is None:
            return role, []
        raise llm.ModelError(
            "chatgpt-plan found unsupported content in the conversation history"
        )
    role = getattr(message, "role", None)
    parts = getattr(message, "parts", None)
    if parts is None:
        raise llm.ModelError(
            "chatgpt-plan found an unsupported message in the conversation history"
        )
    return role, [_part_text(part, role) for part in parts]


def responses_input(prompt):
    """Return ``(input_items, instructions)`` for the Responses API.

    System and developer text becomes ``instructions``; no explicit
    system-role input item is sent. User messages keep the plain-string
    content form; assistant turns are replayed as completed messages with
    ``output_text`` parts, matching the reference implementation.
    """
    instruction_parts = []
    items = []
    for message in getattr(prompt, "messages", None) or []:
        role, texts = _message_texts(message)
        text = "".join(t for t in texts if t)
        if role in ("system", "developer"):
            if text:
                instruction_parts.append(text)
        elif role == "user":
            if text:
                items.append({"role": "user", "content": text})
        elif role == "assistant":
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "id": f"msg_{len(items)}",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                            }
                        ],
                    }
                )
        else:
            raise llm.ModelError(
                f"chatgpt-plan does not support {role!r} messages in the "
                "conversation history"
            )
    if not items:
        raise llm.ModelError("Nothing to send: the request has no user input")
    if getattr(prompt, "_explicit_messages", None) is not None:
        # llm.Prompt(messages=..., system=...) leaves ``system`` outside
        # the authoritative chain; conversation.prompt() bakes it into
        # the first message instead, so skip what is already there.
        system = prompt.system
        if system and system not in instruction_parts:
            instruction_parts.insert(0, system)
    instructions = "\n\n".join(instruction_parts) or None
    return items, instructions
