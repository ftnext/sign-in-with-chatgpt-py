"""Request and public response contracts for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BODY_BYTES = 1 * 1024 * 1024
MAX_TOTAL_TEXT_CHARS = 200_000
MAX_MESSAGES = 200

ALLOWED_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


class InputMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Reasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: str


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    input: list[InputMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    instructions: str | None = None
    reasoning: Reasoning | None = None
    stream: Literal[True] | None = None
    store: Literal[False] | None = None

    @model_validator(mode="after")
    def _check_limits(self):
        total = sum(len(message.content) for message in self.input)
        if self.instructions:
            total += len(self.instructions)
        if total > MAX_TOTAL_TEXT_CHARS:
            raise ValueError(
                f"Total text exceeds {MAX_TOTAL_TEXT_CHARS} characters"
            )
        if self.reasoning and self.reasoning.effort not in ALLOWED_EFFORTS:
            raise ValueError(
                "Unsupported reasoning effort: " + str(self.reasoning.effort)
            )
        return self

    def upstream_payload(self) -> dict:
        payload = {
            "model": self.model,
            "input": [
                {"role": message.role, "content": message.content}
                for message in self.input
            ],
            "store": False,
            "stream": True,
        }
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.reasoning is not None:
            payload["reasoning"] = {"effort": self.reasoning.effort}
        return payload
