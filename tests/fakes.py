"""A stand-in LLM client for tests: scripted replies, no network."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from claimssql.llm.base import LLMResponse, Message

GOOD = '{"answerable": true, "sql": "SELECT 1", "confidence": 80, "reason": "ok"}'
REFUSAL = '{"answerable": false, "sql": null, "confidence": 90, "reason": "not in the data"}'


class FakeClient:
    provider = "fake"
    model = "fake-1"

    def __init__(self, replies: list[str] | Callable[[dict], str] | None = None) -> None:
        self.options: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self._replies = replies if replies is not None else (lambda call: GOOD)

    def chat(
        self,
        messages: list[Message],
        temperature: float,
        seed: int,
        json_mode: bool = True,
    ) -> LLMResponse:
        call = {"messages": messages, "temperature": temperature, "seed": seed,
                "json_mode": json_mode}  # fmt: skip
        self.calls.append(call)
        if callable(self._replies):
            text = self._replies(call)
        else:
            text = self._replies[len(self.calls) - 1]
        return LLMResponse(text=text, latency_ms=100.0, prompt_tokens=50, completion_tokens=10)
