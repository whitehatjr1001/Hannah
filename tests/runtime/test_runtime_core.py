"""RuntimeCore event and tool-roundtrip tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from hannah.runtime.core import RuntimeCore


@dataclass
class _RecordingBus:
    event_types: list[str] = field(default_factory=list)

    async def publish(self, envelope: Any) -> None:
        self.event_types.append(envelope.event_type)


@dataclass
class _StubRegistry:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "race_data",
                    "description": "fetch race data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        return {"tool": name, "args": args}


class _StubProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: list[dict[str, Any]] = []

    def queue_text(self, text: str) -> None:
        self._responses.append(
            {"choices": [{"message": {"role": "assistant", "content": text}}]}
        )

    def queue_tool_then_text(
        self,
        name: str,
        arguments: dict[str, Any],
        final_text: str,
    ) -> None:
        self._responses.append(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": name, "arguments": arguments},
                                }
                            ],
                        }
                    }
                ]
            }
        )
        self.queue_text(final_text)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        snapshot = json.loads(json.dumps(messages))
        self.calls.append(
            {
                "messages": snapshot,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self._responses.pop(0)


@pytest.mark.anyio
async def test_runtime_core_emits_events_for_plain_final_answer() -> None:
    provider = _StubProvider()
    provider.queue_text("box this lap")
    registry = _StubRegistry()
    bus = _RecordingBus()
    core = RuntimeCore(provider=provider, registry=registry, event_bus=bus)

    reply = await core.run_turn(messages=[{"role": "user", "content": "call strategy"}])

    assert reply["content"] == "box this lap"
    assert bus.event_types == [
        "user_message_received",
        "provider_request_started",
        "provider_response_received",
        "final_answer_emitted",
    ]


@pytest.mark.anyio
async def test_runtime_core_emits_tool_events_and_reuses_tool_registry() -> None:
    provider = _StubProvider()
    provider.queue_tool_then_text("race_data", {"race": "bahrain"}, "strategy locked")
    registry = _StubRegistry()
    bus = _RecordingBus()
    core = RuntimeCore(provider=provider, registry=registry, event_bus=bus)

    reply = await core.run_turn(messages=[{"role": "user", "content": "get bahrain data"}])

    assert reply["content"] == "strategy locked"
    assert registry.calls == [("race_data", {"race": "bahrain"})]
    assert bus.event_types == [
        "user_message_received",
        "provider_request_started",
        "provider_response_received",
        "tool_call_started",
        "tool_call_finished",
        "provider_request_started",
        "provider_response_received",
        "final_answer_emitted",
    ]

    second_messages = provider.calls[1]["messages"]
    assistant_tool_message = second_messages[-2]
    tool_result_message = second_messages[-1]

    assert assistant_tool_message["role"] == "assistant"
    assert assistant_tool_message["tool_calls"][0]["function"]["name"] == "race_data"
    assert tool_result_message["role"] == "tool"
    assert tool_result_message["name"] == "race_data"
    assert json.loads(tool_result_message["content"]) == {
        "tool": "race_data",
        "args": {"race": "bahrain"},
    }


@pytest.mark.anyio
async def test_runtime_core_reinvokes_provider_with_retry_guidance_once() -> None:
    provider = _StubProvider()
    provider.queue_text("I can analyze that. Let me know if you'd like me to proceed.")
    provider.queue_text("strategy locked")
    core = RuntimeCore(provider=provider, registry=_StubRegistry(), event_bus=_RecordingBus())

    reply = await core.run_turn(
        messages=[{"role": "user", "content": "analyze bahrain"}],
        should_retry=lambda final_text, retry_used: (
            (not retry_used) and "let me know if you'd like" in final_text.lower()
        ),
        retry_guidance="Do the analysis now.",
    )

    assert reply["content"] == "strategy locked"
    assert len(provider.calls) == 2
    second_messages = provider.calls[1]["messages"]
    assert second_messages[-2] == {
        "role": "assistant",
        "content": "I can analyze that. Let me know if you'd like me to proceed.",
    }
    assert second_messages[-1] == {"role": "system", "content": "Do the analysis now."}
