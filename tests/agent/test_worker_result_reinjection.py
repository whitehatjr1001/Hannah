"""Spawn-result reinjection tests for the shared runtime core."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from hannah.agent.tool_registry import ToolRegistry
from hannah.runtime.core import RuntimeCore


@dataclass
class _RecordingBus:
    event_types: list[str] = field(default_factory=list)

    async def publish(self, envelope: Any) -> None:
        self.event_types.append(envelope.event_type)


class _StubProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: list[dict[str, Any]] = []

    def queue_tool_then_text(
        self,
        name: str,
        arguments: dict[str, Any],
        worker_text: str,
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
        self._responses.append(
            {"choices": [{"message": {"role": "assistant", "content": worker_text}}]}
        )
        self._responses.append(
            {"choices": [{"message": {"role": "assistant", "content": final_text}}]}
        )

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
                "tools": tools or [],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self._responses.pop(0)


@pytest.mark.anyio
async def test_spawn_tool_result_is_reinjected_with_stable_message_shape() -> None:
    provider = _StubProvider()
    provider.queue_tool_then_text(
        "spawn",
        {
            "task": "compare strategies",
            "system_prompt": "You are a strategy worker.",
            "allowed_tools": ["race_data", "race_sim"],
            "result_contract": {"summary": "string"},
        },
        '{"summary":"Prefer the two-stop."}',
        "Use the two-stop.",
    )
    runtime_core = RuntimeCore(
        provider=provider,
        registry=ToolRegistry(),
        event_bus=_RecordingBus(),
    )

    reply = await runtime_core.run_turn(
        messages=[{"role": "user", "content": "compare strategies"}]
    )

    assert reply["content"] == "Use the two-stop."
    second_main_messages = provider.calls[2]["messages"]
    subagent_messages = [
        message
        for message in second_main_messages
        if message["role"] == "system" and message.get("name") == "subagent_result"
    ]
    assert len(subagent_messages) == 1
    assert subagent_messages[0]["worker_id"].startswith("worker-")
    assert json.loads(subagent_messages[0]["content"]) == {
        "worker_id": subagent_messages[0]["worker_id"],
        "status": "completed",
        "result": {"summary": "Prefer the two-stop."},
    }
