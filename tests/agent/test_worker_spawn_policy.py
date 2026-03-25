"""Worker spawn policy and runtime-bound spawn roundtrip tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from hannah.agent.tool_registry import ToolRegistry
from hannah.agent.worker_runtime import WorkerPolicyError, WorkerRuntime, WorkerSpec, validate_worker_spec
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
                "tools": tools or [],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self._responses.pop(0)


def test_worker_spec_requires_allowed_tools() -> None:
    spec = WorkerSpec(
        worker_id="telemetry-1",
        task="fetch race data",
        system_prompt="You are a telemetry worker.",
        allowed_tools=[],
        result_contract={"summary": "string"},
    )

    with pytest.raises(WorkerPolicyError, match="allowed_tools"):
        validate_worker_spec(spec)


def test_nested_spawn_is_rejected() -> None:
    spec = WorkerSpec(
        worker_id="planner-1",
        task="spawn another worker",
        system_prompt="You are a planner worker.",
        allowed_tools=["spawn"],
        result_contract={"summary": "string"},
    )

    with pytest.raises(WorkerPolicyError, match="nested spawn"):
        validate_worker_spec(spec)


@pytest.mark.anyio
async def test_run_worker_rejects_unknown_allowed_tools() -> None:
    provider = _StubProvider()
    runtime = WorkerRuntime(
        provider=provider,
        registry=ToolRegistry(),
        event_bus=_RecordingBus(),
    )
    spec = WorkerSpec(
        worker_id="worker-unknown",
        task="analyze Bahrain strategy",
        system_prompt="You are a strategy worker.",
        allowed_tools=["race_data", "ghost_tool"],
        result_contract={"summary": "string"},
    )

    with pytest.raises(WorkerPolicyError, match="ghost_tool"):
        await runtime.run_worker(spec, parent_session_id="session-1")


@pytest.mark.anyio
async def test_run_worker_returns_error_result_on_contract_violation() -> None:
    provider = _StubProvider()
    provider.queue_text('{"summary": 42}')
    runtime = WorkerRuntime(
        provider=provider,
        registry=ToolRegistry(),
        event_bus=_RecordingBus(),
    )
    spec = WorkerSpec(
        worker_id="worker-contract",
        task="summarize Bahrain strategy",
        system_prompt="You are a strategy worker.",
        allowed_tools=["race_data"],
        result_contract={"summary": "string"},
    )

    result = await runtime.run_worker(spec, parent_session_id="session-1")

    assert result["status"] == "error"
    assert "summary" in result["error"]
    assert result["result"] == {"summary": 42}


@pytest.mark.anyio
async def test_main_runtime_executes_spawn_tool_via_normal_tool_roundtrip() -> None:
    provider = _StubProvider()
    provider._responses.append(
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
                                "function": {
                                    "name": "spawn",
                                    "arguments": {
                                        "task": "compare Bahrain one-stop versus two-stop",
                                        "system_prompt": "You are a strategy worker.",
                                        "allowed_tools": ["race_data", "race_sim", "pit_strategy"],
                                        "result_contract": {"summary": "string", "evidence": "list"},
                                    },
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    provider.queue_text('{"summary":"two-stop is stronger","evidence":["sim delta","traffic window"]}')
    provider.queue_text("Use the two-stop window.")
    bus = _RecordingBus()
    core = RuntimeCore(provider=provider, registry=ToolRegistry(), event_bus=bus)

    reply = await core.run_turn(messages=[{"role": "user", "content": "compare Bahrain strategies"}])

    assert reply["content"] == "Use the two-stop window."
    assert len(provider.calls) == 3
    main_tool_names = {tool["function"]["name"] for tool in provider.calls[0]["tools"]}
    worker_tool_names = {tool["function"]["name"] for tool in provider.calls[1]["tools"]}
    assert "spawn" in main_tool_names
    assert "spawn" not in worker_tool_names
    assert "subagent_spawned" in bus.event_types
    assert "subagent_completed" in bus.event_types

    second_main_messages = provider.calls[2]["messages"]
    spawn_result_message = second_main_messages[-1]
    assert spawn_result_message["role"] == "tool"
    assert spawn_result_message["name"] == "spawn"
    payload = json.loads(spawn_result_message["content"])
    assert payload["status"] == "completed"
    assert payload["result"] == {
        "summary": "two-stop is stronger",
        "evidence": ["sim delta", "traffic window"],
    }
