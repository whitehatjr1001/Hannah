"""Focused seam tests for AgentLoop tool orchestration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import hannah.agent.loop as agent_loop
from hannah.agent.loop import AgentLoop


@dataclass
class _StubMemory:
    recent: list[dict[str, str]] = field(default_factory=list)
    added: list[tuple[str, str]] = field(default_factory=list)

    def get_recent(self, n: int) -> list[dict[str, str]]:
        return self.recent[-n:]

    def add(self, role: str, content: str) -> None:
        self.added.append((role, content))


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


class _RoundTripProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        if len(self.calls) == 1:
            return {
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
                                        "name": "race_data",
                                        "arguments": {"race": "bahrain", "year": 2025},
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "Final recommendation."}}]}


def _make_tool_call(name: str, arguments: Any, call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_run_command_completes_tool_roundtrip_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _StubMemory()
    registry = _StubRegistry()
    provider = _RoundTripProvider()
    loop = AgentLoop(memory=memory, registry=registry, provider=provider)

    monkeypatch.setattr(agent_loop.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_loop, "make_hannah_panel", lambda text: text)

    user_input = "Run simulation for Bahrain 2025."
    asyncio.run(loop.run_command(user_input))

    assert registry.calls == [("race_data", {"race": "bahrain", "year": 2025})]
    assert memory.added == [("user", user_input), ("assistant", "Final recommendation.")]
    assert len(provider.calls) == 2

    second_messages = provider.calls[1]["messages"]
    assistant_tool_msg = second_messages[-2]
    tool_result_msg = second_messages[-1]
    assert assistant_tool_msg["role"] == "assistant"
    assert assistant_tool_msg["tool_calls"][0]["function"]["name"] == "race_data"
    assert tool_result_msg["role"] == "tool"
    assert tool_result_msg["name"] == "race_data"
    assert json.loads(tool_result_msg["content"]) == {
        "tool": "race_data",
        "args": {"race": "bahrain", "year": 2025},
    }


def test_run_turn_delegates_to_runtime_core_with_built_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _StubMemory(recent=[{"role": "assistant", "content": "Previous context."}])
    loop = AgentLoop(memory=memory, registry=_StubRegistry(), provider=_RoundTripProvider())
    captured: dict[str, Any] = {}

    async def _fake_runtime_run_turn(
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        turn_tools: list[dict[str, Any]] | None,
        should_retry,
        retry_guidance: str | None,
        execute_tool_calls,
    ) -> dict[str, str]:
        captured["messages"] = messages
        captured["session_id"] = session_id
        captured["turn_tools"] = turn_tools
        captured["retry_guidance"] = retry_guidance
        captured["execute_tool_calls"] = execute_tool_calls
        captured["retry_permission_deferral"] = should_retry(
            "Let me know if you'd like me to proceed with that analysis!",
            False,
        )
        captured["retry_on_final_answer"] = should_retry("strategy locked", False)
        return {"role": "assistant", "content": "Adapter reply"}

    monkeypatch.setattr(loop.runtime, "run_turn", _fake_runtime_run_turn)

    user_input = "predict the race strategy for the upcoming japanese grand prix"
    result = asyncio.run(loop.run_turn(user_input))

    assert result == "Adapter reply"
    assert captured["session_id"] == "default"
    assert captured["messages"][-1] == {"role": "user", "content": user_input}
    assert captured["messages"][1]["content"].startswith(
        "This turn is a race analysis or prediction request"
    )
    assert captured["messages"][2] == {"role": "assistant", "content": "Previous context."}
    assert {tool["function"]["name"] for tool in captured["turn_tools"]} == {"race_data"}
    assert callable(captured["execute_tool_calls"])
    assert captured["retry_permission_deferral"] is True
    assert captured["retry_on_final_answer"] is False
    assert captured["retry_guidance"] == (
        "The user already asked you to do the analysis. "
        "Do not ask for permission or defer. "
        "Call the relevant tools now and answer decisively."
    )
    assert memory.added == [("user", user_input), ("assistant", "Adapter reply")]


def test_run_turn_uses_adapter_execute_tool_calls_hook_in_real_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _StubMemory()
    registry = _StubRegistry()
    provider = _RoundTripProvider()
    loop = AgentLoop(memory=memory, registry=registry, provider=provider)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(agent_loop.console, "print", lambda *args, **kwargs: None)

    async def _fake_execute_tool_calls(
        tool_calls: list[SimpleNamespace],
        *,
        state: Any = None,
    ) -> list[dict[str, str]]:
        captured["tool_names"] = [tool_call.function.name for tool_call in tool_calls]
        captured["state"] = state
        return [
            {
                "role": "tool",
                "tool_call_id": tool_calls[0].id,
                "name": tool_calls[0].function.name,
                "content": json.dumps({"hook": "adapter_execute"}),
            }
        ]

    monkeypatch.setattr(loop, "_execute_tool_calls", _fake_execute_tool_calls)

    result = asyncio.run(loop.run_turn("Run simulation for Bahrain 2025."))

    assert result == "Final recommendation."
    assert captured["tool_names"] == ["race_data"]
    assert captured["state"] is not None
    second_messages = provider.calls[1]["messages"]
    assert json.loads(second_messages[-1]["content"]) == {"hook": "adapter_execute"}


@pytest.mark.parametrize(
    ("raw_arguments", "expected"),
    [
        ({"lap": 18, "driver": "VER"}, {"lap": 18, "driver": "VER"}),
        ('{"lap": 30, "driver": "NOR"}', {"lap": 30, "driver": "NOR"}),
        ("not-json", {}),
    ],
)
def test_call_tool_coerces_arguments_for_dict_json_and_invalid(
    raw_arguments: Any,
    expected: dict[str, Any],
) -> None:
    registry = _StubRegistry()
    loop = AgentLoop(memory=_StubMemory(), registry=registry, provider=_RoundTripProvider())

    tool_call = _make_tool_call("race_data", raw_arguments, call_id="arg-coerce")
    result = asyncio.run(loop._call_tool(tool_call))

    assert registry.calls[-1] == ("race_data", expected)
    assert result == {"tool": "race_data", "args": expected}


def test_execute_tool_calls_serializes_errors_into_tool_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop(memory=_StubMemory(), registry=_StubRegistry(), provider=_RoundTripProvider())
    monkeypatch.setattr(agent_loop.console, "print", lambda *args, **kwargs: None)

    async def _fake_call_tool(tool_call: SimpleNamespace) -> dict[str, Any]:
        if tool_call.function.name == "race_data":
            return {"status": "ok", "payload": {"race": "bahrain"}}
        raise RuntimeError("tool exploded")

    monkeypatch.setattr(loop, "_call_tool", _fake_call_tool)
    tool_calls = [
        _make_tool_call("race_data", {"race": "bahrain"}, "ok-1"),
        _make_tool_call("pit_strategy", {"race": "bahrain"}, "err-1"),
    ]

    messages = asyncio.run(loop._execute_tool_calls(tool_calls))

    assert len(messages) == 2
    assert messages[0]["role"] == "tool"
    assert json.loads(messages[0]["content"]) == {"status": "ok", "payload": {"race": "bahrain"}}
    assert messages[1]["role"] == "tool"
    error_payload = json.loads(messages[1]["content"])
    assert error_payload["status"] == "error"
    assert error_payload["tool"] == "pit_strategy"
    assert "tool exploded" in error_payload["error"]


def test_execute_tool_calls_serializes_non_json_python_values(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop(memory=_StubMemory(), registry=_StubRegistry(), provider=_RoundTripProvider())
    monkeypatch.setattr(agent_loop.console, "print", lambda *args, **kwargs: None)

    async def _fake_call_tool(tool_call: SimpleNamespace) -> dict[str, Any]:
        del tool_call
        return {"status": "ok", "delta": timedelta(seconds=95)}

    monkeypatch.setattr(loop, "_call_tool", _fake_call_tool)
    tool_calls = [_make_tool_call("race_data", {"race": "bahrain"}, "ok-serial")]

    messages = asyncio.run(loop._execute_tool_calls(tool_calls))

    assert len(messages) == 1
    assert json.loads(messages[0]["content"]) == {"status": "ok", "delta": "0:01:35"}


def test_execute_tool_calls_compacts_large_race_data_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop(memory=_StubMemory(), registry=_StubRegistry(), provider=_RoundTripProvider())
    monkeypatch.setattr(agent_loop.console, "print", lambda *args, **kwargs: None)

    async def _fake_call_tool(tool_call: SimpleNamespace) -> dict[str, Any]:
        del tool_call
        laps = [
            {"lap_number": lap, "driver": "VER", "lap_time": 90.0 + (lap * 0.01)}
            for lap in range(1, 2501)
        ]
        return {
            "laps": laps,
            "stints": [],
            "weather": [{"rainfall": 0.0, "air_temp": 30.0}],
            "drivers": ["VER"],
            "session_info": {"race": "bahrain", "year": 2025, "session": "R"},
        }

    monkeypatch.setattr(loop, "_call_tool", _fake_call_tool)
    tool_calls = [_make_tool_call("race_data", {"race": "bahrain"}, "ok-compact")]

    messages = asyncio.run(loop._execute_tool_calls(tool_calls))

    assert len(messages) == 1
    payload = json.loads(messages[0]["content"])
    assert payload["session_info"] == {"race": "bahrain", "year": 2025, "session": "R"}
    assert payload["drivers"] == ["VER"]
    assert payload["available_telemetry"] == ["laps", "weather"]
    assert payload["telemetry_counts"] == {"laps": 2500, "stints": 0, "weather": 1}
    assert payload["raw_payload_chars"] > 20_000
    assert "laps" not in payload
