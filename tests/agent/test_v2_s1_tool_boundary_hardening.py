"""V2-S1 tests for tool-boundary and dict-style payload hardening."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import hannah.agent.loop as agent_loop
from hannah.agent.loop import AgentLoop


@dataclass
class _StubMemory:
    history: list[dict[str, str]] = field(default_factory=list)

    def get_recent(self, n: int = 10) -> list[dict[str, str]]:
        if n <= 0:
            return []
        return self.history[-n:]

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})


class _SchemaAwareRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "race_data",
                    "description": "Fetch race data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "race": {"type": "string"},
                            "year": {"type": "integer"},
                            "session": {"type": "string"},
                            "driver": {"type": "string"},
                        },
                        "required": ["race"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pit_strategy",
                    "description": "Return pit strategy.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "race": {"type": "string"},
                            "driver": {"type": "string"},
                            "lap": {"type": "integer"},
                        },
                        "required": ["race", "driver"],
                    },
                },
            },
        ]

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        return {"status": "ok", "tool": name, "args": args}


class _DictPayloadProvider:
    def __init__(self, first_content: Any) -> None:
        self.first_content = first_content
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
                            "content": self.first_content,
                            "tool_calls": [
                                {
                                    "id": "dict-call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "race_data",
                                        "arguments": {"race": "bahrain", "year": 2025},
                                    },
                                },
                                {
                                    "id": "dict-call-2",
                                    "type": "function",
                                    "function": {
                                        "name": "pit_strategy",
                                        "arguments": '{"race":"bahrain","driver":"VER","lap":18}',
                                    },
                                },
                            ],
                        }
                    }
                ]
            }

        return {"choices": [{"message": {"role": "assistant", "content": "Tool pass complete."}}]}


def _make_tool_call(name: str, arguments: Any, call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_call_tool_ignores_unexpected_hosted_args_and_preserves_valid_args() -> None:
    registry = _SchemaAwareRegistry()
    loop = AgentLoop(memory=_StubMemory(), registry=registry, provider=object())

    result = asyncio.run(
        loop._call_tool(
            _make_tool_call(
                name="race_data",
                arguments={"race": "bahrain", "year": 2025, "lap": 18},
                call_id="unexpected-arg",
            )
        )
    )

    assert registry.calls == [("race_data", {"race": "bahrain", "year": 2025})]
    assert result == {
        "status": "ok",
        "tool": "race_data",
        "args": {"race": "bahrain", "year": 2025},
    }


def test_call_tool_raises_clear_error_when_required_args_missing_after_normalization() -> None:
    registry = _SchemaAwareRegistry()
    loop = AgentLoop(memory=_StubMemory(), registry=registry, provider=object())

    with pytest.raises(ValueError, match=r"race_data.*race"):
        asyncio.run(
            loop._call_tool(
                _make_tool_call(
                    name="race_data",
                    arguments={"lap": 18},
                    call_id="missing-required",
                )
            )
        )

    assert registry.calls == []


@pytest.mark.parametrize(
    "assistant_content",
    [
        None,
        [{"type": "text", "text": "checking tools"}],
    ],
)
def test_run_command_normalizes_dict_style_tool_payloads_for_content_and_arguments(
    assistant_content: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DictPayloadProvider(first_content=assistant_content)
    registry = _SchemaAwareRegistry()
    memory = _StubMemory()
    loop = AgentLoop(memory=memory, registry=registry, provider=provider)

    monkeypatch.setattr(agent_loop.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_loop, "make_hannah_panel", lambda text: text)

    asyncio.run(loop.run_command("strategy check for bahrain"))

    assert registry.calls == [
        ("race_data", {"race": "bahrain", "year": 2025}),
        ("pit_strategy", {"race": "bahrain", "driver": "VER", "lap": 18}),
    ]

    second_pass_messages = provider.calls[1]["messages"]
    assistant_tool_msg = next(
        message
        for message in second_pass_messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_tool_msg["content"] == ""
    assert all(
        isinstance(tool_call["function"]["arguments"], str)
        for tool_call in assistant_tool_msg["tool_calls"]
    )

