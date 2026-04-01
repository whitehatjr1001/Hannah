"""Compatibility tests for the retired runtime-core ownership surface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from hannah.runtime import MainAgentContext, RuntimeContextBuilder, RuntimeCore


@dataclass
class _RecordingProvider:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "RuntimeCore still works through the compatibility shim.",
                    }
                }
            ]
        }


@dataclass
class _RecordingRegistry:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return []

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        return {"tool": name, "args": args}


def test_runtime_package_keeps_compatibility_exports() -> None:
    builder = RuntimeContextBuilder()
    context = MainAgentContext(persona="system", user_input="hello")
    messages = builder.build_main_turn(context)

    assert RuntimeCore.__module__ == "hannah.runtime.core"
    assert messages[-1] == {"role": "user", "content": "hello"}
    assert len(messages) == 6
    assert "Identity/Runtime block:" in messages[0]["content"]
    assert "Bootstrap docs block:" in messages[1]["content"]
    assert "Memory context block:" in messages[2]["content"]
    assert "Skills summary hook block:" in messages[3]["content"]
    assert "Hannah F1 persona block:" in messages[4]["content"]


@pytest.mark.anyio
async def test_runtime_core_warns_but_still_runs_turns() -> None:
    provider = _RecordingProvider()
    registry = _RecordingRegistry()

    with pytest.warns(DeprecationWarning, match="compatibility shim"):
        core = RuntimeCore(provider=provider, registry=registry)

    assert core.registry is registry

    reply = await core.run_turn(messages=[{"role": "user", "content": "hello"}])

    assert reply["content"] == "RuntimeCore still works through the compatibility shim."
    assert provider.calls == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": None,
            "temperature": 0.2,
            "max_tokens": 2048,
        }
    ]
