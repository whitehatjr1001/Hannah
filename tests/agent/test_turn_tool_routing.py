"""Turn-level tool routing tests for ambiguous strategy prompts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

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


class _SnapshotProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools or [],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class _DeferredThenFinalProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools or [],
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
                            "content": (
                                "I can analyze the race strategy for the upcoming Japanese Grand Prix. "
                                "Please let me know if you'd like me to proceed with that analysis!"
                            ),
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "analysis complete"}}]}


def test_strategy_analysis_turn_hides_train_model_and_injects_guidance() -> None:
    provider = _SnapshotProvider()
    loop = AgentLoop(memory=_StubMemory(), provider=provider)

    asyncio.run(
        loop.run_turn(
            "can model the 2026 make ai models to predict the race startagy for the upcoming japanese grand prix??"
        )
    )

    first_call = provider.calls[0]
    tool_names = {tool["function"]["name"] for tool in first_call["tools"]}
    system_messages = [message["content"] for message in first_call["messages"] if message["role"] == "system"]

    assert "train_model" not in tool_names
    assert any("Do not call train_model" in message for message in system_messages)


def test_explicit_training_turn_keeps_train_model_available() -> None:
    provider = _SnapshotProvider()
    loop = AgentLoop(memory=_StubMemory(), provider=provider)

    asyncio.run(loop.run_turn("train the strategy model for japan using 2024 and 2025 data"))

    first_call = provider.calls[0]
    tool_names = {tool["function"]["name"] for tool in first_call["tools"]}
    system_messages = [message["content"] for message in first_call["messages"] if message["role"] == "system"]

    assert "train_model" in tool_names
    assert all("Do not call train_model" not in message for message in system_messages)


def test_strategy_analysis_turn_retries_after_permission_seeking_response() -> None:
    provider = _DeferredThenFinalProvider()
    loop = AgentLoop(memory=_StubMemory(), provider=provider)

    result = asyncio.run(
        loop.run_turn(
            "can model the 2026 make ai models to predict the race startagy for the up caomming japanese grand prix??"
        )
    )

    second_call = provider.calls[1]
    system_messages = [message["content"] for message in second_call["messages"] if message["role"] == "system"]

    assert result == "analysis complete"
    assert len(provider.calls) == 2
    assert any("user already asked you to do the analysis" in message.lower() for message in system_messages)
