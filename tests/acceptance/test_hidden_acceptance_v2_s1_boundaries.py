"""Masked acceptance scenarios for v2-s1 provider/tool boundary hardening."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Final

import pytest

from hannah.agent.loop import AgentLoop
from hannah.agent.tool_registry import ToolRegistry
from hannah.providers.local_fallback import LocalChoice, LocalCompletion, LocalFunction, LocalMessage, LocalToolCall


@dataclass
class _FakeMemory:
    history: list[dict[str, str]] = field(default_factory=list)

    def get_recent(self, n: int = 10) -> list[dict[str, str]]:
        if n <= 0:
            return []
        return self.history[-n:]

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})


@dataclass
class _ScriptedProvider:
    first_response: object
    final_text: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> object:
        del temperature, max_tokens
        snapshot = json.loads(json.dumps(messages))
        self.calls.append({"messages": snapshot, "tools": tools})
        if len(self.calls) == 1:
            return self.first_response
        return {"choices": [{"message": {"role": "assistant", "content": self.final_text}}]}


@dataclass(frozen=True)
class HiddenNormalizationScenario:
    scenario_id: str
    first_response_factory: Callable[[], object]


def _openai_payload_with_single_race_data_call() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "hacc-v2s1-openai-race-data",
                            "type": "function",
                            "function": {
                                "name": "race_data",
                                "arguments": {
                                    "race": "bahrain",
                                    "year": 2025,
                                    "session": "R",
                                    "driver": "VER",
                                    "lap": 17,
                                },
                            },
                        }
                    ],
                }
            }
        ]
    }


def _local_payload_with_single_race_data_call() -> LocalCompletion:
    arguments = json.dumps(
        {
            "race": "bahrain",
            "year": 2025,
            "session": "R",
            "driver": "VER",
            "lap": 17,
        }
    )
    return LocalCompletion(
        choices=[
            LocalChoice(
                message=LocalMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        LocalToolCall(
                            id="hacc-v2s1-local-race-data",
                            function=LocalFunction(name="race_data", arguments=arguments),
                        )
                    ],
                )
            )
        ]
    )


NORMALIZATION_SCENARIOS: Final[tuple[HiddenNormalizationScenario, ...]] = (
    HiddenNormalizationScenario("HACC_V2S1_N01", _openai_payload_with_single_race_data_call),
    HiddenNormalizationScenario("HACC_V2S1_N02", _local_payload_with_single_race_data_call),
)


@pytest.fixture(autouse=True)
def _stub_race_data_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    import hannah.tools.race_data.tool as race_data_tool

    def _fake_fetch_session(race: str, year: int, session_type: str) -> dict[str, Any]:
        return {
            "laps": [{"lap_number": 1, "driver": "VER", "lap_time": 90.0}],
            "weather": [{"rainfall": False, "air_temp": 30.0}],
            "car_data": [],
            "results": [{"position": 1, "driver": "VER"}],
            "source": f"{race}-{year}-{session_type}",
        }

    class _FakeOpenF1Client:
        def get_sessions(self, year: int, race_name: str) -> list[dict[str, Any]]:
            return [
                {"session_key": 9101, "meeting_name": race_name, "year": year},
                {"session_key": 9102, "meeting_name": race_name, "year": year},
            ]

    monkeypatch.setattr(race_data_tool, "fetch_session", _fake_fetch_session)
    monkeypatch.setattr(race_data_tool, "OpenF1Client", _FakeOpenF1Client)


def _tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "tool"]


def _tool_payload(messages: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for message in messages:
        if message.get("name") != name:
            continue
        return json.loads(message.get("content", "{}"))
    raise AssertionError(f"missing tool payload for {name}")


def test_hidden_v2s1_noisy_race_data_args_and_invalid_tool_error_serialization() -> None:
    provider = _ScriptedProvider(
        first_response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "hacc-v2s1-race-data",
                                "type": "function",
                                "function": {
                                    "name": "race_data",
                                    "arguments": {
                                        "race": "bahrain",
                                        "year": 2025,
                                        "session": "R",
                                        "driver": "VER",
                                        "lap": 21,
                                    },
                                },
                            },
                            {
                                "id": "hacc-v2s1-invalid-tool",
                                "type": "function",
                                "function": {"name": "ghost_tool", "arguments": {"race": "bahrain"}},
                            },
                        ],
                    }
                }
            ]
        },
        final_text="Boundary-safe final answer.",
    )
    memory = _FakeMemory()
    loop = AgentLoop(memory=memory, registry=ToolRegistry(), provider=provider)

    asyncio.run(loop.run_command("Need strategy context for Bahrain with noisy tool calls."))

    assert len(provider.calls) == 2
    second_pass_messages = provider.calls[1]["messages"]
    tool_messages = _tool_messages(second_pass_messages)
    assert {message["name"] for message in tool_messages} == {"race_data", "ghost_tool"}

    race_data_payload = _tool_payload(tool_messages, "race_data")
    assert race_data_payload["session_info"]["race"] == "bahrain"
    assert race_data_payload["session_info"]["year"] == 2025
    assert race_data_payload["drivers"] == ["VER"]

    error_payload = _tool_payload(tool_messages, "ghost_tool")
    assert error_payload["status"] == "error"
    assert error_payload["tool"] == "ghost_tool"
    assert "unknown tool" in error_payload["error"]

    assert memory.history[-2:] == [
        {"role": "user", "content": "Need strategy context for Bahrain with noisy tool calls."},
        {"role": "assistant", "content": "Boundary-safe final answer."},
    ]


@pytest.mark.parametrize(
    "scenario",
    NORMALIZATION_SCENARIOS,
    ids=[scenario.scenario_id for scenario in NORMALIZATION_SCENARIOS],
)
def test_hidden_v2s1_provider_payload_normalization_roundtrip(
    scenario: HiddenNormalizationScenario,
) -> None:
    provider = _ScriptedProvider(
        first_response=scenario.first_response_factory(),
        final_text="Normalization-safe final answer.",
    )
    memory = _FakeMemory()
    loop = AgentLoop(memory=memory, registry=ToolRegistry(), provider=provider)

    asyncio.run(loop.run_command("Fetch race data and normalize provider payloads."))

    assert len(provider.calls) == 2
    second_pass_messages = provider.calls[1]["messages"]
    assistant_tool_message = second_pass_messages[-2]
    tool_result_message = second_pass_messages[-1]

    assert assistant_tool_message["role"] == "assistant"
    assert assistant_tool_message["tool_calls"][0]["function"]["name"] == "race_data"
    assert isinstance(assistant_tool_message["tool_calls"][0]["function"]["arguments"], str)

    assert tool_result_message["role"] == "tool"
    assert tool_result_message["name"] == "race_data"
    race_data_payload = json.loads(tool_result_message["content"])
    assert set(race_data_payload.keys()) == {"laps", "stints", "weather", "drivers", "session_info"}
    assert race_data_payload["session_info"]["openf1_sessions"] == 2
    assert race_data_payload["drivers"] == ["VER"]

    assert memory.history[-2:] == [
        {"role": "user", "content": "Fetch race data and normalize provider payloads."},
        {"role": "assistant", "content": "Normalization-safe final answer."},
    ]
