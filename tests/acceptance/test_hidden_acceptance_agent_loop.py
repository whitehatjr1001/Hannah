"""Masked acceptance scenarios for agent-loop and sub-agent seam behavior."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Final

import pytest
from click.testing import CliRunner

import hannah.cli.agent_command as agent_command_module
import hannah.cli.app as app_module
from hannah.agent.context import RaceContext
from hannah.agent.loop import AgentLoop
from hannah.agent.tool_registry import normalize_tool_args
from hannah.agent.subagents import PredictAgent, RivalAgent, SimAgent, StrategyAgent, SubAgentResult, spawn_all
from hannah.agent.worker_runtime import SPAWN_TOOL_SPEC


@dataclass
class _FakeMemory:
    history: list[dict[str, str]] = field(default_factory=list)

    def get_recent(self, n: int = 10) -> list[dict[str, str]]:
        if n <= 0:
            return []
        return self.history[-n:]

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})


class _FakeToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "race_data",
                    "description": "Fetch deterministic race data.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pit_strategy",
                    "description": "Return deterministic pit strategy.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        if name == "pit_strategy":
            raise ValueError("pit_strategy exploded")
        return {"status": "ok", "tool": name, "args": args}


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        del temperature, max_tokens
        self.calls.append({"messages": messages, "tools": tools})
        if len(self.calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "hidden-call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "race_data",
                                        "arguments": '{"race":"bahrain","year":2025}',
                                    },
                                },
                                {
                                    "id": "hidden-call-2",
                                    "type": "function",
                                    "function": {
                                        "name": "pit_strategy",
                                        "arguments": '{"lap":',
                                    },
                                },
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Decision locked. Continue on current stint.",
                    }
                }
            ]
        }


class _AcceptanceRuntimeRegistry:
    def __init__(self, handlers: dict[str, Any] | None = None) -> None:
        self._handlers = handlers or {}

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "spawn",
                    "description": "Spawn a bounded worker.",
                    "parameters": dict(SPAWN_TOOL_SPEC["parameters"]),
                },
            }
        ]

    def with_runtime_tools(self, handlers: dict[str, Any]) -> "_AcceptanceRuntimeRegistry":
        return _AcceptanceRuntimeRegistry(handlers=dict(handlers))

    def normalize_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert name == "spawn"
        return normalize_tool_args(name, args, parameters=SPAWN_TOOL_SPEC["parameters"])

    async def call(self, name: str, args: dict[str, Any], *, state: Any = None) -> dict[str, Any]:
        handler = self._handlers[name]
        if "state" in inspect.signature(handler).parameters:
            result = handler(**args, state=state)
        else:
            result = handler(**args)
        if inspect.isawaitable(result):
            return await result
        return result


class _HiddenNestedSpawnProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        del temperature, max_tokens
        snapshot = json.loads(json.dumps(messages))
        self.calls.append({"messages": snapshot, "tools": tools})
        if len(self.calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "hidden-nested-spawn",
                                    "type": "function",
                                    "function": {
                                        "name": "spawn",
                                        "arguments": {
                                            "task": "delegate Bahrain strategy",
                                            "system_prompt": "You are a planner worker.",
                                            "allowed_tools": ["spawn"],
                                            "result_contract": {"summary": "string"},
                                        },
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Nested spawn was blocked and reported cleanly.",
                    }
                }
            ]
        }


def _tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "tool"]


def test_hidden_agent_loop_toolflow_and_failure_contract() -> None:
    fake_provider = _FakeProvider()
    fake_registry = _FakeToolRegistry()
    fake_memory = _FakeMemory()
    loop = AgentLoop(memory=fake_memory, registry=fake_registry, provider=fake_provider)

    asyncio.run(loop.run_command("strategy check for bahrain"))

    assert len(fake_provider.calls) == 2
    assert fake_provider.calls[0]["tools"] is not None
    assert [name for name, _ in fake_registry.calls] == ["race_data", "pit_strategy"]
    assert fake_registry.calls[0][1] == {"race": "bahrain", "year": 2025}
    assert fake_registry.calls[1][1] == {}

    second_pass_messages = fake_provider.calls[1]["messages"]
    tool_messages = _tool_messages(second_pass_messages)
    assert len(tool_messages) == 2
    success_payload = json.loads(tool_messages[0]["content"])
    assert success_payload["status"] == "ok"
    error_payload = json.loads(tool_messages[1]["content"])
    assert error_payload["status"] == "error"
    assert error_payload["tool"] == "pit_strategy"
    assert "exploded" in error_payload["error"]

    assert fake_memory.history[-2:] == [
        {"role": "user", "content": "strategy check for bahrain"},
        {"role": "assistant", "content": "Decision locked. Continue on current stint."},
    ]


def test_hidden_primary_agent_and_wrappers_preserve_runtime_path_boundaries(monkeypatch) -> None:
    runner = CliRunner()
    shared_runtime_calls: list[tuple[str | None, bool, str, bool, bool]] = []
    direct_runtime_calls: list[str] = []

    async def _fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
        persist_session: bool,
    ) -> str:
        shared_runtime_calls.append((message, interactive, session_id, new_session, persist_session))
        return "ok"

    class _FakeDirectLoop:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def run_command(self, command: str) -> None:
            direct_runtime_calls.append(command)

    monkeypatch.setattr(agent_command_module, "run_agent_command", _fake_run_agent_command)
    monkeypatch.setattr(app_module, "AgentLoop", _FakeDirectLoop, raising=False)

    agent_result = runner.invoke(app_module.cli, ["agent", "--message", "Should we undercut now?"])
    chat_result = runner.invoke(app_module.cli, ["chat", "--message", "Should we undercut now?"])
    ask_result = runner.invoke(app_module.cli, ["ask", "Should we undercut now?"])
    sandbox_result = runner.invoke(
        app_module.cli,
        ["sandbox", "--agents", "VER,NOR,LEC", "--race", "bahrain"],
    )

    assert agent_result.exit_code == 0
    assert chat_result.exit_code == 0
    assert ask_result.exit_code == 0
    assert sandbox_result.exit_code == 0
    assert shared_runtime_calls == [
        ("Should we undercut now?", False, "cli:direct", False, True),
        ("Should we undercut now?", False, "cli:direct", False, True),
        ("Should we undercut now?", False, "cli:direct", False, False),
    ]
    assert len(direct_runtime_calls) == 1
    assert "Run a full sandbox race at bahrain" in direct_runtime_calls[0]


def test_hidden_nested_spawn_policy_error_is_contained_and_reported() -> None:
    provider = _HiddenNestedSpawnProvider()
    memory = _FakeMemory()
    loop = AgentLoop(
        memory=memory,
        registry=_AcceptanceRuntimeRegistry(),
        provider=provider,
    )

    final_text = asyncio.run(loop.run_turn("delegate Bahrain strategy"))

    assert final_text == "Nested spawn was blocked and reported cleanly."
    assert len(provider.calls) == 2

    second_pass_messages = provider.calls[1]["messages"]
    spawn_tool_message = next(
        message
        for message in second_pass_messages
        if message.get("role") == "tool" and message.get("name") == "spawn"
    )
    payload = json.loads(spawn_tool_message["content"])

    assert payload["status"] == "error"
    assert payload["tool"] == "spawn"
    assert "nested spawn is not allowed in slice 1" in payload["error"]
    assert not any(
        message.get("role") == "system" and message.get("name") == "subagent_result"
        for message in second_pass_messages
    )
    assert memory.history[-2:] == [
        {"role": "user", "content": "delegate Bahrain strategy"},
        {"role": "assistant", "content": "Nested spawn was blocked and reported cleanly."},
    ]


@dataclass(frozen=True)
class HiddenLoopScenario:
    scenario_id: str
    drivers: list[str]
    expected_agents: tuple[str, ...]


LOOP_SCENARIOS: Final[tuple[HiddenLoopScenario, ...]] = (
    HiddenLoopScenario(
        "HACC_L01",
        ["VER", "NOR", "LEC"],
        ("sim_agent", "strategy_agent", "predict_agent"),
    ),
)


@pytest.mark.parametrize("scenario", LOOP_SCENARIOS, ids=[s.scenario_id for s in LOOP_SCENARIOS])
def test_hidden_spawn_all_failure_containment_contract(
    scenario: HiddenLoopScenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ok_sim(self, ctx: RaceContext) -> SubAgentResult:
        del self, ctx
        return SubAgentResult(agent="sim_agent", success=True, data={"signal": "sim"})

    async def _ok_strategy(self, ctx: RaceContext) -> SubAgentResult:
        del self, ctx
        return SubAgentResult(agent="strategy_agent", success=True, data={"signal": "strategy"})

    async def _ok_predict(self, ctx: RaceContext) -> SubAgentResult:
        del self, ctx
        return SubAgentResult(agent="predict_agent", success=True, data={"signal": "predict"})

    async def _rival_mixed(self, ctx: RaceContext) -> SubAgentResult:
        del ctx
        if self.driver_code == "NOR":
            raise RuntimeError("rival provider failure")
        return SubAgentResult(agent=self.name, success=False, error="rejected call")

    monkeypatch.setattr(SimAgent, "run", _ok_sim)
    monkeypatch.setattr(StrategyAgent, "run", _ok_strategy)
    monkeypatch.setattr(PredictAgent, "run", _ok_predict)
    monkeypatch.setattr(RivalAgent, "run", _rival_mixed)

    ctx = RaceContext(
        race="bahrain",
        year=2025,
        laps=57,
        weather="dry",
        drivers=scenario.drivers,
        race_data={"session_info": {"current_lap": 20}},
    )
    output = asyncio.run(spawn_all(ctx))

    assert set(output.keys()) == set(scenario.expected_agents)
    assert output["sim_agent"]["signal"] == "sim"
    assert output["strategy_agent"]["signal"] == "strategy"
    assert output["predict_agent"]["signal"] == "predict"
    assert "rival_nor" not in output
    assert "rival_lec" not in output
