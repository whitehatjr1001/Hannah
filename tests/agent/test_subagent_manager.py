"""Background worker and bus reporting tests for subagents."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from hannah.agent.context import RaceContext
from hannah.agent.subagent_manager import SubagentManager
from hannah.agent.subagents import SubAgentResult


@dataclass
class _RecordingBus:
    release: asyncio.Event
    expected_spawn_count: int
    envelopes: list[Any] = field(default_factory=list)

    async def publish(self, envelope: Any) -> None:
        self.envelopes.append(envelope)
        spawned = [item for item in self.envelopes if item.event_type == "subagent_spawned"]
        if len(spawned) >= self.expected_spawn_count:
            self.release.set()


class _BlockingAgent:
    def __init__(self, name: str, payload: dict[str, Any], release: asyncio.Event) -> None:
        self.name = name
        self._payload = payload
        self._release = release

    async def run(self, ctx: RaceContext) -> SubAgentResult:
        del ctx
        await self._release.wait()
        return SubAgentResult(agent=self.name, success=True, data=self._payload)


@pytest.mark.anyio
async def test_background_workers_announce_completion_through_bus() -> None:
    release = asyncio.Event()
    bus = _RecordingBus(release=release, expected_spawn_count=2)
    manager = SubagentManager(event_bus=bus, session_id="session-42")
    started: list[str] = []

    class _TrackedBlockingAgent(_BlockingAgent):
        async def run(self, ctx: RaceContext) -> SubAgentResult:
            started.append(f"{self.name}:start")
            result = await super().run(ctx)
            started.append(f"{self.name}:end")
            return result

    ctx = RaceContext(
        race="bahrain",
        year=2025,
        laps=57,
        weather="dry",
        drivers=["VER", "NOR"],
        race_data={"session_info": {"current_lap": 20}},
    )
    workers = [
        _TrackedBlockingAgent("sim_agent", {"signal": "sim"}, release),
        _TrackedBlockingAgent("strategy_agent", {"signal": "strategy"}, release),
    ]

    output = await asyncio.wait_for(manager.run_all(ctx, workers), timeout=1.0)

    assert output == {
        "sim_agent": {"signal": "sim"},
        "strategy_agent": {"signal": "strategy"},
    }
    assert started.count("sim_agent:start") == 1
    assert started.count("strategy_agent:start") == 1
    assert started.index("sim_agent:start") < started.index("sim_agent:end")
    assert started.index("strategy_agent:start") < started.index("strategy_agent:end")
    assert [envelope.event_type for envelope in bus.envelopes].count("subagent_spawned") == 2
    assert [envelope.event_type for envelope in bus.envelopes].count("subagent_progress") == 4
    assert [envelope.event_type for envelope in bus.envelopes].count("subagent_completed") == 2
    completed = [envelope for envelope in bus.envelopes if envelope.event_type == "subagent_completed"]
    assert len(completed) == 2
    assert {envelope.worker_id for envelope in completed} == {"sim_agent", "strategy_agent"}
    assert all(envelope.payload["status"] == "completed" for envelope in completed)
    assert {tuple(sorted(envelope.payload["result"].items())) for envelope in completed} == {
        tuple(sorted({"signal": "sim"}.items())),
        tuple(sorted({"signal": "strategy"}.items())),
    }


def test_stable_subagent_result_message_shape() -> None:
    manager = SubagentManager(event_bus=None, session_id="session-1")
    result = SubAgentResult(
        agent="rival_nor",
        success=False,
        data={"decision": "pit now"},
        error="provider timeout",
    )

    message = manager.build_result_message(result)

    assert message == {
        "role": "system",
        "name": "subagent_result",
        "worker_id": "rival_nor",
        "content": json.dumps(
            {
                "worker_id": "rival_nor",
                "status": "error",
                "result": {"decision": "pit now"},
                "error": "provider timeout",
            }
        ),
    }
