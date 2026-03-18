"""V2-S3 public contracts for RivalAgent Q-learning pit-policy backend wiring."""

from __future__ import annotations

import asyncio

from hannah.agent.context import RaceContext
from hannah.agent.subagents import RivalAgent


def test_rival_agent_emits_q_policy_backend_metadata(monkeypatch) -> None:
    async def _fake_ask(self: RivalAgent, prompt: str) -> str:
        del self, prompt
        return ""

    monkeypatch.setattr(RivalAgent, "_ask", _fake_ask)

    ctx = RaceContext(
        race="bahrain",
        year=2025,
        laps=57,
        weather="dry",
        drivers=["VER", "NOR", "LEC"],
        race_data={"session_info": {"current_lap": 21}},
    )
    result = asyncio.run(RivalAgent("NOR").run(ctx))

    assert result.success is True
    required_backend_keys = {"policy_backend", "policy_action_id", "policy_artifact"}
    assert required_backend_keys <= set(result.data.keys())
    assert result.data["policy_backend"] == "q_learning"
    assert int(result.data["policy_action_id"]) in {0, 1}
    assert str(result.data["policy_artifact"]).endswith("models/saved/pit_policy_q_v1.pkl")
