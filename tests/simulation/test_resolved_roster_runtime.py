from __future__ import annotations

from types import SimpleNamespace

from hannah.agent.context import RaceContext
from hannah.agent.subagents import _build_default_subagents
from hannah.agent.worker_registry import build_legacy_worker_specs
from hannah.simulation.competitor_agents import default_rival_grid
from hannah.simulation.sandbox import RaceState


def test_race_state_from_context_prefers_resolved_roster() -> None:
    ctx = RaceContext(
        race="monaco",
        year=2026,
        laps=78,
        weather="dry",
        drivers=["VER", "NOR", "LEC"],
        race_data={
            "resolved_roster": ["HAM", "ALO", "STR"],
            "positions": {"HAM": 1, "ALO": 2, "STR": 3},
            "gaps": {"HAM": 0.0, "ALO": 1.7, "STR": 4.2},
        },
    )

    state = RaceState.from_context(ctx)

    assert state.drivers == ["HAM", "ALO", "STR"]
    assert state.positions == [1, 2, 3]
    assert state.gaps == [0.0, 1.7, 4.2]


def test_worker_and_subagent_rosters_follow_resolved_roster(monkeypatch) -> None:
    class _FakeProvider:
        async def complete(self, **kwargs):  # pragma: no cover - not used
            return {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setattr(
        "hannah.agent.subagents.ProviderRegistry.from_config",
        lambda cfg: _FakeProvider(),
    )

    ctx = RaceContext(
        race="bahrain",
        year=2026,
        laps=57,
        weather="dry",
        drivers=["VER", "NOR", "LEC"],
        race_data={"resolved_roster": ["VER", "HAM", "ALO", "STR"]},
    )

    specs = build_legacy_worker_specs(ctx)
    rival_worker_ids = [spec.worker_id for spec in specs if spec.worker_id.startswith("rival_")]
    subagents = _build_default_subagents(ctx)
    rival_agent_names = [agent.name for agent in subagents if agent.name.startswith("rival_")]

    assert rival_worker_ids == ["rival_ham", "rival_alo", "rival_str"]
    assert rival_agent_names == ["rival_ham", "rival_alo", "rival_str"]


def test_default_rival_grid_uses_race_state_context_instead_of_static_grid() -> None:
    race_state = SimpleNamespace(
        drivers=["HAM", "ALO", "STR"],
        positions=[3, 1, 2],
        gaps=[4.5, 0.0, 1.2],
        tyre_ages=[18, 11, 13],
        weather="dry",
        current_lap=31,
        event_at=lambda lap: None,
    )

    opinions = default_rival_grid(["HAM", "ALO", "STR"], race_state=race_state)

    assert [opinion.driver for opinion in opinions] == ["HAM", "ALO", "STR"]
    assert all(opinion.recommended_pit_lap >= 31 for opinion in opinions)
    assert "lap 31" not in opinions[0].reasoning
