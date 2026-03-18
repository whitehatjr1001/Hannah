"""Masked acceptance scenarios for v1 toolflow behavior."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from itertools import count
from typing import Final

import numpy as np
import pytest

from hannah.agent.tool_registry import ToolRegistry

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in power:RuntimeWarning")


@dataclass(frozen=True)
class HiddenFlowScenario:
    scenario_id: str
    race: str
    year: int
    weather: str
    laps: int
    drivers: list[str]
    focus_driver: str
    focus_lap: int


FLOW_SCENARIOS: Final[tuple[HiddenFlowScenario, ...]] = (
    HiddenFlowScenario("HACC_F01", "bahrain", 2025, "dry", 57, ["VER", "NOR", "LEC"], "VER", 18),
    HiddenFlowScenario("HACC_F02", "bahrain", 2024, "mixed", 57, ["NOR", "PIA", "VER"], "NOR", 16),
    HiddenFlowScenario("HACC_F03", "monaco", 2025, "dry", 78, ["LEC", "VER", "NOR"], "LEC", 22),
    HiddenFlowScenario("HACC_F04", "singapore", 2025, "wet", 62, ["VER", "RUS", "LEC"], "RUS", 21),
    HiddenFlowScenario("HACC_F05", "silverstone", 2024, "dry", 52, ["NOR", "HAM", "VER"], "HAM", 17),
    HiddenFlowScenario("HACC_F06", "bahrain", 2023, "wet", 57, ["VER", "LEC", "HAM"], "LEC", 19),
    HiddenFlowScenario("HACC_F07", "monaco", 2024, "mixed", 78, ["NOR", "LEC", "RUS"], "NOR", 20),
    HiddenFlowScenario("HACC_F08", "singapore", 2023, "dry", 62, ["VER", "NOR", "HAM"], "HAM", 24),
    HiddenFlowScenario("HACC_F09", "silverstone", 2025, "mixed", 52, ["VER", "PIA", "ALO"], "ALO", 18),
    HiddenFlowScenario("HACC_F10", "bahrain", 2022, "dry", 57, ["VER", "PER", "SAI"], "PER", 15),
)


@pytest.fixture(autouse=True)
def _stub_external_data_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    import hannah.tools.race_data.tool as race_data_tool

    def _fake_fetch_session(race: str, year: int, session_type: str) -> dict:
        return {
            "laps": [{"lap_number": 1, "driver": "VER", "lap_time": 90.0}],
            "weather": [{"rainfall": False, "air_temp": 29.0}],
            "car_data": [],
            "results": [{"position": 1, "driver": "VER"}],
            "source": f"{race}-{year}-{session_type}",
        }

    class _FakeOpenF1Client:
        def get_sessions(self, year: int, race_name: str) -> list[dict]:
            return [
                {"session_key": 1001, "meeting_name": race_name, "year": year},
                {"session_key": 1002, "meeting_name": race_name, "year": year},
            ]

    monkeypatch.setattr(race_data_tool, "fetch_session", _fake_fetch_session)
    monkeypatch.setattr(race_data_tool, "OpenF1Client", _FakeOpenF1Client)


@pytest.fixture(autouse=True)
def _deterministic_simulation_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    import hannah.simulation.monte_carlo as monte_carlo

    np.random.seed(20260318)
    original_default_rng = np.random.default_rng
    seeds = count(start=8000)

    def _seeded_rng() -> np.random.Generator:
        return original_default_rng(next(seeds))

    monkeypatch.setattr(monte_carlo.np.random, "default_rng", _seeded_rng)


def _run_tool(registry: ToolRegistry, name: str, args: dict) -> dict:
    return asyncio.run(registry.call(name, args))


@pytest.mark.parametrize("scenario", FLOW_SCENARIOS, ids=[s.scenario_id for s in FLOW_SCENARIOS])
def test_hidden_end_to_end_toolflow_contract(scenario: HiddenFlowScenario) -> None:
    registry = ToolRegistry()

    race_data = _run_tool(
        registry,
        "race_data",
        {
            "race": scenario.race,
            "year": scenario.year,
            "session": "R",
            "driver": scenario.focus_driver,
        },
    )

    assert set(race_data.keys()) == {"laps", "stints", "weather", "drivers", "session_info"}
    assert isinstance(race_data["laps"], list)
    assert race_data["drivers"] == [scenario.focus_driver]
    assert race_data["session_info"]["race"] == scenario.race
    assert race_data["session_info"]["openf1_sessions"] == 2

    sim_payload = {
        "race": scenario.race,
        "year": scenario.year,
        "weather": scenario.weather,
        "drivers": scenario.drivers,
        "laps": scenario.laps,
    }
    sim_result = _run_tool(registry, "race_sim", sim_payload)

    simulation = sim_result["simulation"]
    strategy = sim_result["strategy"]
    assert set(sim_result.keys()) == {"simulation", "strategy"}
    assert set(simulation.keys()) == {
        "winner_probs",
        "optimal_pit_laps",
        "optimal_compounds",
        "p50_race_time_s",
        "undercut_windows",
    }
    assert set(strategy.keys()) == {
        "recommended_pit_lap",
        "recommended_compound",
        "strategy_type",
        "confidence",
        "undercut_window",
        "rival_threats",
        "reasoning",
    }

    winner_probs = simulation["winner_probs"]
    assert len(winner_probs) == len(scenario.drivers)
    assert all(0.0 <= prob <= 1.0 for prob in winner_probs)
    assert math.isclose(sum(winner_probs), 1.0, rel_tol=0.0, abs_tol=1e-9)

    min_pit_lap = 18
    max_pit_lap = min(scenario.laps - 5, 38) - 1
    assert max_pit_lap >= min_pit_lap
    assert all(min_pit_lap <= lap <= max_pit_lap for lap in simulation["optimal_pit_laps"])
    assert all(compound in {"MEDIUM", "HARD"} for compound in simulation["optimal_compounds"])
    assert simulation["p50_race_time_s"] > 0

    assert min_pit_lap <= strategy["recommended_pit_lap"] <= max_pit_lap
    assert strategy["recommended_compound"] in {"MEDIUM", "HARD"}
    assert strategy["strategy_type"] in {"undercut", "stay_out"}
    assert 0.0 <= strategy["confidence"] <= 1.0
    assert strategy["undercut_window"] in simulation["optimal_pit_laps"]
    assert isinstance(strategy["rival_threats"], list)
    assert isinstance(strategy["reasoning"], str) and strategy["reasoning"]

    pit_strategy = _run_tool(
        registry,
        "pit_strategy",
        {
            "race": scenario.race,
            "year": scenario.year,
            "driver": scenario.focus_driver,
            "lap": scenario.focus_lap,
        },
    )

    assert set(pit_strategy.keys()) == set(strategy.keys())
    assert pit_strategy["recommended_compound"] in {"MEDIUM", "HARD"}
    assert pit_strategy["strategy_type"] in {"undercut", "stay_out"}
    assert 0.0 <= pit_strategy["confidence"] <= 1.0
    assert pit_strategy["recommended_pit_lap"] >= min_pit_lap
    assert isinstance(pit_strategy["reasoning"], str) and pit_strategy["reasoning"]
