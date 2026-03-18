"""Masked acceptance scenarios for deterministic trace/replay behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pytest

from hannah.agent.tool_registry import ToolRegistry


@dataclass(frozen=True)
class HiddenTraceScenario:
    scenario_id: str
    race: str
    year: int
    weather: str
    drivers: list[str]
    laps: int


TRACE_SCENARIOS: Final[tuple[HiddenTraceScenario, ...]] = (
    HiddenTraceScenario("HACC_R01", "bahrain", 2025, "dry", ["VER", "NOR", "LEC"], 57),
    HiddenTraceScenario("HACC_R02", "silverstone", 2025, "mixed", ["VER", "PIA", "ALO"], 52),
)


def _run_tool(registry: ToolRegistry, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(registry.call(name, args))


def _trace_payload_from_scenario(scenario: HiddenTraceScenario) -> dict[str, Any]:
    return {
        "race": scenario.race,
        "year": scenario.year,
        "weather": scenario.weather,
        "drivers": scenario.drivers,
        "laps": scenario.laps,
        "trace": True,
    }


@pytest.mark.parametrize("scenario", TRACE_SCENARIOS, ids=[s.scenario_id for s in TRACE_SCENARIOS])
def test_hidden_trace_mode_adds_replay_bundle_without_breaking_sim_payload(
    scenario: HiddenTraceScenario,
) -> None:
    registry = ToolRegistry()

    baseline = _run_tool(
        registry,
        "race_sim",
        {
            "race": scenario.race,
            "year": scenario.year,
            "weather": scenario.weather,
            "drivers": scenario.drivers,
            "laps": scenario.laps,
        },
    )
    assert set(baseline.keys()) == {"simulation", "strategy"}

    traced = _run_tool(registry, "race_sim", _trace_payload_from_scenario(scenario))
    assert {"simulation", "strategy", "trace"} <= set(traced.keys())

    trace = traced["trace"]
    assert isinstance(trace, dict)
    assert {"trace_id", "moments", "replay"} <= set(trace.keys())
    assert isinstance(trace["trace_id"], str) and trace["trace_id"]
    assert isinstance(trace["moments"], list) and trace["moments"]
    assert all(isinstance(moment, dict) for moment in trace["moments"])
    assert all("lap" in moment for moment in trace["moments"])
    assert isinstance(trace["replay"], dict) and trace["replay"]


@pytest.mark.parametrize("scenario", TRACE_SCENARIOS, ids=[s.scenario_id for s in TRACE_SCENARIOS])
def test_hidden_trace_replay_contract_is_stable_for_same_replay_bundle(
    scenario: HiddenTraceScenario,
) -> None:
    registry = ToolRegistry()

    # Keep the baseline call site deterministic even before replay is applied.
    np.random.seed(20260318)
    first = _run_tool(registry, "race_sim", _trace_payload_from_scenario(scenario))
    replay = first["trace"]["replay"]
    assert isinstance(replay, dict) and replay

    np.random.seed(20260318)
    replayed = _run_tool(
        registry,
        "race_sim",
        {
            **_trace_payload_from_scenario(scenario),
            "replay": replay,
        },
    )

    assert replayed["trace"]["trace_id"] == first["trace"]["trace_id"]
    assert replayed["strategy"]["recommended_pit_lap"] == first["strategy"]["recommended_pit_lap"]
    assert replayed["strategy"]["recommended_compound"] == first["strategy"]["recommended_compound"]
