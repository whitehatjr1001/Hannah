"""Masked acceptance scenarios for v2-s2 evaluation depth and trace coherence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Final

import pytest

from hannah.agent.tool_registry import ToolRegistry
from hannah.models.evaluate import evaluate_model
from tests.scenarios.contracts import get_scenario_by_id

pytestmark = pytest.mark.filterwarnings("ignore:urllib3 v2 only supports OpenSSL 1.1.1+")


@dataclass(frozen=True)
class HiddenEvaluationDepthScenario:
    scenario_id: str
    public_scenario_id: str


EVALUATION_DEPTH_SCENARIOS: Final[tuple[HiddenEvaluationDepthScenario, ...]] = (
    HiddenEvaluationDepthScenario("HACC_V2S2_E01", "S01"),
    HiddenEvaluationDepthScenario("HACC_V2S2_E02", "S04"),
    HiddenEvaluationDepthScenario("HACC_V2S2_E03", "S08"),
    HiddenEvaluationDepthScenario("HACC_V2S2_E04", "S09"),
)


@dataclass(frozen=True)
class HiddenTraceCoherenceScenario:
    scenario_id: str
    race: str
    year: int
    weather: str
    laps: int
    drivers: list[str]
    checkpoints: list[int]
    expected_event_kinds: tuple[str, ...]


TRACE_COHERENCE_SCENARIOS: Final[tuple[HiddenTraceCoherenceScenario, ...]] = (
    HiddenTraceCoherenceScenario(
        "HACC_V2S2_T01",
        "silverstone",
        2025,
        "mixed",
        52,
        ["VER", "NOR", "LEC"],
        [17, 18, 19, 20, 26, 27, 52],
        ("rain", "vsc"),
    ),
    HiddenTraceCoherenceScenario(
        "HACC_V2S2_T02",
        "singapore",
        2025,
        "mixed",
        62,
        ["VER", "RUS", "LEC"],
        [20, 21, 22, 30, 31, 32, 62],
        ("rain", "safety_car"),
    ),
)


def _run_tool(registry: ToolRegistry, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(registry.call(name, args))


def _build_scenario_backing(registry: ToolRegistry) -> list[dict[str, Any]]:
    backing: list[dict[str, Any]] = []
    for hidden in EVALUATION_DEPTH_SCENARIOS:
        scenario = get_scenario_by_id(hidden.public_scenario_id)
        sim_args = dict(scenario.tool_inputs["race_sim"])
        sim_args["n_worlds"] = 128
        sim_payload = _run_tool(registry, "race_sim", sim_args)
        backing.append(
            {
                "scenario_id": scenario.scenario_id,
                "weather": str(scenario.input_context["weather"]),
                "recommended_pit_lap": int(sim_payload["strategy"]["recommended_pit_lap"]),
                "trace_events": len(sim_payload["trace"]["timeline"]),
            }
        )
    return backing


def _trace_payload(scenario: HiddenTraceCoherenceScenario) -> dict[str, Any]:
    return {
        "race": scenario.race,
        "year": scenario.year,
        "weather": scenario.weather,
        "drivers": scenario.drivers,
        "laps": scenario.laps,
        "n_worlds": 128,
        "trace": True,
        "trace_checkpoints": scenario.checkpoints,
    }


def test_hidden_v2s2_scenario_backed_evaluation_depth_contract() -> None:
    registry = ToolRegistry()
    scenario_backing = _build_scenario_backing(registry)

    assert len(scenario_backing) == len(EVALUATION_DEPTH_SCENARIOS)
    expected_ids = {row["scenario_id"] for row in scenario_backing}
    expected_weather = {row["weather"] for row in scenario_backing}
    assert {"dry", "mixed"} <= expected_weather

    payload = evaluate_model("winner_ensemble")

    assert set(payload.keys()) >= {"model", "score", "artifact", "artifact_exists", "evaluation_depth"}
    depth = payload["evaluation_depth"]
    assert isinstance(depth, dict)
    assert set(depth.keys()) >= {"scenario_scorecard", "coverage", "stability"}

    scorecard = depth["scenario_scorecard"]
    assert isinstance(scorecard, list) and scorecard
    observed_ids = {str(row["scenario_id"]) for row in scorecard if isinstance(row, dict)}
    assert expected_ids <= observed_ids
    for row in scorecard:
        assert set(row.keys()) >= {"scenario_id", "score", "pit_alignment", "trace_alignment"}
        assert 0.0 <= float(row["score"]) <= 1.0

    coverage = depth["coverage"]
    assert set(coverage.keys()) >= {"by_weather", "event_windows", "scenarios_total"}
    by_weather = coverage["by_weather"]
    assert isinstance(by_weather, dict)
    assert expected_weather <= set(by_weather.keys())

    event_windows = coverage["event_windows"]
    assert set(event_windows.keys()) >= {
        "scenarios_with_event_windows",
        "coherent_timeline_events",
        "total_timeline_events",
    }
    assert event_windows["coherent_timeline_events"] == event_windows["total_timeline_events"]
    assert int(event_windows["scenarios_with_event_windows"]) >= 1

    stability = depth["stability"]
    assert set(stability.keys()) >= {"deterministic", "score_variance"}
    assert stability["deterministic"] is True
    assert float(stability["score_variance"]) <= 0.02


@pytest.mark.parametrize(
    "scenario",
    TRACE_COHERENCE_SCENARIOS,
    ids=[scenario.scenario_id for scenario in TRACE_COHERENCE_SCENARIOS],
)
def test_hidden_v2s2_trace_event_window_coherence_and_replay_determinism(
    scenario: HiddenTraceCoherenceScenario,
) -> None:
    registry = ToolRegistry()

    first = _run_tool(registry, "race_sim", _trace_payload(scenario))
    trace = first["trace"]

    events = trace["events"]
    assert isinstance(events, list) and events
    event_kinds = {str(event["kind"]) for event in events}
    for expected_kind in scenario.expected_event_kinds:
        assert expected_kind in event_kinds

    timeline = trace["timeline"]
    assert isinstance(timeline, list) and timeline
    coherent_entries = [entry for entry in timeline if str(entry.get("event")) in event_kinds]
    assert coherent_entries

    for entry in coherent_entries:
        assert "event_window" in entry
        window = entry["event_window"]
        assert set(window.keys()) >= {"kind", "start_lap", "end_lap", "intensity", "window_index"}
        assert window["kind"] == entry["event"]
        assert int(window["start_lap"]) <= int(entry["lap"]) <= int(window["end_lap"])
        index = int(window["window_index"])
        assert 0 <= index < len(events)
        event = events[index]
        assert event["kind"] == window["kind"]
        assert int(event["start_lap"]) == int(window["start_lap"])
        assert int(event["end_lap"]) == int(window["end_lap"])

    replayed = _run_tool(
        registry,
        "race_sim",
        {
            **_trace_payload(scenario),
            "replay": trace["replay"],
        },
    )

    assert replayed["trace"]["trace_id"] == trace["trace_id"]
    assert replayed["trace"]["events"] == trace["events"]
    assert replayed["trace"]["timeline"] == trace["timeline"]
