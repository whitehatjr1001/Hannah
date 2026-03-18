"""Deterministic harness for public scenario contract tests."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from hannah.models.evaluate import evaluate_model
from hannah.simulation import monte_carlo
from hannah.simulation.monte_carlo import SimResult
from hannah.tools.pit_strategy import tool as pit_strategy_tool
from hannah.tools.predict_winner import tool as predict_winner_tool
from hannah.tools.race_data import tool as race_data_tool
from hannah.tools.race_sim import tool as race_sim_tool
from hannah.tools.train_model import tool as train_model_tool

from tests.scenarios.contracts import ScenarioContract

TOOL_RUNNERS: dict[str, Any] = {
    "race_data": race_data_tool.run,
    "race_sim": race_sim_tool.run,
    "pit_strategy": pit_strategy_tool.run,
    "predict_winner": predict_winner_tool.run,
    "train_model": train_model_tool.run,
    "evaluate_model": evaluate_model,
}


def _deterministic_sim_result(n_drivers: int) -> SimResult:
    weights = np.arange(n_drivers, 0, -1, dtype=float)
    probs = weights / weights.sum()
    optimal_pit_laps = np.array([18 + idx for idx in range(n_drivers)], dtype=int)
    compounds = ["MEDIUM" if idx % 2 == 0 else "HARD" for idx in range(n_drivers)]
    undercut_windows = {idx: int(lap) for idx, lap in enumerate(optimal_pit_laps)}
    base_times = np.linspace(5200.0, 5200.0 + float(n_drivers - 1) * 4.0, n_drivers)
    all_times = np.tile(base_times, (64, 1))
    return SimResult(
        winner_probs=probs,
        optimal_pit_laps=optimal_pit_laps,
        optimal_compounds=compounds,
        p50_race_time=float(np.percentile(all_times[:, 0], 50)),
        undercut_windows=undercut_windows,
        all_times=all_times,
    )


def patch_external_dependencies(monkeypatch) -> None:
    def _fake_fetch_session(race: str, year: int, session_type: str) -> dict[str, Any]:
        del race, year, session_type
        return {
            "laps": [{"lap_number": 1, "driver": "VER", "lap_time": 90.1}],
            "weather": [{"lap_number": 1, "track_temp": 33.0, "rainfall": 0}],
            "car_data": [],
            "results": [{"driver": "VER", "position": 1}],
        }

    async def _fake_run_fast(state, n_worlds: int = 1000) -> SimResult:
        del n_worlds
        return _deterministic_sim_result(state.n_drivers)

    def _fake_get_sessions(self, year: int, race_name: str) -> list[dict[str, Any]]:
        del self, year, race_name
        return [{"session_key": 2025001, "meeting_name": "stubbed"}]

    monkeypatch.setattr(race_data_tool, "fetch_session", _fake_fetch_session)
    monkeypatch.setattr(race_data_tool.OpenF1Client, "get_sessions", _fake_get_sessions)
    monkeypatch.setattr(monte_carlo, "run_fast", _fake_run_fast)


def _run_tool(tool_name: str, payload: dict[str, Any]) -> Any:
    runner = TOOL_RUNNERS[tool_name]
    response = runner(**payload)
    if asyncio.iscoroutine(response):
        return asyncio.run(response)
    return response


def execute_public_contract(
    scenario: ScenarioContract,
    monkeypatch,
) -> tuple[list[str], dict[str, Any]]:
    patch_external_dependencies(monkeypatch)
    executed_path: list[str] = []
    outputs: dict[str, Any] = {}
    for tool_name in scenario.expected_tool_path:
        payload = dict(scenario.tool_inputs.get(tool_name, {}))
        outputs[tool_name] = _run_tool(tool_name, payload)
        executed_path.append(tool_name)
    return executed_path, outputs


def execute_evaluation_contract(scenario: ScenarioContract) -> dict[str, Any]:
    payload = scenario.tool_inputs.get("evaluate_model")
    if payload is None:
        raise ValueError(f"scenario has no evaluate_model input: {scenario.scenario_id}")
    result = _run_tool("evaluate_model", dict(payload))
    if not isinstance(result, dict):
        raise TypeError(
            f"evaluate_model returned non-dict payload for {scenario.scenario_id}: {type(result)!r}"
        )
    return result


def assert_required_keys(payload: dict[str, Any], required_keys: tuple[str, ...]) -> None:
    missing = set(required_keys) - set(payload.keys())
    assert not missing, f"missing keys: {sorted(missing)}"
