from __future__ import annotations

import asyncio

import numpy as np

from hannah.simulation.monte_carlo import SimResult
from hannah.tools.race_sim import tool


def test_race_sim_uses_resolved_roster_when_drivers_omitted(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_race_data_run(**kwargs) -> dict:
        return {
            "resolved_roster": ["HAM", "ALO", "STR"],
            "drivers": ["VER", "NOR", "LEC"],
            "positions": {"HAM": 1, "ALO": 2, "STR": 3},
            "gaps": {"HAM": 0.0, "ALO": 1.8, "STR": 4.0},
            "session_info": {"race": "monaco", "year": 2026, "laps": 78, "weather": "dry"},
        }

    async def _fake_run_fast(state, n_worlds: int = 1000) -> SimResult:
        captured["drivers"] = list(state.drivers)
        captured["n_worlds"] = n_worlds
        return SimResult(
            winner_probs=np.array([0.5, 0.3, 0.2], dtype=float),
            optimal_pit_laps=np.array([22, 23, 24], dtype=int),
            optimal_compounds=["MEDIUM", "HARD", "HARD"],
            p50_race_time=5032.1,
            undercut_windows={0: 22, 1: 23, 2: 24},
        )

    monkeypatch.setattr("hannah.tools.race_data.tool.run", _fake_race_data_run)
    monkeypatch.setattr("hannah.simulation.monte_carlo.run_fast", _fake_run_fast)

    result = asyncio.run(tool.run(race="monaco", year=2026, laps=78))

    assert captured == {"drivers": ["HAM", "ALO", "STR"], "n_worlds": 1000}
    assert set(result.keys()) == {"simulation", "strategy"}
    assert len(result["simulation"]["winner_probs"]) == 3
