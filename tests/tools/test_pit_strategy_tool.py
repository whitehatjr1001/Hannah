from __future__ import annotations

import asyncio

import numpy as np

from hannah.simulation.monte_carlo import SimResult
from hannah.tools.pit_strategy import tool


def test_pit_strategy_seeds_from_resolved_session_state(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_race_data_run(**kwargs) -> dict:
        return {
            "resolved_roster": ["HAM", "ALO", "STR", "VER"],
            "positions": {"HAM": 3, "ALO": 1, "STR": 2, "VER": 4},
            "gaps": {"HAM": 4.2, "ALO": 0.0, "STR": 1.1, "VER": 7.5},
            "compounds": {"HAM": "HARD", "ALO": "MEDIUM", "STR": "HARD", "VER": "MEDIUM"},
            "tyre_ages": {"HAM": 17, "ALO": 12, "STR": 15, "VER": 10},
            "session_info": {"race": "monaco", "year": 2026, "laps": 78, "weather": "dry"},
        }

    async def _fake_run_fast(state, n_worlds: int = 500) -> SimResult:
        captured["drivers"] = list(state.drivers[:4])
        captured["positions"] = list(state.positions[:4])
        captured["current_lap"] = state.current_lap
        return SimResult(
            winner_probs=np.array([0.4, 0.3, 0.2, 0.1], dtype=float),
            optimal_pit_laps=np.array([24, 25, 26, 27], dtype=int),
            optimal_compounds=["MEDIUM", "HARD", "HARD", "MEDIUM"],
            p50_race_time=5050.0,
            undercut_windows={0: 24, 1: 25, 2: 26, 3: 27},
        )

    monkeypatch.setattr("hannah.tools.race_data.tool.run", _fake_race_data_run)
    monkeypatch.setattr("hannah.simulation.monte_carlo.run_fast", _fake_run_fast)

    result = asyncio.run(tool.run(race="monaco", year=2026, driver="HAM", lap=21))

    assert captured == {
        "drivers": ["HAM", "ALO", "STR", "VER"],
        "positions": [3, 1, 2, 4],
        "current_lap": 21,
    }
    assert "recommended_pit_lap" in result
    assert "reasoning" in result
