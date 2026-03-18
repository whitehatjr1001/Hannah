"""Pit strategy tool wrapper."""

from __future__ import annotations

SKILL = {
    "name": "pit_strategy",
    "description": "Returns a pit stop recommendation and confidence score.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string"},
            "year": {"type": "integer"},
            "lap": {"type": "integer"},
            "driver": {"type": "string"},
        },
        "required": ["race", "driver"],
    },
}


async def run(race: str, driver: str, year: int = 2025, lap: int = 1) -> dict:
    """Return a single-driver pit-wall recommendation."""
    import numpy as np

    from hannah.simulation.monte_carlo import run_fast
    from hannah.simulation.sandbox import RaceState
    from hannah.simulation.strategy_engine import StrategyEngine

    state = RaceState(
        race=race,
        year=year,
        laps=57,
        n_drivers=3,
        drivers=[driver, "NOR", "LEC"],
        compounds=["SOFT", "SOFT", "MEDIUM"],
        base_lap_times=np.array([90.0, 90.5, 90.8], dtype=float),
        weather="dry",
        current_lap=lap,
        positions=[1, 2, 3],
        gaps=[0.0, 1.5, 3.2],
        tyre_ages=[lap, lap - 1, lap - 3],
    )
    sim_result = await run_fast(state, n_worlds=500)
    return StrategyEngine().analyse(state, sim_result)
