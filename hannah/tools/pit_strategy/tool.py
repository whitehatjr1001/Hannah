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
    from hannah.tools.race_data import tool as race_data_tool

    session_state = await race_data_tool.run(race=race, year=year, session="R")
    race_state = RaceState.from_race_data(session_state)
    if driver in race_state.drivers:
        driver_index = race_state.drivers.index(driver)
        order = [driver_index, *[index for index in range(race_state.n_drivers) if index != driver_index]]
        race_state.drivers = [race_state.drivers[index] for index in order]
        race_state.compounds = [race_state.compounds[index] for index in order]
        race_state.positions = [race_state.positions[index] for index in order]
        race_state.gaps = [race_state.gaps[index] for index in order]
        race_state.tyre_ages = [race_state.tyre_ages[index] for index in order]
        race_state.base_lap_times = race_state.base_lap_times[order]
        race_state.n_drivers = len(race_state.drivers)
        race_state.current_lap = lap
    else:
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

    sim_result = await run_fast(race_state, n_worlds=500)
    return StrategyEngine().analyse(race_state, sim_result)
