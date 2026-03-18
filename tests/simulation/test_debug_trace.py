"""Deterministic replay/debug trace contracts for simulation outputs."""

from __future__ import annotations

import asyncio

import numpy as np

import hannah.simulation.monte_carlo as monte_carlo
from hannah.simulation.sandbox import RaceState


def _race_state() -> RaceState:
    return RaceState(
        race="silverstone",
        year=2025,
        laps=52,
        n_drivers=3,
        drivers=["VER", "NOR", "LEC"],
        compounds=["SOFT", "SOFT", "MEDIUM"],
        base_lap_times=np.array([90.2, 90.35, 90.5], dtype=float),
        weather="mixed",
        current_lap=7,
        positions=[1, 2, 3],
        gaps=[0.0, 1.5, 3.0],
        tyre_ages=[10, 9, 8],
        seed=20260318,
    )


def test_build_replay_trace_is_deterministic_and_structured() -> None:
    state = _race_state()
    result = asyncio.run(monte_carlo.run_fast(state, n_worlds=80))

    trace_a = monte_carlo.build_replay_trace(state, result)
    trace_b = monte_carlo.build_replay_trace(state, result)

    assert trace_a == trace_b
    assert set(trace_a.keys()) >= {"race", "year", "weather", "seed", "focus_driver", "timeline"}
    assert trace_a["race"] == state.race
    assert trace_a["year"] == state.year
    assert trace_a["weather"] == state.weather
    assert trace_a["seed"] == state.seed
    assert trace_a["focus_driver"] == state.drivers[0]

    timeline = trace_a["timeline"]
    assert isinstance(timeline, list)
    assert timeline
    assert set(timeline[0].keys()) >= {
        "lap",
        "event",
        "recommended_pit_lap",
        "recommended_compound",
        "summary",
    }
    laps = [int(event["lap"]) for event in timeline]
    assert laps == sorted(laps)
    assert laps[0] >= state.current_lap + 1
    assert all(isinstance(event["summary"], str) and event["summary"] for event in timeline)
