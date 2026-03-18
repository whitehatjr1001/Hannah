"""Deterministic tests for Monte Carlo pit window behavior."""

from __future__ import annotations

import asyncio

import numpy as np

import hannah.simulation.monte_carlo as monte_carlo
from hannah.simulation.sandbox import RaceState


def _race_state() -> RaceState:
    return RaceState(
        race="bahrain",
        year=2025,
        laps=57,
        n_drivers=3,
        drivers=["VER", "NOR", "LEC"],
        compounds=["SOFT", "SOFT", "MEDIUM"],
        base_lap_times=np.array([90.0, 90.25, 90.4], dtype=float),
        weather="dry",
        current_lap=6,
        positions=[1, 2, 3],
        gaps=[0.0, 1.3, 3.1],
        tyre_ages=[9, 8, 7],
        seed=1234,
    )


def test_run_fast_respects_v1_pit_bounds_and_compound_set() -> None:
    result = asyncio.run(monte_carlo.run_fast(_race_state(), n_worlds=120))
    min_pit_lap, max_pit_lap = _race_state().pit_lap_bounds()
    assert all(min_pit_lap <= int(lap) <= max_pit_lap for lap in result.optimal_pit_laps)
    assert all(compound in {"MEDIUM", "HARD"} for compound in result.optimal_compounds)
    assert all(min_pit_lap <= int(window) <= max_pit_lap for window in result.undercut_windows.values())
    assert abs(float(np.sum(result.winner_probs)) - 1.0) < 1e-9


def test_run_fast_handles_zero_arg_default_rng_monkeypatch(monkeypatch) -> None:
    original_default_rng = np.random.default_rng
    call_count = {"calls": 0}

    def _seeded_rng():
        call_count["calls"] += 1
        return original_default_rng(999)

    monkeypatch.setattr(monte_carlo.np.random, "default_rng", _seeded_rng)
    result = asyncio.run(monte_carlo.run_fast(_race_state(), n_worlds=24))
    assert call_count["calls"] >= 1
    assert result.winner_probs.shape[0] == 3
