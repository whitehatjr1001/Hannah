"""Fast async prediction-mode simulation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from hannah.domain.teams import get_driver_info
from hannah.simulation.replay_trace import build_replay_trace as _build_replay_trace
from hannah.simulation.sandbox import RaceState
from hannah.simulation.tyre_model import TyreModel

WEATHER_LAP_TIME_DELTA = {
    "dry": 0.0,
    "wet": 8.5,
    "mixed": 3.2,
}


@dataclass
class SimResult:
    winner_probs: np.ndarray
    optimal_pit_laps: np.ndarray
    optimal_compounds: list[str]
    p50_race_time: float
    undercut_windows: dict[int, int]
    all_times: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "winner_probs": self.winner_probs.tolist(),
            "optimal_pit_laps": self.optimal_pit_laps.tolist(),
            "optimal_compounds": self.optimal_compounds,
            "p50_race_time_s": round(float(self.p50_race_time), 3),
            "undercut_windows": self.undercut_windows,
        }


def _next_compound(current_compound: str, state: RaceState, pit_lap: int) -> str:
    # Keep v1 contract stable: post-stop choices are MEDIUM/HARD only.
    del state, pit_lap
    if current_compound == "SOFT":
        return "MEDIUM"
    if current_compound == "MEDIUM":
        return "HARD"
    return "MEDIUM"


def _plan_pit_lap(state: RaceState, driver_index: int, rng: np.random.Generator, tyre_model: TyreModel) -> int:
    low, high = state.ideal_stop_window()
    profile = get_driver_info(state.drivers[driver_index])
    base_target = int(round((low + high) / 2.0))
    style_bias = {
        "aggressive": -2,
        "balanced": 0,
        "defensive": 2,
        "opportunistic": -1 if any(event.kind in {"safety_car", "vsc"} for event in state.event_windows) else 1,
    }[profile.strategy_style]
    recommended_age = tyre_model.recommended_pit_age(
        compound=state.compounds[driver_index],
        wear_factor=state.tyre_wear_factor / max(profile.tyre_management, 0.75),
    )
    age_pressure = max(state.tyre_ages[driver_index] - recommended_age, 0) // 2
    jitter = int(round(rng.normal(0.0, 1.3)))
    min_pit_lap, max_pit_lap = state.pit_lap_bounds()
    return int(np.clip(base_target + style_bias - age_pressure + jitter, min_pit_lap, max_pit_lap))


def _event_bonus(state: RaceState, pit_lap: int) -> float:
    event = state.event_at(pit_lap)
    if event is None:
        return 0.0
    if event.kind == "safety_car":
        return 10.5 * event.intensity
    if event.kind == "vsc":
        return 6.0 * event.intensity
    if event.kind == "rain":
        return 4.5 * event.intensity
    return 0.0


def _simulate_world(
    state: RaceState,
    rng: np.random.Generator,
    tyre_model: TyreModel,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n_drivers = state.n_drivers
    remaining_laps = max(state.remaining_laps, 1)
    pit_laps = np.zeros(n_drivers, dtype=int)
    total_times = np.zeros(n_drivers, dtype=float)
    next_compounds: list[str] = []
    weather_delta = WEATHER_LAP_TIME_DELTA[state.weather]

    for index in range(n_drivers):
        profile = get_driver_info(state.drivers[index])
        pit_lap = _plan_pit_lap(state, index, rng, tyre_model)
        next_compound = _next_compound(state.compounds[index], state, pit_lap)
        pre_laps = max(pit_lap - state.current_lap, 1)
        post_laps = max(remaining_laps - pre_laps, 1)
        wear_factor = state.tyre_wear_factor / max(profile.tyre_management, 0.75)
        pre_penalty = tyre_model.stint_penalty(
            compound=state.compounds[index],
            starting_age=state.tyre_ages[index],
            laps=pre_laps,
            wear_factor=wear_factor,
            rain_intensity=state.rain_intensity if state.weather != "dry" else 0.0,
        )
        post_penalty = tyre_model.stint_penalty(
            compound=next_compound,
            starting_age=0,
            laps=post_laps,
            wear_factor=state.tyre_wear_factor,
            rain_intensity=state.rain_intensity if state.weather != "dry" else 0.0,
        )
        traffic_penalty = max(state.positions[index] - 1, 0) * state.overtake_difficulty * 1.4
        track_position_penalty = state.gaps[index] * state.track_position_bias * 0.75
        pit_gain = _event_bonus(state, pit_lap)
        noise = rng.normal(0.0, 1.1)
        total_times[index] = (
            (state.base_lap_times[index] + weather_delta) * remaining_laps
            + pre_penalty
            + post_penalty
            + state.pit_loss
            + traffic_penalty
            + track_position_penalty
            - pit_gain
            + noise
        )
        pit_laps[index] = pit_lap
        next_compounds.append(next_compound)

    return total_times, pit_laps, next_compounds


async def run_fast(state: RaceState, n_worlds: int = 1000) -> SimResult:
    """Run a deterministic Monte Carlo race simulation."""
    safe_worlds = max(int(n_worlds), 1)

    def _batch() -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        tyre_model = TyreModel()
        rng = _create_rng(state.seed + safe_worlds)
        all_times = np.zeros((safe_worlds, state.n_drivers), dtype=float)
        all_pits = np.zeros((safe_worlds, state.n_drivers), dtype=int)
        all_compounds: list[list[str]] = []
        for index in range(safe_worlds):
            times, pits, post_compounds = _simulate_world(state, rng, tyre_model)
            all_times[index] = times
            all_pits[index] = pits
            all_compounds.append(post_compounds)
        return all_times, all_pits, all_compounds

    all_times, all_pits, all_compounds = await asyncio.to_thread(_batch)
    winners = np.argmin(all_times, axis=1)
    win_counts = np.bincount(winners, minlength=state.n_drivers)
    winner_probs = win_counts / safe_worlds
    optimal_pit_laps = np.array(
        [int(np.bincount(all_pits[:, idx]).argmax()) for idx in range(state.n_drivers)],
        dtype=int,
    )
    winning_times = all_times[np.arange(safe_worlds), winners]
    optimal_compounds: list[str] = []
    for driver_index in range(state.n_drivers):
        counts: dict[str, int] = {}
        for compounds in all_compounds:
            counts[compounds[driver_index]] = counts.get(compounds[driver_index], 0) + 1
        optimal_compounds.append(max(counts.items(), key=lambda item: item[1])[0])
    undercut_windows = _build_undercut_windows(optimal_pit_laps, state)

    return SimResult(
        winner_probs=winner_probs,
        optimal_pit_laps=optimal_pit_laps,
        optimal_compounds=optimal_compounds,
        p50_race_time=float(np.percentile(winning_times, 50)),
        undercut_windows=undercut_windows,
        all_times=all_times,
    )


def _create_rng(seed: int) -> np.random.Generator:
    """Create a seeded RNG while remaining compatible with monkeypatched default_rng."""
    try:
        return np.random.default_rng(seed)
    except TypeError:
        # Some tests monkeypatch default_rng with a zero-arg callable.
        return np.random.default_rng()


def _build_undercut_windows(optimal_pit_laps: np.ndarray, state: RaceState) -> dict[int, int]:
    """Build deterministic undercut windows that remain in the accepted pit-lap bounds."""
    min_pit_lap, max_pit_lap = state.pit_lap_bounds()
    windows: dict[int, int] = {}
    for index, lap in enumerate(optimal_pit_laps):
        bounded = int(np.clip(int(lap), min_pit_lap, max_pit_lap))
        windows[index] = bounded
    return windows


def build_replay_trace(
    state: RaceState,
    sim_result: SimResult,
    checkpoints: list[int] | None = None,
) -> dict:
    """Expose deterministic trace building from the simulation module boundary."""
    return _build_replay_trace(race_state=state, sim_result=sim_result, checkpoints=checkpoints)
