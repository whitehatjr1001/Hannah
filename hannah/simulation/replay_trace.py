"""Deterministic replay/debug trace builders for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hannah.domain.race_state import RaceSnapshot
from hannah.simulation.sandbox import RaceState

_WEATHER_TRACE_DELTA = {
    "dry": 0.0,
    "mixed": 1.6,
    "wet": 3.8,
}


@dataclass(frozen=True)
class TraceCheckpoint:
    lap: int
    leader: str
    order: list[str]
    gaps_s: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "lap": self.lap,
            "leader": self.leader,
            "order": self.order,
            "gaps_s": self.gaps_s,
        }


def build_replay_trace(
    race_state: RaceState,
    sim_result: "SimResult",
    checkpoints: list[int] | None = None,
) -> dict:
    """Build a deterministic replay payload owned by the simulation stack."""
    normalized_checkpoints = _normalize_checkpoints(race_state, checkpoints)
    expected_times = _expected_race_times(race_state, sim_result)
    projected_order = _projected_order(race_state, sim_result, expected_times)
    trace_checkpoints = [
        _build_checkpoint(race_state, lap)
        for lap in normalized_checkpoints
    ]
    focus_pit_lap = int(sim_result.optimal_pit_laps[0])
    focus_compound = str(sim_result.optimal_compounds[0])
    timeline = _timeline(
        race_state=race_state,
        checkpoints=trace_checkpoints,
        focus_pit_lap=focus_pit_lap,
        focus_compound=focus_compound,
    )
    return {
        "race": race_state.race,
        "year": race_state.year,
        "weather": race_state.weather,
        "seed": race_state.seed,
        "focus_driver": race_state.drivers[0],
        "timeline": timeline,
        "projected_order": projected_order,
        "pit_plan": _pit_plan(race_state, sim_result),
        "events": [event.to_dict() for event in race_state.event_windows],
    }


def _timeline(
    race_state: RaceState,
    checkpoints: list[TraceCheckpoint],
    focus_pit_lap: int,
    focus_compound: str,
) -> list[dict]:
    timeline: list[dict] = []
    indexed_events = list(enumerate(race_state.event_windows))
    for checkpoint in checkpoints:
        event = race_state.event_at(checkpoint.lap)
        event_window = None
        if event is not None:
            window_index = next(
                (
                    index
                    for index, candidate in indexed_events
                    if candidate.kind == event.kind
                    and candidate.start_lap == event.start_lap
                    and candidate.end_lap == event.end_lap
                ),
                0,
            )
            event_window = {
                "kind": event.kind,
                "start_lap": event.start_lap,
                "end_lap": event.end_lap,
                "intensity": event.intensity,
                "window_index": window_index,
            }
        if checkpoint.lap == focus_pit_lap:
            event_label = "pit_window"
        elif event is not None:
            event_label = event.kind
        else:
            event_label = "pace_projection"
        summary = (
            f"Lap {checkpoint.lap}: {checkpoint.leader} leads; "
            f"pit target lap {focus_pit_lap} on {focus_compound}."
        )
        entry = {
            "lap": checkpoint.lap,
            "event": event_label,
            "recommended_pit_lap": focus_pit_lap,
            "recommended_compound": focus_compound,
            "summary": summary,
        }
        if event_window is not None:
            entry["event_window"] = event_window
        timeline.append(entry)
    return timeline


def _normalize_checkpoints(race_state: RaceState, checkpoints: list[int] | None) -> list[int]:
    first = min(max(race_state.current_lap + 1, 1), race_state.laps)
    middle = min(max((race_state.current_lap + race_state.laps) // 2, first), race_state.laps)
    focus = first
    default = [first, focus, middle, race_state.laps]
    source = default if checkpoints is None else checkpoints
    normalized = sorted({int(np.clip(int(lap), first, race_state.laps)) for lap in source})
    return normalized or [first, middle, race_state.laps]


def _expected_race_times(race_state: RaceState, sim_result: "SimResult") -> np.ndarray:
    if sim_result.all_times is not None and sim_result.all_times.size:
        return np.mean(sim_result.all_times, axis=0)
    laps_remaining = max(race_state.remaining_laps, 1)
    weather_delta = _WEATHER_TRACE_DELTA.get(race_state.weather, 0.0)
    expected = []
    for index in range(race_state.n_drivers):
        tyre_penalty = float(race_state.tyre_ages[index]) * 0.08 * race_state.tyre_wear_factor
        traffic_penalty = max(race_state.positions[index] - 1, 0) * race_state.overtake_difficulty * 0.9
        expected.append(
            (race_state.base_lap_times[index] + weather_delta) * laps_remaining
            + race_state.gaps[index]
            + tyre_penalty
            + traffic_penalty
        )
    return np.array(expected, dtype=float)


def _projected_order(
    race_state: RaceState,
    sim_result: "SimResult",
    expected_times: np.ndarray,
) -> list[dict]:
    ordered_indices = list(np.argsort(expected_times))
    output: list[dict] = []
    for index in ordered_indices:
        output.append(
            {
                "driver": race_state.drivers[index],
                "win_prob": round(float(sim_result.winner_probs[index]), 4),
                "expected_race_time_s": round(float(expected_times[index]), 3),
                "current_position": int(race_state.positions[index]),
                "current_gap_s": round(float(race_state.gaps[index]), 3),
            }
        )
    return output


def _pit_plan(race_state: RaceState, sim_result: "SimResult") -> list[dict]:
    snapshot = RaceSnapshot(
        race=race_state.race,
        year=race_state.year,
        total_laps=race_state.laps,
        current_lap=race_state.current_lap,
        weather=race_state.weather,
        drivers=list(race_state.drivers),
        compounds={driver: race_state.compounds[index] for index, driver in enumerate(race_state.drivers)},
        tyre_ages={driver: int(race_state.tyre_ages[index]) for index, driver in enumerate(race_state.drivers)},
        gaps={driver: float(race_state.gaps[index]) for index, driver in enumerate(race_state.drivers)},
        positions={driver: int(race_state.positions[index]) for index, driver in enumerate(race_state.drivers)},
    )
    output: list[dict] = []
    for index, driver in enumerate(race_state.drivers):
        projected_rejoin = snapshot.projected_pit_rejoin(driver, pit_loss=race_state.pit_loss)
        output.append(
            {
                "driver": driver,
                "current_compound": race_state.compounds[index],
                "target_compound": sim_result.optimal_compounds[index],
                "current_tyre_age": int(race_state.tyre_ages[index]),
                "optimal_pit_lap": int(sim_result.optimal_pit_laps[index]),
                "undercut_window": int(sim_result.undercut_windows.get(index, sim_result.optimal_pit_laps[index])),
                "projected_rejoin_position": projected_rejoin.projected_position,
                "projected_rejoin_gap_s": round(float(projected_rejoin.projected_gap), 3),
                "projected_car_ahead": projected_rejoin.car_ahead,
            }
        )
    return output


def _build_checkpoint(race_state: RaceState, lap: int) -> TraceCheckpoint:
    laps_remaining = max(race_state.laps - lap, 0)
    lap_offset = max(lap - race_state.current_lap, 0)
    weather_delta = _WEATHER_TRACE_DELTA.get(race_state.weather, 0.0)
    projections: list[tuple[int, float]] = []
    for index in range(race_state.n_drivers):
        projected_age = race_state.tyre_ages[index] + lap_offset
        degradation = projected_age * 0.09 * race_state.tyre_wear_factor
        traffic = max(race_state.positions[index] - 1, 0) * race_state.overtake_difficulty * 0.7
        projected = (
            race_state.gaps[index]
            + (race_state.base_lap_times[index] + weather_delta + degradation) * laps_remaining
            + traffic
        )
        projections.append((index, float(projected)))

    ordered = sorted(projections, key=lambda item: item[1])
    leader_time = ordered[0][1] if ordered else 0.0
    order = [race_state.drivers[index] for index, _ in ordered]
    gaps = {race_state.drivers[index]: round(float(time - leader_time), 3) for index, time in ordered}
    return TraceCheckpoint(lap=int(lap), leader=order[0] if order else "", order=order, gaps_s=gaps)


if TYPE_CHECKING:  # pragma: no cover
    from hannah.simulation.monte_carlo import SimResult
