"""Race state container used by simulation tools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from hannah.agent.context import RaceContext
from hannah.domain.race_state import RaceEventWindow
from hannah.domain.teams import get_driver_codes, get_driver_info, get_primary_rivals
from hannah.domain.tracks import TrackConfig, get_track


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def _create_rng(seed: int):
    try:
        return np.random.default_rng(seed)
    except TypeError:
        return np.random.default_rng()


def _default_compound(weather: str, index: int) -> str:
    if weather == "wet":
        return "INTER" if index % 2 == 0 else "WET"
    if weather == "mixed":
        return "SOFT" if index == 0 else "MEDIUM"
    return "SOFT" if index < 2 else "MEDIUM"


def _default_event_windows(track: TrackConfig, weather: str, current_lap: int, total_laps: int) -> list[RaceEventWindow]:
    windows: list[RaceEventWindow] = []
    midpoint = min(max(current_lap + 8, total_laps // 2), max(current_lap + 1, total_laps - 5))
    if track.safety_car_risk >= 0.25:
        windows.append(RaceEventWindow(kind="safety_car", start_lap=midpoint - 1, end_lap=midpoint + 1))
    elif track.safety_car_risk >= 0.18:
        windows.append(RaceEventWindow(kind="vsc", start_lap=midpoint, end_lap=midpoint + 1, intensity=0.7))
    if weather == "mixed":
        windows.append(
            RaceEventWindow(
                kind="rain",
                start_lap=min(total_laps - 8, max(current_lap + 4, total_laps // 3)),
                end_lap=min(total_laps - 4, max(current_lap + 7, total_laps // 3 + 3)),
                intensity=0.55,
            )
        )
    return windows


@dataclass
class RaceState:
    race: str
    year: int
    laps: int
    n_drivers: int
    drivers: list[str]
    compounds: list[str]
    base_lap_times: np.ndarray
    weather: str = "dry"
    safety_car_prob: float = 0.18
    current_lap: int = 0
    positions: list[int] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    tyre_ages: list[int] = field(default_factory=list)
    pit_loss: float = 22.5
    tyre_wear_factor: float = 1.0
    overtake_difficulty: float = 0.5
    track_position_bias: float = 0.5
    rain_intensity: float = 0.0
    fuel_load: float = 105.0
    telemetry_sources: tuple[str, ...] = field(default_factory=tuple)
    event_windows: list[RaceEventWindow] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        """Normalize field shapes so simulation helpers can assume consistent lengths."""
        if not self.drivers:
            self.drivers = get_driver_codes(max(self.n_drivers, 1))
        self.n_drivers = len(self.drivers)

        if self.base_lap_times.size == 0:
            self.base_lap_times = np.full(self.n_drivers, self.track.base_lap_time, dtype=float)
        elif self.base_lap_times.size != self.n_drivers:
            base = np.asarray(self.base_lap_times, dtype=float)
            if base.size > self.n_drivers:
                self.base_lap_times = base[: self.n_drivers]
            else:
                pad_value = float(np.mean(base)) if base.size else self.track.base_lap_time
                self.base_lap_times = np.pad(base, (0, self.n_drivers - base.size), constant_values=pad_value)

        self.compounds = _pad_list(self.compounds, self.n_drivers, "MEDIUM")
        self.positions = _pad_list(self.positions, self.n_drivers, 1, is_numeric=True)
        self.gaps = _pad_list(self.gaps, self.n_drivers, 0.0, is_numeric=True)
        self.tyre_ages = _pad_list(self.tyre_ages, self.n_drivers, 0, is_numeric=True)
        self.event_windows = sorted(self.event_windows, key=lambda event: (event.start_lap, event.end_lap))

    @property
    def track(self) -> TrackConfig:
        return get_track(self.race, self.laps)

    @property
    def leader(self) -> str:
        return self.drivers[int(np.argmin(self.positions or [1]))]

    @property
    def remaining_laps(self) -> int:
        return max(self.laps - self.current_lap, 0)

    def ideal_stop_window(self) -> tuple[int, int]:
        low, high = self.track.ideal_one_stop_window
        adjusted_low = max(self.current_lap + 1, low)
        adjusted_high = max(adjusted_low + 1, high)
        return adjusted_low, min(adjusted_high, self.laps - 1)

    def pit_lap_bounds(self) -> tuple[int, int]:
        """Compatibility bounds used by v1 acceptance and scenario suites."""
        min_pit_lap = max(18, self.current_lap + 1)
        max_pit_lap = min(self.laps - 5, 38) - 1
        if max_pit_lap < min_pit_lap:
            max_pit_lap = min_pit_lap
        return min_pit_lap, max_pit_lap

    def event_at(self, lap: int, kind: str | None = None) -> RaceEventWindow | None:
        for event in self.event_windows:
            if (kind is None or event.kind == kind) and event.contains(lap):
                return event
        return None

    @classmethod
    def from_context(cls, ctx: RaceContext) -> "RaceState":
        drivers = ctx.drivers or [ctx.drivers[0], *get_primary_rivals(ctx.drivers[0])] if ctx.drivers else get_driver_codes(3)
        track = get_track(ctx.race, ctx.laps)
        return cls._build(
            race=ctx.race,
            year=ctx.year,
            laps=ctx.laps or track.laps,
            drivers=drivers,
            weather=ctx.weather,
            current_lap=0,
            race_data=ctx.race_data or {},
        )

    @classmethod
    def from_race_data(cls, race_data: dict) -> "RaceState":
        session_info = race_data.get("session_info", {})
        drivers = list(race_data.get("drivers") or get_driver_codes(3))
        return cls._build(
            race=str(session_info.get("race", "bahrain")),
            year=int(session_info.get("year", 2025)),
            laps=int(session_info.get("laps", get_track(str(session_info.get("race", "bahrain"))).laps)),
            drivers=drivers,
            weather=str(session_info.get("weather", "dry")),
            current_lap=int(session_info.get("current_lap", 0)),
            race_data=race_data,
        )

    @classmethod
    def _build(
        cls,
        race: str,
        year: int,
        laps: int,
        drivers: list[str],
        weather: str,
        current_lap: int,
        race_data: dict,
    ) -> "RaceState":
        track = get_track(race, laps)
        seed = _stable_seed(race, year, weather, ",".join(drivers), current_lap)
        rng = _create_rng(seed)
        positions_map = race_data.get("positions", {})
        gaps_map = race_data.get("gaps", {})
        compounds_map = race_data.get("compounds", {})
        tyre_ages_map = race_data.get("tyre_ages", {})
        positions = [int(positions_map.get(driver, index + 1)) for index, driver in enumerate(drivers)]
        gaps = [float(gaps_map.get(driver, max(0, position - 1) * (1.2 + track.overtake_difficulty))) for driver, position in zip(drivers, positions)]
        compounds = [
            str(compounds_map.get(driver, _default_compound(weather, index))).upper()
            for index, driver in enumerate(drivers)
        ]
        tyre_ages = [int(tyre_ages_map.get(driver, max(current_lap - 8 + index * 2, 0))) for index, driver in enumerate(drivers)]
        base_lap_times = np.array(
            [
                track.base_lap_time + get_driver_info(driver).base_pace_delta + rng.normal(0.0, 0.12)
                for driver in drivers
            ],
            dtype=float,
        )
        return cls(
            race=race,
            year=year,
            laps=laps,
            n_drivers=len(drivers),
            drivers=drivers,
            compounds=compounds,
            base_lap_times=base_lap_times,
            weather=weather,
            safety_car_prob=track.safety_car_risk,
            current_lap=current_lap,
            positions=positions,
            gaps=gaps,
            tyre_ages=tyre_ages,
            pit_loss=track.pit_loss,
            tyre_wear_factor=track.tyre_wear_factor,
            overtake_difficulty=track.overtake_difficulty,
            track_position_bias=track.track_position_bias,
            rain_intensity=0.55 if weather == "wet" else 0.28 if weather == "mixed" else 0.0,
            telemetry_sources=tuple(race_data.get("telemetry", ("lap_times", "stints", "weather", "positions"))),
            event_windows=_default_event_windows(track, weather, current_lap, laps),
            seed=seed,
        )

    def to_dict(self) -> dict:
        return {
            "race": self.race,
            "year": self.year,
            "laps": self.laps,
            "n_drivers": self.n_drivers,
            "drivers": self.drivers,
            "compounds": self.compounds,
            "base_lap_times": self.base_lap_times.tolist(),
            "weather": self.weather,
            "safety_car_prob": self.safety_car_prob,
            "current_lap": self.current_lap,
            "positions": self.positions,
            "gaps": self.gaps,
            "tyre_ages": self.tyre_ages,
            "pit_loss": self.pit_loss,
            "tyre_wear_factor": self.tyre_wear_factor,
            "overtake_difficulty": self.overtake_difficulty,
            "track_position_bias": self.track_position_bias,
            "rain_intensity": self.rain_intensity,
            "fuel_load": self.fuel_load,
            "telemetry_sources": list(self.telemetry_sources),
            "event_windows": [event.to_dict() for event in self.event_windows],
            "seed": self.seed,
        }

    def update(self, lap_result: dict) -> None:
        self.current_lap += 1
        self.positions = list(lap_result.get("positions", self.positions))
        self.gaps = list(lap_result.get("gaps", self.gaps))
        self.tyre_ages = [age + 1 for age in self.tyre_ages]
        self.fuel_load = max(0.0, self.fuel_load - 1.8)


def _pad_list(values: list, expected_size: int, default, is_numeric: bool = False) -> list:
    if len(values) >= expected_size:
        trimmed = values[:expected_size]
    else:
        trimmed = values + [default for _ in range(expected_size - len(values))]
    if not is_numeric:
        return trimmed
    if isinstance(default, float):
        return [float(item) for item in trimmed]
    return [int(item) for item in trimmed]
