"""Deterministic telemetry baseline builder."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_RACES: tuple[str, ...] = ("bahrain", "monaco", "silverstone")
BASE_TRACK_TEMPS: dict[str, float] = {
    "bahrain": 31.0,
    "monaco": 27.0,
    "silverstone": 24.0,
    "spa": 22.0,
}
BASE_AIR_TEMPS: dict[str, float] = {
    "bahrain": 26.0,
    "monaco": 22.0,
    "silverstone": 18.0,
    "spa": 17.0,
}
COMPOUND_ENCODING: dict[str, int] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
}


@dataclass(frozen=True)
class DriverBaseline:
    driver: str
    base_lap_time: float
    pace_delta: float


BASELINE_DRIVERS: tuple[DriverBaseline, ...] = (
    DriverBaseline("VER", 89.20, -0.22),
    DriverBaseline("NOR", 89.45, -0.12),
    DriverBaseline("LEC", 89.72, 0.00),
)


def _compound_for_lap(lap_number: int) -> str:
    if lap_number <= 18:
        return "SOFT"
    if lap_number <= 38:
        return "MEDIUM"
    return "HARD"


def build_telemetry_baseline(years: list[int], races: list[str] | None = None) -> pd.DataFrame:
    """Build a lightweight deterministic telemetry baseline."""
    selected_races = tuple(str(race).lower().replace(" ", "_") for race in (races or list(DEFAULT_RACES)))
    rows: list[dict[str, float | int | str]] = []
    for year in years:
        year_bias = max(year - min(years), 0) * 0.05 if years else 0.0
        for race in selected_races:
            base_track_temp = BASE_TRACK_TEMPS.get(race, 26.0)
            base_air_temp = BASE_AIR_TEMPS.get(race, 20.0)
            for driver_index, driver in enumerate(BASELINE_DRIVERS):
                for lap_number in range(1, 58):
                    compound = _compound_for_lap(lap_number)
                    compound_encoded = COMPOUND_ENCODING[compound]
                    stint_start = 1 if lap_number <= 18 else 19 if lap_number <= 38 else 39
                    tyre_age = lap_number - stint_start
                    rainfall = 0.18 if race == "spa" and 24 <= lap_number <= 30 else 0.0
                    fuel_load = max(110.0 - lap_number * 1.85, 0.0)
                    gap_to_leader = max(0.0, driver_index * 1.7 + lap_number * 0.03)
                    traffic = min(0.75, driver_index * 0.12 + (lap_number % 6) * 0.03)
                    track_temp = base_track_temp + (lap_number % 7) * 0.45
                    air_temp = base_air_temp + (lap_number % 5) * 0.2
                    deg_penalty_s = 0.018 * tyre_age + 0.0015 * max(track_temp - 28.0, 0.0) + 0.012 * compound_encoded
                    lap_time = (
                        driver.base_lap_time
                        + driver.pace_delta
                        + year_bias
                        + deg_penalty_s
                        + 0.062 * lap_number
                        + 2.7 * traffic
                        + 0.019 * fuel_load
                        + 3.1 * rainfall
                    )
                    rows.append(
                        {
                            "year": int(year),
                            "race": race,
                            "driver": driver.driver,
                            "lap_number": lap_number,
                            "lap_time_s": round(lap_time, 3),
                            "compound": compound,
                            "compound_encoded": compound_encoded,
                            "tyre_age": tyre_age,
                            "track_temp": round(track_temp, 3),
                            "air_temp": round(air_temp, 3),
                            "rainfall": rainfall,
                            "fuel_load": round(fuel_load, 3),
                            "gap_to_leader": round(gap_to_leader, 3),
                            "traffic": round(traffic, 3),
                            "deg_penalty_s": round(deg_penalty_s, 6),
                        }
                    )
    return pd.DataFrame(rows)
