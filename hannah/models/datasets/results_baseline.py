"""Deterministic historical results baseline builder."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_RACES: tuple[str, ...] = ("bahrain", "monaco", "silverstone")
TRACK_TYPES: dict[str, str] = {
    "bahrain": "permanent",
    "monaco": "street",
    "silverstone": "permanent",
    "spa": "permanent",
    "singapore": "street",
}
TRACK_TYPE_ENCODING: dict[str, int] = {
    "permanent": 0,
    "street": 1,
}
TEAM_ENCODING: dict[str, int] = {
    "red_bull": 0,
    "mclaren": 1,
    "ferrari": 2,
    "mercedes": 3,
}
TYRE_STRATEGY_ENCODING: dict[str, int] = {
    "soft_medium_hard": 0,
    "soft_hard": 1,
    "medium_hard": 2,
}


@dataclass(frozen=True)
class BaselineDriver:
    driver: str
    team: str
    grid_position: int
    q3_time: float
    tyre_strategy: str
    avg_pace_delta: float
    win_bias: float


BASELINE_DRIVERS: tuple[BaselineDriver, ...] = (
    BaselineDriver("VER", "red_bull", 1, 88.120, "soft_hard", -0.28, 1.0),
    BaselineDriver("NOR", "mclaren", 2, 88.244, "soft_medium_hard", -0.18, 0.85),
    BaselineDriver("LEC", "ferrari", 3, 88.337, "medium_hard", -0.09, 0.72),
    BaselineDriver("HAM", "mercedes", 5, 88.511, "medium_hard", 0.04, 0.45),
)


def build_results_baseline(years: list[int], races: list[str] | None = None) -> pd.DataFrame:
    """Build a lightweight deterministic results baseline for winner priors."""
    selected_races = tuple(str(race).lower().replace(" ", "_") for race in (races or list(DEFAULT_RACES)))
    rows: list[dict[str, object]] = []
    for year in years:
        year_shift = max(year - min(years), 0) * 0.015 if years else 0.0
        for race in selected_races:
            track_type = TRACK_TYPES.get(race, "permanent")
            safety_car_prob = 0.18 if track_type == "permanent" else 0.34
            for index, driver in enumerate(BASELINE_DRIVERS):
                score = driver.win_bias - year_shift - index * 0.03
                rows.append(
                    {
                        "year": int(year),
                        "race": race,
                        "driver": driver.driver,
                        "team": driver.team,
                        "grid_position": driver.grid_position,
                        "q3_time": round(driver.q3_time + year_shift + index * 0.012, 3),
                        "team_encoded": TEAM_ENCODING[driver.team],
                        "track_type": track_type,
                        "track_type_encoded": TRACK_TYPE_ENCODING[track_type],
                        "tyre_strategy": driver.tyre_strategy,
                        "tyre_strategy_encoded": TYRE_STRATEGY_ENCODING[driver.tyre_strategy],
                        "avg_pace_delta": round(driver.avg_pace_delta + year_shift + index * 0.01, 3),
                        "safety_car_prob": safety_car_prob,
                        "won": 1 if score == max(
                            item.win_bias - year_shift - item_index * 0.03
                            for item_index, item in enumerate(BASELINE_DRIVERS)
                        ) else 0,
                    }
                )
    return pd.DataFrame(rows)
