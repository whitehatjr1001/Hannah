"""Track metadata used by the simulation stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackConfig:
    name: str
    laps: int
    base_lap_time: float
    pit_loss: float
    tyre_wear_factor: float
    overtake_difficulty: float
    track_position_bias: float
    safety_car_risk: float
    rain_sensitivity: float
    ideal_one_stop_window: tuple[int, int]


TRACKS: dict[str, TrackConfig] = {
    "abu_dhabi": TrackConfig("abu_dhabi", 58, 90.2, 21.5, 0.96, 0.48, 0.44, 0.16, 0.3, (18, 25)),
    "bahrain": TrackConfig("bahrain", 57, 92.0, 22.5, 1.3, 0.42, 0.38, 0.18, 0.25, (16, 22)),
    "barcelona": TrackConfig("barcelona", 66, 79.8, 21.3, 1.18, 0.46, 0.45, 0.12, 0.28, (17, 24)),
    "imola": TrackConfig("imola", 63, 80.4, 26.0, 1.04, 0.76, 0.82, 0.21, 0.32, (19, 28)),
    "interlagos": TrackConfig("interlagos", 71, 71.3, 22.0, 1.07, 0.53, 0.48, 0.3, 0.55, (21, 29)),
    "jeddah": TrackConfig("jeddah", 50, 88.1, 19.2, 0.92, 0.37, 0.34, 0.32, 0.18, (17, 24)),
    "miami": TrackConfig("miami", 57, 91.5, 21.8, 1.1, 0.44, 0.4, 0.22, 0.38, (16, 23)),
    "monaco": TrackConfig("monaco", 78, 71.0, 19.5, 1.08, 0.94, 0.95, 0.28, 0.18, (28, 40)),
    "monza": TrackConfig("monza", 53, 81.4, 23.0, 0.84, 0.3, 0.22, 0.14, 0.16, (18, 26)),
    "silverstone": TrackConfig("silverstone", 52, 88.0, 21.0, 1.2, 0.55, 0.41, 0.24, 0.6, (17, 24)),
    "singapore": TrackConfig("singapore", 62, 101.0, 28.0, 1.42, 0.72, 0.74, 0.44, 0.46, (18, 27)),
    "spa": TrackConfig("spa", 44, 104.5, 20.2, 1.12, 0.58, 0.36, 0.26, 0.72, (12, 18)),
}


def get_track(name: str, fallback_laps: int | None = None) -> TrackConfig:
    lookup = name.lower().replace(" ", "_")
    config = TRACKS.get(lookup)
    if config is not None:
        return config
    laps = 57 if fallback_laps is None else fallback_laps
    return TrackConfig(
        name=lookup,
        laps=laps,
        base_lap_time=90.0,
        pit_loss=22.0,
        tyre_wear_factor=1.0,
        overtake_difficulty=0.5,
        track_position_bias=0.5,
        safety_car_risk=0.2,
        rain_sensitivity=0.35,
        ideal_one_stop_window=(max(10, laps // 3), max(12, laps // 2)),
    )
