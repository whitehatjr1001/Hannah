"""Tyre degradation model with trained-model fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hannah.models.inference_v2 import load_joblib_artifact


@dataclass(frozen=True)
class CompoundProfile:
    name: str
    base_loss: float
    variance: float
    cliff_lap: int
    pace_offset: float
    rain_capable: bool = False


COMPOUND_LIBRARY: dict[str, CompoundProfile] = {
    "SOFT": CompoundProfile("SOFT", base_loss=0.075, variance=0.008, cliff_lap=18, pace_offset=-0.35),
    "MEDIUM": CompoundProfile("MEDIUM", base_loss=0.048, variance=0.006, cliff_lap=28, pace_offset=0.0),
    "HARD": CompoundProfile("HARD", base_loss=0.031, variance=0.004, cliff_lap=40, pace_offset=0.42),
    "INTER": CompoundProfile("INTER", base_loss=0.055, variance=0.01, cliff_lap=22, pace_offset=2.4, rain_capable=True),
    "WET": CompoundProfile("WET", base_loss=0.043, variance=0.012, cliff_lap=26, pace_offset=4.7, rain_capable=True),
}


class TyreModel:
    """Predict lap-time degradation from tyre age and compound."""

    def __init__(self, model_path: str | Path = "models/saved/tyre_deg_v1.pkl") -> None:
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            return None
        try:
            return load_joblib_artifact(self.model_path)
        except Exception:
            return None

    def predict(
        self,
        compound: str,
        age: int,
        track_temp: float = 30.0,
        wear_factor: float = 1.0,
        rain_intensity: float = 0.0,
    ) -> float:
        profile = self._profile(compound)
        if self.model is not None:
            predicted = self._predict_from_artifact(
                compound=compound,
                age=age,
                track_temp=track_temp,
                wear_factor=wear_factor,
                rain_intensity=rain_intensity,
            )
            if predicted is not None:
                return predicted

        normalized_age = max(age, 0)
        cliff_age = max(normalized_age - profile.cliff_lap, 0)
        temp_factor = max(track_temp - 30.0, 0.0) * 0.004
        weather_factor = 0.0
        if rain_intensity > 0.0 and not profile.rain_capable:
            weather_factor = 4.5 * rain_intensity
        elif rain_intensity == 0.0 and profile.rain_capable:
            weather_factor = 1.2
        return float(
            profile.pace_offset
            + (profile.base_loss * normalized_age * wear_factor)
            + (np.power(cliff_age, 1.35) * 0.08)
            + temp_factor
            + weather_factor
        )

    def stint_penalty(
        self,
        compound: str,
        starting_age: int,
        laps: int,
        track_temp: float = 30.0,
        wear_factor: float = 1.0,
        rain_intensity: float = 0.0,
    ) -> float:
        ages = np.arange(starting_age, starting_age + max(laps, 0), dtype=float)
        if ages.size == 0:
            return 0.0
        penalties = [
            self.predict(
                compound=compound,
                age=int(age),
                track_temp=track_temp,
                wear_factor=wear_factor,
                rain_intensity=rain_intensity,
            )
            for age in ages
        ]
        return float(np.sum(penalties))

    def predict_batch(
        self,
        compounds: list[str],
        ages: list[int],
        n_worlds: int,
        track_temp: float = 30.0,
        wear_factor: float = 1.0,
        rain_intensity: float = 0.0,
    ) -> np.ndarray:
        base = np.array(
            [
                self.predict(
                    compound=compound,
                    age=age,
                    track_temp=track_temp,
                    wear_factor=wear_factor,
                    rain_intensity=rain_intensity,
                )
                for compound, age in zip(compounds, ages)
            ],
            dtype=float,
        )
        return np.tile(base, (n_worlds, 1))

    def recommended_pit_age(self, compound: str, wear_factor: float = 1.0) -> int:
        profile = self._profile(compound)
        return max(int(profile.cliff_lap / max(wear_factor, 0.8)), 8)

    def _profile(self, compound: str) -> CompoundProfile:
        normalized = compound.strip().upper()
        return COMPOUND_LIBRARY.get(normalized, COMPOUND_LIBRARY["MEDIUM"])

    def _predict_from_artifact(
        self,
        *,
        compound: str,
        age: int,
        track_temp: float,
        wear_factor: float,
        rain_intensity: float,
    ) -> float | None:
        estimator = getattr(self.model, "model", None)
        feature_names = list(getattr(self.model, "feature_names", []) or [])
        if estimator is None or not hasattr(estimator, "predict") or not feature_names:
            return None

        normalized_compound = compound.strip().upper()
        base_features: dict[str, float] = {
            "tyr_e_age_in_stint": float(max(age, 0)),
            "stint_number": 1.0,
            "stint_length": float(max(age, 1)),
            "lap_number": float(max(age, 1)),
            "position": 10.0,
            "position_change": 0.0,
            "gap_normalized": 0.0,
            "track_temp": float(track_temp),
            "air_temp": 25.0,
            "rainfall": float(rain_intensity),
            "race_median": 95.0,
            "compound_encoded": {"SOFT": 2.0, "MEDIUM": 1.0, "HARD": 0.0}.get(
                normalized_compound, 1.0
            ),
            "sector_1": 0.0,
            "sector_2": 0.0,
            "sector_3": 0.0,
            "sector_sum": 0.0,
            "laps_remaining": 0.0,
            "laps_remaining_pct": 0.0,
            "stint_progress": min(float(max(age, 1)) / 30.0, 1.0),
            "compound_x_age": float(max(age, 0)),
            "gap_to_ahead": 0.0,
            "is_leader": 0.0,
            "top_3": 0.0,
            "race_phase": 1.0,
            "safety_car_flag": 0.0,
            "vsc_flag": 0.0,
        }
        base_features[f"compound_{normalized_compound}"] = 1.0

        row = [float(base_features.get(name, 0.0)) for name in feature_names]
        try:
            prediction = estimator.predict(np.array([row], dtype=float))[0]
        except Exception:
            return None
        return float(prediction)
