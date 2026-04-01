"""Deterministic tyre degradation smoke trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hannah.models.artifact_paths import resolve_artifact_path
from hannah.models.artifacts import atomic_pickle_dump
from hannah.models.datasets.telemetry_baseline import build_telemetry_baseline

ARTIFACT_PATH = Path("models/saved/tyre_deg_v1.pkl")
COMPOUND_TO_VALUE = {
    "SOFT": 0.0,
    "MEDIUM": 1.0,
    "HARD": 2.0,
    "INTER": 3.0,
    "WET": 4.0,
}


@dataclass(frozen=True)
class TyreDegArtifact:
    """Serialized linear model metadata for deterministic smoke inference."""

    version: str
    coef_age: float
    coef_temp: float
    coef_compound: float
    intercept: float
    years: tuple[int, ...]
    races: tuple[str, ...]


def _build_synthetic_samples(years: list[int], races: list[str] | None) -> tuple[np.ndarray, np.ndarray]:
    dataset = build_telemetry_baseline(years=years, races=races)
    features = dataset[["tyre_age", "track_temp", "compound_encoded"]].to_numpy(dtype=float)
    targets = dataset["deg_penalty_s"].to_numpy(dtype=float)
    return features, targets


def _fit_linear_regression(features: np.ndarray, targets: np.ndarray) -> tuple[float, float, float, float]:
    # Closed-form least squares keeps training deterministic and dependency-light.
    design = np.column_stack((np.ones(len(features), dtype=float), features))
    weights, *_ = np.linalg.lstsq(design, targets, rcond=None)
    intercept, coef_age, coef_temp, coef_compound = weights.tolist()
    return float(intercept), float(coef_age), float(coef_temp), float(coef_compound)


def train(years: list[int], races: list[str] | None = None) -> str:
    """Train a tiny deterministic tyre model and save an artifact."""
    features, targets = _build_synthetic_samples(years=years, races=races)
    intercept, coef_age, coef_temp, coef_compound = _fit_linear_regression(features, targets)

    artifact = TyreDegArtifact(
        version="v1",
        coef_age=coef_age,
        coef_temp=coef_temp,
        coef_compound=coef_compound,
        intercept=intercept,
        years=tuple(years),
        races=tuple(races or []),
    )
    return atomic_pickle_dump(resolve_artifact_path("tyre_model"), artifact)
