"""Deterministic lap-time predictor smoke trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hannah.models.artifact_paths import resolve_artifact_path
from hannah.models.artifacts import atomic_pickle_dump
from hannah.models.datasets.telemetry_baseline import build_telemetry_baseline

ARTIFACT_PATH = Path("models/saved/laptime_v1.pkl")


@dataclass(frozen=True)
class LapTimeArtifact:
    """Serialized linear model parameters for smoke inference."""

    version: str
    intercept: float
    coefficients: dict[str, float]
    years: tuple[int, ...]
    races: tuple[str, ...]


def _build_synthetic_dataset(years: list[int], races: list[str] | None) -> tuple[np.ndarray, np.ndarray]:
    dataset = build_telemetry_baseline(years=years, races=races)
    features = dataset[["lap_number", "tyre_age", "traffic", "fuel_load", "rainfall"]].to_numpy(dtype=float)
    targets = dataset["lap_time_s"].to_numpy(dtype=float)
    return features, targets


def _fit_linear_regression(features: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    design = np.column_stack((np.ones(len(features), dtype=float), features))
    weights, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return float(weights[0]), np.array(weights[1:], dtype=float)


def train(years: list[int], races: list[str] | None = None) -> str:
    """Train a deterministic linear lap-time model and save it."""
    features, targets = _build_synthetic_dataset(years=years, races=races)
    intercept, coefficients = _fit_linear_regression(features, targets)
    feature_names = ("lap", "tyre_wear", "traffic", "fuel_weight", "rainfall")

    artifact = LapTimeArtifact(
        version="v1",
        intercept=intercept,
        coefficients={name: float(value) for name, value in zip(feature_names, coefficients)},
        years=tuple(years),
        races=tuple(races or []),
    )
    return atomic_pickle_dump(resolve_artifact_path("laptime_model"), artifact)
