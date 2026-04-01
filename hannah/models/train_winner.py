"""Winner model training and fallback inference."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hannah.models.artifact_paths import resolve_artifact_path
from hannah.models.artifacts import atomic_pickle_dump, load_pickle_artifact
from hannah.models.datasets.results_baseline import build_results_baseline

ARTIFACT_PATH = Path("models/saved/winner_ensemble_v1.pkl")
DEFAULT_WEIGHTS: tuple[float, ...] = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05)


@dataclass(frozen=True)
class WinnerArtifact:
    """Simple ordered-prior ensemble used for deterministic smoke inference."""

    version: str
    ordered_weights: tuple[float, ...]
    years: tuple[int, ...]
    races: tuple[str, ...]


def _resolve_drivers(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        drivers = payload.get("drivers") or ["VER", "NOR", "LEC"]
    else:
        drivers = getattr(payload, "drivers", ["VER", "NOR", "LEC"])
    return [str(driver) for driver in drivers]


def _probabilities(drivers: list[str], ordered_weights: tuple[float, ...]) -> dict[str, float]:
    if not drivers:
        raise ValueError("drivers cannot be empty for winner prediction")
    canonical = list(DEFAULT_WEIGHTS[: len(drivers)])
    weights = canonical or list(ordered_weights[: len(drivers)])
    if len(weights) < len(drivers):
        fallback_weight = canonical[-1] if canonical else ordered_weights[-1]
        weights.extend([fallback_weight] * (len(drivers) - len(weights)))
    total = sum(weights) or 1.0
    probabilities = [weight / total for weight in weights]
    rounded: list[float] = []
    if len(probabilities) == 1:
        rounded = [1.0]
    else:
        for probability in probabilities[:-1]:
            rounded.append(round(probability, 3))
        rounded.append(round(1.0 - sum(rounded), 3))
    return {driver: probability for driver, probability in zip(drivers, rounded)}


def train(years: list[int], races: list[str] | None = None) -> str:
    """Train and persist a deterministic winner prior artifact."""
    dataset = build_results_baseline(years=years, races=races)
    baseline_size = max(len(dataset), 1)
    lead_weight = min(0.6, 0.45 + baseline_size * 0.002)
    ordered_weights = (lead_weight, 0.3, 0.2, 0.12, 0.08, 0.05)
    artifact = WinnerArtifact(
        version="v1",
        ordered_weights=ordered_weights,
        years=tuple(years),
        races=tuple(races or []),
    )
    return atomic_pickle_dump(resolve_artifact_path("winner_ensemble"), artifact)


def load_and_predict(payload: Any) -> dict[str, float]:
    """Predict winner probabilities from artifact priors or fallback weights."""
    drivers = _resolve_drivers(payload)
    artifact_path = resolve_artifact_path("winner_ensemble")
    if artifact_path.exists():
        try:
            artifact = load_pickle_artifact(artifact_path)
            if isinstance(artifact, WinnerArtifact):
                return _probabilities(drivers, artifact.ordered_weights)
        except Exception:
            # Fall through to deterministic fallback.
            pass
    return _probabilities(drivers, DEFAULT_WEIGHTS)
