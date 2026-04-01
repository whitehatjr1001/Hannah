"""Deterministic pit-policy smoke trainer."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from hannah.models.artifact_paths import resolve_artifact_path
from hannah.models.artifacts import atomic_zip_bundle

ARTIFACT_PATH = Path("models/saved/pit_rl_v1.zip")


@dataclass(frozen=True)
class PitPolicyArtifact:
    """Minimal policy parameters for deterministic pit decisions."""

    version: str
    pit_if_tyre_wear_above: float
    pit_if_safety_car_lap_window: tuple[int, int]
    default_target_compound: str
    years: tuple[int, ...]
    races: tuple[str, ...]


def _derive_policy(years: list[int], races: list[str] | None) -> PitPolicyArtifact:
    years_span = (max(years) - min(years)) if years else 0
    race_bonus = float(len(races or [])) * 0.8
    threshold = max(55.0, min(72.0, 64.0 - years_span + race_bonus))
    return PitPolicyArtifact(
        version="v1",
        pit_if_tyre_wear_above=round(threshold, 2),
        pit_if_safety_car_lap_window=(12, 42),
        default_target_compound="MEDIUM",
        years=tuple(years),
        races=tuple(races or []),
    )


def train(years: list[int], races: list[str] | None = None) -> str:
    """Train a deterministic policy surrogate and persist it as a zip artifact."""
    artifact = _derive_policy(years=years, races=races)
    return atomic_zip_bundle(
        resolve_artifact_path("pit_rl"),
        {
            "policy.json": json.dumps(asdict(artifact), sort_keys=True, indent=2),
            "README.txt": "Deterministic pit-policy smoke artifact. Replace with PPO weights in advanced runs.\n",
        },
    )
