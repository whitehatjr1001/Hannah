"""Shared model artifact path resolution."""

from __future__ import annotations

from pathlib import Path

from hannah.config.loader import load_config
from hannah.config.schema import AppConfig

PUBLIC_MODEL_NAMES: tuple[str, ...] = (
    "tyre_model",
    "laptime_model",
    "pit_rl",
    "pit_policy_q",
    "winner_ensemble",
)

_MODEL_NAME_ALIASES: dict[str, str] = {
    "tyre_deg": "tyre_model",
    "laptime": "laptime_model",
    "winner": "winner_ensemble",
}


def normalize_model_name(model_name: str) -> str:
    """Normalize public and legacy model ids into the canonical public name."""
    normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _MODEL_NAME_ALIASES.get(normalized, normalized)
    if canonical not in PUBLIC_MODEL_NAMES:
        raise ValueError(f"unknown model_name: {model_name}")
    return canonical


def resolve_artifact_path(model_name: str, *, config: AppConfig | None = None) -> Path:
    """Resolve a model artifact path from typed config."""
    canonical_name = normalize_model_name(model_name)
    app_config = config or load_config()
    return Path(getattr(app_config.models, canonical_name))


def resolve_artifact_paths(*, config: AppConfig | None = None) -> dict[str, Path]:
    """Resolve every public model artifact path from typed config."""
    app_config = config or load_config()
    return {name: resolve_artifact_path(name, config=app_config) for name in PUBLIC_MODEL_NAMES}
