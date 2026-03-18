"""Load Hannah configuration from YAML with typed defaults."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from hannah.config.schema import AppConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from disk, falling back to defaults."""
    config_path = Path(path or "config.yaml")
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    normalised = _resolve_env_placeholders(_normalise_yaml_keys(raw))
    return AppConfig.model_validate(normalised)


def _normalise_yaml_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate reserved words like 'async' into Python-safe keys."""
    simulation = raw.get("simulation")
    if isinstance(simulation, dict) and "async" in simulation and "async_enabled" not in simulation:
        simulation["async_enabled"] = simulation.pop("async")
    return raw


def _resolve_env_placeholders(raw: Any) -> Any:
    if isinstance(raw, dict):
        return {key: _resolve_env_placeholders(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return [_resolve_env_placeholders(item) for item in raw]
    if isinstance(raw, str):
        return _resolve_string(raw)
    return raw


def _resolve_string(value: str) -> str:
    match = re.fullmatch(r"\$\{([A-Z0-9_]+):-([^}]+)\}", value)
    if match is None:
        return value
    env_name, default = match.groups()
    return os.getenv(env_name, default)
