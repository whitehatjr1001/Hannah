"""Hosted provider setup helpers for the Hannah CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ProviderPreset:
    """Supported hosted-provider preset."""

    name: str
    label: str
    default_model: str
    api_key_env_vars: tuple[str, ...]
    model_hints: tuple[str, ...]


@dataclass(frozen=True)
class ProviderStatus:
    """Resolved provider status for display in the CLI."""

    name: str
    label: str
    default_model: str
    api_key_env_vars: tuple[str, ...]
    configured: bool
    configured_env_var: str | None
    active: bool


PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        name="openai",
        label="OpenAI",
        default_model="gpt-4o-mini",
        api_key_env_vars=("OPENAI_API_KEY",),
        model_hints=("openai/", "gpt-", "o1", "o3", "o4", "chatgpt"),
    ),
    ProviderPreset(
        name="anthropic",
        label="Anthropic",
        default_model="claude-sonnet-4-6",
        api_key_env_vars=("ANTHROPIC_API_KEY",),
        model_hints=("anthropic/", "claude"),
    ),
    ProviderPreset(
        name="google",
        label="Google",
        default_model="gemini/gemini-2.0-flash",
        api_key_env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        model_hints=("gemini/", "google/", "gemini"),
    ),
)

_CONFLICTING_HOSTED_KEYS = (
    "HANNAH_RLM_API_BASE",
    "HANNAH_RLM_API_KEY",
    "HANNAH_FORCE_LOCAL_PROVIDER",
)


def get_provider_preset(name: str) -> ProviderPreset:
    normalized = name.strip().lower()
    for preset in PROVIDER_PRESETS:
        if preset.name == normalized:
            return preset
    raise ValueError(f"unknown provider preset: {name}")


def list_provider_presets() -> list[ProviderPreset]:
    return list(PROVIDER_PRESETS)


def load_env_context(
    env_path: str | Path = ".env",
    *,
    include_process_env: bool = True,
) -> dict[str, str]:
    """Load env values from file and current process env."""
    file_values = _read_env_file(Path(env_path))
    merged: dict[str, str] = {}
    if include_process_env:
        for key, value in os.environ.items():
            if key in _interesting_env_keys():
                merged[key] = value
    merged.update(file_values)
    return merged


def summarize_provider_statuses(env: Mapping[str, str]) -> list[ProviderStatus]:
    model = env.get("HANNAH_MODEL", "")
    active_preset = detect_provider_from_model(model)
    statuses: list[ProviderStatus] = []
    for preset in PROVIDER_PRESETS:
        configured_env_var = next(
            (
                key
                for key in preset.api_key_env_vars
                if _has_real_api_key(env.get(key))
            ),
            None,
        )
        statuses.append(
            ProviderStatus(
                name=preset.name,
                label=preset.label,
                default_model=preset.default_model,
                api_key_env_vars=preset.api_key_env_vars,
                configured=configured_env_var is not None,
                configured_env_var=configured_env_var,
                active=active_preset == preset.name,
            )
        )
    return statuses


def detect_provider_from_model(model: str | None) -> str | None:
    if not model:
        return None
    normalized = model.strip().lower()
    for preset in PROVIDER_PRESETS:
        if any(hint in normalized for hint in preset.model_hints):
            return preset.name
    return None


def primary_api_key_env_var(provider_name: str) -> str | None:
    try:
        preset = get_provider_preset(provider_name)
    except ValueError:
        return None
    return preset.api_key_env_vars[0]


def apply_provider_configuration(
    *,
    env_path: str | Path,
    provider: ProviderPreset,
    api_key: str,
    model: str,
) -> dict[str, str]:
    """Write provider settings into the local .env file."""
    path = Path(env_path)
    updates = {
        "HANNAH_MODEL": model,
        provider.api_key_env_vars[0]: api_key,
    }
    _apply_env_file_changes(path=path, updates=updates, remove_keys=_CONFLICTING_HOSTED_KEYS)
    os.environ["HANNAH_MODEL"] = model
    os.environ[provider.api_key_env_vars[0]] = api_key
    for key in _CONFLICTING_HOSTED_KEYS:
        os.environ.pop(key, None)
    return {
        "provider": provider.name,
        "label": provider.label,
        "model": model,
        "api_key_env_var": provider.api_key_env_vars[0],
        "env_path": str(path),
    }


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _apply_env_file_changes(
    *,
    path: Path,
    updates: Mapping[str, str],
    remove_keys: tuple[str, ...],
) -> None:
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining_updates = dict(updates)
    rendered_lines: list[str] = []
    managed_keys = set(updates) | set(remove_keys)

    for raw_line in existing_lines:
        if "=" not in raw_line or raw_line.lstrip().startswith("#"):
            rendered_lines.append(raw_line)
            continue
        key, _ = raw_line.split("=", 1)
        normalized_key = key.strip()
        if normalized_key in remove_keys:
            continue
        if normalized_key in remaining_updates:
            rendered_lines.append(f"{normalized_key}={remaining_updates.pop(normalized_key)}")
            continue
        rendered_lines.append(raw_line)

    for key, value in remaining_updates.items():
        if rendered_lines and rendered_lines[-1] != "":
            rendered_lines.append("")
        rendered_lines.append(f"{key}={value}")

    filtered_lines: list[str] = []
    for line in rendered_lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key, _ = line.split("=", 1)
            if key.strip() in remove_keys and key.strip() not in updates:
                continue
        filtered_lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(filtered_lines).rstrip() + "\n", encoding="utf-8")


def _interesting_env_keys() -> set[str]:
    keys = {"HANNAH_MODEL", *_CONFLICTING_HOSTED_KEYS}
    for preset in PROVIDER_PRESETS:
        keys.update(preset.api_key_env_vars)
    return keys


def _has_real_api_key(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    if normalized in {"none", "changeme", "placeholder"}:
        return False
    if "key-here" in normalized or "your-key" in normalized:
        return False
    return True
