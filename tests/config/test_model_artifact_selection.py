"""Config-driven artifact selection tests."""

from __future__ import annotations

from pathlib import Path

from hannah.config.loader import load_config
from hannah.models.artifact_paths import PUBLIC_MODEL_NAMES, resolve_artifact_path


def test_load_config_normalizes_legacy_and_public_model_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  tyre_deg: artifacts/tyre-legacy.pkl",
                "  laptime_model: artifacts/laptime-public.pkl",
                "  pit_rl: artifacts/pit-rl.zip",
                "  pit_policy_q: artifacts/pit-policy.pkl",
                "  winner: artifacts/winner-legacy.pkl",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(path=config_path)

    assert config.models.tyre_model == "artifacts/tyre-legacy.pkl"
    assert config.models.laptime_model == "artifacts/laptime-public.pkl"
    assert config.models.pit_rl == "artifacts/pit-rl.zip"
    assert config.models.pit_policy_q == "artifacts/pit-policy.pkl"
    assert config.models.winner_ensemble == "artifacts/winner-legacy.pkl"
    assert config.models.tyre_deg == config.models.tyre_model
    assert config.models.laptime == config.models.laptime_model
    assert config.models.winner == config.models.winner_ensemble


def test_resolve_artifact_path_supports_all_public_model_names(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  tyre_model: custom/tyre.pkl",
                "  laptime_model: custom/laptime.pkl",
                "  pit_rl: custom/pit.zip",
                "  pit_policy_q: custom/pit-policy.pkl",
                "  winner_ensemble: custom/winner.pkl",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(path=config_path)

    resolved = {name: resolve_artifact_path(name, config=config) for name in PUBLIC_MODEL_NAMES}

    assert resolved == {
        "tyre_model": Path("custom/tyre.pkl"),
        "laptime_model": Path("custom/laptime.pkl"),
        "pit_rl": Path("custom/pit.zip"),
        "pit_policy_q": Path("custom/pit-policy.pkl"),
        "winner_ensemble": Path("custom/winner.pkl"),
    }
