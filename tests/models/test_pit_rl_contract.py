"""Pit RL artifact contract tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from hannah.models.artifact_paths import resolve_artifact_path
from hannah.models.train_pit_rl import train


def test_train_pit_rl_uses_configured_artifact_path_and_writes_policy_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  pit_rl: runtime-artifacts/pit-rl-policy.zip",
                "",
            ]
        ),
        encoding="utf-8",
    )

    saved_path = train(years=[2023, 2024], races=["bahrain", "monaco"])

    assert saved_path == "runtime-artifacts/pit-rl-policy.zip"
    assert resolve_artifact_path("pit_rl") == Path(saved_path)
    with zipfile.ZipFile(saved_path, mode="r") as archive:
        payload = json.loads(archive.read("policy.json").decode("utf-8"))
    assert payload["version"] == "v1"
    assert payload["years"] == [2023, 2024]
    assert payload["races"] == ["bahrain", "monaco"]
