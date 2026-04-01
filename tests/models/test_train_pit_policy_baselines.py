"""Pit-policy baseline trainer contract tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from hannah.models.train_pit_q import QPitPolicyArtifact, load_artifact, train


def test_train_pit_policy_q_uses_configured_artifact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  pit_policy_q: runtime-artifacts/pit-policy-baseline.pkl",
                "",
            ]
        ),
        encoding="utf-8",
    )

    saved_path = train(years=[2024], races=["bahrain"])

    assert saved_path == "runtime-artifacts/pit-policy-baseline.pkl"
    with Path(saved_path).open("rb") as handle:
        artifact = pickle.load(handle)
    assert isinstance(artifact, QPitPolicyArtifact)


def test_load_artifact_raises_when_configured_pit_policy_q_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  pit_policy_q: runtime-artifacts/missing.pkl",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_artifact()
