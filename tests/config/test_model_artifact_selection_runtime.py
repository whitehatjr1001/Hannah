"""Runtime artifact selection tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hannah.models import train_pit_q
from hannah.tools.train_model import tool as train_model_tool


def test_train_model_runtime_uses_configured_pit_policy_q_artifact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  pit_policy_q: runtime-artifacts/pit-policy-q.pkl",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = asyncio.run(
        train_model_tool.run(
            model_name="pit_policy_q",
            years=[2024],
            races=["bahrain"],
        )
    )

    assert result == {"saved": "runtime-artifacts/pit-policy-q.pkl"}
    assert Path("runtime-artifacts/pit-policy-q.pkl").exists()


def test_pit_policy_q_runtime_load_does_not_silently_train_missing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  pit_policy_q: runtime-artifacts/missing-pit-policy-q.pkl",
                "",
            ]
        ),
        encoding="utf-8",
    )
    train_called = False

    def _fake_train(years: list[int], races: list[str] | None = None) -> str:
        del years, races
        nonlocal train_called
        train_called = True
        return "runtime-artifacts/should-not-exist.pkl"

    monkeypatch.setattr(train_pit_q, "train", _fake_train)

    with pytest.raises(FileNotFoundError):
        train_pit_q.load_artifact()

    assert train_called is False
