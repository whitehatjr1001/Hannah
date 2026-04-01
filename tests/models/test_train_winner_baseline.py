"""Winner baseline dataset and trainer tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from hannah.models.datasets.results_baseline import build_results_baseline
from hannah.models.train_winner import WinnerArtifact, load_and_predict, train


def test_results_baseline_builder_filters_requested_seasons_and_races() -> None:
    dataset = build_results_baseline(years=[2023, 2024], races=["bahrain"])

    assert not dataset.empty
    assert set(dataset["year"]) == {2023, 2024}
    assert set(dataset["race"]) == {"bahrain"}
    assert {
        "driver",
        "team",
        "grid_position",
        "q3_time",
        "track_type",
        "tyre_strategy_encoded",
        "avg_pace_delta",
        "safety_car_prob",
        "won",
    } <= set(dataset.columns)


def test_train_winner_uses_configured_artifact_path_and_persists_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  winner_ensemble: runtime-artifacts/winner-baseline.pkl",
                "",
            ]
        ),
        encoding="utf-8",
    )

    saved_path = train(years=[2023, 2024], races=["bahrain"])

    assert saved_path == "runtime-artifacts/winner-baseline.pkl"
    with Path(saved_path).open("rb") as handle:
        artifact = pickle.load(handle)
    assert isinstance(artifact, WinnerArtifact)
    assert artifact.years == (2023, 2024)
    assert artifact.races == ("bahrain",)

    probabilities = load_and_predict({"drivers": ["VER", "NOR", "LEC"]})
    assert set(probabilities) == {"VER", "NOR", "LEC"}
    assert pytest.approx(sum(probabilities.values()), rel=0, abs=0.01) == 1.0
