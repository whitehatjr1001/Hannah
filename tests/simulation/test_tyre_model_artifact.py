"""Tests for tyre-model artifact-backed prediction."""

from __future__ import annotations

from dataclasses import dataclass

from hannah.simulation.tyre_model import TyreModel


@dataclass
class _FakeArtifact:
    model: object
    feature_names: list[str]


class _FakeEstimator:
    def predict(self, rows):
        return [float(rows[0][0]) + 1.5]


def test_tyre_model_uses_artifact_estimator_when_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "hannah.simulation.tyre_model.load_joblib_artifact",
        lambda path: _FakeArtifact(
            model=_FakeEstimator(),
            feature_names=["tyr_e_age_in_stint", "track_temp", "compound_SOFT"],
        ),
    )

    model = TyreModel(model_path=tmp_path / "trained.pkl")
    model.model = _FakeArtifact(
        model=_FakeEstimator(),
        feature_names=["tyr_e_age_in_stint", "track_temp", "compound_SOFT"],
    )

    prediction = model.predict("SOFT", age=7, track_temp=32.0)

    assert prediction == 8.5
