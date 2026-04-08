"""Tests for v2 winner inference fallback through train_winner.load_and_predict."""

from __future__ import annotations

from hannah.models.train_winner import load_and_predict


def test_load_and_predict_uses_v2_inference_when_race_context_is_available(monkeypatch) -> None:
    class _FakeArtifact:
        ensemble = b"placeholder"

    monkeypatch.setattr(
        "hannah.models.train_winner.resolve_artifact_path",
        lambda model_name: type("P", (), {"exists": lambda self: True})(),
    )
    monkeypatch.setattr(
        "hannah.models.train_winner.load_pickle_artifact",
        lambda path: _FakeArtifact(),
    )
    monkeypatch.setattr(
        "hannah.models.inference_v2.load_race_frame",
        lambda year, race: object(),
    )
    monkeypatch.setattr(
        "hannah.models.inference_v2.infer_winner",
        lambda df: {
            "winner_probs": [
                {"driver": "NOR", "win_probability": 0.6},
                {"driver": "VER", "win_probability": 0.3},
                {"driver": "LEC", "win_probability": 0.1},
            ]
        },
    )

    result = load_and_predict(
        {"race": "australian_grand_prix", "year": 2026, "drivers": ["VER", "NOR"]}
    )

    assert result == {"NOR": 0.667, "VER": 0.333}
