"""Deterministic unit tests for tyre degradation behavior."""

from __future__ import annotations

from hannah.simulation.tyre_model import TyreModel


def test_tyre_penalty_increases_with_age(tmp_path) -> None:
    model = TyreModel(model_path=tmp_path / "no-model.pkl")
    young = model.predict("SOFT", age=4, track_temp=32.0, wear_factor=1.0, rain_intensity=0.0)
    old = model.predict("SOFT", age=16, track_temp=32.0, wear_factor=1.0, rain_intensity=0.0)
    assert old > young


def test_slicks_get_larger_penalty_than_rain_tyre_in_rain(tmp_path) -> None:
    model = TyreModel(model_path=tmp_path / "no-model.pkl")
    slick_penalty = model.predict("SOFT", age=10, rain_intensity=0.7)
    rain_penalty = model.predict("INTER", age=10, rain_intensity=0.7)
    assert slick_penalty > rain_penalty


def test_recommended_pit_age_respects_wear_factor(tmp_path) -> None:
    model = TyreModel(model_path=tmp_path / "no-model.pkl")
    conservative = model.recommended_pit_age("MEDIUM", wear_factor=0.9)
    aggressive = model.recommended_pit_age("MEDIUM", wear_factor=1.4)
    assert aggressive < conservative
