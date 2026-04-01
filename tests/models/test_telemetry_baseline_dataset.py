"""Telemetry baseline dataset tests."""

from __future__ import annotations

from hannah.models.datasets.telemetry_baseline import build_telemetry_baseline


def test_telemetry_baseline_builder_returns_expected_feature_columns() -> None:
    dataset = build_telemetry_baseline(years=[2024], races=["bahrain"])

    assert not dataset.empty
    assert set(dataset["year"]) == {2024}
    assert set(dataset["race"]) == {"bahrain"}
    assert {
        "driver",
        "lap_number",
        "lap_time_s",
        "compound",
        "compound_encoded",
        "tyre_age",
        "track_temp",
        "air_temp",
        "rainfall",
        "fuel_load",
        "gap_to_leader",
    } <= set(dataset.columns)
    assert float(dataset["lap_time_s"].min()) > 0.0
    assert float(dataset["lap_number"].max()) >= 50.0
