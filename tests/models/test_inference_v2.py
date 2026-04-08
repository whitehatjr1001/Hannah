"""Tests for direct v2 inference helpers."""

from __future__ import annotations

import pandas as pd

from hannah.models.inference_v2 import build_tyre_features


def test_build_tyre_features_reindexes_to_training_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "race": "bahrain",
                "driver_code": "VER",
                "lap_time_s": 90.1,
                "compound": "SOFT",
                "stint_number": 1,
                "lap_number": 1,
                "position": 1,
                "gap_to_leader_s": 0.0,
                "track_temp": 31.0,
                "air_temp": 24.0,
                "rainfall": 0.0,
                "tyre_age_in_stint": 1.0,
            }
        ]
    )

    features, race_medians = build_tyre_features(
        df,
        ["tyr_e_age_in_stint", "track_temp", "compound_SOFT", "race_bahrain"],
    )

    assert list(features.columns) == [
        "tyr_e_age_in_stint",
        "track_temp",
        "compound_SOFT",
        "race_bahrain",
    ]
    assert float(race_medians.iloc[0]) == 90.1
