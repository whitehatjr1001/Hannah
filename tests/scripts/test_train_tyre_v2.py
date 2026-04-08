"""Tests for train_tyre_v2 feature alignment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_tyre_v2.py"
    spec = importlib.util.spec_from_file_location("train_tyre_v2", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_features_v2_reuses_training_columns_for_test_split() -> None:
    module = _load_module()

    train_df = pd.DataFrame(
        [
            {
                "race": "bahrain",
                "driver_code": "VER",
                "lap_time_s": 91.0,
                "compound": "SOFT",
                "stint_number": 1,
                "lap_number": 1,
                "position": 1,
                "gap_to_leader_s": 0.0,
                "track_temp": 32.0,
                "air_temp": 24.0,
                "rainfall": 0.0,
                "tyre_age_in_stint": 1.0,
            },
            {
                "race": "monaco",
                "driver_code": "LEC",
                "lap_time_s": 74.0,
                "compound": "MEDIUM",
                "stint_number": 1,
                "lap_number": 1,
                "position": 1,
                "gap_to_leader_s": 0.0,
                "track_temp": 35.0,
                "air_temp": 23.0,
                "rainfall": 0.0,
                "tyre_age_in_stint": 1.0,
            },
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                "race": "bahrain",
                "driver_code": "NOR",
                "lap_time_s": 92.0,
                "compound": "SOFT",
                "stint_number": 1,
                "lap_number": 2,
                "position": 2,
                "gap_to_leader_s": 1.2,
                "track_temp": 31.0,
                "air_temp": 25.0,
                "rainfall": 0.0,
                "tyre_age_in_stint": 2.0,
            }
        ]
    )

    _, _, feature_names = module._build_features_v2(train_df)
    x_test, _, test_feature_names = module._build_features_v2(
        test_df, feature_names=feature_names
    )

    assert test_feature_names == feature_names
    assert x_test.shape[1] == len(feature_names)
