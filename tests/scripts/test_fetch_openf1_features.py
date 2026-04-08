"""Tests for the feature-store fetch script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "fetch_openf1_features.py"
    )
    spec = importlib.util.spec_from_file_location("fetch_openf1_features", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_fastf1_session_data_uses_lightweight_profile(monkeypatch) -> None:
    module = _load_script_module()

    class FakeSession:
        def __init__(self) -> None:
            self.load_kwargs: dict | None = None
            self.laps = pd.DataFrame([{"LapNumber": 1, "Driver": "VER"}])
            self.weather_data = pd.DataFrame([{"AirTemp": 30.0}])

        def load(self, **kwargs) -> None:
            self.load_kwargs = dict(kwargs)

    fake_session = FakeSession()

    class FakeFastF1:
        class Cache:
            @staticmethod
            def enable_cache(path: str) -> None:
                return None

        @staticmethod
        def get_session(year: int, round_num: int, session_code: str):
            assert (year, round_num, session_code) == (2024, 4, "R")
            return fake_session

    monkeypatch.setitem(sys.modules, "fastf1", FakeFastF1)

    laps_df, weather_df = module._load_fastf1_session_data(2024, 4)

    assert fake_session.load_kwargs == module.FASTF1_LOAD_KWARGS
    assert list(laps_df["Driver"]) == ["VER"]
    assert list(weather_df["AirTemp"]) == [30.0]
