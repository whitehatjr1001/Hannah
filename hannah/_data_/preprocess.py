"""Feature engineering helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight environments
    pd = None  # type: ignore[assignment]


def _as_records(value: Any) -> list[dict[str, Any]]:
    if pd is not None and isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _to_seconds(value: Any) -> float:
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_table(records: list[dict[str, Any]], *, prefer_pandas: bool):
    if pd is None or not prefer_pandas:
        return records
    return pd.DataFrame(records)


def build_features(laps_df: pd.DataFrame, stints_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized feature table for simulation and model training."""
    laps = _as_records(laps_df)
    stints = _as_records(stints_df)
    weather = _as_records(weather_df)

    stint_by_id = {
        str(row.get("Stint")): row
        for row in stints
        if row.get("Stint") is not None
    }
    weather_row = weather[0] if weather else {}
    air_temp = float(weather_row.get("AirTemp", weather_row.get("air_temperature", 25.0)))
    track_temp = float(weather_row.get("TrackTemp", weather_row.get("track_temperature", 30.0)))
    rainfall = float(weather_row.get("Rainfall", weather_row.get("rainfall", 0.0)))

    prefer_pandas = pd is not None and isinstance(laps_df, pd.DataFrame)
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(laps, start=1):
        stint_number = row.get("Stint", 1)
        normalized = {
            "driver": row.get("Driver", "UNK"),
            "lap_number": int(row.get("LapNumber", index)),
            "lap_time_s": _to_seconds(row.get("LapTime", row.get("LapTimeSeconds", 0.0))),
            "compound": row.get("Compound", "UNKNOWN"),
            "tyre_age": int(row.get("TyreLife", 0)),
            "stint_number": int(stint_number) if str(stint_number).isdigit() else 1,
            "air_temp": air_temp,
            "track_temp": track_temp,
            "rainfall": rainfall,
            "position": int(row.get("Position", 0)),
            "gap_to_leader_s": float(row.get("GapToLeader", 0.0)),
        }
        stint_info = stint_by_id.get(str(stint_number))
        if stint_info:
            normalized["stint_compound"] = stint_info.get("Compound", normalized["compound"])
        normalized_rows.append(normalized)
    return _as_table(normalized_rows, prefer_pandas=prefer_pandas)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize numeric columns."""
    if pd is not None and isinstance(df, pd.DataFrame):
        normalised = df.copy()
        numeric_columns = normalised.select_dtypes(include=["number"]).columns
        for column in numeric_columns:
            col_min = normalised[column].min()
            col_max = normalised[column].max()
            if col_min == col_max:
                normalised[column] = 0.0
            else:
                normalised[column] = (normalised[column] - col_min) / (col_max - col_min)
        return normalised

    rows = _as_records(df)
    if not rows:
        return rows
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
    )
    bounds: dict[str, tuple[float, float]] = {}
    for key in numeric_keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        bounds[key] = (min(values), max(values))

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        for key in numeric_keys:
            value = float(row.get(key, 0.0))
            lower, upper = bounds[key]
            out[key] = 0.0 if lower == upper else (value - lower) / (upper - lower)
        normalized_rows.append(out)
    return normalized_rows


def build_laptime_training_matrix(records: Iterable[dict[str, Any]]) -> tuple[list[list[float]], list[float]]:
    """Create a deterministic matrix used by lightweight lap-time smoke models."""
    features: list[list[float]] = []
    targets: list[float] = []
    for row in records:
        lap = float(row.get("lap_number", row.get("lap", 0.0)))
        tyre_age = float(row.get("tyre_age", 0.0))
        gap = float(row.get("gap_to_leader_s", 0.0))
        rain = float(row.get("rainfall", 0.0))
        lap_time = float(row.get("lap_time_s", row.get("lap_time", 0.0)))
        features.append([lap, tyre_age, gap, rain])
        targets.append(lap_time)
    return features, targets
