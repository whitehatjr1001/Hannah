"""Unified feature table builder from FastF1 + OpenF1 session data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hannah.utils.console import Console

console = Console()

REQUIRED_COLUMNS = [
    "year",
    "race",
    "session_key",
    "driver_number",
    "driver_code",
    "team_name",
    "lap_number",
    "stint_number",
    "compound",
    "tyre_age_at_start",
    "tyre_age_in_stint",
    "lap_time_s",
    "sector_1",
    "sector_2",
    "sector_3",
    "is_pit_out_lap",
    "pit_lap",
    "pit_duration_s",
    "air_temp",
    "track_temp",
    "rainfall",
    "gap_to_leader_s",
    "safety_car",
    "vsc",
    "position",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_lap_time(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    if isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) == 3:
            try:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                pass
        elif len(parts) == 2:
            try:
                return float(parts[0]) * 60 + float(parts[1])
            except ValueError:
                pass
    return _safe_float(value, 0.0)


def _find_stint_for_lap(
    lap_number: int,
    driver_number: int,
    stints_df: list[dict],
) -> dict[str, Any]:
    for stint in stints_df:
        stint_driver = stint.get("driver_number") or stint.get("DriverNumber")
        if stint_driver is None:
            continue
        try:
            if int(stint_driver) != driver_number:
                continue
        except (TypeError, ValueError):
            continue

        lap_start = stint.get("lap_start") or stint.get("LapStart")
        lap_end = stint.get("lap_end") or stint.get("LapEnd")
        if lap_start is None or lap_end is None:
            continue

        try:
            if int(lap_start) <= lap_number <= int(lap_end):
                compound = stint.get("compound") or stint.get("Compound", "UNKNOWN")
                tyre_age_at_start = _safe_int(
                    stint.get("tyre_age_at_start") or stint.get("TyreLifeAtStart", 0)
                )
                stint_number = _safe_int(
                    stint.get("stint_number") or stint.get("Stint", 1), 1
                )
                return {
                    "compound": compound,
                    "tyre_age_at_start": tyre_age_at_start,
                    "stint_number": stint_number,
                    "lap_start": int(lap_start),
                }
        except (TypeError, ValueError):
            continue

    return {
        "compound": "UNKNOWN",
        "tyre_age_at_start": 0,
        "stint_number": 1,
        "lap_start": lap_number,
    }


def _check_pit_stop(
    lap_number: int,
    driver_number: int,
    pit_stops: list[dict],
) -> tuple[bool, float]:
    for pit in pit_stops:
        pit_driver = pit.get("driver_number") or pit.get("DriverNumber")
        pit_lap = pit.get("lap") or pit.get("Lap")
        if pit_driver is None or pit_lap is None:
            continue
        try:
            if int(pit_driver) == driver_number and int(pit_lap) == lap_number:
                duration = pit.get("lane_duration") or pit.get("PitDuration", 0.0)
                return True, _safe_float(duration, 0.0)
        except (TypeError, ValueError):
            continue
    return False, 0.0


def _interpolate_weather(
    lap_timestamp: Any,
    weather_data: pd.DataFrame | list[dict],
) -> dict[str, float]:
    defaults = {"air_temp": 25.0, "track_temp": 30.0, "rainfall": 0.0}

    if isinstance(weather_data, pd.DataFrame):
        if weather_data.empty:
            return defaults

        time_col = None
        for col in ("Time", "date", "Date", "timestamp", "Timestamp"):
            if col in weather_data.columns:
                time_col = col
                break

        if time_col is None:
            row = weather_data.iloc[0]
            return {
                "air_temp": _safe_float(
                    row.get("AirTemp") or row.get("air_temperature", 25.0), 25.0
                ),
                "track_temp": _safe_float(
                    row.get("TrackTemp") or row.get("track_temperature", 30.0), 30.0
                ),
                "rainfall": _safe_float(
                    row.get("Rainfall") or row.get("rainfall", 0.0), 0.0
                ),
            }

        if lap_timestamp is None:
            row = weather_data.iloc[0]
        else:
            try:
                weather_times = pd.to_datetime(weather_data[time_col])
                lap_time = pd.to_datetime(lap_timestamp)
                diffs = (weather_times - lap_time).abs()
                idx = diffs.idxmin()
                row = weather_data.loc[idx]
            except Exception:
                row = weather_data.iloc[0]
    elif isinstance(weather_data, list):
        if not weather_data:
            return defaults

        if lap_timestamp is None:
            row = weather_data[0]
        else:
            best_row = weather_data[0]
            best_diff = None
            for w_row in weather_data:
                w_time = w_row.get("Time") or w_row.get("date") or w_row.get("Date")
                if w_time is None:
                    continue
                try:
                    w_dt = pd.to_datetime(w_time)
                    lap_dt = pd.to_datetime(lap_timestamp)
                    diff = abs((w_dt - lap_dt).total_seconds())
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_row = w_row
                except Exception:
                    continue
            row = best_row
    else:
        return defaults

    return {
        "air_temp": _safe_float(
            row.get("AirTemp") or row.get("air_temperature", 25.0), 25.0
        ),
        "track_temp": _safe_float(
            row.get("TrackTemp") or row.get("track_temperature", 30.0), 30.0
        ),
        "rainfall": _safe_float(row.get("Rainfall") or row.get("rainfall", 0.0), 0.0),
    }


def _get_driver_info(
    driver_number: int,
    drivers: list[dict],
) -> tuple[str, str]:
    for driver in drivers:
        dnum = (
            driver.get("driver_number")
            or driver.get("DriverNumber")
            or driver.get("BroadcastName")
        )
        if dnum is None:
            continue
        try:
            if int(dnum) == driver_number:
                code = (
                    driver.get("name_acronym")
                    or driver.get("DriverCode")
                    or driver.get("Abbreviation", "UNK")
                )
                team = (
                    driver.get("team_name")
                    or driver.get("TeamName")
                    or driver.get("Team", "")
                )
                return str(code), str(team)
        except (TypeError, ValueError):
            continue

    acronym = (
        driver.get("name_acronym") or driver.get("DriverCode") if drivers else None
    )
    team = driver.get("team_name") or driver.get("TeamName") if drivers else None
    return str(acronym or "UNK"), str(team or "")


def _find_interval_for_lap(
    lap_timestamp: Any,
    driver_number: int,
    intervals: list[dict],
) -> float:
    if not intervals:
        return 0.0

    best_gap = 0.0
    best_diff = None

    for interval in intervals:
        int_driver = interval.get("driver_number") or interval.get("DriverNumber")
        int_time = (
            interval.get("date") or interval.get("Time") or interval.get("Timestamp")
        )
        if int_driver is None or int_time is None:
            continue

        try:
            if int(int_driver) != driver_number:
                continue
        except (TypeError, ValueError):
            continue

        if lap_timestamp is None:
            gap = (
                interval.get("gap_to_leader")
                or interval.get("GapToLeader")
                or interval.get("Interval", 0.0)
            )
            return _safe_float(gap, 0.0)

        try:
            int_dt = pd.to_datetime(int_time)
            lap_dt = pd.to_datetime(lap_timestamp)
            diff = abs((int_dt - lap_dt).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                gap = (
                    interval.get("gap_to_leader")
                    or interval.get("GapToLeader")
                    or interval.get("Interval", 0.0)
                )
                best_gap = _safe_float(gap, 0.0)
        except Exception:
            continue

    return best_gap


def _check_sc_vsc(
    lap_timestamp: Any,
    race_control: list[dict],
) -> tuple[bool, bool]:
    if not race_control:
        return False, False

    safety_car = False
    vsc = False

    for msg in race_control:
        msg_time = msg.get("date") or msg.get("Time") or msg.get("Timestamp")
        if msg_time is None:
            continue

        category = msg.get("category") or msg.get("Category", "")
        if isinstance(category, str):
            category_lower = category.lower()
        else:
            category_lower = ""

        message = msg.get("message") or msg.get("Message", "")
        if isinstance(message, str):
            message_lower = message.lower()
        else:
            message_lower = ""

        is_sc = "safety car" in category_lower or "safety car" in message_lower
        is_vsc = (
            "virtual safety car" in category_lower
            or "vsc" in message_lower
            or "virtual safety car" in message_lower
        )

        if is_sc or is_vsc:
            if lap_timestamp is None:
                if is_sc:
                    safety_car = True
                if is_vsc:
                    vsc = True
                continue

            try:
                msg_dt = pd.to_datetime(msg_time)
                lap_dt = pd.to_datetime(lap_timestamp)
                diff = abs((msg_dt - lap_dt).total_seconds())
                if diff < 30:
                    if is_vsc:
                        vsc = True
                    elif is_sc:
                        safety_car = True
            except Exception:
                continue

    return safety_car, vsc


def build_race_features(
    laps_df: pd.DataFrame,
    stints_df: list[dict],
    pit_stops: list[dict],
    weather_df: pd.DataFrame | list[dict],
    drivers: list[dict],
    session_info: dict,
    intervals: list[dict] | None = None,
    race_control: list[dict] | None = None,
) -> pd.DataFrame:
    """Build a unified feature table from FastF1 + OpenF1 session data.

    Args:
        laps_df: Laps DataFrame from FastF1 (orient="records" dicts accepted).
        stints_df: Stint data from OpenF1.
        pit_stops: Pit stop data from OpenF1.
        weather_df: Weather data (DataFrame or list of dicts).
        drivers: Driver info from OpenF1.
        session_info: Session metadata dict.
        intervals: Interval data for gap_to_leader (2023+).
        race_control: Race control messages for SC/VSC flags (2023+).

    Returns:
        DataFrame with all required feature columns.
    """
    if isinstance(laps_df, list):
        laps_df = pd.DataFrame(laps_df)

    if laps_df.empty:
        console.print("[dim]no lap data available, returning empty feature table[/dim]")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    year = session_info.get("year", 0)
    race = session_info.get("race", session_info.get("MeetingName", "unknown"))
    session_key = session_info.get("session_key", session_info.get("SessionKey", 0))

    weather_is_df = isinstance(weather_df, pd.DataFrame)

    driver_lookup = {}
    for driver in drivers:
        dnum = driver.get("driver_number") or driver.get("DriverNumber")
        if dnum is not None:
            try:
                driver_lookup[int(dnum)] = driver
            except (TypeError, ValueError):
                continue

    rows: list[dict[str, Any]] = []

    for _, lap in laps_df.iterrows():
        driver_number = _safe_int(
            lap.get("DriverNumber") or lap.get("driver_number"), 0
        )
        lap_number = _safe_int(lap.get("LapNumber") or lap.get("lap_number"), 0)

        stint_info = _find_stint_for_lap(lap_number, driver_number, stints_df)
        compound = stint_info["compound"]
        tyre_age_at_start = stint_info["tyre_age_at_start"]
        stint_number = stint_info["stint_number"]
        tyre_age_in_stint = lap_number - stint_info["lap_start"] + 1

        pit_lap, pit_duration_s = _check_pit_stop(lap_number, driver_number, pit_stops)

        lap_timestamp = (
            lap.get("Time")
            or lap.get("date")
            or lap.get("Timestamp")
            or lap.get("LapStartDate")
        )

        weather = _interpolate_weather(lap_timestamp, weather_df)

        driver_info = driver_lookup.get(driver_number)
        if driver_info:
            driver_code = (
                driver_info.get("name_acronym")
                or driver_info.get("DriverCode")
                or "UNK"
            )
            team_name = (
                driver_info.get("team_name") or driver_info.get("TeamName") or ""
            )
        else:
            driver_code = lap.get("Driver") or lap.get("driver_code", "UNK")
            team_name = lap.get("Team") or lap.get("team_name", "")

        gap_to_leader = _safe_float(
            lap.get("GapToLeader") or lap.get("gap_to_leader_s", 0.0), 0.0
        )
        if intervals and gap_to_leader == 0.0:
            gap_to_leader = _find_interval_for_lap(
                lap_timestamp, driver_number, intervals
            )

        safety_car, vsc = _check_sc_vsc(lap_timestamp, race_control)

        position = _safe_int(lap.get("Position") or lap.get("position", 0), 0)

        lap_time_s = _parse_lap_time(
            lap.get("LapTime") or lap.get("LapTimeSeconds") or lap.get("lap_time_s")
        )

        sector_1 = _parse_lap_time(
            lap.get("Sector1Time") or lap.get("sector_1") or lap.get("Sector1", 0.0)
        )
        sector_2 = _parse_lap_time(
            lap.get("Sector2Time") or lap.get("sector_2") or lap.get("Sector2", 0.0)
        )
        sector_3 = _parse_lap_time(
            lap.get("Sector3Time") or lap.get("sector_3") or lap.get("Sector3", 0.0)
        )

        is_pit_out_lap = bool(
            lap.get("IsPitOutLap") or lap.get("is_pit_out_lap", False)
        )

        rows.append(
            {
                "year": year,
                "race": race,
                "session_key": session_key,
                "driver_number": driver_number,
                "driver_code": str(driver_code),
                "team_name": str(team_name),
                "lap_number": lap_number,
                "stint_number": stint_number,
                "compound": compound,
                "tyre_age_at_start": tyre_age_at_start,
                "tyre_age_in_stint": max(tyre_age_in_stint, 0),
                "lap_time_s": lap_time_s,
                "sector_1": sector_1,
                "sector_2": sector_2,
                "sector_3": sector_3,
                "is_pit_out_lap": is_pit_out_lap,
                "pit_lap": pit_lap,
                "pit_duration_s": pit_duration_s,
                "air_temp": weather["air_temp"],
                "track_temp": weather["track_temp"],
                "rainfall": weather["rainfall"],
                "gap_to_leader_s": gap_to_leader,
                "safety_car": safety_car,
                "vsc": vsc,
                "position": position,
            }
        )

    df = pd.DataFrame(rows)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col in ("pit_lap", "safety_car", "vsc", "is_pit_out_lap"):
                df[col] = False
            elif col in (
                "year",
                "race",
                "session_key",
                "driver_number",
                "lap_number",
                "stint_number",
                "tyre_age_at_start",
                "tyre_age_in_stint",
                "position",
            ):
                df[col] = 0
            elif col == "driver_code":
                df[col] = "UNK"
            elif col == "team_name":
                df[col] = ""
            elif col == "compound":
                df[col] = "UNKNOWN"
            else:
                df[col] = 0.0

    df = df[REQUIRED_COLUMNS]

    console.print(f"[dim]built {len(df)} feature rows for {race} {year}[/dim]")

    return df


def save_feature_parquet(
    df: pd.DataFrame,
    year: int,
    race: str,
    output_dir: str = "data/feature_store",
) -> str:
    """Save feature DataFrame to parquet with snappy compression.

    Args:
        df: Feature DataFrame from build_race_features.
        year: Season year.
        race: Race name.
        output_dir: Base output directory.

    Returns:
        Path to the saved parquet file.
    """
    race_slug = race.lower().replace(" ", "_").replace("/", "_")
    target_dir = Path(output_dir) / str(year) / race_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / "race_features.parquet"

    df.to_parquet(
        str(output_path),
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    console.print(f"[dim]saved {len(df)} rows to {output_path}[/dim]")

    return str(output_path)
