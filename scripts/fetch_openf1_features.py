"""CLI script to fetch historical F1 data and build the parquet feature store.

Usage:
    python scripts/fetch_openf1_features.py --years 2024 --races bahrain
    python scripts/fetch_openf1_features.py --years 2023,2024,2025 --races bahrain,monaco,silverstone
    python scripts/fetch_openf1_features.py --resume  # skip already-fetched year/race combos
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click
import pandas as pd

from hannah._data_.fastf1_loader import FASTF1_LOAD_KWARGS
from hannah.utils.console import Console

console = Console()

FEATURE_STORE_ROOT = Path("data/feature_store")
DEFAULT_WORKERS = 2


def _parse_years(years_str: str) -> list[int]:
    """Parse comma-separated year string into sorted unique ints."""
    return sorted({int(y.strip()) for y in years_str.split(",") if y.strip()})


def _parse_races(races_str: str | None) -> list[str] | None:
    """Parse comma-separated race string into lowercase list, or None."""
    if not races_str:
        return None
    return [r.strip().lower() for r in races_str.split(",") if r.strip()]


def _get_race_schedule(year: int) -> list[dict[str, Any]]:
    """Get canonical race names for a year from FastF1 schedule."""
    try:
        import fastf1

        fastf1.Cache.enable_cache("data/fastf1_cache")
        schedule = fastf1.get_event_schedule(year)
        races: list[dict[str, Any]] = []
        for _, row in schedule.iterrows():
            races.append(
                {
                    "round": int(row.get("RoundNumber", 0)),
                    "event_name": str(row.get("EventName", "")),
                    "country": str(row.get("Country", "")),
                    "location": str(row.get("Location", "")),
                }
            )
        return races
    except Exception as err:
        console.print(
            f"[yellow]warning:[/yellow] Could not fetch schedule for {year}: {err}"
        )
        return []


def _match_race_name(
    user_race: str, schedule: list[dict[str, Any]]
) -> tuple[str, int] | None:
    """Match a lowercase user race name to the canonical FastF1 event name.

    Returns (event_name, round_number) or None.
    Skips round 0 (Pre-Season Testing, etc.)
    """
    user_normalized = user_race.lower().replace("_", " ").replace("-", " ")
    for token in ("grand prix", "formula 1", "fia", "world championship", "gp"):
        user_normalized = user_normalized.replace(token, " ").strip()
    user_tokens = set(user_normalized.split())

    for race in schedule:
        if race.get("round", 0) == 0:
            continue

        # Check all possible match fields
        candidates = [
            race.get("event_name", ""),
            race.get("country", ""),
            race.get("location", ""),
        ]

        for candidate in candidates:
            if not candidate:
                continue
            candidate_normalized = candidate.lower().replace("_", " ").replace("-", " ")
            for token in ("grand prix", "formula 1", "fia", "world championship", "gp"):
                candidate_normalized = candidate_normalized.replace(token, " ").strip()
            candidate_normalized = " ".join(candidate_normalized.split())

            if user_normalized == candidate_normalized:
                return race["event_name"], race["round"]
            if user_normalized in candidate_normalized:
                return race["event_name"], race["round"]
            if user_tokens and user_tokens <= set(candidate_normalized.split()):
                return race["event_name"], race["round"]

    return None


def _parquet_path(year: int, race_slug: str) -> Path:
    """Get the parquet file path for a year/race combo."""
    return FEATURE_STORE_ROOT / str(year) / race_slug / "race_features.parquet"


def _already_fetched(year: int, race_slug: str) -> bool:
    """Check if a year/race parquet already exists."""
    return _parquet_path(year, race_slug).exists()


def _parse_lap_time(value: Any) -> float:
    """Parse lap time to seconds."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_features_from_fastf1(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    year: int,
    race: str,
    round_num: int,
) -> pd.DataFrame:
    """Build full feature table from FastF1 session data.

    Uses FastF1's rich column names directly.
    """
    rows: list[dict[str, Any]] = []

    # Build weather lookup by time (store as total seconds for safe comparison)
    weather_lookup: list[dict[str, float]] = []
    if not weather_df.empty and "Time" in weather_df.columns:
        for _, w in weather_df.iterrows():
            try:
                wt = w["Time"]
                if hasattr(wt, "total_seconds"):
                    wt_sec = wt.total_seconds()
                else:
                    wt_sec = pd.to_datetime(wt).timestamp()
                weather_lookup.append(
                    {
                        "time_sec": wt_sec,
                        "air_temp": float(w.get("AirTemp", 25.0)),
                        "track_temp": float(w.get("TrackTemp", 30.0)),
                        "rainfall": float(w.get("Rainfall", 0.0)),
                    }
                )
            except Exception:
                pass

    for _, lap in laps_df.iterrows():
        lap_number = (
            int(lap.get("LapNumber", 0)) if pd.notna(lap.get("LapNumber")) else 0
        )
        driver_number = (
            int(lap.get("DriverNumber", 0)) if pd.notna(lap.get("DriverNumber")) else 0
        )
        driver_code = str(lap.get("Driver", "UNK"))
        team_name = str(lap.get("Team", ""))
        stint_number = int(lap.get("Stint", 1)) if pd.notna(lap.get("Stint")) else 1
        compound = str(lap.get("Compound", "UNKNOWN"))
        tyre_age = (
            float(lap.get("TyreLife", 0)) if pd.notna(lap.get("TyreLife")) else 0.0
        )
        position = int(lap.get("Position", 0)) if pd.notna(lap.get("Position")) else 0

        # Parse lap time
        lap_time_s = _parse_lap_time(lap.get("LapTime"))

        # Parse sector times
        sector_1 = _parse_lap_time(lap.get("Sector1Time"))
        sector_2 = _parse_lap_time(lap.get("Sector2Time"))
        sector_3 = _parse_lap_time(lap.get("Sector3Time"))

        # Pit info
        pit_out_time = lap.get("PitOutTime")
        pit_in_time = lap.get("PitInTime")
        pit_lap = bool(pd.notna(pit_in_time) or pd.notna(pit_out_time))
        pit_duration_s = 0.0
        if pd.notna(pit_out_time) and pd.notna(pit_in_time):
            try:
                if hasattr(pit_out_time, "total_seconds") and hasattr(
                    pit_in_time, "total_seconds"
                ):
                    pit_duration_s = float(
                        pit_in_time.total_seconds() - pit_out_time.total_seconds()
                    )
                else:
                    pit_duration_s = float(
                        (
                            pd.to_timedelta(pit_in_time) - pd.to_timedelta(pit_out_time)
                        ).total_seconds()
                    )
            except Exception:
                pit_duration_s = 0.0

        # Is pit out lap
        is_pit_out_lap = bool(lap.get("IsPitOutLap", False))

        # Weather interpolation
        lap_time_val = lap.get("LapStartTime") or lap.get("Time")
        air_temp, track_temp, rainfall = 25.0, 30.0, 0.0
        if lap_time_val is not None and weather_lookup:
            try:
                if hasattr(lap_time_val, "total_seconds"):
                    lap_sec = lap_time_val.total_seconds()
                else:
                    lap_sec = pd.to_datetime(lap_time_val).timestamp()
                best_diff = None
                best_w = None
                for w in weather_lookup:
                    diff = abs(w["time_sec"] - lap_sec)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_w = w
                if best_w:
                    air_temp = best_w["air_temp"]
                    track_temp = best_w["track_temp"]
                    rainfall = best_w["rainfall"]
            except Exception:
                pass

        # Tyre age: FastF1 TyreLife is already the age within the current stint.
        # tyre_age_at_start = TyreLife at the first lap of this stint
        # tyre_age_in_stint = current TyreLife (same as tyre_age)
        tyre_age_in_stint = max(tyre_age, 0)
        tyre_age_at_start = max(tyre_age - 1, 0)  # approximate; will be refined below

        rows.append(
            {
                "year": year,
                "race": race,
                "session_key": 0,
                "driver_number": driver_number,
                "driver_code": driver_code,
                "team_name": team_name,
                "lap_number": lap_number,
                "stint_number": stint_number,
                "compound": compound,
                "tyre_age_at_start": max(tyre_age - 1, 0),
                "tyre_age_in_stint": max(tyre_age_in_stint, 0),
                "lap_time_s": lap_time_s,
                "sector_1": sector_1,
                "sector_2": sector_2,
                "sector_3": sector_3,
                "is_pit_out_lap": is_pit_out_lap,
                "pit_lap": pit_lap,
                "pit_duration_s": pit_duration_s,
                "air_temp": air_temp,
                "track_temp": track_temp,
                "rainfall": rainfall,
                "gap_to_leader_s": 0.0,
                "safety_car": False,
                "vsc": False,
                "position": position,
            }
        )

    return pd.DataFrame(rows)


def _load_fastf1_session_data(year: int, round_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a race session with a reduced FastF1 profile.

    Keep lap and weather data for the training pipeline, but skip expensive
    telemetry/messages fetches that do not feed the parquet feature store.
    """
    import fastf1

    fastf1.Cache.enable_cache("data/fastf1_cache")
    session = fastf1.get_session(year, round_num, "R")
    session.load(**FASTF1_LOAD_KWARGS)
    return session.laps, session.weather_data


def _fetch_and_build_race(
    year: int,
    canonical_race_name: str,
    race_slug: str,
    round_num: int,
) -> dict[str, Any]:
    """Fetch data for a single race and build features."""
    result: dict[str, Any] = {
        "year": year,
        "race": race_slug,
        "status": "PENDING",
        "rows": 0,
        "drivers": 0,
        "laps": 0,
        "compounds": [],
        "error": None,
    }

    console.print(
        f"[cyan]→[/cyan] Fetching {race_slug} {year} (R{round_num}) from FastF1..."
    )
    t0 = time.time()

    try:
        laps_df, weather_df = _load_fastf1_session_data(year, round_num)
    except Exception as err:
        console.print(f"[red]✗[/red] FastF1 error for {race_slug} {year}: {err}")
        result["status"] = "ERROR"
        result["error"] = str(err)
        return result

    console.print(f"[dim]  FastF1 loaded in {time.time() - t0:.1f}s[/dim]")

    if laps_df is None or laps_df.empty:
        console.print(f"[yellow]⚠[/yellow] No lap data for {race_slug} {year}")
        result["status"] = "SKIPPED_NO_DATA"
        return result

    # Build features
    features_df = _build_features_from_fastf1(
        laps_df=laps_df,
        weather_df=weather_df,
        year=year,
        race=canonical_race_name,
        round_num=round_num,
    )

    if features_df.empty:
        console.print(
            f"[yellow]⚠[/yellow] Could not build features for {race_slug} {year}"
        )
        result["status"] = "SKIPPED_NO_DATA"
        return result

    # Save parquet
    output_path = _parquet_path(year, race_slug)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(str(output_path), index=False, engine="pyarrow")

    # Summary
    result["status"] = "OK"
    result["rows"] = len(features_df)
    result["drivers"] = features_df["driver_code"].nunique()
    result["laps"] = int(features_df["lap_number"].max())
    result["compounds"] = sorted(features_df["compound"].dropna().unique().tolist())

    return result


@click.command()
@click.option(
    "--years", required=True, help="Comma-separated years, e.g. 2018,2019,2020"
)
@click.option(
    "--races",
    default=None,
    help="Comma-separated race names (optional, fetches all if not specified)",
)
@click.option("--resume", is_flag=True, help="Skip already-fetched year/race combos")
@click.option(
    "--workers",
    default=DEFAULT_WORKERS,
    show_default=True,
    type=click.IntRange(1, 8),
    help="Number of race fetches to run in parallel",
)
def fetch(years: str, races: str | None, resume: bool, workers: int) -> None:
    """Fetch historical F1 data and build the parquet feature store."""
    year_list = _parse_years(years)
    race_filter = _parse_races(races)

    if not year_list:
        console.print("[red]No valid years provided[/red]")
        raise SystemExit(1)

    console.print(f"[bold green]F1 Feature Store Builder[/bold green]")
    console.print(f"[dim]Years: {year_list}[/dim]")
    if race_filter:
        console.print(f"[dim]Races: {race_filter}[/dim]")
    if resume:
        console.print(f"[dim]Resume mode: skipping already-fetched combos[/dim]")
    console.print(f"[dim]Workers: {workers}[/dim]")
    console.print()

    all_results: list[dict[str, Any]] = []

    for year in year_list:
        console.print(f"[bold blue]═══ Year {year} ═══[/bold blue]")

        schedule = _get_race_schedule(year)
        if not schedule:
            console.print(f"[yellow]⚠ No schedule for {year}, skipping[/yellow]")
            continue

        # Filter races if specified
        if race_filter:
            races_to_fetch: list[tuple[str, str, int]] = []
            for user_race in race_filter:
                match = _match_race_name(user_race, schedule)
                if match:
                    event_name, round_num = match
                    races_to_fetch.append((user_race, event_name, round_num))
                else:
                    console.print(
                        f"[yellow]⚠ Could not match race '{user_race}' for {year}[/yellow]"
                    )
        else:
            races_to_fetch = [
                (
                    r["event_name"].lower().replace(" ", "_").replace("'", ""),
                    r["event_name"],
                    r["round"],
                )
                for r in schedule
                if r["round"] > 0  # Skip Pre-Season Testing
            ]

        pending_fetches: list[tuple[str, str, int]] = []
        for race_slug, canonical_name, round_num in races_to_fetch:
            if resume and _already_fetched(year, race_slug):
                console.print(f"[dim]  ⊘ {race_slug} already fetched, skipping[/dim]")
                all_results.append(
                    {
                        "year": year,
                        "race": race_slug,
                        "status": "SKIPPED_EXISTS",
                        "rows": 0,
                        "drivers": 0,
                        "laps": 0,
                        "compounds": [],
                        "error": None,
                    }
                )
                continue
            pending_fetches.append((race_slug, canonical_name, round_num))

        if workers == 1:
            for race_slug, canonical_name, round_num in pending_fetches:
                try:
                    result = _fetch_and_build_race(
                        year, canonical_name, race_slug, round_num
                    )
                except Exception as err:
                    result = {
                        "year": year,
                        "race": race_slug,
                        "status": "ERROR",
                        "rows": 0,
                        "drivers": 0,
                        "laps": 0,
                        "compounds": [],
                        "error": str(err),
                    }
                all_results.append(result)
                if result["status"] == "OK":
                    compounds_str = (
                        ", ".join(result["compounds"]) if result["compounds"] else "N/A"
                    )
                    console.print(
                        f"[green]✓[/green] {race_slug}: {result['rows']:,} rows, "
                        f"{result['drivers']} drivers, {result['laps']} laps "
                        f"[dim]({compounds_str})[/dim]"
                    )
                elif result["status"].startswith("SKIPPED"):
                    console.print(f"[yellow]⊘[/yellow] {race_slug}: {result['status']}")
                else:
                    console.print(f"[red]✗[/red] {race_slug}: {result['status']}")
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending_fetches) or 1)) as executor:
                future_map = {
                    executor.submit(_fetch_and_build_race, year, canonical_name, race_slug, round_num): race_slug
                    for race_slug, canonical_name, round_num in pending_fetches
                }
                for future in as_completed(future_map):
                    race_slug = future_map[future]
                    try:
                        result = future.result()
                    except Exception as err:
                        result = {
                            "year": year,
                            "race": race_slug,
                            "status": "ERROR",
                            "rows": 0,
                            "drivers": 0,
                            "laps": 0,
                            "compounds": [],
                            "error": str(err),
                        }
                    all_results.append(result)
                    if result["status"] == "OK":
                        compounds_str = (
                            ", ".join(result["compounds"]) if result["compounds"] else "N/A"
                        )
                        console.print(
                            f"[green]✓[/green] {race_slug}: {result['rows']:,} rows, "
                            f"{result['drivers']} drivers, {result['laps']} laps "
                            f"[dim]({compounds_str})[/dim]"
                        )
                    elif result["status"].startswith("SKIPPED"):
                        console.print(f"[yellow]⊘[/yellow] {race_slug}: {result['status']}")
                    else:
                        console.print(f"[red]✗[/red] {race_slug}: {result['status']}")

        console.print()

    # Summary table
    console.print(f"[bold green]═══ Summary ═══[/bold green]")
    console.print()

    from rich.table import Table as RichTable

    table = RichTable(title="Feature Store Build Results")
    table.add_column("Year", style="cyan", justify="center")
    table.add_column("Race", style="green")
    table.add_column("Rows", justify="right")
    table.add_column("Drivers", justify="center")
    table.add_column("Laps", justify="center")
    table.add_column("Status", justify="center")

    ok_count = 0
    total_rows = 0

    for r in all_results:
        status_style = {
            "OK": "green",
            "SKIPPED_EXISTS": "dim",
            "SKIPPED_NO_DATA": "yellow",
            "ERROR": "red",
        }.get(r["status"], "white")

        table.add_row(
            str(r["year"]),
            r["race"],
            f"{r['rows']:,}" if r["rows"] else "-",
            str(r["drivers"]) if r["drivers"] else "-",
            str(r["laps"]) if r["laps"] else "-",
            f"[{status_style}]{r['status']}[/{status_style}]",
        )

        if r["status"] == "OK":
            ok_count += 1
            total_rows += r["rows"]

    console.print(table)
    console.print()
    console.print(
        f"[bold]Total:[/bold] {ok_count} races fetched, {total_rows:,} rows in feature store"
    )


if __name__ == "__main__":
    fetch()
