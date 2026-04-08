"""FastF1 session loader."""

from __future__ import annotations

from pathlib import Path

from hannah.utils.console import Console

console = Console()

FASTF1_LOAD_KWARGS = {
    "laps": True,
    "telemetry": False,
    "weather": True,
    "messages": False,
}


def fetch_session(race: str, year: int, session_type: str) -> dict:
    """Fetch a FastF1 session and return JSON-serializable payloads."""
    cache_dir = Path("data/fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]fetching {race} {year} {session_type} via FastF1...[/dim]")

    try:
        import fastf1

        fastf1.Cache.enable_cache(str(cache_dir))
        session = fastf1.get_session(year, race, session_type)
        session.load(**FASTF1_LOAD_KWARGS)
        return {
            "laps": session.laps.to_dict(orient="records"),
            "weather": session.weather_data.to_dict(orient="records"),
            "car_data": [],
            "results": session.results.to_dict(orient="records"),
        }
    except Exception as err:
        return {
            "laps": [],
            "weather": [],
            "car_data": [],
            "results": [],
            "error": str(err),
        }
