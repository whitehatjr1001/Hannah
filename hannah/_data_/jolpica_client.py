"""Jolpica-F1 API client (Ergast successor) for historical F1 data from 2016 onwards."""

from __future__ import annotations

import time
from typing import Any

import requests

from hannah._data_.cache import JsonCache
from hannah.utils.console import Console

console = Console()
_JOLPICA_MIN_YEAR = 2016
_JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1/"
_JOLPICA_RATE_LIMIT_DELAY = 0.35
_REQUEST_TIMEOUT = 10


class JolpicaClient:
    """Cached Jolpica-F1 (Ergast successor) client for historical F1 data."""

    def __init__(
        self,
        base_url: str = _JOLPICA_BASE_URL,
        timeout: int = _REQUEST_TIMEOUT,
        rate_limit_delay: float = _JOLPICA_RATE_LIMIT_DELAY,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.cache = JsonCache()
        self._last_request_time: float = 0.0

    def get_race_results(self, year: int, round: int | None = None) -> list[dict]:
        """Fetch race results for a given year and optional round.

        Returns list of dicts with keys:
        year, round, race_name, circuit_id, driver_id, constructor,
        grid_position, final_position, laps_completed, status,
        points, fastest_lap, fastest_lap_speed
        """
        params: dict[str, Any] = {"year": year}
        if round is not None:
            params["round"] = round

        raw = self._get("results", params)
        return [
            _flatten_race_result(year, round, race)
            for race in raw
            for _ in race.get("Results", [])
        ]

    def get_lap_times(self, year: int, round: int, lap: int) -> list[dict]:
        """Fetch lap times for all drivers in a specific lap.

        Returns list of dicts with keys:
        year, round, race_name, circuit_id, driver_id, position,
        lap_number, lap_time_seconds
        """
        params: dict[str, Any] = {"year": year, "round": round, "lap": lap}
        raw = self._get("laps", params)
        rows: list[dict] = []
        for race in raw:
            race_name = race.get("raceName", "")
            circuit_id = race.get("Circuit", {}).get("circuitId", "")
            for result in race.get("Results", []):
                driver_id = result.get("Driver", {}).get("driverId", "")
                for timing in result.get("Timings", []):
                    rows.append(
                        {
                            "year": year,
                            "round": round,
                            "race_name": race_name,
                            "circuit_id": circuit_id,
                            "driver_id": driver_id,
                            "position": int(timing.get("position", 0)),
                            "lap_number": int(timing.get("number", lap)),
                            "lap_time_seconds": _parse_duration_to_seconds(
                                timing.get("time", "")
                            ),
                        }
                    )
        return rows

    def get_pit_stops(self, year: int, round: int) -> list[dict]:
        """Fetch pit stop data for a given year and round.

        Returns list of dicts with keys:
        year, round, race_name, driver_id, stop_number, lap_number,
        pit_time_seconds, pit_duration_seconds
        """
        params: dict[str, Any] = {"year": year, "round": round}
        raw = self._get("pitstops", params)
        rows: list[dict] = []
        for race in raw:
            race_name = race.get("raceName", "")
            for stop in race.get("PitStops", []):
                rows.append(
                    {
                        "year": year,
                        "round": round,
                        "race_name": race_name,
                        "driver_id": stop.get("driverId", ""),
                        "stop_number": int(stop.get("stop", 0)),
                        "lap_number": int(stop.get("lap", 0)),
                        "pit_time_seconds": _parse_time_of_day(stop.get("time", "")),
                        "pit_duration_seconds": _parse_duration_to_seconds(
                            stop.get("duration", "")
                        ),
                    }
                )
        return rows

    def get_qualifying(self, year: int, round: int | None = None) -> list[dict]:
        """Fetch qualifying results for a given year and optional round.

        Returns list of dicts with keys:
        year, round, race_name, driver_id, position, q1, q2, q3
        (Q1/Q2/Q3 are ISO 8601 durations parsed to seconds)
        """
        params: dict[str, Any] = {"year": year}
        if round is not None:
            params["round"] = round

        raw = self._get("qualifying", params)
        rows: list[dict] = []
        for race in raw:
            race_name = race.get("raceName", "")
            for result in race.get("QualifyingResults", []):
                rows.append(
                    {
                        "year": year,
                        "round": round or 0,
                        "race_name": race_name,
                        "driver_id": result.get("Driver", {}).get("driverId", ""),
                        "position": int(result.get("position", 0)),
                        "q1": _parse_duration_to_seconds(result.get("Q1", "")),
                        "q2": _parse_duration_to_seconds(result.get("Q2", "")),
                        "q3": _parse_duration_to_seconds(result.get("Q3", "")),
                    }
                )
        return rows

    def get_circuits(self, year: int) -> list[dict]:
        """Fetch circuit information for a given season.

        Returns list of dicts with keys:
        year, round, circuit_id, circuit_name, location, country,
        latitude, longitude
        """
        params: dict[str, Any] = {"year": year}
        raw = self._get("circuits", params)
        rows: list[dict] = []
        for race in raw:
            circuit = race.get("Circuit", {})
            rows.append(
                {
                    "year": year,
                    "round": int(race.get("round", 0)),
                    "circuit_id": circuit.get("circuitId", ""),
                    "circuit_name": circuit.get("circuitName", ""),
                    "location": circuit.get("Location", {}).get("locality", ""),
                    "country": circuit.get("Location", {}).get("country", ""),
                    "latitude": float(circuit.get("Location", {}).get("lat", 0)),
                    "longitude": float(circuit.get("Location", {}).get("long", 0)),
                }
            )
        return rows

    def get_constructor_standings(self, year: int) -> list[dict]:
        """Fetch constructor standings for a given year.

        Returns list of dicts with keys:
        year, position, constructor_id, constructor_name, points, wins
        """
        params: dict[str, Any] = {"year": year}
        raw = self._get("constructorStandings", params)
        rows: list[dict] = []
        for standing in raw:
            constructor = standing.get("Constructor", {})
            rows.append(
                {
                    "year": year,
                    "position": int(standing.get("position", 0)),
                    "constructor_id": constructor.get("constructorId", ""),
                    "constructor_name": constructor.get("name", ""),
                    "points": float(standing.get("points", 0)),
                    "wins": int(standing.get("wins", 0)),
                }
            )
        return rows

    def get_driver_standings(self, year: int) -> list[dict]:
        """Fetch driver standings for a given year.

        Returns list of dicts with keys:
        year, position, driver_id, driver_name, constructor, points, wins
        """
        params: dict[str, Any] = {"year": year}
        raw = self._get("driverStandings", params)
        rows: list[dict] = []
        for standing in raw:
            driver = standing.get("Driver", {})
            constructor = (
                standing.get("Constructors", [{}])[0]
                if standing.get("Constructors")
                else {}
            )
            rows.append(
                {
                    "year": year,
                    "position": int(standing.get("position", 0)),
                    "driver_id": driver.get("driverId", ""),
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "constructor": constructor.get("name", ""),
                    "points": float(standing.get("points", 0)),
                    "wins": int(standing.get("wins", 0)),
                }
            )
        return rows

    def get_season_schedule(self, year: int) -> list[dict]:
        """Fetch the full season schedule for a given year.

        Returns list of dicts with keys:
        year, round, race_name, circuit_id, circuit_name, location,
        country, date, time
        """
        params: dict[str, Any] = {"year": year}
        raw = self._get("schedule", params)
        rows: list[dict] = []
        for race in raw:
            circuit = race.get("Circuit", {})
            rows.append(
                {
                    "year": year,
                    "round": int(race.get("round", 0)),
                    "race_name": race.get("raceName", ""),
                    "circuit_id": circuit.get("circuitId", ""),
                    "circuit_name": circuit.get("circuitName", ""),
                    "location": circuit.get("Location", {}).get("locality", ""),
                    "country": circuit.get("Location", {}).get("country", ""),
                    "date": race.get("date", ""),
                    "time": race.get("time", ""),
                }
            )
        return rows

    def _get(self, endpoint: str, params: dict[str, Any]) -> list[dict]:
        """Internal cached GET request with rate limiting and Jolpica response parsing."""
        cached = self.cache.load(f"jolpica_{endpoint}", params)
        if cached is not None:
            return cached

        self._apply_rate_limit()

        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as err:
            console.print(f"[yellow]warning:[/yellow] Jolpica {endpoint} failed: {err}")
            return []

        self._last_request_time = time.monotonic()

        parsed = self._parse_jolpica_response(endpoint, payload)
        self.cache.save(f"jolpica_{endpoint}", params, parsed)
        return parsed

    def _apply_rate_limit(self) -> None:
        """Enforce rate limit delay between requests."""
        if self._last_request_time > 0:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)

    def _parse_jolpica_response(
        self, endpoint: str, payload: dict[str, Any]
    ) -> list[dict]:
        """Extract the relevant list from Jolpica's nested MRData structure."""
        try:
            mr_data = payload.get("MRData", {})
            if endpoint == "results":
                return mr_data.get("RaceTable", {}).get("Races", [])
            elif endpoint == "laps":
                return mr_data.get("RaceTable", {}).get("Races", [])
            elif endpoint == "pitstops":
                return mr_data.get("RaceTable", {}).get("Races", [])
            elif endpoint == "qualifying":
                return mr_data.get("RaceTable", {}).get("Races", [])
            elif endpoint == "circuits":
                return mr_data.get("RaceTable", {}).get("Races", [])
            elif endpoint == "constructorStandings":
                return (
                    mr_data.get("StandingsTable", {})
                    .get("StandingsLists", [{}])[0]
                    .get("ConstructorStandings", [])
                )
            elif endpoint == "driverStandings":
                return (
                    mr_data.get("StandingsTable", {})
                    .get("StandingsLists", [{}])[0]
                    .get("DriverStandings", [])
                )
            elif endpoint == "schedule":
                return mr_data.get("RaceTable", {}).get("Races", [])
            else:
                console.print(
                    f"[yellow]warning:[/yellow] Unknown Jolpica endpoint: {endpoint}"
                )
                return []
        except Exception as err:
            console.print(
                f"[yellow]warning:[/yellow] Failed to parse Jolpica response for {endpoint}: {err}"
            )
            return []


def _flatten_race_result(
    year: int, round_override: int | None, race: dict[str, Any]
) -> list[dict]:
    """Flatten a single race result into per-driver rows."""
    rows: list[dict] = []
    race_name = race.get("raceName", "")
    circuit_id = race.get("Circuit", {}).get("circuitId", "")
    round_num = round_override or int(race.get("round", 0))

    for result in race.get("Results", []):
        driver = result.get("Driver", {})
        constructor = result.get("Constructor", {})
        fastest_lap = result.get("FastestLap", {})
        avg_speed = fastest_lap.get("AverageSpeed", {})

        rows.append(
            {
                "year": year,
                "round": round_num,
                "race_name": race_name,
                "circuit_id": circuit_id,
                "driver_id": driver.get("driverId", ""),
                "constructor": constructor.get("name", ""),
                "grid_position": int(result.get("grid", 0)),
                "final_position": int(result.get("position", 0)),
                "laps_completed": int(result.get("laps", 0)),
                "status": result.get("status", ""),
                "points": float(result.get("points", 0)),
                "fastest_lap": fastest_lap.get("lap", ""),
                "fastest_lap_speed": float(avg_speed.get("speed", 0))
                if avg_speed.get("speed")
                else 0.0,
            }
        )
    return rows


def _parse_duration_to_seconds(duration: str) -> float | None:
    """Parse ISO 8601 duration (e.g., 'PT1M23.456S') to seconds."""
    if not duration or not isinstance(duration, str):
        return None
    try:
        import re

        match = re.match(r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", duration)
        if not match:
            return None
        minutes = int(match.group(1)) if match.group(1) else 0
        seconds = float(match.group(2)) if match.group(2) else 0.0
        return minutes * 60 + seconds
    except Exception:
        return None


def _parse_time_of_day(time_str: str) -> float | None:
    """Parse time of day (e.g., '14:32:45') to seconds since midnight."""
    if not time_str or not isinstance(time_str, str):
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return None
    except Exception:
        return None


def should_use_jolpica(year: int) -> bool:
    """Check if Jolpica should be used for a given year (2016 onwards)."""
    return year >= _JOLPICA_MIN_YEAR
