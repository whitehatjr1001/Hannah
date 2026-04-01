"""Tool wrapper for race data access."""

from __future__ import annotations

import asyncio
from typing import Any

from hannah._data_.fastf1_loader import fetch_session
from hannah._data_.openf1_client import OpenF1Client, should_enrich_from_openf1
from hannah._data_.season_roster_resolver import resolve_season_roster

SKILL = {
    "name": "race_data",
    "description": "Fetches F1 race data from FastF1 and OpenF1.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string", "description": "Race name e.g. bahrain"},
            "year": {"type": "integer", "description": "Season year"},
            "session": {"type": "string", "enum": ["R", "Q", "FP1", "FP2", "FP3"]},
            "driver": {"type": "string", "description": "Driver code e.g. VER"},
        },
        "required": ["race"],
    },
}


async def run(
    race: str,
    year: int = 2025,
    session: str = "R",
    driver: str | None = None,
) -> dict:
    """Fetch and merge race data from available sources."""
    if should_enrich_from_openf1(year):
        client = OpenF1Client()
        fastf1_task = asyncio.to_thread(fetch_session, race, year, session)
        sessions_task = asyncio.to_thread(client.get_sessions, year, race)
        fastf1_payload, openf1_sessions = await asyncio.gather(fastf1_task, sessions_task)
        session_key = _resolve_openf1_session_key(openf1_sessions, session)
        if session_key is not None:
            stints_task = asyncio.to_thread(_call_openf1_method, client, "get_stints", session_key)
            weather_task = asyncio.to_thread(_call_openf1_method, client, "get_weather", session_key)
            drivers_task = asyncio.to_thread(_call_openf1_method, client, "get_drivers", session_key)
            openf1_stints, openf1_weather, openf1_drivers = await asyncio.gather(
                stints_task,
                weather_task,
                drivers_task,
            )
        else:
            openf1_stints = []
            openf1_weather = []
            openf1_drivers = []
    else:
        fastf1_payload = await asyncio.to_thread(fetch_session, race, year, session)
        openf1_sessions = []
        openf1_stints = []
        openf1_weather = []
        openf1_drivers = []

    resolved_roster = resolve_season_roster(
        year,
        fastf1_payload=fastf1_payload,
        openf1_drivers=openf1_drivers,
    )
    roster_codes = resolved_roster.get("codes", [])
    drivers_payload = [driver] if driver else list(roster_codes) if isinstance(roster_codes, list) else []
    if not drivers_payload:
        drivers_payload = ["VER", "NOR", "LEC"]

    session_info = {
        "race": race,
        "year": year,
        "session": session,
        "driver": driver,
        "openf1_sessions": len(openf1_sessions),
        "laps": 57,
        "weather": "dry",
        "resolved_roster": resolved_roster,
    }
    return {
        "laps": fastf1_payload.get("laps", []),
        "stints": openf1_stints,
        "weather": openf1_weather or fastf1_payload.get("weather", []),
        "drivers": drivers_payload,
        "session_info": session_info,
    }


def _resolve_openf1_session_key(sessions: list[dict[str, Any]], session: str) -> int | None:
    target = _session_lookup_value(session)
    for candidate in sessions:
        if not isinstance(candidate, dict):
            continue
        session_key = candidate.get("session_key")
        if not isinstance(session_key, int):
            continue
        name_fields = (
            candidate.get("session_name"),
            candidate.get("session_type"),
            candidate.get("session_code"),
        )
        normalized = {str(value).strip().lower() for value in name_fields if value is not None}
        if target in normalized:
            return session_key
    for candidate in sessions:
        session_key = candidate.get("session_key")
        if isinstance(session_key, int):
            return session_key
    return None


def _session_lookup_value(session: str) -> str:
    return {
        "R": "race",
        "Q": "qualifying",
        "FP1": "practice 1",
        "FP2": "practice 2",
        "FP3": "practice 3",
    }.get(session, session).lower()


def _call_openf1_method(client: Any, method_name: str, session_key: int) -> list[dict[str, Any]]:
    method = getattr(client, method_name, None)
    if not callable(method):
        return []
    result = method(session_key)
    return result if isinstance(result, list) else []
