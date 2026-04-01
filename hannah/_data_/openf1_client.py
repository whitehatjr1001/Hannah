"""OpenF1 REST client with local caching."""

from __future__ import annotations

from typing import Any

import requests

from hannah._data_.cache import JsonCache
from hannah.utils.console import Console

console = Console()


class OpenF1Client:
    """Minimal cached OpenF1 client."""

    def __init__(self, base_url: str = "https://api.openf1.org/v1", timeout: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache = JsonCache()

    def get_sessions(self, year: int, race_name: str) -> list[dict]:
        sessions: list[dict] = []
        seen_session_keys: set[int] = set()
        for meeting in self.get_meetings(year, race_name):
            meeting_key = meeting.get("meeting_key")
            if not isinstance(meeting_key, int):
                continue
            for session in self._get("sessions", {"meeting_key": meeting_key}):
                session_key = session.get("session_key")
                if isinstance(session_key, int):
                    if session_key in seen_session_keys:
                        continue
                    seen_session_keys.add(session_key)
                sessions.append(session)
        return sessions

    def get_meetings(self, year: int, race_name: str | None = None) -> list[dict]:
        meetings = self._get("meetings", {"year": year})
        if not race_name:
            return meetings

        lookup = _normalise_lookup(race_name)
        return [meeting for meeting in meetings if _meeting_matches(meeting, lookup)]

    def get_laps(self, session_key: int, driver_number: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return self._get("laps", params)

    def get_stints(self, session_key: int) -> list[dict]:
        return self._get("stints", {"session_key": session_key})

    def get_weather(self, session_key: int) -> list[dict]:
        return self._get("weather", {"session_key": session_key})

    def get_drivers(self, session_key: int) -> list[dict]:
        return self._get("drivers", {"session_key": session_key})

    def _get(self, endpoint: str, params: dict[str, Any]) -> list[dict]:
        cached = self.cache.load(f"openf1_{endpoint}", params)
        if cached is not None:
            return cached
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as err:
            console.print(f"[yellow]warning:[/yellow] OpenF1 {endpoint} failed: {err}")
            return []
        self.cache.save(f"openf1_{endpoint}", params, payload)
        return payload


def _meeting_matches(meeting: dict[str, Any], lookup: str) -> bool:
    candidate_fields = (
        meeting.get("meeting_name"),
        meeting.get("meeting_official_name"),
        meeting.get("country_name"),
        meeting.get("location"),
        meeting.get("circuit_short_name"),
    )
    lookup_tokens = set(lookup.split())
    for candidate in candidate_fields:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized_candidate = _normalise_lookup(candidate)
        if lookup == normalized_candidate or lookup in normalized_candidate:
            return True
        candidate_tokens = set(normalized_candidate.split())
        if lookup_tokens and lookup_tokens <= candidate_tokens:
            return True
    return False


def _normalise_lookup(value: str) -> str:
    normalized = value.lower().replace("_", " ").replace("-", " ")
    for token in ("grand prix", "formula 1", "fia", "world championship"):
        normalized = normalized.replace(token, " ")
    return " ".join(normalized.split())
