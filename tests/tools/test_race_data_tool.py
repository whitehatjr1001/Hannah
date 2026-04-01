from __future__ import annotations

import asyncio
import json

from hannah.agent.loop import AgentLoop
from hannah.runtime.core import RuntimeCore
from hannah.tools.race_data import tool as race_data_tool


def test_race_data_tool_keeps_top_level_contract_and_nests_resolved_roster(monkeypatch) -> None:
    def _fake_fetch_session(race: str, year: int, session_type: str) -> dict:
        assert (race, year, session_type) == ("bahrain", 2025, "R")
        return {
            "laps": [{"Driver": "VER"}],
            "weather": [{"air_temp": 30.0}],
            "car_data": [],
            "results": [{"Abbreviation": "VER"}, {"Abbreviation": "NOR"}],
        }

    class _FakeOpenF1Client:
        def get_sessions(self, year: int, race_name: str) -> list[dict]:
            assert (year, race_name) == (2025, "bahrain")
            return [{"session_key": 1001, "session_name": "Race"}]

        def get_stints(self, session_key: int) -> list[dict]:
            assert session_key == 1001
            return [{"driver_number": 1, "lap_start": 1, "lap_end": 20}]

        def get_weather(self, session_key: int) -> list[dict]:
            assert session_key == 1001
            return [{"air_temperature": 31.0}]

        def get_drivers(self, session_key: int) -> list[dict]:
            assert session_key == 1001
            return [
                {"name_acronym": "VER", "full_name": "Max Verstappen", "team_name": "Red Bull Racing"},
                {"name_acronym": "NOR", "full_name": "Lando Norris", "team_name": "McLaren"},
            ]

    monkeypatch.setattr(race_data_tool, "fetch_session", _fake_fetch_session)
    monkeypatch.setattr(race_data_tool, "OpenF1Client", _FakeOpenF1Client)

    payload = asyncio.run(race_data_tool.run(race="bahrain", year=2025, session="R"))

    assert set(payload.keys()) == {
        "laps",
        "stints",
        "weather",
        "drivers",
        "session_info",
    }
    assert payload["drivers"] == ["VER", "NOR"]
    assert payload["stints"] == [{"driver_number": 1, "lap_start": 1, "lap_end": 20}]
    assert payload["weather"] == [{"air_temperature": 31.0}]
    assert payload["session_info"]["openf1_sessions"] == 1
    assert payload["session_info"]["resolved_roster"]["source"] == "fastf1"
    assert payload["session_info"]["resolved_roster"]["codes"] == ["VER", "NOR"]


def test_race_data_tool_skips_openf1_for_legacy_windows(monkeypatch) -> None:
    def _fake_fetch_session(race: str, year: int, session_type: str) -> dict:
        return {
            "laps": [],
            "weather": [{"air_temp": 24.0}],
            "car_data": [],
            "results": [{"Abbreviation": "ALO"}, {"Abbreviation": "VET"}],
        }

    class _ExplodingOpenF1Client:
        def __init__(self) -> None:
            raise AssertionError("OpenF1 should be gated off for historical windows")

    monkeypatch.setattr(race_data_tool, "fetch_session", _fake_fetch_session)
    monkeypatch.setattr(race_data_tool, "OpenF1Client", _ExplodingOpenF1Client)

    payload = asyncio.run(race_data_tool.run(race="bahrain", year=2021, session="R"))

    assert payload["stints"] == []
    assert payload["weather"] == [{"air_temp": 24.0}]
    assert payload["drivers"] == ["ALO", "VET"]
    assert payload["session_info"]["openf1_sessions"] == 0
    assert payload["session_info"]["resolved_roster"]["source"] == "fastf1"


def test_race_data_compaction_preserves_resolved_roster_summary() -> None:
    large_payload = {
        "laps": [{"lap_number": lap, "driver": "VER", "lap_time": 90.0} for lap in range(2500)],
        "stints": [],
        "weather": [{"air_temp": 30.0}],
        "drivers": ["VER", "NOR"],
        "session_info": {"race": "bahrain", "year": 2025, "session": "R"},
        "resolved_roster": {
            "season": 2025,
            "source": "openf1",
            "count": 2,
            "codes": ["VER", "NOR"],
            "drivers": [
                {"code": "VER", "name": "Max Verstappen", "team": "Red Bull Racing"},
                {"code": "NOR", "name": "Lando Norris", "team": "McLaren"},
            ],
        },
    }

    loop_summary = AgentLoop()._compact_tool_payload(large_payload, tool_name="race_data")
    runtime_summary = RuntimeCore(provider=object(), registry=object(), allow_spawn_tool=False)._compact_tool_payload(
        large_payload,
        tool_name="race_data",
    )

    assert loop_summary["resolved_roster"] == {
        "season": 2025,
        "source": "openf1",
        "count": 2,
        "codes": ["VER", "NOR"],
    }
    assert runtime_summary["resolved_roster"] == loop_summary["resolved_roster"]
    assert "laps" not in loop_summary
    assert "laps" not in runtime_summary


def test_race_data_tool_keeps_explicit_driver_contract(monkeypatch) -> None:
    def _fake_fetch_session(race: str, year: int, session_type: str) -> dict:
        return {
            "laps": [{"Driver": "VER"}],
            "weather": [],
            "car_data": [],
            "results": [{"Abbreviation": "VER"}, {"Abbreviation": "NOR"}],
        }

    class _FakeOpenF1Client:
        def get_sessions(self, year: int, race_name: str) -> list[dict]:
            return []

    monkeypatch.setattr(race_data_tool, "fetch_session", _fake_fetch_session)
    monkeypatch.setattr(race_data_tool, "OpenF1Client", _FakeOpenF1Client)

    payload = asyncio.run(race_data_tool.run(race="bahrain", year=2025, session="R", driver="VER"))

    assert payload["drivers"] == ["VER"]
    assert payload["session_info"]["resolved_roster"]["codes"] == ["VER", "NOR"]
