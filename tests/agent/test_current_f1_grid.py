"""Current F1 grid and rival prompt coverage."""

from __future__ import annotations

from hannah.agent.subagents import RivalAgent
from hannah.agent.worker_registry import RIVAL_TEAM_PERSONAS
from hannah.domain.teams import (
    build_current_resolved_roster,
    canonical_driver_code,
    get_driver_codes,
    get_driver_info,
)


def test_2026_driver_grid_reflects_current_official_lineup() -> None:
    codes = get_driver_codes()

    assert len(codes) == 22
    assert {"VER", "HAD", "LEC", "HAM", "RUS", "ANT", "LAW", "LIN", "GAS", "COL", "PER", "BOT"} <= set(codes)

    assert get_driver_info("HAD").team == "Red Bull Racing"
    assert get_driver_info("HAD").teammate == "VER"
    assert get_driver_info("HAM").team == "Ferrari"
    assert get_driver_info("HAM").teammate == "LEC"
    assert get_driver_info("SAI").team == "Williams"
    assert get_driver_info("SAI").teammate == "ALB"
    assert get_driver_info("PER").team == "Cadillac"
    assert get_driver_info("PER").teammate == "BOT"
    assert get_driver_info("COL").team == "Alpine"
    assert get_driver_info("COL").teammate == "GAS"
    assert get_driver_info("BOR").team == "Audi"
    assert get_driver_info("BOR").teammate == "HUL"
    assert get_driver_info("ANT").team == "Mercedes"
    assert get_driver_info("ANT").teammate == "RUS"
    assert get_driver_info("LIN").team == "Racing Bulls"
    assert get_driver_info("LIN").teammate == "LAW"


def test_current_aliases_cover_modern_2026_rookies_and_returns() -> None:
    assert canonical_driver_code("antonelli") == "ANT"
    assert canonical_driver_code("hadjar") == "HAD"
    assert canonical_driver_code("lindblad") == "LIN"
    assert canonical_driver_code("bortoleto") == "BOR"
    assert canonical_driver_code("colapinto") == "COL"
    assert canonical_driver_code("hulkenberg") == "HUL"
    assert canonical_driver_code("bottas") == "BOT"
    assert canonical_driver_code("bearman") == "BEA"
    assert canonical_driver_code("ocon") == "OCO"


def test_rival_personas_cover_current_front_running_reference_teams() -> None:
    assert "Red Bull" in RivalAgent("HAD").persona
    assert "Ferrari" in RivalAgent("HAM").persona
    assert "Mercedes" in RivalAgent("RUS").persona
    assert "McLaren" in RivalAgent("NOR").persona
    assert "Cadillac" in RivalAgent("PER").persona

    assert "Red Bull" in RIVAL_TEAM_PERSONAS["HAD"]
    assert "Ferrari" in RIVAL_TEAM_PERSONAS["HAM"]
    assert "Mercedes" in RIVAL_TEAM_PERSONAS["RUS"]
    assert "McLaren" in RIVAL_TEAM_PERSONAS["NOR"]
    assert "Cadillac" in RIVAL_TEAM_PERSONAS["PER"]


def test_current_resolved_roster_matches_live_grid_helpers() -> None:
    roster = build_current_resolved_roster()

    assert roster.driver_codes() == get_driver_codes()
    assert roster.get("VER").team == get_driver_info("VER").team
    assert roster.get("HAD").teammate == get_driver_info("HAD").teammate
