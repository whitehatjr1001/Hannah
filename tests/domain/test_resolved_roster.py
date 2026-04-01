"""Resolved roster domain contracts."""

from __future__ import annotations

import pytest

from hannah.domain.prompts import build_race_snapshot_prompt, build_team_strategist_persona
from hannah.domain.race_state import RaceSnapshot
from hannah.domain.resolved_roster import ResolvedDriverProfile, ResolvedRoster


def _profile(
    code: str,
    *,
    driver: str,
    team: str,
    teammate: str,
) -> ResolvedDriverProfile:
    return ResolvedDriverProfile(
        code=code,
        driver=driver,
        team=team,
        teammate=teammate,
        color="#123456",
        base_pace_delta=0.0,
        tyre_management=0.9,
        wet_weather_skill=0.9,
        strategy_style="balanced",
    )


def test_resolved_roster_supports_case_insensitive_profile_lookup() -> None:
    roster = ResolvedRoster(
        year=2009,
        source="historical_seed",
        drivers=(
            _profile("VET", driver="Sebastian Vettel", team="Red Bull Racing", teammate="WEB"),
            _profile("WEB", driver="Mark Webber", team="Red Bull Racing", teammate="VET"),
        ),
    )

    assert roster.driver_codes() == ["VET", "WEB"]
    assert roster.get("vet").driver == "Sebastian Vettel"
    assert roster.get("WEB").teammate == "VET"


def test_resolved_roster_raises_for_unknown_driver() -> None:
    roster = ResolvedRoster(
        year=2012,
        source="historical_seed",
        drivers=(
            _profile("ALO", driver="Fernando Alonso", team="Ferrari", teammate="MAS"),
        ),
    )

    with pytest.raises(ValueError, match="unknown driver code"):
        roster.get("VET")


def test_resolved_roster_prompt_lines_include_team_and_driver_names() -> None:
    roster = ResolvedRoster(
        year=1998,
        source="historical_seed",
        drivers=(
            _profile("MSC", driver="Michael Schumacher", team="Ferrari", teammate="IRV"),
            _profile("IRV", driver="Eddie Irvine", team="Ferrari", teammate="MSC"),
        ),
    )

    assert roster.to_prompt_lines() == [
        "Resolved roster source: historical_seed (1998).",
        "- MSC: Michael Schumacher, Ferrari teammate IRV",
        "- IRV: Eddie Irvine, Ferrari teammate MSC",
    ]


def test_team_strategist_persona_falls_back_when_partial_roster_omits_teammate() -> None:
    roster = ResolvedRoster(
        year=2012,
        source="historical_seed",
        drivers=(
            _profile("ALO", driver="Fernando Alonso", team="Ferrari", teammate="MAS"),
        ),
    )

    persona = build_team_strategist_persona("ALO", resolved_roster=roster)

    assert "Fernando Alonso (ALO)" in persona
    assert "sister car" in persona
    assert "MAS" in persona


def test_race_snapshot_synthesizes_standings_and_gaps_from_position_order() -> None:
    snapshot = RaceSnapshot(
        race="monza",
        year=2025,
        total_laps=53,
        drivers=["NOR", "VER", "LEC"],
        positions={"VER": 1, "LEC": 2, "NOR": 3},
    )

    assert [driver.code for driver in snapshot.driver_states] == ["VER", "LEC", "NOR"]
    assert [driver.gap_to_leader for driver in snapshot.driver_states] == [0.0, 1.8, 3.6]

    prompt = build_race_snapshot_prompt(snapshot)
    assert "Standings: P1 VER" in prompt
    assert "P2 LEC" in prompt
    assert "P3 NOR" in prompt


def test_projected_pit_rejoin_rejects_unknown_driver() -> None:
    snapshot = RaceSnapshot(
        race="monza",
        year=2025,
        total_laps=53,
        drivers=["VER", "NOR", "LEC"],
    )

    with pytest.raises(ValueError, match="unknown driver code"):
        snapshot.projected_pit_rejoin("XYZ")
