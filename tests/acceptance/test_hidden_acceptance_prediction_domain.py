"""Masked acceptance scenarios for prediction and domain contracts."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Final

import pytest

from hannah.agent.tool_registry import ToolRegistry
from hannah.domain.commands import StrategyCommand, parse_command
from hannah.domain.prompts import build_race_snapshot_prompt
from hannah.domain.race_state import RaceSnapshot

pytestmark = pytest.mark.filterwarnings("ignore:urllib3 v2 only supports OpenSSL 1.1.1+")


@dataclass(frozen=True)
class HiddenPredictionScenario:
    scenario_id: str
    race: str
    year: int
    drivers: list[str]
    expected: dict[str, float]


PREDICTION_SCENARIOS: Final[tuple[HiddenPredictionScenario, ...]] = (
    HiddenPredictionScenario(
        "HACC_P01",
        "bahrain",
        2025,
        ["VER", "NOR", "LEC"],
        {"VER": 0.5, "NOR": 0.3, "LEC": 0.2},
    ),
    HiddenPredictionScenario(
        "HACC_P02",
        "monaco",
        2024,
        ["LEC", "VER", "NOR"],
        {"LEC": 0.5, "VER": 0.3, "NOR": 0.2},
    ),
    HiddenPredictionScenario(
        "HACC_P03",
        "singapore",
        2025,
        ["NOR", "PIA"],
        {"NOR": 0.625, "PIA": 0.375},
    ),
    HiddenPredictionScenario(
        "HACC_P04",
        "silverstone",
        2023,
        ["HAM"],
        {"HAM": 1.0},
    ),
    HiddenPredictionScenario(
        "HACC_P05",
        "bahrain",
        2022,
        ["VER", "PER"],
        {"VER": 0.625, "PER": 0.375},
    ),
    HiddenPredictionScenario(
        "HACC_P06",
        "monaco",
        2025,
        ["ALO", "RUS", "HAM"],
        {"ALO": 0.5, "RUS": 0.3, "HAM": 0.2},
    ),
)

VALID_COMMANDS: Final[tuple[tuple[str, StrategyCommand], ...]] = (
    ("VER pit hard", StrategyCommand(driver="VER", action="pit", compound="HARD")),
    ("NOR push", StrategyCommand(driver="NOR", action="push", compound=None)),
    ("lec conserve", StrategyCommand(driver="LEC", action="conserve", compound=None)),
    ("HAM stay_out", StrategyCommand(driver="HAM", action="stay_out", compound=None)),
)

INVALID_COMMANDS: Final[tuple[str, ...]] = (
    "pit hard",
    "VER fly",
    "NOR",
    "",
)


def _run_tool(registry: ToolRegistry, name: str, args: dict) -> dict:
    return asyncio.run(registry.call(name, args))


@pytest.mark.parametrize(
    "scenario",
    PREDICTION_SCENARIOS,
    ids=[scenario.scenario_id for scenario in PREDICTION_SCENARIOS],
)
def test_hidden_prediction_probability_contract(scenario: HiddenPredictionScenario) -> None:
    registry = ToolRegistry()
    result = _run_tool(
        registry,
        "predict_winner",
        {"race": scenario.race, "year": scenario.year, "drivers": scenario.drivers},
    )

    assert set(result.keys()) == {"winner_probs"}
    winner_probs = result["winner_probs"]
    assert list(winner_probs.keys()) == scenario.drivers
    assert math.isclose(sum(winner_probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert winner_probs == scenario.expected


@pytest.mark.parametrize("raw, expected", VALID_COMMANDS)
def test_hidden_domain_command_parser_accepts_valid_shape(
    raw: str,
    expected: StrategyCommand,
) -> None:
    parsed = parse_command(raw)
    assert parsed == expected


@pytest.mark.parametrize("raw", INVALID_COMMANDS)
def test_hidden_domain_command_parser_rejects_invalid_shape(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_command(raw)


def test_hidden_race_snapshot_prompt_contract() -> None:
    snapshot = RaceSnapshot(
        race="monaco",
        year=2025,
        total_laps=78,
        current_lap=41,
        weather="mixed",
        drivers=["LEC", "VER", "NOR"],
        leader="LEC",
        compounds={"LEC": "HARD", "VER": "MEDIUM", "NOR": "HARD"},
        tyre_ages={"LEC": 17, "VER": 13, "NOR": 14},
        gaps={"LEC": 0.0, "VER": 1.4, "NOR": 4.0},
    )

    prompt = build_race_snapshot_prompt(snapshot)
    assert "Race: monaco 2025." in prompt
    assert "Lap 41/78." in prompt
    assert "Weather: mixed." in prompt
    assert "Drivers: LEC, VER, NOR." in prompt
    assert "Leader: LEC." in prompt
