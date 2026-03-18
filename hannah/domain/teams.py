"""Team and driver metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TeamStrategyStyle = Literal["aggressive", "balanced", "defensive", "opportunistic"]


@dataclass(frozen=True)
class TeamInfo:
    code: str
    team: str
    driver: str
    color: str
    teammate: str
    base_pace_delta: float
    tyre_management: float
    wet_weather_skill: float
    strategy_style: TeamStrategyStyle


DRIVER_GRID: dict[str, TeamInfo] = {
    "VER": TeamInfo(
        code="VER",
        team="Red Bull",
        driver="Max Verstappen",
        color="#1E5BC6",
        teammate="PER",
        base_pace_delta=-0.35,
        tyre_management=0.92,
        wet_weather_skill=0.95,
        strategy_style="aggressive",
    ),
    "PER": TeamInfo(
        code="PER",
        team="Red Bull",
        driver="Sergio Perez",
        color="#1E5BC6",
        teammate="VER",
        base_pace_delta=0.15,
        tyre_management=0.88,
        wet_weather_skill=0.82,
        strategy_style="balanced",
    ),
    "NOR": TeamInfo(
        code="NOR",
        team="McLaren",
        driver="Lando Norris",
        color="#FF8000",
        teammate="PIA",
        base_pace_delta=-0.22,
        tyre_management=0.89,
        wet_weather_skill=0.9,
        strategy_style="aggressive",
    ),
    "PIA": TeamInfo(
        code="PIA",
        team="McLaren",
        driver="Oscar Piastri",
        color="#FF8000",
        teammate="NOR",
        base_pace_delta=-0.1,
        tyre_management=0.9,
        wet_weather_skill=0.83,
        strategy_style="balanced",
    ),
    "LEC": TeamInfo(
        code="LEC",
        team="Ferrari",
        driver="Charles Leclerc",
        color="#DC0000",
        teammate="SAI",
        base_pace_delta=-0.18,
        tyre_management=0.85,
        wet_weather_skill=0.86,
        strategy_style="defensive",
    ),
    "SAI": TeamInfo(
        code="SAI",
        team="Ferrari",
        driver="Carlos Sainz",
        color="#DC0000",
        teammate="LEC",
        base_pace_delta=-0.05,
        tyre_management=0.87,
        wet_weather_skill=0.84,
        strategy_style="balanced",
    ),
    "HAM": TeamInfo(
        code="HAM",
        team="Mercedes",
        driver="Lewis Hamilton",
        color="#27F4D2",
        teammate="RUS",
        base_pace_delta=-0.12,
        tyre_management=0.91,
        wet_weather_skill=0.94,
        strategy_style="opportunistic",
    ),
    "RUS": TeamInfo(
        code="RUS",
        team="Mercedes",
        driver="George Russell",
        color="#27F4D2",
        teammate="HAM",
        base_pace_delta=-0.02,
        tyre_management=0.86,
        wet_weather_skill=0.82,
        strategy_style="balanced",
    ),
    "ALO": TeamInfo(
        code="ALO",
        team="Aston Martin",
        driver="Fernando Alonso",
        color="#229971",
        teammate="STR",
        base_pace_delta=0.22,
        tyre_management=0.93,
        wet_weather_skill=0.92,
        strategy_style="opportunistic",
    ),
    "STR": TeamInfo(
        code="STR",
        team="Aston Martin",
        driver="Lance Stroll",
        color="#229971",
        teammate="ALO",
        base_pace_delta=0.44,
        tyre_management=0.82,
        wet_weather_skill=0.78,
        strategy_style="defensive",
    ),
    "ALB": TeamInfo(
        code="ALB",
        team="Williams",
        driver="Alex Albon",
        color="#64C4FF",
        teammate="SAR",
        base_pace_delta=0.58,
        tyre_management=0.84,
        wet_weather_skill=0.79,
        strategy_style="opportunistic",
    ),
    "SAR": TeamInfo(
        code="SAR",
        team="Williams",
        driver="Logan Sargeant",
        color="#64C4FF",
        teammate="ALB",
        base_pace_delta=0.8,
        tyre_management=0.8,
        wet_weather_skill=0.72,
        strategy_style="defensive",
    ),
}

DEFAULT_DRIVER_ORDER: tuple[str, ...] = (
    "VER",
    "NOR",
    "LEC",
    "PIA",
    "HAM",
    "RUS",
    "SAI",
    "PER",
    "ALO",
    "ALB",
)

DRIVER_ALIASES: dict[str, str] = {
    "verstappen": "VER",
    "perez": "PER",
    "norris": "NOR",
    "piastri": "PIA",
    "leclerc": "LEC",
    "sainz": "SAI",
    "hamilton": "HAM",
    "russell": "RUS",
    "alonso": "ALO",
    "stroll": "STR",
    "albon": "ALB",
    "sargeant": "SAR",
}


def get_driver_info(code: str) -> TeamInfo:
    lookup = canonical_driver_code(code)
    if lookup not in DRIVER_GRID:
        raise ValueError(f"unknown driver code: {code}")
    return DRIVER_GRID[lookup]


def get_driver_codes(limit: int | None = None) -> list[str]:
    codes = list(DEFAULT_DRIVER_ORDER)
    return codes if limit is None else codes[:limit]


def get_team_drivers(team: str) -> list[str]:
    return [info.code for info in DRIVER_GRID.values() if info.team.lower() == team.lower()]


def get_primary_rivals(driver: str, field_size: int = 3) -> list[str]:
    driver_code = canonical_driver_code(driver)
    rivals = [code for code in DEFAULT_DRIVER_ORDER if code != driver_code]
    return rivals[:field_size]


def canonical_driver_code(driver: str) -> str:
    """Return a canonical three-letter driver code from a code or surname."""
    token = driver.strip()
    if not token:
        raise ValueError("driver cannot be empty")
    direct = token.upper()
    if direct in DRIVER_GRID:
        return direct
    return DRIVER_ALIASES.get(token.lower(), direct[:3])
