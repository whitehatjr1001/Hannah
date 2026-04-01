"""Current team and driver metadata for the active F1 grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hannah.domain.resolved_roster import ResolvedDriverProfile, ResolvedRoster

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


@dataclass(frozen=True)
class TeamCatalogEntry:
    team: str
    color: str
    drivers: tuple[str, ...]


DRIVER_GRID: dict[str, TeamInfo] = {
    "RUS": TeamInfo(
        code="RUS",
        team="Mercedes",
        driver="George Russell",
        color="#27F4D2",
        teammate="ANT",
        base_pace_delta=-0.28,
        tyre_management=0.89,
        wet_weather_skill=0.84,
        strategy_style="balanced",
    ),
    "ANT": TeamInfo(
        code="ANT",
        team="Mercedes",
        driver="Kimi Antonelli",
        color="#27F4D2",
        teammate="RUS",
        base_pace_delta=-0.18,
        tyre_management=0.87,
        wet_weather_skill=0.82,
        strategy_style="aggressive",
    ),
    "LEC": TeamInfo(
        code="LEC",
        team="Ferrari",
        driver="Charles Leclerc",
        color="#DC0000",
        teammate="HAM",
        base_pace_delta=-0.22,
        tyre_management=0.87,
        wet_weather_skill=0.87,
        strategy_style="aggressive",
    ),
    "HAM": TeamInfo(
        code="HAM",
        team="Ferrari",
        driver="Lewis Hamilton",
        color="#DC0000",
        teammate="LEC",
        base_pace_delta=-0.16,
        tyre_management=0.92,
        wet_weather_skill=0.95,
        strategy_style="opportunistic",
    ),
    "NOR": TeamInfo(
        code="NOR",
        team="McLaren",
        driver="Lando Norris",
        color="#FF8000",
        teammate="PIA",
        base_pace_delta=-0.2,
        tyre_management=0.9,
        wet_weather_skill=0.91,
        strategy_style="aggressive",
    ),
    "PIA": TeamInfo(
        code="PIA",
        team="McLaren",
        driver="Oscar Piastri",
        color="#FF8000",
        teammate="NOR",
        base_pace_delta=-0.17,
        tyre_management=0.91,
        wet_weather_skill=0.85,
        strategy_style="balanced",
    ),
    "VER": TeamInfo(
        code="VER",
        team="Red Bull Racing",
        driver="Max Verstappen",
        color="#1E5BC6",
        teammate="HAD",
        base_pace_delta=-0.12,
        tyre_management=0.93,
        wet_weather_skill=0.97,
        strategy_style="aggressive",
    ),
    "HAD": TeamInfo(
        code="HAD",
        team="Red Bull Racing",
        driver="Isack Hadjar",
        color="#1E5BC6",
        teammate="VER",
        base_pace_delta=0.06,
        tyre_management=0.85,
        wet_weather_skill=0.79,
        strategy_style="aggressive",
    ),
    "OCO": TeamInfo(
        code="OCO",
        team="Haas F1 Team",
        driver="Esteban Ocon",
        color="#B6BABD",
        teammate="BEA",
        base_pace_delta=0.1,
        tyre_management=0.86,
        wet_weather_skill=0.83,
        strategy_style="balanced",
    ),
    "BEA": TeamInfo(
        code="BEA",
        team="Haas F1 Team",
        driver="Oliver Bearman",
        color="#B6BABD",
        teammate="OCO",
        base_pace_delta=0.08,
        tyre_management=0.84,
        wet_weather_skill=0.78,
        strategy_style="aggressive",
    ),
    "GAS": TeamInfo(
        code="GAS",
        team="Alpine",
        driver="Pierre Gasly",
        color="#0090FF",
        teammate="COL",
        base_pace_delta=0.14,
        tyre_management=0.86,
        wet_weather_skill=0.85,
        strategy_style="opportunistic",
    ),
    "COL": TeamInfo(
        code="COL",
        team="Alpine",
        driver="Franco Colapinto",
        color="#0090FF",
        teammate="GAS",
        base_pace_delta=0.26,
        tyre_management=0.82,
        wet_weather_skill=0.77,
        strategy_style="balanced",
    ),
    "LAW": TeamInfo(
        code="LAW",
        team="Racing Bulls",
        driver="Liam Lawson",
        color="#6692FF",
        teammate="LIN",
        base_pace_delta=0.2,
        tyre_management=0.84,
        wet_weather_skill=0.81,
        strategy_style="aggressive",
    ),
    "LIN": TeamInfo(
        code="LIN",
        team="Racing Bulls",
        driver="Arvid Lindblad",
        color="#6692FF",
        teammate="LAW",
        base_pace_delta=0.3,
        tyre_management=0.8,
        wet_weather_skill=0.74,
        strategy_style="aggressive",
    ),
    "HUL": TeamInfo(
        code="HUL",
        team="Audi",
        driver="Nico Hulkenberg",
        color="#C0C0C0",
        teammate="BOR",
        base_pace_delta=0.32,
        tyre_management=0.87,
        wet_weather_skill=0.82,
        strategy_style="balanced",
    ),
    "BOR": TeamInfo(
        code="BOR",
        team="Audi",
        driver="Gabriel Bortoleto",
        color="#C0C0C0",
        teammate="HUL",
        base_pace_delta=0.38,
        tyre_management=0.81,
        wet_weather_skill=0.76,
        strategy_style="balanced",
    ),
    "SAI": TeamInfo(
        code="SAI",
        team="Williams",
        driver="Carlos Sainz",
        color="#64C4FF",
        teammate="ALB",
        base_pace_delta=0.3,
        tyre_management=0.88,
        wet_weather_skill=0.84,
        strategy_style="balanced",
    ),
    "ALB": TeamInfo(
        code="ALB",
        team="Williams",
        driver="Alexander Albon",
        color="#64C4FF",
        teammate="SAI",
        base_pace_delta=0.36,
        tyre_management=0.86,
        wet_weather_skill=0.81,
        strategy_style="opportunistic",
    ),
    "PER": TeamInfo(
        code="PER",
        team="Cadillac",
        driver="Sergio Perez",
        color="#0B3A6E",
        teammate="BOT",
        base_pace_delta=0.48,
        tyre_management=0.87,
        wet_weather_skill=0.83,
        strategy_style="opportunistic",
    ),
    "BOT": TeamInfo(
        code="BOT",
        team="Cadillac",
        driver="Valtteri Bottas",
        color="#0B3A6E",
        teammate="PER",
        base_pace_delta=0.44,
        tyre_management=0.88,
        wet_weather_skill=0.81,
        strategy_style="balanced",
    ),
    "ALO": TeamInfo(
        code="ALO",
        team="Aston Martin",
        driver="Fernando Alonso",
        color="#229971",
        teammate="STR",
        base_pace_delta=0.54,
        tyre_management=0.94,
        wet_weather_skill=0.93,
        strategy_style="opportunistic",
    ),
    "STR": TeamInfo(
        code="STR",
        team="Aston Martin",
        driver="Lance Stroll",
        color="#229971",
        teammate="ALO",
        base_pace_delta=0.68,
        tyre_management=0.81,
        wet_weather_skill=0.78,
        strategy_style="defensive",
    ),
}

DEFAULT_DRIVER_ORDER: tuple[str, ...] = (
    "RUS",
    "ANT",
    "LEC",
    "HAM",
    "NOR",
    "PIA",
    "VER",
    "HAD",
    "OCO",
    "BEA",
    "GAS",
    "COL",
    "LAW",
    "LIN",
    "HUL",
    "BOR",
    "SAI",
    "ALB",
    "PER",
    "BOT",
    "ALO",
    "STR",
)

DRIVER_ALIASES: dict[str, str] = {
    "albon": "ALB",
    "alex": "ALB",
    "alonso": "ALO",
    "antonelli": "ANT",
    "arvid": "LIN",
    "bearman": "BEA",
    "bortoleto": "BOR",
    "bottas": "BOT",
    "carlos": "SAI",
    "charles": "LEC",
    "colapinto": "COL",
    "esteban": "OCO",
    "fernando": "ALO",
    "franco": "COL",
    "gabriel": "BOR",
    "gasly": "GAS",
    "george": "RUS",
    "hadjar": "HAD",
    "hamilton": "HAM",
    "hulkenberg": "HUL",
    "isack": "HAD",
    "kimi": "ANT",
    "lando": "NOR",
    "lawson": "LAW",
    "leclerc": "LEC",
    "liam": "LAW",
    "lindblad": "LIN",
    "max": "VER",
    "nico": "HUL",
    "norris": "NOR",
    "ocon": "OCO",
    "oliver": "BEA",
    "ollie": "BEA",
    "oscar": "PIA",
    "perez": "PER",
    "piastri": "PIA",
    "pierre": "GAS",
    "russell": "RUS",
    "sainz": "SAI",
    "sergio": "PER",
    "stroll": "STR",
    "valtteri": "BOT",
    "verstappen": "VER",
}

TEAM_ALIASES: dict[str, str] = {
    "aston martin": "Aston Martin",
    "audi": "Audi",
    "cadillac": "Cadillac",
    "ferrari": "Ferrari",
    "haas": "Haas F1 Team",
    "haas f1 team": "Haas F1 Team",
    "mclaren": "McLaren",
    "mercedes": "Mercedes",
    "racing bulls": "Racing Bulls",
    "red bull": "Red Bull Racing",
    "red bull racing": "Red Bull Racing",
    "williams": "Williams",
    "alpine": "Alpine",
}


def build_current_resolved_roster(codes: list[str] | tuple[str, ...] | None = None) -> ResolvedRoster:
    selected_codes = tuple(codes) if codes is not None else DEFAULT_DRIVER_ORDER
    return ResolvedRoster(
        year=2026,
        source="current_f1_2026_fallback",
        drivers=tuple(_to_resolved_driver_profile(DRIVER_GRID[code]) for code in selected_codes),
    )


def get_team_catalog() -> dict[str, TeamCatalogEntry]:
    catalog: dict[str, TeamCatalogEntry] = {}
    for team in dict.fromkeys(info.team for info in DRIVER_GRID.values()):
        drivers = tuple(code for code in DEFAULT_DRIVER_ORDER if DRIVER_GRID[code].team == team)
        catalog[team] = TeamCatalogEntry(
            team=team,
            color=DRIVER_GRID[drivers[0]].color,
            drivers=drivers,
        )
    return catalog


def canonical_team_name(team: str) -> str:
    token = team.strip()
    if not token:
        raise ValueError("team cannot be empty")
    return TEAM_ALIASES.get(token.lower(), token)


def get_driver_info(code: str) -> TeamInfo:
    lookup = canonical_driver_code(code)
    roster = build_current_resolved_roster()
    if lookup not in roster.driver_codes():
        raise ValueError(f"unknown driver code: {code}")
    return DRIVER_GRID[lookup]


def get_driver_codes(limit: int | None = None) -> list[str]:
    codes = build_current_resolved_roster().driver_codes()
    return codes if limit is None else codes[:limit]


def get_team_drivers(team: str) -> list[str]:
    canonical = canonical_team_name(team).lower()
    return [info.code for info in DRIVER_GRID.values() if info.team.lower() == canonical]


def get_primary_rivals(driver: str, field_size: int = 3) -> list[str]:
    driver_code = canonical_driver_code(driver)
    rivals = [code for code in DEFAULT_DRIVER_ORDER if code != driver_code]
    return rivals[:field_size]


def canonical_driver_code(driver: str) -> str:
    """Return a canonical three-letter driver code from a code or known name."""
    token = driver.strip()
    if not token:
        raise ValueError("driver cannot be empty")
    direct = token.upper()
    if direct in DRIVER_GRID:
        return direct
    alias = DRIVER_ALIASES.get(token.lower())
    if alias is not None:
        return alias
    raise ValueError(f"unknown driver: {driver}")


def _to_resolved_driver_profile(info: TeamInfo) -> ResolvedDriverProfile:
    return ResolvedDriverProfile(
        code=info.code,
        driver=info.driver,
        team=info.team,
        teammate=info.teammate,
        color=info.color,
        base_pace_delta=info.base_pace_delta,
        tyre_management=info.tyre_management,
        wet_weather_skill=info.wet_weather_skill,
        strategy_style=info.strategy_style,
    )
