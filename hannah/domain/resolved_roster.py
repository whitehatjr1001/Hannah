"""Resolved roster carriers shared across runtime and historical lookups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TeamStrategyStyle = Literal["aggressive", "balanced", "defensive", "opportunistic"]


@dataclass(frozen=True, slots=True)
class ResolvedDriverProfile:
    code: str
    driver: str
    team: str
    teammate: str
    color: str
    base_pace_delta: float
    tyre_management: float
    wet_weather_skill: float
    strategy_style: TeamStrategyStyle


@dataclass(frozen=True, slots=True)
class ResolvedRoster:
    drivers: tuple[ResolvedDriverProfile, ...]
    year: int | None = None
    source: str = "unknown"

    def get(self, code: str) -> ResolvedDriverProfile:
        lookup = code.strip().upper()
        for profile in self.drivers:
            if profile.code == lookup:
                return profile
        raise ValueError(f"unknown driver code: {code}")

    def driver_codes(self) -> list[str]:
        return [profile.code for profile in self.drivers]

    def to_prompt_lines(self) -> list[str]:
        season = f" ({self.year})" if self.year is not None else ""
        lines = [f"Resolved roster source: {self.source}{season}."]
        lines.extend(
            f"- {profile.code}: {profile.driver}, {profile.team} teammate {profile.teammate}"
            for profile in self.drivers
        )
        return lines
