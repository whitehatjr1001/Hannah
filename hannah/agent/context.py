"""Command and race context objects for the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentCommandContext:
    command: str
    race: str | None = None
    year: int | None = None
    driver: str | None = None
    laps: int | None = None
    weather: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RaceContext:
    race: str
    year: int
    laps: int
    weather: str
    drivers: list[str]
    race_data: dict | None = None

