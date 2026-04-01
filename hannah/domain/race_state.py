"""Shared race-state DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hannah.domain.resolved_roster import ResolvedRoster
from hannah.domain.teams import get_driver_info
from hannah.domain.tracks import get_track

WeatherState = Literal["dry", "mixed", "wet"]


@dataclass(frozen=True)
class DriverSnapshot:
    code: str
    team: str
    position: int
    compound: str
    tyre_age: int
    gap_to_leader: float
    last_lap_time: float
    status: str = "Racing"

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "team": self.team,
            "position": self.position,
            "compound": self.compound,
            "tyre_age": self.tyre_age,
            "gap_to_leader": self.gap_to_leader,
            "last_lap_time": self.last_lap_time,
            "status": self.status,
        }


@dataclass(frozen=True)
class PitProjection:
    projected_position: int
    projected_gap: float
    car_ahead: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "projected_position": self.projected_position,
            "projected_gap": self.projected_gap,
            "car_ahead": self.car_ahead,
        }


@dataclass(frozen=True)
class RaceEventWindow:
    kind: str
    start_lap: int
    end_lap: int
    intensity: float = 1.0

    def contains(self, lap: int) -> bool:
        return self.start_lap <= lap <= self.end_lap

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start_lap": self.start_lap,
            "end_lap": self.end_lap,
            "intensity": self.intensity,
        }


@dataclass(frozen=True)
class RaceSnapshot:
    race: str
    year: int
    total_laps: int
    current_lap: int = 0
    weather: WeatherState = "dry"
    drivers: list[str] = field(default_factory=list)
    leader: str | None = None
    compounds: dict[str, str] = field(default_factory=dict)
    tyre_ages: dict[str, int] = field(default_factory=dict)
    gaps: dict[str, float] = field(default_factory=dict)
    positions: dict[str, int] = field(default_factory=dict)
    telemetry: tuple[str, ...] = field(default_factory=tuple)
    event_windows: tuple[RaceEventWindow, ...] = field(default_factory=tuple)
    driver_states: tuple[DriverSnapshot, ...] = field(default_factory=tuple)
    resolved_roster: ResolvedRoster | None = None

    def __post_init__(self) -> None:
        if self.driver_states:
            return
        synthesized: list[DriverSnapshot] = []
        ordered_codes = self._ordered_driver_codes()
        for index, code in enumerate(ordered_codes, start=1):
            profile = self.resolved_roster.get(code) if self.resolved_roster is not None else get_driver_info(code)
            position = self.positions.get(code, index)
            synthesized.append(
                DriverSnapshot(
                    code=code,
                    team=profile.team,
                    position=position,
                    compound=self.compounds.get(code, "MEDIUM"),
                    tyre_age=self.tyre_ages.get(code, 0),
                    gap_to_leader=self.gaps.get(code, float(max(position - 1, 0)) * 1.8),
                    last_lap_time=90.0 + profile.base_pace_delta,
                )
            )
        object.__setattr__(self, "driver_states", tuple(synthesized))
        if self.leader is None and synthesized:
            leader = min(synthesized, key=lambda driver: driver.position).code
            object.__setattr__(self, "leader", leader)

    def to_dict(self) -> dict:
        return {
            "race": self.race,
            "year": self.year,
            "total_laps": self.total_laps,
            "current_lap": self.current_lap,
            "weather": self.weather,
            "drivers": self.drivers,
            "leader": self.leader,
            "compounds": self.compounds,
            "tyre_ages": self.tyre_ages,
            "gaps": self.gaps,
            "positions": self.positions,
            "telemetry": list(self.telemetry),
            "event_windows": [event.to_dict() for event in self.event_windows],
            "driver_states": [driver.to_dict() for driver in self.driver_states],
            "resolved_roster": None if self.resolved_roster is None else self.resolved_roster.to_prompt_lines(),
        }

    def projected_pit_rejoin(self, driver: str, pit_loss: float | None = None) -> PitProjection:
        """Estimate where a driver would rejoin after a pit stop."""
        ordered = sorted(self.driver_states, key=lambda state: state.position)
        target = next((state for state in ordered if state.code == driver.upper()), None)
        if target is None:
            raise ValueError(f"unknown driver code: {driver}")

        delta = float(pit_loss if pit_loss is not None else get_track(self.race).pit_loss)
        projected_gap = target.gap_to_leader + delta
        ahead = [state for state in ordered if state.code != target.code and state.gap_to_leader < projected_gap]
        ahead_sorted = sorted(ahead, key=lambda state: state.gap_to_leader)
        car_ahead = ahead_sorted[-1].code if ahead_sorted else None
        return PitProjection(
            projected_position=len(ahead_sorted) + 1,
            projected_gap=projected_gap,
            car_ahead=car_ahead,
        )

    def _ordered_driver_codes(self) -> list[str]:
        unique_codes = list(dict.fromkeys(self.drivers))
        return sorted(unique_codes, key=lambda code: (self.positions.get(code, len(unique_codes) + 1), unique_codes.index(code)))
