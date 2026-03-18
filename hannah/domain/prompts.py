"""Domain-specific prompt helpers."""

from __future__ import annotations

from hannah.domain.race_state import RaceSnapshot


def build_command_grammar() -> str:
    """Return the strategist command grammar Hannah understands."""
    return (
        "Commands: <DRIVER> pit soft|medium|hard|inter|wet; "
        "<DRIVER> pit cancel; <DRIVER> push; <DRIVER> normal; "
        "<DRIVER> conserve; <DRIVER> stay out."
    )


def build_team_scope_prompt(team: str, drivers: list[str]) -> str:
    """Build a narrow control scope for a team strategist."""
    return f"You control {team}. Only issue commands for: {', '.join(drivers)}."


def build_race_snapshot_prompt(snapshot: RaceSnapshot) -> str:
    """Serialize race state into a compact strategist-facing prompt."""
    standings = "; ".join(
        (
            f"P{driver.position} {driver.code} {driver.compound}"
            f" age {driver.tyre_age} gap {driver.gap_to_leader:+.1f}s"
        )
        for driver in snapshot.driver_states[:6]
    )
    telemetry = ", ".join(snapshot.telemetry) if snapshot.telemetry else "synthetic fallback"
    drivers = ", ".join(snapshot.drivers)
    return (
        f"Race: {snapshot.race} {snapshot.year}. "
        f"Lap {snapshot.current_lap}/{snapshot.total_laps}. "
        f"Weather: {snapshot.weather}. "
        f"Drivers: {drivers}. "
        f"Telemetry: {telemetry}. "
        f"Leader: {snapshot.leader or 'unknown'}. "
        f"Standings: {standings}."
    )


def build_strategist_prompt(snapshot: RaceSnapshot, team: str, drivers: list[str]) -> str:
    """Build the in-race strategist prompt used by sub-agents."""
    return " ".join(
        [
            build_team_scope_prompt(team, drivers),
            "The simulator owns state transitions. Use gaps, tyre ages, track position, and weather.",
            build_race_snapshot_prompt(snapshot),
            build_command_grammar(),
            "If the best call is to continue, issue stay out.",
        ]
    )
