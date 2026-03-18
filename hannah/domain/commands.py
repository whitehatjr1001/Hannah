"""Structured race strategy commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StrategyAction = Literal["pit", "push", "normal", "conserve", "stay_out", "pit_cancel", "nothing"]
TyreCompound = Literal["SOFT", "MEDIUM", "HARD", "INTER", "WET"]
DrivingStyle = Literal["push", "normal", "conserve"]

COMPOUND_ALIASES = {
    "s": "SOFT",
    "soft": "SOFT",
    "m": "MEDIUM",
    "medium": "MEDIUM",
    "h": "HARD",
    "hard": "HARD",
    "inter": "INTER",
    "intermediate": "INTER",
    "wet": "WET",
}


@dataclass(frozen=True)
class StrategyCommand:
    driver: str
    action: StrategyAction
    compound: TyreCompound | None = None
    note: str | None = None


def parse_commands(raw: str) -> list[StrategyCommand]:
    """Parse a semicolon-delimited command list."""
    commands = [parse_command(chunk) for chunk in raw.split(";") if chunk.strip()]
    if not commands:
        raise ValueError("no commands found")
    return commands


def parse_command(raw: str) -> StrategyCommand:
    """Parse a FormulaGPT-style command into a typed object."""
    parts = raw.strip().replace(",", " ").replace(";", "").split()
    if len(parts) < 2:
        raise ValueError(f"invalid strategy command: {raw}")

    driver = parts[0].upper()
    action = parts[1].lower()
    remaining = [part.lower() for part in parts[2:]]

    if action == "pit":
        if remaining and remaining[0] in {"cancel", "abort"}:
            return StrategyCommand(driver=driver, action="pit_cancel", note="cancel pit call")
        compound = _normalise_compound(parts[2] if len(parts) > 2 else "MEDIUM")
        return StrategyCommand(driver=driver, action="pit", compound=compound)

    if action in {"cancel", "abort"} and remaining and remaining[0] == "pit":
        return StrategyCommand(driver=driver, action="pit_cancel", note="cancel pit call")

    if action == "pace":
        if not remaining:
            raise ValueError(f"pace command missing driving style: {raw}")
        action = remaining[0]

    if action == "nothing":
        return StrategyCommand(driver=driver, action="nothing")

    if action == "stay_out":
        return StrategyCommand(driver=driver, action="stay_out")

    if action == "stay":
        if not remaining or remaining[0] != "out":
            raise ValueError(f"unsupported action: {' '.join(parts[1:])}")
        return StrategyCommand(driver=driver, action="stay_out", note="hold track position")

    if action not in {"push", "normal", "conserve"}:
        raise ValueError(f"unsupported action: {action}")
    return StrategyCommand(driver=driver, action=action)


def _normalise_compound(raw: str) -> TyreCompound:
    compound = COMPOUND_ALIASES.get(raw.strip().lower())
    if compound is None:
        raise ValueError(f"unsupported pit compound: {raw}")
    return compound  # type: ignore[return-value]
