"""Prompt builders for Hannah sub-agents."""

from __future__ import annotations

from hannah.agent.context import RaceContext


def build_strategy_prompt(ctx: RaceContext) -> str:
    """Create a concise strategist prompt from race context."""
    return (
        f"Race: {ctx.race} {ctx.year}, {ctx.laps} laps, {ctx.weather}. "
        f"Drivers: {', '.join(ctx.drivers)}. "
        f"Race data snapshot: {ctx.race_data or {}}."
    )

