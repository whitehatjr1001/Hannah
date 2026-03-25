"""Shared wrapper prompt builders for the Hannah CLI."""

from __future__ import annotations


def build_ask_intent(question: str) -> str:
    """Preserve direct freeform asks as-is."""
    return question


def build_simulate_intent(
    *,
    race: str,
    year: int,
    driver: str | None,
    laps: int,
    weather: str,
) -> str:
    return (
        f"Run a race simulation for {race} {year}. "
        f"Driver: {driver or 'all'}. Laps: {laps}. Weather: {weather}."
    )


def build_predict_intent(*, race: str, year: int) -> str:
    return (
        f"Predict the winner for the {race} Grand Prix {year}. "
        "Fetch current qualifying and historical data, run the winner ensemble model, "
        "and give me the podium probabilities."
    )


def build_strategy_intent(
    *,
    race: str,
    lap: int,
    driver: str,
    strategy_type: str,
) -> str:
    return (
        f"Strategy call for {driver} at {race}, lap {lap}. "
        f"Strategy type requested: {strategy_type}. "
        "Check tyre state, competitor positions, and give me a decisive call."
    )
