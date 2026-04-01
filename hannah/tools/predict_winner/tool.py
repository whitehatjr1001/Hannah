"""Winner prediction tool."""

from __future__ import annotations

from hannah.models.train_winner import load_and_predict

SKILL = {
    "name": "predict_winner",
    "description": "Predicts winner probabilities for the requested race.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string"},
            "year": {"type": "integer"},
            "drivers": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["race"],
    },
}


def _resolved_roster(race_data: dict | None, fallback: list[str] | None = None) -> list[str]:
    if isinstance(race_data, dict):
        session_info = race_data.get("session_info", {})
        roster = None
        if isinstance(session_info, dict):
            roster = session_info.get("resolved_roster")
        if not roster:
            roster = race_data.get("resolved_roster")
        if not roster:
            roster = race_data.get("drivers")
        if isinstance(roster, (list, tuple)):
            resolved = [str(driver) for driver in roster if str(driver)]
            if resolved:
                return resolved
    return list(fallback or [])


async def run(race: str, year: int = 2025, drivers: list[str] | None = None) -> dict:
    """Return winner probabilities from the saved or fallback model."""
    if drivers is None:
        from hannah.tools.race_data import tool as race_data_tool

        race_data = await race_data_tool.run(race=race, year=year, session="R")
        resolved_drivers = _resolved_roster(race_data, ["VER", "NOR", "LEC"])
    else:
        resolved_drivers = list(drivers)
    payload = {"race": race, "year": year, "drivers": resolved_drivers}
    return {"winner_probs": load_and_predict(payload)}
