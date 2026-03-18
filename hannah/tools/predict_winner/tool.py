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


async def run(race: str, year: int = 2025, drivers: list[str] | None = None) -> dict:
    """Return winner probabilities from the saved or fallback model."""
    payload = {"race": race, "year": year, "drivers": drivers or ["VER", "NOR", "LEC"]}
    return {"winner_probs": load_and_predict(payload)}

