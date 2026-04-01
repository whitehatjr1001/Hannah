from __future__ import annotations

import asyncio

from hannah.tools.predict_winner import tool


def test_predict_winner_prefers_resolved_roster_over_static_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_race_data_run(**kwargs) -> dict:
        return {
            "drivers": ["VER", "NOR", "LEC"],
            "session_info": {
                "race": "monaco",
                "year": 2026,
                "resolved_roster": ["HAM", "ALO", "STR"],
            },
        }

    def _fake_load_and_predict(payload) -> dict:
        captured["payload"] = payload
        return {"HAM": 0.5, "ALO": 0.3, "STR": 0.2}

    monkeypatch.setattr("hannah.tools.race_data.tool.run", _fake_race_data_run)
    monkeypatch.setattr("hannah.tools.predict_winner.tool.load_and_predict", _fake_load_and_predict)

    result = asyncio.run(tool.run(race="monaco", year=2026))

    assert captured["payload"] == {"race": "monaco", "year": 2026, "drivers": ["HAM", "ALO", "STR"]}
    assert result == {"winner_probs": {"HAM": 0.5, "ALO": 0.3, "STR": 0.2}}
