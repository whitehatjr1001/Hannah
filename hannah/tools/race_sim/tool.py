"""Simulation tool wrapper."""

from __future__ import annotations

import hashlib
import json

SKILL = {
    "name": "race_sim",
    "description": "Runs the fast Monte Carlo simulation and returns strategy outputs.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string"},
            "year": {"type": "integer"},
            "weather": {"type": "string"},
            "drivers": {"type": "array", "items": {"type": "string"}},
            "laps": {"type": "integer"},
            "n_worlds": {"type": "integer"},
            "trace": {"type": "boolean"},
            "trace_checkpoints": {"type": "array", "items": {"type": "integer"}},
            "replay": {"type": "object"},
        },
        "required": ["race"],
    },
}


async def run(
    race: str,
    year: int = 2025,
    weather: str = "dry",
    drivers: list[str] | None = None,
    laps: int = 57,
    n_worlds: int = 1000,
    trace: bool = False,
    trace_checkpoints: list[int] | None = None,
    replay: dict | None = None,
) -> dict:
    """Run Monte Carlo and strategy heuristics."""
    from hannah.agent.context import RaceContext
    from hannah.simulation.monte_carlo import build_replay_trace, run_fast
    from hannah.simulation.strategy_engine import StrategyEngine

    selected_drivers = drivers or ["VER", "NOR", "LEC"]
    seed = _stable_seed(race=race, year=year, weather=weather, drivers=selected_drivers, laps=laps)
    state = RaceContext(
        race=race,
        year=year,
        laps=laps,
        drivers=selected_drivers,
        weather=weather,
    )
    from hannah.simulation.sandbox import RaceState

    race_state = RaceState.from_context(state)
    race_state.seed = seed
    sim_result = await run_fast(race_state, n_worlds=n_worlds)
    strategy = StrategyEngine().analyse(race_state, sim_result)
    result = {"simulation": sim_result.to_dict(), "strategy": strategy}
    if trace:
        replay_payload = replay
        if replay_payload is None:
            replay_payload = build_replay_trace(
                state=race_state,
                sim_result=sim_result,
                checkpoints=trace_checkpoints,
            )
        result["trace"] = {
            **replay_payload,
            "trace_id": _trace_id(replay_payload),
            "moments": list(replay_payload.get("timeline", [])),
            "replay": replay_payload,
        }
    return result


def _trace_id(replay_payload: dict) -> str:
    encoded = json.dumps(replay_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]
    return f"trace-{digest}"


def _stable_seed(race: str, year: int, weather: str, drivers: list[str], laps: int) -> int:
    payload = f"{race}|{year}|{weather}|{','.join(drivers)}|{laps}"
    return int(hashlib.md5(payload.encode("utf-8")).hexdigest()[:8], 16)


def _create_rng(np_module, seed: int):
    try:
        return np_module.random.default_rng(seed)
    except TypeError:
        return np_module.random.default_rng()
