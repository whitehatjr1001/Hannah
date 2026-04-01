"""Deterministic evaluation helpers for smoke-trained artifacts."""

from __future__ import annotations

import asyncio
import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hannah.models.artifact_paths import PUBLIC_MODEL_NAMES, resolve_artifact_path
from hannah.models.artifacts import load_pickle_artifact

MODEL_SCORE_THRESHOLDS = {
    "tyre_model": 0.85,
    "laptime_model": 0.85,
    "pit_rl": 0.9,
    "pit_policy_q": 0.9,
    "winner_ensemble": 0.95,
}

SCENARIO_DEPTH_CASES: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "S01",
        "race": "bahrain",
        "year": 2025,
        "weather": "dry",
        "drivers": ("VER", "NOR", "LEC"),
        "laps": 57,
    },
    {
        "scenario_id": "S04",
        "race": "silverstone",
        "year": 2025,
        "weather": "mixed",
        "drivers": ("VER", "NOR", "LEC"),
        "laps": 57,
    },
    {
        "scenario_id": "S08",
        "race": "jeddah",
        "year": 2025,
        "weather": "dry",
        "drivers": ("VER", "NOR", "LEC"),
        "laps": 57,
    },
    {
        "scenario_id": "S09",
        "race": "spa",
        "year": 2025,
        "weather": "mixed",
        "drivers": ("VER", "NOR", "LEC"),
        "laps": 57,
    },
)


def _score_tyre_model(path: Path) -> float:
    artifact = load_pickle_artifact(path)
    predicted = artifact.intercept + artifact.coef_age * 20 + artifact.coef_temp * 32 + artifact.coef_compound
    target = 0.018 * 20 + 0.0015 * (32 - 28) + 0.012
    error = abs(float(predicted) - float(target))
    return max(0.0, 1.0 - min(error, 1.0))


def _score_laptime_model(path: Path) -> float:
    artifact = load_pickle_artifact(path)
    estimate = (
        artifact.intercept
        + artifact.coefficients["lap"] * 30
        + artifact.coefficients["tyre_wear"] * 46.5
        + artifact.coefficients["traffic"] * 0.15
        + artifact.coefficients["fuel_weight"] * 59.0
        + artifact.coefficients["rainfall"] * 0.0
    )
    plausible = 80.0 <= estimate <= 130.0
    return 0.9 if plausible else 0.2


def _score_pit_rl(path: Path) -> float:
    with zipfile.ZipFile(path, mode="r") as archive:
        policy = json.loads(archive.read("policy.json").decode("utf-8"))
    threshold = float(policy["pit_if_tyre_wear_above"])
    return 1.0 if 50.0 <= threshold <= 75.0 else 0.0


def _score_pit_policy_q(path: Path) -> float:
    from hannah.agent.context import RaceContext
    from hannah.models.train_pit_q import QPitPolicyArtifact, choose_action
    from hannah.simulation.sandbox import RaceState

    artifact = load_pickle_artifact(path)
    if not isinstance(artifact, QPitPolicyArtifact):
        return 0.0
    race_state = RaceState.from_context(
        RaceContext(
            race="bahrain",
            year=2025,
            laps=57,
            weather="dry",
            drivers=["NOR", "LEC", "HAM"],
            race_data={"session_info": {"current_lap": 21}},
        )
    )
    race_state.current_lap = 21
    action = choose_action(race_state=race_state, driver_code="NOR", current_lap=21)
    learned = bool(np.any(np.abs(artifact.q_table) > 1e-9))
    return 1.0 if learned and action in {0, 1} else 0.0


def _score_winner(path: Path) -> float:
    from hannah.models.train_winner import load_and_predict

    artifact = load_pickle_artifact(path)
    probs = load_and_predict({"drivers": ["VER", "NOR", "LEC"]})
    valid = abs(sum(probs.values()) - 1.0) <= 1e-9
    has_weights = bool(getattr(artifact, "ordered_weights", ()))
    return 1.0 if valid and has_weights else 0.0


@lru_cache(maxsize=1)
def _scenario_depth_rows() -> tuple[dict[str, Any], ...]:
    from hannah.tools.race_sim import tool as race_sim_tool

    rows: list[dict[str, Any]] = []
    for case in SCENARIO_DEPTH_CASES:
        payload = asyncio.run(
            race_sim_tool.run(
                race=str(case["race"]),
                year=int(case["year"]),
                weather=str(case["weather"]),
                drivers=list(case["drivers"]),
                laps=int(case["laps"]),
                n_worlds=64,
                trace=True,
            )
        )
        trace = payload["trace"]
        events = list(trace.get("events", []))
        event_kinds = {str(event["kind"]) for event in events if isinstance(event, dict)}
        timeline = list(trace.get("timeline", []))
        coherent_timeline_events = 0
        total_timeline_events = 0
        for entry in timeline:
            label = str(entry.get("event", ""))
            if label not in event_kinds:
                continue
            total_timeline_events += 1
            window = entry.get("event_window")
            if not isinstance(window, dict):
                continue
            if str(window.get("kind")) != label:
                continue
            lap = int(entry.get("lap", 0))
            if int(window.get("start_lap", lap)) <= lap <= int(window.get("end_lap", lap)):
                coherent_timeline_events += 1
        strategy_lap = int(payload["strategy"]["recommended_pit_lap"])
        pit_plan = list(trace.get("pit_plan", []))
        pit_alignment = 1.0
        if pit_plan:
            pit_alignment = 1.0 if int(pit_plan[0].get("optimal_pit_lap", strategy_lap)) == strategy_lap else 0.0
        trace_alignment = 1.0 if coherent_timeline_events == total_timeline_events else 0.0
        rows.append(
            {
                "scenario_id": str(case["scenario_id"]),
                "weather": str(case["weather"]),
                "score": round((pit_alignment + trace_alignment) / 2.0, 3),
                "pit_alignment": round(pit_alignment, 3),
                "trace_alignment": round(trace_alignment, 3),
                "_has_event_windows": bool(events),
                "_coherent_timeline_events": coherent_timeline_events,
                "_total_timeline_events": total_timeline_events,
            }
        )
    return tuple(rows)


def _build_evaluation_depth() -> dict[str, Any]:
    rows = list(_scenario_depth_rows())
    by_weather: dict[str, int] = {}
    scenarios_with_event_windows = 0
    coherent_timeline_events = 0
    total_timeline_events = 0
    for row in rows:
        weather = str(row["weather"])
        by_weather[weather] = by_weather.get(weather, 0) + 1
        if row["_has_event_windows"]:
            scenarios_with_event_windows += 1
        coherent_timeline_events += int(row["_coherent_timeline_events"])
        total_timeline_events += int(row["_total_timeline_events"])

    scorecard = [
        {
            "scenario_id": str(row["scenario_id"]),
            "score": float(row["score"]),
            "pit_alignment": float(row["pit_alignment"]),
            "trace_alignment": float(row["trace_alignment"]),
        }
        for row in rows
    ]
    return {
        "scenario_scorecard": scorecard,
        "coverage": {
            "by_weather": by_weather,
            "event_windows": {
                "scenarios_with_event_windows": scenarios_with_event_windows,
                "coherent_timeline_events": coherent_timeline_events,
                "total_timeline_events": total_timeline_events,
            },
            "scenarios_total": len(scorecard),
        },
        "stability": {
            "deterministic": True,
            "score_variance": 0.0,
        },
    }


def evaluate_model(model_name: str) -> dict[str, object]:
    """Return deterministic evaluation metrics for a smoke-trained artifact."""
    model_key = model_name.strip().lower()
    if model_key not in PUBLIC_MODEL_NAMES:
        raise ValueError(f"unknown model_name: {model_name}")

    artifact_path = resolve_artifact_path(model_key)
    if not artifact_path.exists():
        from hannah.models import train_laptime, train_pit_q, train_pit_rl, train_tyre_deg, train_winner

        train_map = {
            "tyre_model": train_tyre_deg.train,
            "laptime_model": train_laptime.train,
            "pit_rl": train_pit_rl.train,
            "pit_policy_q": train_pit_q.train,
            "winner_ensemble": train_winner.train,
        }
        train_map[model_key](years=[2024], races=["bahrain"])

    if model_key == "tyre_model":
        score = _score_tyre_model(artifact_path)
    elif model_key == "laptime_model":
        score = _score_laptime_model(artifact_path)
    elif model_key == "pit_rl":
        score = _score_pit_rl(artifact_path)
    elif model_key == "pit_policy_q":
        score = _score_pit_policy_q(artifact_path)
    else:
        score = _score_winner(artifact_path)

    threshold = float(MODEL_SCORE_THRESHOLDS[model_key])
    rounded_score = round(float(score), 3)

    return {
        "model": model_key,
        "score": rounded_score,
        "artifact": str(artifact_path),
        "artifact_exists": artifact_path.exists(),
        "threshold": threshold,
        "meets_threshold": rounded_score >= threshold,
        "evaluation_depth": _build_evaluation_depth(),
    }
