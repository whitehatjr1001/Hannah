"""Training launcher tool."""

from __future__ import annotations

from typing import Any

from hannah.models import train_laptime, train_pit_q, train_pit_rl, train_tyre_deg, train_winner

SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "tyre_model",
    "laptime_model",
    "pit_rl",
    "pit_policy_q",
    "winner_ensemble",
    "all",
)

_MODEL_NAME_ALIASES: dict[str, str] = {
    "lap_time_model": "laptime_model",
    "laptime": "laptime_model",
    "pit_policy": "pit_policy_q",
    "pit_policy_model": "pit_policy_q",
    "pit_strategy_model": "pit_policy_q",
    "pit_strategy_q": "pit_policy_q",
    "q_learning": "pit_policy_q",
    "q_policy": "pit_policy_q",
    "race_strategy": "pit_policy_q",
    "race_strategy_model": "pit_policy_q",
    "strategy": "pit_policy_q",
    "strategy_backend": "pit_policy_q",
    "strategy_model": "pit_policy_q",
    "strategy_predictor": "pit_policy_q",
    "tyre_deg": "tyre_model",
    "tire_model": "tyre_model",
    "winner_model": "winner_ensemble",
}

SKILL = {
    "name": "train_model",
    "description": (
        "Launches offline retraining jobs for Hannah's backend artifacts. "
        "Use this only when the user explicitly asks to train or retrain a supported model. "
        "Do not use it for one-off race strategy analysis; use race_data, race_sim, and pit_strategy instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "enum": list(SUPPORTED_MODEL_NAMES),
                "description": (
                    "Supported offline artifact to retrain. "
                    "Use pit_policy_q for the strategy backend."
                ),
            },
            "years": {"type": "array", "items": {"type": "integer"}},
            "races": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["model_name"],
    },
}


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize common hosted-model aliases into canonical train_model ids."""
    normalized = dict(args)
    model_name = normalized.get("model_name")
    if isinstance(model_name, str):
        normalized["model_name"] = _normalize_model_name(model_name)
    return normalized


async def run(model_name: str, years: list[int] | None = None, races: list[str] | None = None) -> dict:
    """Dispatch to a training script."""
    years = years or [2024]
    canonical_model_name = _normalize_model_name(model_name)
    trainers = {
        "tyre_model": train_tyre_deg.train,
        "laptime_model": train_laptime.train,
        "pit_rl": train_pit_rl.train,
        "pit_policy_q": train_pit_q.train,
        "winner_ensemble": train_winner.train,
    }
    if canonical_model_name == "all":
        paths = {name: trainer(years=years, races=races) for name, trainer in trainers.items()}
        return {"saved": paths}
    if canonical_model_name not in trainers:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(f"unknown model_name: {model_name}. supported model_name values: {supported}")
    return {"saved": trainers[canonical_model_name](years=years, races=races)}


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    return _MODEL_NAME_ALIASES.get(normalized, normalized)
