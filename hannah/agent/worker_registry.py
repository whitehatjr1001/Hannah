"""Legacy worker-spec registration and compatibility helpers."""

from __future__ import annotations

from hannah.agent.context import RaceContext
from hannah.agent.worker_runtime import WorkerSpec
from hannah.domain.prompts import build_team_strategist_persona

RIVAL_TEAM_PERSONAS = {
    code: build_team_strategist_persona(code)
    for code in (
        "NOR",
        "PIA",
        "RUS",
        "ANT",
        "LEC",
        "HAM",
        "VER",
        "HAD",
        "LAW",
        "LIN",
        "GAS",
        "COL",
        "HUL",
        "BOR",
        "ALB",
        "SAI",
        "PER",
        "BOT",
        "ALO",
        "STR",
        "OCO",
        "BEA",
    )
}


def build_legacy_worker_specs(ctx: RaceContext) -> list[WorkerSpec]:
    base_prompt = f"{ctx.race.title()} {ctx.year}, weather {ctx.weather}, {ctx.laps} laps."
    specs = [
        WorkerSpec(
            worker_id="sim_agent",
            task=f"Run a race simulation for {base_prompt}",
            system_prompt="You are a race simulation worker. Summarize the strongest strategy signal.",
            allowed_tools=["race_data", "race_sim"],
            result_contract={"summary": "string", "evidence": "list"},
        ),
        WorkerSpec(
            worker_id="strategy_agent",
            task=f"Recommend the best pit window for the lead driver in {base_prompt}",
            system_prompt="You are a chief strategist worker. Return one concise pit-wall recommendation.",
            allowed_tools=["race_data", "race_sim", "pit_strategy"],
            result_contract={"summary": "string", "recommended_pit_lap": "integer"},
        ),
        WorkerSpec(
            worker_id="predict_agent",
            task=f"Project winner odds for {base_prompt}",
            system_prompt="You are a winner-probability worker. Focus on probabilities and the top evidence.",
            allowed_tools=["race_data", "predict_winner"],
            result_contract={"summary": "string", "winner_probs": "object"},
        ),
    ]

    for driver in ctx.drivers[1:]:
        specs.append(
            WorkerSpec(
                worker_id=f"rival_{driver.lower()}",
                task=f"Return the best rival strategy call for {driver} in {base_prompt}",
                system_prompt=RIVAL_TEAM_PERSONAS.get(
                    driver,
                    f"You are the strategist for {driver}. Return one sharp race call.",
                ),
                allowed_tools=["race_data", "pit_strategy"],
                result_contract={"summary": "string", "decision": "string"},
            )
        )
    return specs
