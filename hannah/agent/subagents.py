"""Async sub-agent helpers for Hannah."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from hannah.agent.context import RaceContext
from hannah.agent.prompts import build_strategy_prompt
from hannah.agent.worker_registry import build_legacy_worker_specs as _build_legacy_worker_specs
from hannah.agent.worker_runtime import WorkerSpec
from hannah.config.loader import load_config
from hannah.models.train_pit_q import ARTIFACT_PATH as PIT_POLICY_Q_ARTIFACT_PATH
from hannah.models.train_pit_q import choose_action as choose_q_policy_action
from hannah.providers.registry import ProviderRegistry
from hannah.simulation.competitor_agents import default_rival_grid
from hannah.simulation.sandbox import RaceState
from hannah.utils.console import Console

console = Console()


@dataclass
class SubAgentResult:
    agent: str
    success: bool
    data: dict = field(default_factory=dict)
    error: str | None = None


class BaseSubAgent:
    """Small sub-agent wrapper around the shared provider."""

    name: str = "base"
    persona: str = "You are an F1 strategy analyst."

    def __init__(self) -> None:
        self.provider = ProviderRegistry.from_config(load_config())

    async def run(self, ctx: RaceContext) -> SubAgentResult:
        raise NotImplementedError(f"{self.__class__.__name__}.run must be implemented")

    async def _ask(self, prompt: str) -> str:
        response = await self.provider.complete(
            messages=[
                {"role": "system", "content": self.persona},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            temperature=0.1,
            max_tokens=512,
        )
        return self._extract_text(response)

    def _extract_text(self, response: object) -> str:
        if hasattr(response, "choices") and getattr(response, "choices"):
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if hasattr(message, "content"):
                return str(message.content or "")
            if isinstance(message, dict):
                return str(message.get("content", ""))
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    return str(message.get("content", ""))
        return ""

    def _safe_json(self, payload: Any) -> str:
        try:
            return json.dumps(payload, default=str)
        except Exception:
            return str(payload)


class SimAgent(BaseSubAgent):
    name = "sim_agent"
    persona = "You are a Red Bull race simulation engineer. Summarize the strategy signal."

    async def run(self, ctx: RaceContext) -> SubAgentResult:
        try:
            from hannah.simulation.monte_carlo import run_fast
            from hannah.simulation.sandbox import RaceState

            state = RaceState.from_context(ctx)
            result = await run_fast(state)
            return SubAgentResult(agent=self.name, success=True, data=result.to_dict())
        except Exception as err:
            return SubAgentResult(agent=self.name, success=False, error=str(err))


class StrategyAgent(BaseSubAgent):
    name = "strategy_agent"
    persona = "You are Red Bull's chief strategist. Be concise and data-backed."

    async def run(self, ctx: RaceContext) -> SubAgentResult:
        try:
            strategy = await self._ask(build_strategy_prompt(ctx))
            return SubAgentResult(
                agent=self.name,
                success=True,
                data={
                    "strategy": strategy,
                    "mode": "local_fallback" if not strategy else "provider",
                },
            )
        except Exception as err:
            return SubAgentResult(agent=self.name, success=False, error=str(err))


class PredictAgent(BaseSubAgent):
    name = "predict_agent"
    persona = "You are a race prediction analyst. Report winning probabilities only."

    async def run(self, ctx: RaceContext) -> SubAgentResult:
        try:
            from hannah.models.train_winner import load_and_predict

            probs = await asyncio.to_thread(load_and_predict, ctx)
            return SubAgentResult(agent=self.name, success=True, data={"winner_probs": probs})
        except Exception as err:
            return SubAgentResult(agent=self.name, success=False, error=str(err))


class RivalAgent(BaseSubAgent):
    TEAM_PERSONAS = {
        "NOR": "You are the McLaren strategist. Prefer aggressive undercuts.",
        "LEC": "You are the Ferrari strategist. Protect track position.",
        "HAM": "You are the Mercedes strategist. Default to conservative calls.",
        "ALO": "You are the Aston Martin strategist. Exploit safety car windows.",
    }

    def __init__(self, driver_code: str) -> None:
        super().__init__()
        self.driver_code = driver_code
        self.name = f"rival_{driver_code.lower()}"
        self.persona = self.TEAM_PERSONAS.get(
            driver_code,
            f"You are the strategist for {driver_code}. Return a sharp race call.",
        )

    async def run(self, ctx: RaceContext) -> SubAgentResult:
        try:
            current_lap = 20
            if ctx.race_data and isinstance(ctx.race_data, dict):
                session_info = ctx.race_data.get("session_info", {})
                if isinstance(session_info, dict):
                    current_lap = int(session_info.get("current_lap", current_lap))
            race_state = RaceState.from_context(ctx)
            race_state.current_lap = current_lap
            opinions = default_rival_grid(
                drivers=[self.driver_code],
                race_state=race_state,
                current_lap=current_lap,
            )
            opinion = opinions[0]
            policy_action_id = choose_q_policy_action(
                race_state=race_state,
                driver_code=self.driver_code,
                current_lap=current_lap,
            )
            prompt = (
                f"{build_strategy_prompt(ctx)} "
                f"Return one concise recommendation for {self.driver_code} in JSON."
            )
            provider_text = await self._ask(prompt)
            decision = {
                **opinion.to_dict(),
                "team": self.driver_code,
                "policy_backend": "q_learning",
                "policy_action_id": policy_action_id,
                "policy_artifact": str(PIT_POLICY_Q_ARTIFACT_PATH),
                "decision": provider_text
                or (
                    (
                        f"pit now at lap {current_lap} for {opinion.recommended_compound}"
                        if policy_action_id == 1
                        else f"{opinion.action} at lap {opinion.recommended_pit_lap} "
                        f"for {opinion.recommended_compound}"
                    )
                ),
            }
            if policy_action_id == 1:
                decision["recommended_pit_lap"] = current_lap
                decision["action"] = "pit now"
                decision["strategy_type"] = "undercut"
            return SubAgentResult(
                agent=self.name,
                success=True,
                data=decision,
            )
        except Exception as err:
            return SubAgentResult(agent=self.name, success=False, error=str(err))


def build_legacy_worker_specs(ctx: RaceContext) -> list[WorkerSpec]:
    """Compatibility shim exposing the fixed F1 worker roster as generic specs."""
    return _build_legacy_worker_specs(ctx)


async def spawn_all(ctx: RaceContext) -> dict[str, dict]:
    """Run all sub-agents concurrently and collect successful outputs."""
    rivals = [driver for driver in ctx.drivers[1:]]
    console.print(f"  [dim]spawning {3 + len(rivals)} sub-agents concurrently...[/dim]")
    results = await asyncio.gather(
        SimAgent().run(ctx),
        StrategyAgent().run(ctx),
        PredictAgent().run(ctx),
        *(RivalAgent(driver).run(ctx) for driver in rivals),
        return_exceptions=True,
    )
    output: dict[str, dict] = {}
    for result in results:
        if isinstance(result, Exception):
            console.print(f"  [red]sub-agent error: {result}[/red]")
            continue
        if result.success:
            console.print(f"  [green]✓ {result.agent}[/green]")
            output[result.agent] = result.data
        else:
            console.print(f"  [yellow]⚠ {result.agent}: {result.error}[/yellow]")
    return output
