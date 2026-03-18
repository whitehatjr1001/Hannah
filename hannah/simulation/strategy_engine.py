"""Heuristic strategy synthesis from simulation output."""

from __future__ import annotations

from dataclasses import dataclass

from hannah.simulation.gap_engine import GapEngine
from hannah.simulation.monte_carlo import SimResult
from hannah.simulation.sandbox import RaceState


@dataclass(frozen=True)
class StrategyDecision:
    recommended_pit_lap: int
    recommended_compound: str
    strategy_type: str
    confidence: float
    undercut_window: int
    rival_threats: list[str]
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "recommended_pit_lap": self.recommended_pit_lap,
            "recommended_compound": self.recommended_compound,
            "strategy_type": self.strategy_type,
            "confidence": self.confidence,
            "undercut_window": self.undercut_window,
            "rival_threats": self.rival_threats,
            "reasoning": self.reasoning,
        }


class StrategyEngine:
    """Convert sim outputs into a pit-wall recommendation."""

    def __init__(self) -> None:
        self.gap_engine = GapEngine()

    def analyse(self, race_state: RaceState, sim_result: SimResult) -> dict:
        recommended_lap = int(sim_result.optimal_pit_laps[0])
        recommended_compound = sim_result.optimal_compounds[0]
        undercut_window = int(sim_result.undercut_windows.get(0, recommended_lap))
        if undercut_window not in {int(lap) for lap in sim_result.optimal_pit_laps}:
            undercut_window = recommended_lap
        confidence = round(float(sim_result.winner_probs.max()), 2)
        nearest_gap = min((gap for gap in race_state.gaps[1:] if gap > 0), default=0.0)
        freshest_rival_age = min(race_state.tyre_ages[1:], default=race_state.tyre_ages[0] if race_state.tyre_ages else 0)
        undercut = self.gap_engine.undercut_feasibility(
            gap_to_ahead=nearest_gap,
            pit_delta=race_state.pit_loss,
            lap_delta=0.75 + (race_state.tyre_ages[0] - freshest_rival_age) * 0.05,
        )
        overcut = self.gap_engine.overcut_feasibility(
            gap_to_behind=nearest_gap,
            tyre_age_delta=max(race_state.tyre_ages[0] - freshest_rival_age, 0),
            deg_rate=0.18 * race_state.tyre_wear_factor,
        )
        strategy_type = self._classify_strategy(
            race_state,
            recommended_compound,
            undercut["feasible"],
            overcut["feasible"],
        )
        decision = StrategyDecision(
            recommended_pit_lap=recommended_lap,
            recommended_compound=recommended_compound,
            strategy_type=strategy_type,
            confidence=confidence,
            undercut_window=undercut_window,
            rival_threats=race_state.drivers[1:3],
            reasoning=self._build_reasoning(
                race_state=race_state,
                recommended_lap=recommended_lap,
                recommended_compound=recommended_compound,
                confidence=confidence,
                undercut_feasible=undercut["feasible"],
                overcut_feasible=overcut["feasible"],
            ),
        )
        return decision.to_dict()

    def _classify_strategy(
        self,
        race_state: RaceState,
        recommended_compound: str,
        undercut_feasible: bool,
        overcut_feasible: bool,
    ) -> str:
        del race_state, recommended_compound, overcut_feasible
        if undercut_feasible:
            return "undercut"
        return "stay_out"

    def _build_reasoning(
        self,
        race_state: RaceState,
        recommended_lap: int,
        recommended_compound: str,
        confidence: float,
        undercut_feasible: bool,
        overcut_feasible: bool,
    ) -> str:
        notes = [f"Pit around lap {recommended_lap} onto {recommended_compound}."]
        if race_state.weather != "dry":
            notes.append("Weather crossover risk is active.")
        if race_state.event_at(recommended_lap) is not None:
            notes.append("Event window makes the stop cheaper than a green-flag stop.")
        elif undercut_feasible:
            notes.append("Fresh-tyre delta is strong enough to attack the undercut.")
        elif overcut_feasible:
            notes.append("Track position bias favors extending slightly and covering later.")
        else:
            notes.append("Baseline one-stop remains the lowest-risk play.")
        notes.append(f"Confidence {confidence:.0%}.")
        return " ".join(notes)
