"""Helpers for rival strategy opinions."""

from __future__ import annotations

from dataclasses import dataclass

from hannah.domain.teams import get_driver_info
from hannah.simulation.sandbox import RaceState


@dataclass(frozen=True)
class CompetitorOpinion:
    driver: str
    action: str
    confidence: float
    recommended_pit_lap: int
    recommended_compound: str
    strategy_type: str
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "driver": self.driver,
            "action": self.action,
            "confidence": self.confidence,
            "recommended_pit_lap": self.recommended_pit_lap,
            "recommended_compound": self.recommended_compound,
            "strategy_type": self.strategy_type,
            "reasoning": self.reasoning,
        }


def default_rival_grid(
    drivers: list[str],
    race_state: RaceState | None = None,
    current_lap: int = 20,
) -> list[CompetitorOpinion]:
    """Return deterministic rival opinions from team style and track context."""
    opinions: list[CompetitorOpinion] = []
    for index, driver in enumerate(drivers):
        try:
            profile = get_driver_info(driver)
            style = profile.strategy_style
        except Exception:
            style = "balanced"
        action = {
            "aggressive": "cover undercut",
            "balanced": "hold baseline",
            "defensive": "protect track position",
            "opportunistic": "attack event window",
        }[style]
        compound = {
            "aggressive": "MEDIUM",
            "balanced": "HARD",
            "defensive": "HARD",
            "opportunistic": "MEDIUM",
        }[style]
        pit_lap = current_lap + 2 + min(index, 2)
        confidence = 0.56 + min(index, 3) * 0.05
        strategy_type = "undercut" if style in {"aggressive", "opportunistic"} else "stay_out"
        if race_state is not None and race_state.weather != "dry":
            wet_skill = getattr(profile, "wet_weather_skill", 0.85) if "profile" in locals() else 0.85
            action = "watch crossover" if not wet_skill > 0.88 else "attack crossover"
            compound = "INTER" if race_state.weather == "mixed" else "WET"
            confidence += 0.04
            strategy_type = "event_window"
        opinions.append(
            CompetitorOpinion(
                driver=driver,
                action=action,
                confidence=round(min(confidence, 0.92), 2),
                recommended_pit_lap=max(pit_lap, 1),
                recommended_compound=compound,
                strategy_type=strategy_type,
                reasoning=(
                    f"{driver} likely to {action} around lap {pit_lap} on {compound} "
                    f"given {race_state.weather if race_state else 'dry'} conditions."
                ),
            )
        )
    return opinions
