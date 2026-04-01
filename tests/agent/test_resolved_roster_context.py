"""Resolved roster propagation through agent runtime context objects."""

from __future__ import annotations

from hannah.agent.context import MainAgentContext, NanobotContextBuilder, RaceContext
from hannah.agent.prompts import build_strategy_prompt
from hannah.domain.resolved_roster import ResolvedDriverProfile, ResolvedRoster


def _roster() -> ResolvedRoster:
    return ResolvedRoster(
        year=2001,
        source="historical_seed",
        drivers=(
            ResolvedDriverProfile(
                code="MSC",
                driver="Michael Schumacher",
                team="Ferrari",
                teammate="BAR",
                color="#DC0000",
                base_pace_delta=-0.3,
                tyre_management=0.95,
                wet_weather_skill=0.97,
                strategy_style="aggressive",
            ),
            ResolvedDriverProfile(
                code="BAR",
                driver="Rubens Barrichello",
                team="Ferrari",
                teammate="MSC",
                color="#DC0000",
                base_pace_delta=-0.12,
                tyre_management=0.92,
                wet_weather_skill=0.91,
                strategy_style="balanced",
            ),
        ),
    )


def test_build_strategy_prompt_prefers_resolved_roster_summary() -> None:
    ctx = RaceContext(
        race="monza",
        year=2001,
        laps=53,
        weather="dry",
        drivers=["MSC", "BAR"],
        resolved_roster=_roster(),
        race_data={"session_info": {"current_lap": 18}},
    )

    prompt = build_strategy_prompt(ctx)

    assert "Resolved roster: historical_seed (2001)." in prompt
    assert "MSC Michael Schumacher [Ferrari]" in prompt
    assert "BAR Rubens Barrichello [Ferrari]" in prompt


def test_runtime_context_builder_injects_resolved_roster_block() -> None:
    builder = NanobotContextBuilder()
    messages = builder.build_main_turn(
        MainAgentContext(
            persona="You are Hannah.",
            user_input="Run a historical strategy review.",
            resolved_roster=_roster(),
        )
    )

    system_messages = [message["content"] for message in messages if message["role"] == "system"]

    assert any("Resolved roster block:" in content for content in system_messages)
    assert any("MSC: Michael Schumacher, Ferrari teammate BAR" in content for content in system_messages)
