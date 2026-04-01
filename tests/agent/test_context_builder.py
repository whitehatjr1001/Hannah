"""Tests for Hannah's nanobot-style main context assembly."""

from __future__ import annotations

from hannah.agent.persona import HANNAH_PERSONA
from hannah.runtime.context import MainAgentContext, RuntimeContextBuilder


def test_build_main_turn_emits_structured_system_blocks_before_history() -> None:
    builder = RuntimeContextBuilder()

    messages = builder.build_main_turn(
        MainAgentContext(
            persona=HANNAH_PERSONA,
            user_input="Should we pit on lap 22?",
            recent_messages=(
                {"role": "user", "content": "We have a 1.2s gap to the car ahead."},
                {"role": "assistant", "content": "Undercut is live if the tyre delta holds."},
            ),
        )
    )

    assert [message["role"] for message in messages] == [
        "system",
        "system",
        "system",
        "system",
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "identity/runtime" in messages[0]["content"].lower()
    assert "bootstrap docs" in messages[1]["content"].lower()
    assert "agent_loop.md" in messages[1]["content"].lower()
    assert "memory context" in messages[2]["content"].lower()
    assert "skills summary" in messages[3]["content"].lower()
    assert "hannah f1 persona" in messages[4]["content"].lower()
    assert "red bull racing" in messages[4]["content"].lower()
    assert messages[5] == {"role": "user", "content": "We have a 1.2s gap to the car ahead."}
    assert messages[6] == {
        "role": "assistant",
        "content": "Undercut is live if the tyre delta holds.",
    }
    assert messages[7] == {"role": "user", "content": "Should we pit on lap 22?"}


def test_build_main_turn_uses_skills_summary_hook_output() -> None:
    builder = RuntimeContextBuilder()
    hook_calls: list[str] = []

    def _skills_summary_hook() -> str:
        hook_calls.append("called")
        return "race_data, race_sim, pit_strategy"

    messages = builder.build_main_turn(
        MainAgentContext(
            persona=HANNAH_PERSONA,
            user_input="Predict the winner for Bahrain.",
            recent_messages=(),
            skills_summary_hook=_skills_summary_hook,
        )
    )

    skills_block = messages[3]["content"]
    assert hook_calls == ["called"]
    assert "race_data, race_sim, pit_strategy" in skills_block
    assert "hook" in skills_block.lower()
    assert messages[-1] == {"role": "user", "content": "Predict the winner for Bahrain."}
