"""Prompt builders for Hannah sub-agents and runtime context blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_BOOTSTRAP_DOCS = (
    "AGENT_LOOP.md",
    "ARCHITECTURE.md",
    "GOAL.md",
    "PRD.md",
    "V1_RELEASE.md",
)


def build_identity_runtime_block(*, dynamic_guidance: str | None = None) -> str:
    lines = [
        "Identity/Runtime block:",
        "- Identity: Hannah Smith, Red Bull Racing's virtual Race Director.",
        "- Runtime: CLI-first, tool-using F1 strategy agent.",
        "- Rule: keep orchestration in the agent layer and let tools own deterministic work.",
    ]
    if dynamic_guidance:
        lines.extend(
            [
                "- Dynamic guidance:",
                f"  {dynamic_guidance.strip()}",
            ]
        )
    return "\n".join(lines)


def build_bootstrap_docs_block(docs: Sequence[str] | None = None) -> str:
    docs_to_use = tuple(docs) if docs else DEFAULT_BOOTSTRAP_DOCS
    lines = [
        "Bootstrap docs block:",
        "- Use these docs as the operating reference for the turn:",
    ]
    lines.extend(f"- {doc}" for doc in docs_to_use)
    return "\n".join(lines)


def build_memory_context_block(
    recent_messages: Sequence[Mapping[str, Any]],
    *,
    memory_context: str | None = None,
) -> str:
    lines = [
        "Memory context block:",
        f"- {len(recent_messages)} recent messages are available in the conversation history below.",
        "- Preserve continuity from the prior turn before answering.",
    ]
    if memory_context:
        lines.extend(
            [
                "- Session memory summary:",
                f"  {memory_context.strip()}",
            ]
        )
    return "\n".join(lines)


def build_skills_summary_block(skills_summary: str | None = None) -> str:
    lines = [
        "Skills summary hook block:",
    ]
    if skills_summary:
        lines.extend(
            [
                "- Hook result:",
                f"  {skills_summary.strip()}",
            ]
        )
    else:
        lines.extend(
            [
                "- Hook result: not loaded yet.",
                "- Slice 4 will inject discovered SKILL summaries here.",
            ]
        )
    return "\n".join(lines)


def build_hannah_persona_block(persona: str) -> str:
    return "\n".join(
        [
            "Hannah F1 persona block:",
            persona.strip(),
        ]
    )


def build_strategy_prompt(ctx: Any) -> str:
    """Create a concise strategist prompt from race context."""
    return (
        f"Race: {ctx.race} {ctx.year}, {ctx.laps} laps, {ctx.weather}. "
        f"Drivers: {', '.join(ctx.drivers)}. "
        f"Race data snapshot: {ctx.race_data or {}}."
    )
