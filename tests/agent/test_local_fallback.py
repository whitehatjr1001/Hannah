"""Regression tests for the local fallback provider planner."""

from __future__ import annotations

from hannah.providers.local_fallback import DeterministicFallbackPlanner


def test_extract_drivers_filters_non_driver_english_tokens() -> None:
    planner = DeterministicFallbackPlanner()
    drivers = planner._extract_drivers(
        "Run a race simulation for bahrain 2025. Driver: VER. Laps: 57. Weather: dry."
    )
    assert drivers == ["VER"]


def test_freeform_fallback_message_points_to_provider_setup_commands() -> None:
    planner = DeterministicFallbackPlanner()

    response = planner._synthesize("hi", {})

    assert "hannah providers" in response
    assert "hannah configure" in response
