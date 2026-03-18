"""Time formatting helpers."""

from __future__ import annotations


def seconds_to_gap(value: float) -> str:
    """Format a gap in seconds for the CLI."""
    return f"{value:+.3f}s"

