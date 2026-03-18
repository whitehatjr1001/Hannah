"""Formatting helpers for the Hannah CLI."""

from hannah.utils.console import Panel

BANNER = """[bold magenta]
  ██╗  ██╗ █████╗ ███╗  ██╗███╗  ██╗ █████╗ ██╗  ██╗
  ██║  ██║██╔══██╗████╗ ██║████╗ ██║██╔══██╗██║  ██║
  ███████║███████║██╔██╗██║██╔██╗██║███████║███████║
  ██╔══██║██╔══██║██║╚████║██║╚████║██╔══██║██╔══██║
  ██║  ██║██║  ██║██║ ╚███║██║ ╚███║██║  ██║██║  ██║
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚══╝╚═╝  ╚══╝╚═╝  ╚═╝╚═╝  ╚═╝
[/bold magenta]
[bold pink1]  SMITH  ◆  Red Bull Race Director  ◆  v0.1.0-alpha[/bold pink1]
"""


def make_hannah_panel(text: str) -> Panel:
    """Render Hannah's response consistently."""
    return Panel(
        text,
        title="[bold magenta]Hannah Smith[/bold magenta]",
        border_style="dim",
        padding=(0, 2),
    )


def format_trace_summary(trace: dict) -> str:
    """Render a compact deterministic trace summary for the direct CLI path."""
    replay = trace.get("replay", {})
    timeline = replay.get("timeline", [])
    projected_order = replay.get("projected_order", [])
    pit_plan = replay.get("pit_plan", [])
    events = replay.get("events", [])
    race = replay.get("race", "unknown")
    year = replay.get("year", "")
    seed = replay.get("seed", "?")
    focus = replay.get("focus_driver", "?")

    lines = [f"{trace.get('trace_id', 'trace')} | {race} {year} | focus={focus} seed={seed}"]
    if timeline:
        first = timeline[0]
        lines.append(
            f"Call: lap {first.get('recommended_pit_lap', '?')} on "
            f"{first.get('recommended_compound', '?')} ({first.get('event', 'signal')})"
        )
    if projected_order:
        top = ", ".join(
            f"{item.get('driver', '?')} {float(item.get('win_prob', 0.0)):.3f}"
            for item in projected_order[:3]
        )
        lines.append(f"Projected order: {top}")
    if events:
        event_text = ", ".join(
            f"{event.get('kind', 'event')}[{event.get('start_lap', '?')}-{event.get('end_lap', '?')}]"
            for event in events
        )
        lines.append(f"Event windows: {event_text}")
    if pit_plan:
        compact_plan = ", ".join(
            f"{row.get('driver', '?')} L{row.get('optimal_pit_lap', '?')}->{row.get('target_compound', '?')}"
            for row in pit_plan
        )
        lines.append(f"Pit plan: {compact_plan}")
    if timeline:
        laps = ", ".join(
            f"L{entry.get('lap', '?')}:{entry.get('event', 'signal')}"
            for entry in timeline
        )
        lines.append(f"Moments: {laps}")
    return "\n".join(lines)
