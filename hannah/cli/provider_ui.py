"""Provider onboarding UI helpers shared by CLI commands and chat mode."""

from __future__ import annotations

import os
from pathlib import Path

import click

from hannah.config.provider_setup import (
    apply_provider_configuration,
    get_provider_preset,
    list_provider_presets,
    load_env_context,
    summarize_provider_statuses,
)
from hannah.utils.console import Console, Table


def provider_choice() -> click.Choice:
    return click.Choice([preset.name for preset in list_provider_presets()], case_sensitive=False)


def render_provider_status_table(*, console: Console, env_file: Path = Path(".env")) -> None:
    env = load_env_context(env_path=env_file)
    statuses = summarize_provider_statuses(env)
    table = Table(title="Hannah Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Active", style="magenta")
    table.add_column("Model", style="yellow")
    table.add_column("Key Env", style="blue")
    table.add_column("Preview", style="dim")
    for status in statuses:
        preview = ""
        if status.configured_env_var is not None:
            preview = _mask_secret(env.get(status.configured_env_var))
        table.add_row(
            status.label,
            "configured" if status.configured else "not configured",
            "active" if status.active else "",
            status.default_model,
            " / ".join(status.api_key_env_vars),
            preview,
        )
    console.print()
    console.print(table)
    console.print(f"[dim]  Env file:[/dim] [cyan]{env_file}[/cyan]\n")


def render_model_status(*, console: Console, env_file: Path = Path(".env")) -> None:
    env = load_env_context(env_path=env_file)
    current = env.get("HANNAH_MODEL", "claude-sonnet-4-6")
    statuses = summarize_provider_statuses(env)
    active_provider = next((status.label for status in statuses if status.active), "Unknown")
    console.print(f"\n  [bold]Active model:[/bold] [cyan]{current}[/cyan]")
    console.print(f"  [bold]Provider:[/bold] [cyan]{active_provider}[/cyan]")
    rlm_base = env.get("HANNAH_RLM_API_BASE") or os.getenv("HANNAH_RLM_API_BASE")
    if rlm_base:
        console.print(f"  [bold]RLM base:[/bold] [cyan]{rlm_base}[/cyan]")
    console.print()


def run_provider_configure_flow(
    *,
    console: Console,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    env_file: Path = Path(".env"),
) -> dict[str, str]:
    if provider is None:
        render_provider_status_table(console=console, env_file=env_file)
        provider = click.prompt("Provider", type=provider_choice(), default="openai")
    preset = get_provider_preset(provider)
    resolved_api_key = api_key or click.prompt(
        f"{preset.label} API key",
        hide_input=True,
        confirmation_prompt=True,
    )
    resolved_model = model or click.prompt("Model", default=preset.default_model, show_default=True)

    changes = apply_provider_configuration(
        env_path=env_file,
        provider=preset,
        api_key=resolved_api_key,
        model=resolved_model,
    )
    console.print()
    console.print(f"[green]Configured {preset.label}[/green]")
    console.print(f"  [bold]Model:[/bold] {changes['model']}")
    console.print(f"  [bold]API key env:[/bold] {changes['api_key_env_var']}")
    console.print(f"  [bold]Env file:[/bold] {changes['env_path']}")
    console.print(
        "  [bold]Conflicts cleared:[/bold] "
        "HANNAH_RLM_API_BASE, HANNAH_RLM_API_KEY, HANNAH_FORCE_LOCAL_PROVIDER"
    )
    console.print()
    return changes


def _mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
