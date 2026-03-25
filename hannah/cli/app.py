"""Click entrypoint for Hannah."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

try:
    import click
except Exception:
    class _Choice:
        def __init__(self, choices):
            self.choices = choices

    class _ClickCompat:
        Context = object
        Choice = _Choice

        @staticmethod
        def pass_context(func):
            return func

        @staticmethod
        def option(*args, **kwargs):
            del args, kwargs
            return lambda func: func

        @staticmethod
        def argument(*args, **kwargs):
            del args, kwargs
            return lambda func: func

        @staticmethod
        def group(*args, **kwargs):
            del args, kwargs

            def decorator(func):
                commands: dict[str, object] = {}

                def command(*command_args, **command_kwargs):
                    del command_args

                    def register(subcommand):
                        name = str(command_kwargs.get("name") or getattr(subcommand, "name", subcommand.__name__))
                        commands[name] = subcommand
                        return subcommand

                    return register

                func.command = command
                func.name = func.__name__
                func.commands = commands
                return func

            return decorator

    click = _ClickCompat()

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

from hannah.agent.loop import AgentLoop
from hannah.agent.tool_registry import ToolRegistry
from hannah.cli import agent_command as agent_command_module
from hannah.cli import command_prompts
from hannah.cli.chat import is_interactive_terminal, print_sessions
from hannah.cli.format import BANNER, format_trace_summary, make_hannah_panel
from hannah.cli.provider_ui import (
    provider_choice,
    render_model_status,
    render_provider_status_table,
    run_provider_configure_flow,
)
from hannah.utils.console import Console

load_dotenv()
console = Console()


def _run_agent_command(command: str) -> None:
    asyncio.run(AgentLoop().run_command(command))


def _parse_driver_codes(raw: str) -> list[str]:
    drivers = [driver.strip().upper() for driver in raw.split(",") if driver.strip()]
    return drivers or ["VER", "NOR", "LEC"]


def _parse_lap_csv(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None
    checkpoints: list[int] = []
    for token in tokens:
        checkpoints.append(int(token))
    return checkpoints


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Hannah Smith — F1 strategy simulation CLI agent."""
    if getattr(ctx, "invoked_subcommand", None) is None:
        if is_interactive_terminal():
            ctx.invoke(agent)
            return
        console.print(BANNER)
        console.print("[dim]  Type [bold]hannah --help[/bold] to see all commands.[/dim]\n")


@cli.command()
@click.option("--race", required=True, help="Race name e.g. bahrain, monaco")
@click.option("--year", default=2025, help="Season year")
@click.option("--driver", default=None, help="Driver code e.g. VER, NOR")
@click.option("--laps", default=57, help="Number of laps")
@click.option("--weather", default="dry", type=click.Choice(["dry", "wet", "mixed"]))
def simulate(race: str, year: int, driver: str | None, laps: int, weather: str) -> None:
    """Run a fast async race simulation."""
    asyncio.run(
        agent_command_module.run_agent_command(
            command_prompts.build_simulate_intent(
                race=race,
                year=year,
                driver=driver,
                laps=laps,
                weather=weather,
            ),
            interactive=False,
            session_id="cli:direct",
            new_session=False,
            persist_session=False,
        )
    )


@cli.command(name="trace")
@click.option("--race", required=True, help="Race name e.g. bahrain, monaco")
@click.option("--year", default=2025, help="Season year")
@click.option(
    "--drivers",
    default="VER,NOR,LEC",
    help="Comma-separated driver codes e.g. VER,NOR,LEC",
)
@click.option("--laps", default=57, help="Number of laps")
@click.option("--weather", default="dry", type=click.Choice(["dry", "wet", "mixed"]))
@click.option("--n-worlds", default=1000, type=int, help="Monte Carlo worlds")
@click.option(
    "--checkpoints",
    default=None,
    help="Optional comma-separated checkpoint laps e.g. 12,28,57",
)
@click.option("--json-output", is_flag=True, help="Print raw trace JSON instead of summary")
def trace_command(
    race: str,
    year: int,
    drivers: str,
    laps: int,
    weather: str,
    n_worlds: int,
    checkpoints: str | None,
    json_output: bool,
) -> None:
    """Generate deterministic replay/debug trace directly from the simulation stack."""
    from hannah.tools.race_sim import tool as race_sim_tool

    try:
        trace_checkpoints = _parse_lap_csv(checkpoints)
    except ValueError as err:
        console.print(f"\n[red]invalid checkpoints:[/red] {err}\n")
        return

    payload = asyncio.run(
        race_sim_tool.run(
            race=race,
            year=year,
            weather=weather,
            drivers=_parse_driver_codes(drivers),
            laps=laps,
            n_worlds=n_worlds,
            trace=True,
            trace_checkpoints=trace_checkpoints,
        )
    )
    trace_payload = payload.get("trace", {})
    if json_output:
        console.print()
        console.print(json.dumps(trace_payload, indent=2, sort_keys=True))
        console.print()
        return

    console.print()
    console.print(make_hannah_panel(format_trace_summary(trace_payload)))
    console.print()


@cli.command()
@click.option("--race", required=True, help="Race name")
@click.option("--year", default=2025, help="Season year")
def predict(race: str, year: int) -> None:
    """Predict the winner for an upcoming race."""
    asyncio.run(
        agent_command_module.run_agent_command(
            command_prompts.build_predict_intent(race=race, year=year),
            interactive=False,
            session_id="cli:direct",
            new_session=False,
            persist_session=False,
        )
    )


@cli.command()
@click.option("--race", required=True, help="Race name")
@click.option("--lap", required=True, type=int, help="Current lap number")
@click.option("--driver", required=True, help="Driver code")
@click.option(
    "--type",
    "strategy_type",
    default="optimal",
    type=click.Choice(["optimal", "undercut", "overcut", "alternate"]),
)
def strategy(race: str, lap: int, driver: str, strategy_type: str) -> None:
    """Get a pit strategy call for the current race situation."""
    asyncio.run(
        agent_command_module.run_agent_command(
            command_prompts.build_strategy_intent(
                race=race,
                lap=lap,
                driver=driver,
                strategy_type=strategy_type,
            ),
            interactive=False,
            session_id="cli:direct",
            new_session=False,
            persist_session=False,
        )
    )


@cli.command()
@click.option("--agents", required=True, help="Comma-separated driver codes e.g. VER,NOR,LEC")
@click.option("--race", default="bahrain", help="Race circuit")
@click.option("--laps", default=57, help="Race laps")
@click.option("--weather", default="dry")
def sandbox(agents: str, race: str, laps: int, weather: str) -> None:
    """Run a multi-agent sandbox race."""
    driver_list = [driver.strip().upper() for driver in agents.split(",") if driver.strip()]
    _run_agent_command(
        f"Run a full sandbox race at {race}, {laps} laps, {weather} conditions. "
        f"Drivers: {', '.join(driver_list)}. "
        "Each driver has their own strategy agent. Show lap-by-lap key moments and final result."
    )


@cli.command()
@click.option("--race", required=True, help="Race name")
@click.option("--year", default=2025, help="Season year")
@click.option("--session", default="R", type=click.Choice(["R", "Q", "FP1", "FP2", "FP3"]))
@click.option("--driver", default=None, help="Driver code (optional)")
def fetch(race: str, year: int, session: str, driver: str | None) -> None:
    """Fetch and cache F1 data."""
    suffix = f", driver {driver}" if driver else ""
    _run_agent_command(
        f"Fetch {session} session data for {race} {year}{suffix}. "
        "Cache it locally and confirm what data was retrieved."
    )


@cli.command()
@click.argument(
    "model_name",
    type=click.Choice(["tyre_model", "laptime_model", "pit_rl", "pit_policy_q", "winner_ensemble", "all"]),
)
@click.option("--years", default="2024", help="Comma-separated years e.g. 2022,2023,2024")
@click.option("--races", default=None, help="Specific races only (optional)")
def train(model_name: str, years: str, races: str | None) -> None:
    """Train ML models from the CLI."""
    year_list = [year.strip() for year in years.split(",") if year.strip()]
    race_suffix = f", races: {races}" if races else ""
    _run_agent_command(
        f"Train {model_name} using data from years: {', '.join(year_list)}{race_suffix}. "
        "Stream training progress and save the model checkpoint."
    )


@cli.command()
@click.argument("question")
def ask(question: str) -> None:
    """Ask Hannah a freeform strategy question."""
    asyncio.run(
        agent_command_module.run_agent_command(
            command_prompts.build_ask_intent(question),
            interactive=False,
            session_id="cli:direct",
            new_session=False,
            persist_session=False,
        )
    )


@cli.command(name="agent")
@click.option("--message", "-m", default=None, help="Single message to send through the shared agent runtime")
@click.option("--session", "session_id", default="cli:direct", help="Session id to resume")
@click.option("--new-session", is_flag=True, help="Create a fresh session id before sending the message")
def agent(message: str | None, session_id: str, new_session: bool) -> None:
    """Primary shared runtime surface for one-shot and interactive agent turns."""
    asyncio.run(
        agent_command_module.run_agent_command(
            message,
            interactive=message is None,
            session_id=session_id,
            new_session=new_session,
            persist_session=True,
        )
    )


@cli.command()
@click.option("--message", "-m", default=None, help="Single message to send through the chat compatibility path")
@click.option("--session", "session_id", default="cli:direct", help="Chat session id to resume")
@click.option("--new-session", is_flag=True, help="Create a fresh session id before sending the message")
def chat(message: str | None, session_id: str, new_session: bool) -> None:
    """Compatibility wrapper over the shared agent runtime."""
    asyncio.run(
        agent_command_module.run_agent_command(
            message,
            interactive=message is None,
            session_id=session_id,
            new_session=new_session,
            persist_session=True,
        )
    )


@cli.command(name="sessions")
def sessions_command() -> None:
    """List saved chat sessions."""
    print_sessions(console=console)


@cli.command(name="providers")
@click.option(
    "--env-file",
    default=".env",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Env file to inspect alongside the current process environment",
)
def providers_command(env_file: Path) -> None:
    """Show supported hosted providers and whether they are configured."""
    render_provider_status_table(console=console, env_file=env_file)


@cli.command()
@click.option("--provider", type=provider_choice(), default=None, help="Provider preset to configure")
@click.option("--api-key", default=None, help="API key to store in the env file")
@click.option("--model", default=None, help="Model override; defaults to the preset's hosted model")
@click.option(
    "--env-file",
    default=".env",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Env file to update",
)
def configure(provider: str | None, api_key: str | None, model: str | None, env_file: Path) -> None:
    """Configure a hosted provider key and default model."""
    run_provider_configure_flow(
        console=console,
        provider=provider,
        api_key=api_key,
        model=model,
        env_file=env_file,
    )


@cli.command()
def tools() -> None:
    """List all registered tools."""
    ToolRegistry().list_tools()


@cli.command()
def model() -> None:
    """Show the current model in use."""
    render_model_status(console=console)


@cli.command(name="rlm-probe")
@click.option("--base-url", default=None, help="Optional local OpenAI-compatible base URL override")
@click.option("--api-key", default=None, help="Optional API key override")
@click.option("--model-name", default=None, help="Optional model override")
@click.option("--json-output", is_flag=True, help="Print raw probe JSON")
def rlm_probe(base_url: str | None, api_key: str | None, model_name: str | None, json_output: bool) -> None:
    """Probe the optional local RLM endpoint without entering the main loop."""
    from hannah.rlm.helper import probe_runtime_helper

    payload = probe_runtime_helper(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )
    if json_output:
        console.print()
        console.print(json.dumps(payload, indent=2, sort_keys=True))
        console.print()
        return

    console.print()
    console.print(f"  [bold]RLM probe:[/bold] {'ok' if payload['ok'] else 'failed'}")
    console.print(f"  [bold]Base:[/bold] {payload['base_url']}")
    console.print(f"  [bold]Model:[/bold] {payload['model']}")
    console.print(f"  [bold]Health:[/bold] {payload['health']}")
    console.print(f"  [bold]Chat:[/bold] {payload['chat']}")
    console.print()
