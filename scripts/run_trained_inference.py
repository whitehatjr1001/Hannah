"""Run direct inference against the v2 trained artifacts."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

from hannah.models.inference_v2 import available_races, run_inference

console = Console()


@click.command()
@click.option("--year", required=True, type=int)
@click.option("--race", required=True, type=str, help="Race slug, e.g. australian_grand_prix")
def main(year: int, race: str) -> None:
    """Load trained artifacts and run direct inference for one race parquet."""
    if race not in available_races(year):
        console.print(f"[red]Race parquet not found for {year}: {race}[/red]")
        races = available_races(year)
        if races:
            console.print(f"[dim]Available races: {', '.join(races)}[/dim]")
        raise SystemExit(1)

    result = run_inference(year=year, race=race)

    summary = Table(title=f"Direct Inference — {race} {year}")
    summary.add_column("Model", style="cyan")
    summary.add_column("Signal", style="magenta")
    summary.add_column("Value", justify="right")
    summary.add_row("Tyre", "RMSE", f"{result['tyre']['rmse']:.3f}s")
    summary.add_row("Lap Time", "RMSE", f"{result['laptime']['rmse']:.3f}s")
    summary.add_row("Pit", "Mean Prob", f"{result['pit']['mean_pit_probability']:.4f}")
    top_winner = result["winner"]["winner_probs"][0] if result["winner"]["winner_probs"] else {"driver": "N/A", "win_probability": 0.0}
    summary.add_row("Winner", "Top Pick", f"{top_winner['driver']} ({top_winner['win_probability']:.4f})")
    console.print(summary)
    console.print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
