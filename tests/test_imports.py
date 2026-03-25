"""Smoke tests for the new package scaffold."""

from hannah.agent.memory import Memory
from hannah.agent.tool_registry import ToolRegistry
from hannah.cli.agent_command import run_agent_command
from hannah.cli.app import cli
from hannah.cli.command_prompts import build_simulate_intent
from hannah.models.train_winner import load_and_predict


def test_memory_round_trip() -> None:
    memory = Memory(db_path="data/test_hannah_memory.db")
    memory.clear()
    memory.add("user", "test")
    assert memory.get_recent(1) == [{"role": "user", "content": "test"}]


def test_tool_registry_discovers_tools() -> None:
    tool_names = {tool["function"]["name"] for tool in ToolRegistry().get_tool_specs()}
    assert {"race_data", "race_sim", "pit_strategy", "predict_winner", "train_model"} <= tool_names


def test_winner_predictor_stub() -> None:
    probs = load_and_predict({"drivers": ["VER", "NOR", "LEC"]})
    assert set(probs.keys()) == {"VER", "NOR", "LEC"}
    assert round(sum(probs.values()), 3) == 1.0


def test_cli_object_exists() -> None:
    assert cli is not None


def test_cli_trace_command_exists() -> None:
    assert hasattr(cli, "commands")
    assert "trace" in cli.commands


def test_cli_agent_command_exists() -> None:
    assert hasattr(cli, "commands")
    assert "agent" in cli.commands


def test_primary_cli_helpers_are_importable() -> None:
    assert callable(run_agent_command)
    assert callable(build_simulate_intent)
