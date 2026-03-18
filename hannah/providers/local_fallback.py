"""Deterministic local fallback provider for offline orchestration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from hannah.domain.teams import DRIVER_ALIASES, DRIVER_GRID


@dataclass(frozen=True)
class LocalFunction:
    """Function call metadata for local tool-calling responses."""

    name: str
    arguments: str

    def model_dump(self) -> dict[str, str]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class LocalToolCall:
    """Tool call envelope compatible with the agent loop."""

    id: str
    function: LocalFunction
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


@dataclass(frozen=True)
class LocalMessage:
    """Message object compatible with LiteLLM response usage in the loop."""

    role: str
    content: str
    tool_calls: list[LocalToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True)
class LocalChoice:
    """Single-choice wrapper matching the expected provider response shape."""

    message: LocalMessage


@dataclass(frozen=True)
class LocalCompletion:
    """Completion response wrapper matching `response.choices[0].message`."""

    choices: list[LocalChoice]


@dataclass
class DeterministicFallbackPlanner:
    """Plan tool calls and synthesize an answer without external model access."""

    default_race: str = "bahrain"
    default_year: int = 2025
    tool_counter: int = field(default=0, init=False)

    _RACES: tuple[str, ...] = (
        "abu_dhabi",
        "bahrain",
        "barcelona",
        "imola",
        "interlagos",
        "jeddah",
        "miami",
        "monaco",
        "monza",
        "silverstone",
        "singapore",
        "spa",
    )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> LocalCompletion:
        """Return a deterministic tool-call or final synthesis completion."""
        user_text = self._latest_user_text(messages)
        tool_outputs = self._collect_tool_outputs(messages)
        available_tools = self._available_tool_names(tools)

        if available_tools and not tool_outputs:
            tool_calls = self._plan_tool_calls(user_text, available_tools)
            if tool_calls:
                return LocalCompletion(
                    choices=[LocalChoice(message=LocalMessage(role="assistant", content="", tool_calls=tool_calls))]
                )

        final_text = self._synthesize(user_text, tool_outputs)
        return LocalCompletion(choices=[LocalChoice(message=LocalMessage(role="assistant", content=final_text))])

    def _latest_user_text(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", "")).strip()
        return ""

    def _collect_tool_outputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for message in messages:
            if message.get("role") != "tool":
                continue
            name = str(message.get("name", "")).strip()
            if not name:
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                parsed = self._maybe_json(content)
                outputs[name] = parsed
            else:
                outputs[name] = content
        return outputs

    def _available_tool_names(self, tools: list[dict] | None) -> set[str]:
        names: set[str] = set()
        for tool in tools or []:
            function = tool.get("function", {})
            name = function.get("name")
            if isinstance(name, str) and name:
                names.add(name)
        return names

    def _plan_tool_calls(self, user_text: str, available: set[str]) -> list[LocalToolCall]:
        lowered = user_text.lower()
        race = self._extract_race(user_text)
        year = self._extract_year(user_text)
        drivers = self._extract_drivers(user_text)
        weather = self._extract_weather(user_text)
        lap = self._extract_lap(user_text)

        calls: list[tuple[str, dict[str, Any]]] = []
        if any(token in lowered for token in ("train", "training")):
            model_name = self._extract_model_name(lowered)
            years = self._extract_years(user_text)
            args: dict[str, Any] = {"model_name": model_name}
            if years:
                args["years"] = years
            calls.append(("train_model", args))
            return self._build_tool_calls(calls, available)

        if any(token in lowered for token in ("predict", "winner", "podium", "probability")):
            calls.append(("race_data", {"race": race, "year": year, "session": "Q", "driver": None}))
            calls.append(("predict_winner", {"race": race, "year": year, "drivers": drivers}))
            return self._build_tool_calls(calls, available)

        if any(token in lowered for token in ("simulate", "simulation", "sandbox", "strategy", "pit")):
            calls.append(("race_data", {"race": race, "year": year, "session": "R", "driver": drivers[0]}))
            calls.append(
                (
                    "race_sim",
                    {
                        "race": race,
                        "year": year,
                        "weather": weather,
                        "drivers": drivers,
                        "laps": 57,
                    },
                )
            )
            if any(token in lowered for token in ("strategy", "pit")):
                calls.append(
                    (
                        "pit_strategy",
                        {"race": race, "year": year, "driver": drivers[0], "lap": lap},
                    )
                )
            return self._build_tool_calls(calls, available)

        if any(token in lowered for token in ("fetch", "data", "telemetry", "session")):
            calls.append(("race_data", {"race": race, "year": year, "session": "R", "driver": None}))
            return self._build_tool_calls(calls, available)

        return []

    def _build_tool_calls(self, calls: list[tuple[str, dict[str, Any]]], available: set[str]) -> list[LocalToolCall]:
        planned: list[LocalToolCall] = []
        for name, args in calls:
            if name not in available:
                continue
            self.tool_counter += 1
            planned.append(
                LocalToolCall(
                    id=f"local-call-{self.tool_counter}",
                    function=LocalFunction(name=name, arguments=json.dumps(args)),
                )
            )
        return planned

    def _synthesize(self, user_text: str, tool_outputs: dict[str, Any]) -> str:
        if "train_model" in tool_outputs:
            saved = tool_outputs["train_model"]
            if isinstance(saved, dict) and "saved" in saved:
                artifact = saved["saved"]
            else:
                artifact = saved
            return f"Training complete. Saved artifacts: {artifact}."

        if "pit_strategy" in tool_outputs:
            strategy_payload = tool_outputs["pit_strategy"]
            if isinstance(strategy_payload, dict):
                lap = strategy_payload.get("recommended_pit_lap", "unknown")
                compound = strategy_payload.get("recommended_compound", "unknown")
                confidence = strategy_payload.get("confidence", 0.0)
                reasoning = strategy_payload.get("reasoning", "No reasoning provided.")
                return (
                    f"Recommendation: pit around lap {lap} for {compound}. "
                    f"Confidence: {confidence}. {reasoning}"
                )

        if "predict_winner" in tool_outputs:
            winner_payload = tool_outputs["predict_winner"]
            probs = winner_payload.get("winner_probs", {}) if isinstance(winner_payload, dict) else {}
            if isinstance(probs, dict) and probs:
                ordered = sorted(probs.items(), key=lambda pair: pair[1], reverse=True)
                top = ", ".join(f"{driver} {prob:.3f}" for driver, prob in ordered[:3])
                return f"Prediction complete. Top probabilities: {top}."
            return "Prediction complete. No winner probabilities were produced."

        if "race_sim" in tool_outputs:
            sim_payload = tool_outputs["race_sim"]
            if isinstance(sim_payload, dict):
                strategy = sim_payload.get("strategy", {})
                simulation = sim_payload.get("simulation", {})
                lap = strategy.get("recommended_pit_lap", "unknown")
                compound = strategy.get("recommended_compound", "unknown")
                p50 = simulation.get("p50_race_time_s", "unknown")
                return (
                    f"Simulation complete. Recommended pit lap {lap} on {compound}. "
                    f"Projected p50 race time: {p50} seconds."
                )

        if "race_data" in tool_outputs:
            data = tool_outputs["race_data"]
            if isinstance(data, dict):
                session = data.get("session_info", {})
                race = session.get("race", self.default_race)
                year = session.get("year", self.default_year)
                drivers = data.get("drivers", [])
                return f"Data fetched for {race} {year}. Drivers in context: {drivers}."

        if user_text.strip():
            return (
                "No external model was available, but local planning is active. "
                "Ask for simulation, strategy, prediction, fetch, or training commands, "
                "or run `hannah providers` / `hannah configure` to connect OpenAI, Claude, or Gemini."
            )
        return "No input received."

    def _extract_race(self, text: str) -> str:
        normalized = text.lower().replace(" ", "_")
        for race in self._RACES:
            if race in normalized:
                return race
        return self.default_race

    def _extract_year(self, text: str) -> int:
        years = self._extract_years(text)
        return years[0] if years else self.default_year

    def _extract_years(self, text: str) -> list[int]:
        years = [int(match) for match in re.findall(r"\b20\d{2}\b", text)]
        return sorted(set(years))

    def _extract_drivers(self, text: str) -> list[str]:
        matches = re.findall(r"\b[A-Z]{3}\b", text.upper())
        unique: list[str] = []
        for code in matches:
            if code in DRIVER_GRID and code not in unique:
                unique.append(code)
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        for word in words:
            alias = DRIVER_ALIASES.get(word.lower())
            if alias is not None and alias not in unique:
                unique.append(alias)
        if unique:
            return unique[:3]
        return ["VER", "NOR", "LEC"]

    def _extract_weather(self, text: str) -> str:
        lowered = text.lower()
        if "wet" in lowered:
            return "wet"
        if "mixed" in lowered or "crossover" in lowered:
            return "mixed"
        return "dry"

    def _extract_lap(self, text: str) -> int:
        match = re.search(r"\blap\s+(\d{1,2})\b", text.lower())
        if match is None:
            return 20
        return max(int(match.group(1)), 1)

    def _extract_model_name(self, lowered_text: str) -> str:
        if "all" in lowered_text:
            return "all"
        if "tyre" in lowered_text or "tire" in lowered_text:
            return "tyre_model"
        if "laptime" in lowered_text or "lap time" in lowered_text:
            return "laptime_model"
        if "winner" in lowered_text or "ensemble" in lowered_text:
            return "winner_ensemble"
        if "pit" in lowered_text and "rl" in lowered_text:
            return "pit_rl"
        return "all"

    def _maybe_json(self, content: str) -> Any:
        stripped = content.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return content
