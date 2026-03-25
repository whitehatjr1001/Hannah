"""Generic bounded worker execution for depth-1 spawn flows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from hannah.runtime.context import RuntimeContextBuilder
from hannah.runtime.events import EventEnvelope
from hannah.utils.console import Console


SPAWN_TOOL_SPEC = {
    "name": "spawn",
    "description": "Spawn a bounded worker with an allowlisted tool surface.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "system_prompt": {"type": "string", "minLength": 1},
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "result_contract": {"type": "object"},
        },
        "required": ["task", "system_prompt", "allowed_tools", "result_contract"],
        "additionalProperties": False,
    },
}


class WorkerPolicyError(ValueError):
    """Raised when a worker spec violates the slice policy."""


@dataclass(slots=True)
class WorkerSpec:
    worker_id: str
    task: str
    system_prompt: str
    allowed_tools: list[str]
    result_contract: dict[str, Any]


@dataclass(slots=True)
class WorkerResult:
    worker_id: str
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "worker_id": self.worker_id,
            "status": self.status,
            "result": self.result,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


def validate_worker_spec(spec: WorkerSpec) -> None:
    if not spec.worker_id.strip():
        raise WorkerPolicyError("worker_id must not be empty")
    if not spec.task.strip():
        raise WorkerPolicyError("task must not be empty")
    if not spec.system_prompt.strip():
        raise WorkerPolicyError("system_prompt must not be empty")
    if not spec.allowed_tools:
        raise WorkerPolicyError("allowed_tools must not be empty")
    if "spawn" in spec.allowed_tools:
        raise WorkerPolicyError("nested spawn is not allowed in slice 1")
    if not isinstance(spec.result_contract, dict) or not spec.result_contract:
        raise WorkerPolicyError("result_contract must not be empty")


def make_worker_id(prefix: str = "worker") -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


class WorkerRuntime:
    """Run a bounded worker through the shared provider/tool runtime shape."""

    def __init__(
        self,
        *,
        provider: object,
        registry: object,
        event_bus: object,
        context_builder: RuntimeContextBuilder | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        console: Console | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.event_bus = event_bus
        self.context_builder = context_builder or RuntimeContextBuilder()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.console = console or Console()

    async def run_worker(self, spec: WorkerSpec, *, parent_session_id: str) -> dict[str, Any]:
        validate_worker_spec(spec)
        allowed_tools = self._resolve_allowed_tools(spec.allowed_tools)
        await self._publish(
            "subagent_spawned",
            session_id=parent_session_id,
            worker_id=spec.worker_id,
            payload={"task": spec.task, "allowed_tools": list(allowed_tools)},
        )

        try:
            from hannah.runtime.core import RuntimeCore

            restricted_registry = self._restricted_registry(allowed_tools)
            await self._publish(
                "subagent_progress",
                session_id=parent_session_id,
                worker_id=spec.worker_id,
                payload={
                    "message": (
                        f"Running worker task with tools: {', '.join(allowed_tools)}"
                    )
                },
            )
            child_core = RuntimeCore(
                provider=self.provider,
                registry=restricted_registry,
                event_bus=self.event_bus,
                context_builder=self.context_builder,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                console=self.console,
                allow_spawn_tool=False,
            )
            reply = await child_core.run_turn(
                messages=[
                    {"role": "system", "content": spec.system_prompt},
                    {"role": "user", "content": spec.task},
                ],
                session_id=parent_session_id,
            )
            result_payload = self._coerce_result(reply.get("content", ""), spec.result_contract)
            await self._publish(
                "subagent_progress",
                session_id=parent_session_id,
                worker_id=spec.worker_id,
                payload={"message": "Validating worker result"},
            )
            contract_errors = self._validate_result_contract(result_payload, spec.result_contract)
            if contract_errors:
                result = WorkerResult(
                    worker_id=spec.worker_id,
                    status="error",
                    result=result_payload,
                    error=f"result_contract violation: {'; '.join(contract_errors)}",
                )
                await self._publish(
                    "subagent_completed",
                    session_id=parent_session_id,
                    worker_id=spec.worker_id,
                    payload={"status": result.status, "error": result.error},
                )
                return result.to_dict()
            result = WorkerResult(
                worker_id=spec.worker_id,
                status="completed",
                result=result_payload,
            )
            await self._publish(
                "subagent_completed",
                session_id=parent_session_id,
                worker_id=spec.worker_id,
                payload={"status": result.status},
            )
            return result.to_dict()
        except Exception as err:
            await self._publish(
                "subagent_completed",
                session_id=parent_session_id,
                worker_id=spec.worker_id,
                payload={"status": "error", "error": str(err)},
            )
            raise

    def _resolve_allowed_tools(self, requested_tools: list[str]) -> list[str]:
        known_tools = self._known_tool_names()
        unknown_tools = sorted({name for name in requested_tools if name not in known_tools})
        if unknown_tools:
            joined = ", ".join(unknown_tools)
            raise WorkerPolicyError(f"unknown allowed_tools requested: {joined}")
        return list(requested_tools)

    def _restricted_registry(self, allowed_tools: list[str]) -> object:
        subset = getattr(self.registry, "subset", None)
        restricted_registry = subset(allowed_tools) if callable(subset) else self.registry
        tool_specs = getattr(restricted_registry, "get_tool_specs", None)
        if callable(tool_specs) and not tool_specs():
            raise WorkerPolicyError("allowed_tools resolved to an empty worker tool surface")
        return restricted_registry

    def _known_tool_names(self) -> set[str]:
        tool_names = getattr(self.registry, "tool_names", None)
        if callable(tool_names):
            return set(tool_names())
        tool_specs = getattr(self.registry, "get_tool_specs", None)
        if not callable(tool_specs):
            return set()
        names: set[str] = set()
        for spec in tool_specs():
            function = spec.get("function", {})
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
        return names

    async def _publish(
        self,
        event_type: str,
        *,
        session_id: str,
        worker_id: str,
        payload: dict[str, Any],
    ) -> None:
        await self.event_bus.publish(
            EventEnvelope.create(
                event_type=event_type,
                session_id=session_id,
                message_id=worker_id,
                worker_id=worker_id,
                payload=payload,
            )
        )

    def _coerce_result(
        self,
        raw_content: str,
        result_contract: dict[str, Any],
    ) -> dict[str, Any]:
        stripped = raw_content.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            summary_key = "summary" if "summary" in result_contract else "content"
            return {summary_key: stripped}
        if isinstance(parsed, dict):
            return parsed
        summary_key = "summary" if "summary" in result_contract else "content"
        if isinstance(parsed, str):
            return {summary_key: parsed}
        return {summary_key: json.dumps(parsed, default=str)}

    def _validate_result_contract(
        self,
        result: dict[str, Any],
        result_contract: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        for key, expected in result_contract.items():
            if key not in result:
                errors.append(f"missing required {key}")
                continue
            error = self._validate_contract_value(key, result[key], expected)
            if error is not None:
                errors.append(error)
        return errors

    def _validate_contract_value(
        self,
        key: str,
        value: Any,
        expected: Any,
    ) -> str | None:
        if not isinstance(expected, str):
            return None
        expected_type = expected.strip().lower()
        if expected_type == "string" and not isinstance(value, str):
            return f"{key} should be string"
        if expected_type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return f"{key} should be integer"
        if expected_type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"{key} should be number"
        if expected_type == "boolean" and not isinstance(value, bool):
            return f"{key} should be boolean"
        if expected_type in {"list", "array"} and not isinstance(value, list):
            return f"{key} should be list"
        if expected_type in {"object", "dict", "map"} and not isinstance(value, dict):
            return f"{key} should be object"
        return None
