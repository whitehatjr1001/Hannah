# Hannah Agent Runtime Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the first bounded slice of the approved redesign: a new evented runtime core, a primary `hannah agent` CLI surface, generic depth-1 spawned workers, streamed worker/runtime events, and backward-compatible wrapper commands over the same core.

**Architecture:** Add a new `hannah/runtime/` package that owns event types, the async event bus, turn state, and the shared runtime core. Keep `hannah/agent/loop.py` as a compatibility adapter, move generic worker spawning into explicit worker runtime modules, and preserve F1 domain ownership in the existing tool and simulation layers.

**Tech Stack:** Python 3.11, `asyncio`, `click`, `rich`, LiteLLM, local fallback provider, `pytest`

---

## Scope

This plan intentionally covers **slice 1** of the approved redesign, not the entire end-state architecture.

**In scope for this plan**

- event envelope and async bus
- shared runtime core extracted from `AgentLoop`
- primary `hannah agent` command
- wrapper-equivalence for `ask` and legacy command entrypoints
- generic spawned worker runtime with `allowed_tools`
- depth-1 spawn policy (`spawn` allowed only for the main agent)
- streamed runtime and worker events in CLI/chat

**Out of scope for this plan**

- full retirement of legacy F1 worker helpers
- provider-native token streaming
- runtime event replay persistence
- dynamic runtime skill execution infrastructure

## File Structure

### New files

- Create: `hannah/runtime/__init__.py`
  Responsibility: export the runtime package surface.
- Create: `hannah/runtime/events.py`
  Responsibility: typed event envelope and event names.
- Create: `hannah/runtime/bus.py`
  Responsibility: async publish/subscribe with subscriber isolation.
- Create: `hannah/runtime/turn_state.py`
  Responsibility: correlation IDs, mutable turn state, and event-safe snapshots.
- Create: `hannah/runtime/context.py`
  Responsibility: main-agent and subagent prompt/context assembly aligned with the approved prompt guide.
- Create: `hannah/runtime/core.py`
  Responsibility: shared turn runtime for provider calls, tool execution, worker orchestration, and event emission.
- Create: `hannah/cli/agent_command.py`
  Responsibility: canonical `agent` command execution path plus wrapper prompt/intent helpers.
- Create: `hannah/cli/command_prompts.py`
  Responsibility: shared structured intents and wrapper prompt builders for legacy commands.
- Create: `hannah/agent/worker_runtime.py`
  Responsibility: `WorkerSpec`, spawn policy, worker execution, and result envelopes.
- Create: `hannah/agent/worker_registry.py`
  Responsibility: allowlisted worker templates and compatibility registration for legacy F1 roles.
- Create: `hannah/session/event_records.py`
  Responsibility: JSONL-safe serialization helpers for message and event records.
- Create: `tests/runtime/test_event_bus.py`
- Create: `tests/runtime/test_runtime_core.py`
- Create: `tests/cli/test_agent_command.py`
- Create: `tests/agent/test_worker_spawn_policy.py`
- Create: `tests/agent/test_worker_result_reinjection.py`
- Create: `tests/acceptance/test_hidden_acceptance_runtime_events.py`

### Existing files to modify

- Modify: `hannah/agent/loop.py`
  Responsibility after this slice: thin adapter over `RuntimeCore`.
- Modify: `hannah/cli/app.py`
  Responsibility after this slice: top-level CLI registration, including new `agent` command and compatibility wrappers.
- Modify: `hannah/cli/chat.py`
  Responsibility after this slice: interactive shell over the shared `agent` runtime path.
- Modify: `hannah/cli/format.py`
  Responsibility after this slice: Rich render helpers for runtime and worker events.
- Modify: `hannah/agent/subagents.py`
  Responsibility after this slice: compatibility shims that map legacy fixed-role helpers onto the generic worker runtime.
- Modify: `hannah/agent/tool_registry.py`
  Responsibility after this slice: continue schema/discovery/dispatch, with support for runtime-bound tools such as `spawn`.
- Modify: `hannah/providers/registry.py`
  Responsibility after this slice: continue provider selection while exposing whatever the new runtime core needs.
- Modify: `hannah/providers/litellm_provider.py`
  Responsibility after this slice: continue request/response provider work, with optional runtime event hooks if needed.
- Modify: `hannah/session/manager.py`
  Responsibility after this slice: persist session transcripts while preserving compatibility with current JSONL history.
- Modify: `tests/agent/test_agent_loop_seams.py`
- Modify: `tests/agent/test_subagents_provider_seams.py`
- Modify: `tests/agent/test_turn_tool_routing.py`
- Modify: `tests/agent/test_v2_s1_tool_boundary_hardening.py`
- Modify: `tests/agent/test_local_fallback.py`
- Modify: `tests/cli/test_chat_sessions.py`
- Modify: `tests/session/test_manager.py`
- Modify: `tests/acceptance/test_hidden_acceptance_agent_loop.py`
- Modify: `tests/test_imports.py`
- Modify: `hannah/docs/AGENT_LOOP.md`

## Task 1: Create Runtime Event Primitives and Async Bus

**Files:**
- Create: `hannah/runtime/__init__.py`
- Create: `hannah/runtime/events.py`
- Create: `hannah/runtime/bus.py`
- Test: `tests/runtime/test_event_bus.py`

- [ ] **Step 1: Write the failing event-bus tests**

```python
import asyncio

import pytest

from hannah.runtime.bus import RuntimeEventBus
from hannah.runtime.events import RuntimeEvent


@pytest.mark.asyncio
async def test_publish_preserves_order_for_async_subscribers():
    bus = RuntimeEventBus()
    seen: list[str] = []

    async def record(event: RuntimeEvent) -> None:
        seen.append(event.event_type)

    bus.subscribe(record)
    await bus.publish(RuntimeEvent.test("subagent_progress"))
    await bus.publish(RuntimeEvent.test("subagent_completed"))

    assert seen == ["subagent_progress", "subagent_completed"]


@pytest.mark.asyncio
async def test_publish_isolates_subscriber_failures():
    bus = RuntimeEventBus()
    seen: list[str] = []

    async def boom(event: RuntimeEvent) -> None:
        raise RuntimeError("subscriber exploded")

    async def record(event: RuntimeEvent) -> None:
        seen.append(event.event_type)

    bus.subscribe(boom)
    bus.subscribe(record)

    await bus.publish(RuntimeEvent.test("final_answer_emitted"))

    assert seen == ["final_answer_emitted"]


def test_required_event_types_match_slice_contract():
    assert REQUIRED_RUNTIME_EVENTS == {
        "user_message_received",
        "provider_request_started",
        "provider_response_received",
        "tool_call_started",
        "tool_call_finished",
        "subagent_spawned",
        "subagent_progress",
        "subagent_completed",
        "final_answer_emitted",
        "error_emitted",
    }
```

- [ ] **Step 2: Run the targeted tests to confirm the new module is missing**

Run: `.venv/bin/python -m pytest tests/runtime/test_event_bus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hannah.runtime'`

- [ ] **Step 3: Implement the event envelope and event bus**

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

Subscriber = Callable[["RuntimeEvent"], Awaitable[None]]
REQUIRED_RUNTIME_EVENTS = {
    "user_message_received",
    "provider_request_started",
    "provider_response_received",
    "tool_call_started",
    "tool_call_finished",
    "subagent_spawned",
    "subagent_progress",
    "subagent_completed",
    "final_answer_emitted",
    "error_emitted",
}


@dataclass(slots=True)
class RuntimeEvent:
    event_type: str
    timestamp: str
    session_id: str
    message_id: str
    worker_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def test(cls, event_type: str) -> "RuntimeEvent":
        assert event_type in REQUIRED_RUNTIME_EVENTS
        return cls(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="test-session",
            message_id="test-message",
        )


class RuntimeEventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    async def publish(self, event: RuntimeEvent) -> None:
        for subscriber in list(self._subscribers):
            try:
                await subscriber(event)
            except Exception:
                continue
```

- [ ] **Step 4: Re-run the targeted tests**

Run: `.venv/bin/python -m pytest tests/runtime/test_event_bus.py -v`
Expected: PASS

- [ ] **Step 5: Commit the runtime-event foundation**

```bash
git add hannah/runtime/__init__.py hannah/runtime/events.py hannah/runtime/bus.py tests/runtime/test_event_bus.py
git commit -m "feat: add runtime event bus foundation"
```

## Task 2: Extract a Shared Runtime Core From AgentLoop

**Files:**
- Create: `hannah/runtime/turn_state.py`
- Create: `hannah/runtime/context.py`
- Create: `hannah/runtime/core.py`
- Modify: `hannah/agent/loop.py`
- Test: `tests/runtime/test_runtime_core.py`
- Test: `tests/agent/test_agent_loop_seams.py`

- [ ] **Step 1: Write failing runtime-core tests for final answers and tool roundtrips**

```python
import pytest

from hannah.runtime.core import RuntimeCore


@pytest.mark.asyncio
async def test_runtime_core_emits_events_for_plain_final_answer(stub_provider, stub_registry, recording_bus):
    stub_provider.queue_text("box this lap")
    core = RuntimeCore(provider=stub_provider, registry=stub_registry, event_bus=recording_bus)

    reply = await core.run_turn(messages=[{"role": "user", "content": "call strategy"}])

    assert reply["content"] == "box this lap"
    assert recording_bus.event_types == [
        "user_message_received",
        "provider_request_started",
        "provider_response_received",
        "final_answer_emitted",
    ]


@pytest.mark.asyncio
async def test_runtime_core_emits_tool_events_and_reuses_tool_registry(stub_provider, stub_registry, recording_bus):
    stub_provider.queue_tool_then_text("race_data", {"race": "bahrain"}, "strategy locked")
    core = RuntimeCore(provider=stub_provider, registry=stub_registry, event_bus=recording_bus)

    reply = await core.run_turn(messages=[{"role": "user", "content": "get bahrain data"}])

    assert reply["content"] == "strategy locked"
    assert "tool_call_started" in recording_bus.event_types
    assert "tool_call_finished" in recording_bus.event_types
```

- [ ] **Step 2: Run the targeted runtime-core tests**

Run: `.venv/bin/python -m pytest tests/runtime/test_runtime_core.py tests/agent/test_agent_loop_seams.py -v`
Expected: FAIL because `RuntimeCore` and `TurnState` do not exist

- [ ] **Step 3: Implement `TurnState` and the first version of `RuntimeCore`**

```python
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class TurnState:
    session_id: str
    message_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[dict] = field(default_factory=list)


class RuntimeContextBuilder:
    def build_main_messages(self, messages: list[dict]) -> list[dict]:
        # keep instruction/context/input separation explicit
        return list(messages)


class RuntimeCore:
    def __init__(self, provider, registry, event_bus, memory=None, context_builder=None):
        self.provider = provider
        self.registry = registry
        self.event_bus = event_bus
        self.memory = memory
        self.context_builder = context_builder or RuntimeContextBuilder()

    async def run_turn(self, messages: list[dict], *, session_id: str = "default") -> dict:
        state = TurnState(session_id=session_id, messages=self.context_builder.build_main_messages(messages))
        await self.event_bus.publish(RuntimeEvent.test("user_message_received"))
        await self.event_bus.publish(RuntimeEvent.test("provider_request_started"))
        message = await self.provider.complete(state.messages, tools=self.registry.get_tool_specs())
        await self.event_bus.publish(RuntimeEvent.test("provider_response_received"))
        # preserve current tool-call roundtrip behavior here
        return {"role": "assistant", "content": "..."}
```

- [ ] **Step 4: Refactor `hannah/agent/loop.py` into an adapter**

```python
class AgentLoop:
    def __init__(self, ...):
        ...
        self.runtime = RuntimeCore(
            provider=self.provider,
            registry=self.registry,
            event_bus=self.event_bus,
            memory=self.memory,
        )

    async def run_turn(self, user_input: str, **kwargs):
        messages = self._build_messages(user_input=user_input, **kwargs)
        return await self.runtime.run_turn(messages, session_id=self._session_id(kwargs))
```

- [ ] **Step 5: Re-run the targeted tests**

Run: `.venv/bin/python -m pytest tests/runtime/test_runtime_core.py tests/agent/test_agent_loop_seams.py tests/agent/test_turn_tool_routing.py -v`
Expected: PASS

- [ ] **Step 6: Commit the shared runtime extraction**

```bash
git add hannah/runtime/turn_state.py hannah/runtime/context.py hannah/runtime/core.py hannah/agent/loop.py tests/runtime/test_runtime_core.py tests/agent/test_agent_loop_seams.py tests/agent/test_turn_tool_routing.py
git commit -m "refactor: extract shared runtime core from agent loop"
```

## Task 3: Make `hannah agent` the Primary Surface and Preserve Wrappers

**Files:**
- Create: `hannah/cli/agent_command.py`
- Create: `hannah/cli/command_prompts.py`
- Modify: `hannah/cli/app.py`
- Modify: `hannah/cli/chat.py`
- Test: `tests/cli/test_agent_command.py`
- Test: `tests/cli/test_chat_sessions.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write failing CLI tests for the new `agent` command and wrapper equivalence**

```python
from click.testing import CliRunner

from hannah.cli.app import cli


def test_agent_message_mode_invokes_shared_runtime(monkeypatch):
    runner = CliRunner()
    seen: list[tuple[str, str | None]] = []

    async def fake_run(message: str | None, interactive: bool = False) -> str:
        seen.append((message or "", "interactive" if interactive else "oneshot"))
        return "ok"

    monkeypatch.setattr("hannah.cli.agent_command.run_agent_command", fake_run)

    result = runner.invoke(cli, ["agent", "--message", "should we undercut"])

    assert result.exit_code == 0
    assert seen == [("should we undercut", "oneshot")]


def test_ask_is_a_backward_compatible_alias(monkeypatch):
    runner = CliRunner()
    calls: list[str] = []

    async def fake_run(message: str | None, interactive: bool = False) -> str:
        calls.append(message or "")
        return "ok"

    monkeypatch.setattr("hannah.cli.agent_command.run_agent_command", fake_run)

    result = runner.invoke(cli, ["ask", "who wins monza"])

    assert result.exit_code == 0
    assert calls == ["who wins monza"]


def test_simulate_is_a_wrapper_over_shared_agent_runtime(monkeypatch):
    runner = CliRunner()
    intents: list[dict] = []

    monkeypatch.setattr("hannah.cli.agent_command.run_agent_command", lambda *args, **kwargs: "ok")
    monkeypatch.setattr("hannah.cli.command_prompts.build_simulate_intent", lambda race, driver, laps: intents.append({"race": race, "driver": driver, "laps": laps}) or "simulate prompt")

    result = runner.invoke(cli, ["simulate", "--race", "bahrain", "--driver", "VER", "--laps", "57"])

    assert result.exit_code == 0
    assert intents == [{"race": "bahrain", "driver": "VER", "laps": 57}]
```

- [ ] **Step 2: Run the targeted CLI tests**

Run: `.venv/bin/python -m pytest tests/cli/test_agent_command.py tests/cli/test_chat_sessions.py tests/test_imports.py -v`
Expected: FAIL because `agent`, the shared command helper, and centralized wrapper prompt builders do not exist

- [ ] **Step 3: Implement the shared CLI execution helper**

```python
async def run_agent_command(message: str | None, *, interactive: bool, session_name: str | None = None) -> str:
    if interactive:
        return await run_chat_session(session_name=session_name)
    loop = AgentLoop(...)
    return await loop.run_command(message or "")


def build_simulate_intent(*, race: str, driver: str | None, laps: int | None) -> str:
    return f"Simulate race={race} driver={driver or 'auto'} laps={laps or 'default'}"
```

- [ ] **Step 4: Register `agent`, preserve `cli`, and convert legacy command entrypoints into wrappers**

```python
@cli.command("agent")
@click.option("--message", type=str, default=None)
def agent_command(message: str | None) -> None:
    interactive = message is None
    output = asyncio.run(run_agent_command(message, interactive=interactive))
    if output:
        click.echo(output)


@cli.command("ask")
@click.argument("prompt")
def ask(prompt: str) -> None:
    output = asyncio.run(run_agent_command(prompt, interactive=False))
    click.echo(output)


@cli.command("simulate")
def simulate(...) -> None:
    prompt = build_simulate_intent(...)
    output = asyncio.run(run_agent_command(prompt, interactive=False))
    click.echo(output)
```

- [ ] **Step 5: Re-run the targeted CLI tests**

Run: `.venv/bin/python -m pytest tests/cli/test_agent_command.py tests/cli/test_chat_sessions.py tests/test_imports.py -v`
Expected: PASS

- [ ] **Step 6: Commit the new primary CLI surface**

```bash
git add hannah/cli/agent_command.py hannah/cli/command_prompts.py hannah/cli/app.py hannah/cli/chat.py tests/cli/test_agent_command.py tests/cli/test_chat_sessions.py tests/test_imports.py
git commit -m "feat: add primary hannah agent command"
```

## Task 4: Add Generic Depth-1 Worker Spawning

**Files:**
- Create: `hannah/agent/worker_runtime.py`
- Create: `hannah/agent/worker_registry.py`
- Modify: `hannah/agent/tool_registry.py`
- Modify: `hannah/agent/subagents.py`
- Modify: `hannah/runtime/core.py`
- Test: `tests/agent/test_worker_spawn_policy.py`
- Test: `tests/agent/test_subagents_provider_seams.py`
- Test: `tests/agent/test_turn_tool_routing.py`

- [ ] **Step 1: Write failing tests for the `spawn` runtime tool, worker validation, and no-nested-spawn policy**

```python
import pytest

from hannah.agent.worker_runtime import WorkerPolicyError, WorkerSpec, validate_worker_spec


def test_worker_spec_requires_allowed_tools():
    spec = WorkerSpec(
        worker_id="telemetry-1",
        task="fetch race data",
        system_prompt="You are a telemetry worker.",
        allowed_tools=[],
        result_contract={"summary": "string"},
    )

    with pytest.raises(WorkerPolicyError):
        validate_worker_spec(spec)


def test_nested_spawn_is_rejected():
    spec = WorkerSpec(
        worker_id="planner-1",
        task="spawn another worker",
        system_prompt="You are a planner worker.",
        allowed_tools=["spawn"],
        result_contract={"summary": "string"},
    )

    with pytest.raises(WorkerPolicyError):
        validate_worker_spec(spec)


@pytest.mark.asyncio
async def test_main_runtime_executes_spawn_tool_via_normal_tool_roundtrip(stub_provider, runtime_core, recording_bus):
    stub_provider.queue_tool_then_text(
        "spawn",
        {
            "task": "compare Bahrain one-stop versus two-stop",
            "system_prompt": "You are a strategy worker.",
            "allowed_tools": ["race_data", "race_sim", "pit_strategy"],
            "result_contract": {"summary": "string", "evidence": "list"},
        },
        "Use the two-stop window."
    )

    reply = await runtime_core.run_turn(messages=[{"role": "user", "content": "compare Bahrain strategies"}])

    assert reply["content"] == "Use the two-stop window."
    assert "subagent_spawned" in recording_bus.event_types
    assert "subagent_completed" in recording_bus.event_types
```

- [ ] **Step 2: Run the worker-policy tests**

Run: `.venv/bin/python -m pytest tests/agent/test_worker_spawn_policy.py tests/agent/test_subagents_provider_seams.py tests/agent/test_turn_tool_routing.py -v`
Expected: FAIL because the worker runtime modules do not exist and `spawn` is not in the runtime tool surface

- [ ] **Step 3: Implement `WorkerSpec`, `WorkerResult`, validation, and the runtime `spawn` tool contract**

```python
from dataclasses import dataclass


class WorkerPolicyError(ValueError):
    pass


@dataclass(slots=True)
class WorkerSpec:
    worker_id: str
    task: str
    system_prompt: str
    allowed_tools: list[str]
    result_contract: dict[str, str]


def validate_worker_spec(spec: WorkerSpec) -> None:
    if not spec.allowed_tools:
        raise WorkerPolicyError("allowed_tools must not be empty")
    if "spawn" in spec.allowed_tools:
        raise WorkerPolicyError("nested spawn is not allowed in slice 1")


SPAWN_TOOL_SPEC = {
    "name": "spawn",
    "description": "Spawn a bounded subagent with allowed tools and a result contract.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "system_prompt": {"type": "string"},
            "allowed_tools": {"type": "array", "items": {"type": "string"}},
            "result_contract": {"type": "object"},
        },
        "required": ["task", "system_prompt", "allowed_tools", "result_contract"],
    },
}
```

- [ ] **Step 4: Implement runtime-bound `spawn` registration, the generic worker runtime, and legacy compatibility shims**

```python
class ToolRegistry:
    def with_runtime_tools(self, handlers: dict[str, object]) -> "ToolRegistry":
        ...


class WorkerRuntime:
    async def run_worker(self, spec: WorkerSpec, *, parent_session_id: str) -> dict:
        validate_worker_spec(spec)
        # build a restricted tool registry view and run the same provider/tool loop shape
        return {"worker_id": spec.worker_id, "status": "completed", "result": {...}}


def build_legacy_worker_specs(ctx) -> list[WorkerSpec]:
    return [
        WorkerSpec(
            worker_id="strategy",
            task="analyze pit window",
            system_prompt="You are an F1 strategy worker.",
            allowed_tools=["race_data", "race_sim", "pit_strategy"],
            result_contract={"summary": "string", "evidence": "list"},
        )
    ]


async def handle_spawn_tool(args: dict) -> dict:
    spec = WorkerSpec(**args)
    return await worker_runtime.run_worker(spec, parent_session_id=state.session_id)
```

- [ ] **Step 5: Re-run the worker tests**

Run: `.venv/bin/python -m pytest tests/agent/test_worker_spawn_policy.py tests/agent/test_subagents_provider_seams.py tests/agent/test_turn_tool_routing.py tests/agent/test_local_fallback.py -v`
Expected: PASS

- [ ] **Step 6: Commit the worker-runtime foundation**

```bash
git add hannah/agent/worker_runtime.py hannah/agent/worker_registry.py hannah/agent/tool_registry.py hannah/agent/subagents.py hannah/runtime/core.py tests/agent/test_worker_spawn_policy.py tests/agent/test_subagents_provider_seams.py tests/agent/test_turn_tool_routing.py tests/agent/test_local_fallback.py
git commit -m "feat: add generic worker runtime with depth-1 spawn policy"
```

## Task 5: Reinject Worker Results and Stream Runtime Events

**Files:**
- Modify: `hannah/runtime/core.py`
- Modify: `hannah/cli/format.py`
- Modify: `hannah/cli/chat.py`
- Create: `hannah/session/event_records.py`
- Modify: `hannah/session/manager.py`
- Test: `tests/agent/test_worker_result_reinjection.py`
- Test: `tests/runtime/test_runtime_core.py`
- Test: `tests/session/test_manager.py`

- [ ] **Step 1: Write failing tests for `spawn`-result reinjection, event rendering, and event-record persistence**

```python
import pytest

from hannah.runtime.events import RuntimeEvent


@pytest.mark.asyncio
async def test_spawn_tool_result_is_reinjected_with_stable_message_shape(stub_provider, runtime_core):
    stub_provider.queue_tool_then_text(
        "spawn",
        {
            "task": "compare strategies",
            "system_prompt": "You are a strategy worker.",
            "allowed_tools": ["race_data", "race_sim"],
            "result_contract": {"summary": "string"},
        },
        "Prefer the two-stop."
    )

    reply = await runtime_core.run_turn(messages=[{"role": "user", "content": "compare strategies"}])

    subagent_messages = [m for m in reply["messages"] if m["role"] == "system" and m.get("name") == "subagent_result"]
    assert len(subagent_messages) == 1
    assert subagent_messages[0]["worker_id"] == "strategy"


def test_render_runtime_event_formats_subagent_progress():
    event = RuntimeEvent.test("subagent_progress")
    event.worker_id = "strategy"
    event.payload = {"message": "Running race_sim"}

    renderable = render_runtime_event(event)

    assert "strategy" in str(renderable)
    assert "Running race_sim" in str(renderable)


def test_event_records_use_a_stable_jsonl_shape():
    record = serialize_event_record(
        RuntimeEvent.test("subagent_completed"),
        session_id="session-1",
    )

    assert record["record_type"] == "event"
    assert record["session_id"] == "session-1"
    assert record["payload"]["event_type"] == "subagent_completed"
```

- [ ] **Step 2: Run the targeted reinjection and session tests**

Run: `.venv/bin/python -m pytest tests/agent/test_worker_result_reinjection.py tests/runtime/test_runtime_core.py tests/session/test_manager.py -v`
Expected: FAIL because `spawn`-result reinjection, event-record helpers, and render helpers are missing

- [ ] **Step 3: Implement worker-result reinjection in the runtime core**

```python
def _worker_result_message(result: dict) -> dict:
    return {
        "role": "system",
        "name": "subagent_result",
        "worker_id": result["worker_id"],
        "content": json.dumps(result),
    }


async def _handle_spawn_result(self, result: dict, state: TurnState) -> None:
    state.messages.append(_worker_result_message(result))
    await self.event_bus.publish(
        RuntimeEvent(
            event_type="subagent_completed",
            timestamp=...,
            session_id=state.session_id,
            message_id=state.message_id,
            worker_id=result["worker_id"],
            payload=result,
        )
    )
```

- [ ] **Step 4: Implement Rich event rendering and explicit event-record persistence**

```python
def serialize_event_record(event: RuntimeEvent, *, session_id: str) -> dict:
    return {
        "record_type": "event",
        "session_id": session_id,
        "created_at": event.timestamp,
        "payload": {
            "event_type": event.event_type,
            "message_id": event.message_id,
            "worker_id": event.worker_id,
            "payload": event.payload,
        },
    }


class SessionManager:
    def append_event(self, session_name: str, event: RuntimeEvent) -> None:
        self._append_record(session_name, serialize_event_record(event, session_id=session_name))


def render_runtime_event(event: RuntimeEvent):
    if event.event_type == "subagent_spawned":
        return Text(f"[subagent] {event.worker_id} spawned")
    if event.event_type == "subagent_progress":
        return Text(f"[subagent] {event.worker_id}: {event.payload['message']}")
    return Text(event.event_type)
```

- [ ] **Step 5: Re-run the reinjection and session tests**

Run: `.venv/bin/python -m pytest tests/agent/test_worker_result_reinjection.py tests/runtime/test_runtime_core.py tests/session/test_manager.py tests/cli/test_chat_sessions.py -v`
Expected: PASS

- [ ] **Step 6: Commit runtime streaming and reinjection**

```bash
git add hannah/runtime/core.py hannah/cli/format.py hannah/cli/chat.py hannah/session/event_records.py hannah/session/manager.py tests/agent/test_worker_result_reinjection.py tests/runtime/test_runtime_core.py tests/session/test_manager.py tests/cli/test_chat_sessions.py
git commit -m "feat: stream runtime events and reinject worker results"
```

## Task 6: Lock Compatibility, Docs, and Acceptance Validation

**Files:**
- Modify: `tests/agent/test_v2_s1_tool_boundary_hardening.py`
- Modify: `tests/acceptance/test_hidden_acceptance_agent_loop.py`
- Create: `tests/acceptance/test_hidden_acceptance_runtime_events.py`
- Modify: `hannah/docs/AGENT_LOOP.md`

- [ ] **Step 1: Write or extend acceptance tests for wrapper-equivalence and spawn boundaries**

```python
def test_agent_and_ask_follow_the_same_runtime_path(...):
    ...


def test_disallowed_nested_spawn_is_contained_and_reported(...):
    ...


def test_subagent_events_are_streamed_in_order(...):
    ...


def test_simulate_predict_and_strategy_are_wrappers_over_agent_runtime(...):
    ...
```

- [ ] **Step 2: Run the acceptance-focused test targets**

Run: `.venv/bin/python -m pytest tests/agent/test_v2_s1_tool_boundary_hardening.py tests/acceptance/test_hidden_acceptance_agent_loop.py tests/acceptance/test_hidden_acceptance_runtime_events.py -v`
Expected: FAIL until the new runtime path is fully wired

- [ ] **Step 3: Update `hannah/docs/AGENT_LOOP.md` to match the new architecture**

```md
- `hannah agent` is now the primary runtime surface.
- `AgentLoop` is a compatibility adapter over `RuntimeCore`.
- The main tool surface includes a runtime-bound `spawn` tool.
- Generic workers use structured `WorkerSpec` objects with `allowed_tools`.
- Nested spawn is disallowed in slice 1.
- Runtime events use the `subagent_*` naming contract from the approved spec.
- Runtime events drive streaming output and JSONL event records.
```

- [ ] **Step 4: Run the full public suite**

Run: `.venv/bin/python -m pytest`
Expected: PASS

- [ ] **Step 5: Run targeted CLI smokes**

Run: `.venv/bin/python hannah.py agent --message "Should we undercut in Bahrain if the degradation curve is steep?"`
Expected: exits `0` and prints an answer from the shared runtime path

Run: `.venv/bin/python hannah.py ask "Predict the Monza winner from prior pace data"`
Expected: exits `0` and matches the same one-shot runtime behavior

Run: `.venv/bin/python hannah.py chat --message "Summarize likely pit windows for a dry race"`
Expected: exits `0` and routes through the same `agent` runtime compatibility path

Run: `.venv/bin/python hannah.py simulate --race bahrain --driver VER --laps 57`
Expected: exits `0` and routes through the same shared runtime path via wrapper intent

Run: `.venv/bin/python hannah.py predict --race monza --year 2025`
Expected: exits `0` and routes through the same shared runtime path via wrapper intent

Run: `.venv/bin/python hannah.py strategy --race bahrain --lap 22 --driver VER`
Expected: exits `0` and routes through the same shared runtime path via wrapper intent

- [ ] **Step 6: Commit the compatibility/docs finish**

```bash
git add tests/agent/test_v2_s1_tool_boundary_hardening.py tests/acceptance/test_hidden_acceptance_agent_loop.py tests/acceptance/test_hidden_acceptance_runtime_events.py hannah/docs/AGENT_LOOP.md
git commit -m "docs: align agent loop docs with runtime slice 1"
```

## Final Validation Checklist

- [ ] `.venv/bin/python -m pytest tests/runtime/test_event_bus.py tests/runtime/test_runtime_core.py -v`
- [ ] `.venv/bin/python -m pytest tests/cli/test_agent_command.py tests/cli/test_chat_sessions.py tests/test_imports.py -v`
- [ ] `.venv/bin/python -m pytest tests/agent/test_worker_spawn_policy.py tests/agent/test_worker_result_reinjection.py tests/agent/test_subagents_provider_seams.py -v`
- [ ] `.venv/bin/python -m pytest tests/session/test_manager.py tests/agent/test_v2_s1_tool_boundary_hardening.py -v`
- [ ] `.venv/bin/python -m pytest tests/acceptance/test_hidden_acceptance_agent_loop.py tests/acceptance/test_hidden_acceptance_runtime_events.py -v`
- [ ] `.venv/bin/python -m pytest`

## Notes for the Implementer

- Keep `tool_registry.py` as the tool boundary. Do not move orchestration there.
- Keep F1 domain ownership in existing tool and simulation modules.
- Do not let worker execution create a second provider/tool loop shape; reuse the same runtime semantics.
- Do not add nested `spawn` in this slice.
- If a task threatens to pull in provider-native streaming or runtime replay persistence, stop and split it into a later plan.
