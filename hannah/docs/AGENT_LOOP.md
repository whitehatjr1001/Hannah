# AGENT_LOOP.md

## Purpose

This document describes the current Hannah agent runtime after slice 1.

The primary runtime surface is `hannah agent`. Legacy commands such as `ask` and `chat` are compatibility wrappers over the same shared runtime path. `AgentLoop` still exists, but it is now a compatibility adapter over `RuntimeCore`.

The short version:

1. a CLI surface collects a user message
2. `RuntimeCore` owns the turn, provider call, tool execution, worker spawning, and runtime events
3. `AgentLoop` adapts existing memory/persona/tool-selection behavior onto that core
4. runtime events stream to the terminal and persist to JSONL session records
5. tools and simulation layers still own the F1 domain work

---

## Main Files

| File | Responsibility |
| --- | --- |
| `hannah.py` | Root CLI wrapper |
| `hannah/cli/app.py` | Registers `agent` as the primary runtime command plus compatibility wrappers |
| `hannah/cli/agent_command.py` | Shared one-shot and interactive execution path for `agent` and wrapper commands |
| `hannah/cli/chat.py` | Session-aware interactive shell and runtime-event subscription |
| `hannah/agent/loop.py` | Compatibility adapter that builds prompts, selects tools, and delegates turn execution to `RuntimeCore` |
| `hannah/runtime/core.py` | Shared runtime owner for provider calls, tool roundtrips, worker spawning, and event emission |
| `hannah/runtime/context.py` | Main-agent and worker message assembly |
| `hannah/runtime/events.py` | Runtime event envelope and allowed event names |
| `hannah/runtime/bus.py` | Async event bus used for streaming and persistence hooks |
| `hannah/agent/worker_runtime.py` | Generic worker execution, `WorkerSpec`, spawn policy, and worker result handling |
| `hannah/agent/tool_registry.py` | Tool discovery plus runtime-bound tool binding, normalization, and dispatch |
| `hannah/session/manager.py` | JSONL session persistence for chat and `agent` session mode |
| `hannah/session/event_records.py` | JSON-safe serialization for persisted runtime events |

---

## Runtime Surfaces

`hannah agent` is the canonical entrypoint.

- `hannah agent --message "..."` runs a one-shot turn through the shared runtime
- `hannah agent` on a real TTY launches the interactive session path
- `hannah chat` is a compatibility wrapper over that same runtime path
- `hannah ask` is a compatibility wrapper for freeform one-shot turns

The runtime boundary is intentional:

- `agent` and `chat` use session-backed persistence
- `ask` preserves the older one-shot behavior without session persistence
- direct utility commands such as `sandbox`, `fetch`, and `train` can still stay off the shared runtime when they intentionally call legacy command helpers

---

## AgentLoop Compatibility Layer

`AgentLoop` is no longer the core runtime owner.

Its job is now:

1. load config, memory, provider, and registry
2. build main-turn messages from persona, dynamic guidance, memory, and user input
3. choose the per-turn tool surface
4. call `RuntimeCore.run_turn(...)`
5. persist the final user and assistant messages through the existing memory interface

That keeps the old `AgentLoop.run_turn(...)` and `AgentLoop.run_command(...)` API stable while moving actual runtime behavior into `RuntimeCore`.

---

## Tool Surface

`ToolRegistry` still discovers the regular tool modules under `hannah/tools/*/tool.py`, but the main-agent runtime surface now also includes a runtime-bound `spawn` tool.

Main-agent turns can see:

- domain tools such as `race_data`, `race_sim`, `pit_strategy`, `predict_winner`, and `train_model`
- runtime-bound `spawn`, which is injected by `RuntimeCore` through `ToolRegistry.with_runtime_tools(...)`

Worker turns do not get the full main-agent tool surface. `WorkerRuntime` builds a restricted registry from the parent registry and the worker's allowlist.

---

## Worker Model

Generic workers are described with structured `WorkerSpec` objects:

- `worker_id`
- `task`
- `system_prompt`
- `allowed_tools`
- `result_contract`

`allowed_tools` is the hard boundary for worker execution. Workers only receive the explicitly allowed subset of tools.

Slice 1 policy:

- nested spawn is disallowed
- any worker spec that includes `spawn` in `allowed_tools` is rejected before the worker runs
- the parent turn receives a normal tool-error payload for that failed spawn call instead of crashing the whole turn

This keeps worker execution depth-1 and preserves containment.

---

## End-To-End Turn Flow

For the primary runtime path:

1. `hannah/cli/app.py` dispatches to `hannah/cli/agent_command.py`
2. `agent_command.py` chooses one-shot or interactive execution
3. `chat.py` or the one-shot path creates memory/session context
4. `AgentLoop` builds the main prompt and turn-specific tool list
5. `RuntimeCore` emits `user_message_received`
6. `RuntimeCore` calls the provider
7. if the provider returns tool calls, `RuntimeCore` normalizes arguments and executes them
8. if the tool is `spawn`, `WorkerRuntime` runs a bounded worker turn through another `RuntimeCore`
9. tool results and worker results are reinjected into the parent turn
10. `RuntimeCore` emits `final_answer_emitted`
11. `AgentLoop` persists the user/assistant messages

The LLM still orchestrates. The domain tools still own simulation, telemetry, prediction, and strategy computation.

---

## Runtime Events

The event contract is fixed in `hannah/runtime/events.py`.

Core event names:

- `user_message_received`
- `provider_request_started`
- `provider_response_received`
- `tool_call_started`
- `tool_call_finished`
- `subagent_spawned`
- `subagent_progress`
- `subagent_completed`
- `final_answer_emitted`
- `error_emitted`

The worker-facing naming contract is the `subagent_*` family:

- `subagent_spawned`
- `subagent_progress`
- `subagent_completed`

Those names are the compatibility contract for slice 1. They are what the CLI formatter, acceptance tests, and session event persistence layer consume.

---

## Streaming And JSONL Persistence

Runtime events are not just internal tracing.

- `hannah/cli/chat.py` subscribes to the runtime event bus
- `hannah/cli/format.py` renders the `subagent_*` events as streamed terminal output
- `hannah/session/manager.py` persists every runtime event as a JSONL event record through `hannah/session/event_records.py`

That means the same runtime event stream drives:

- live subagent activity in the terminal
- durable event history in session files

The slice 1 acceptance contract depends on that shared event stream, including stable ordering for `subagent_spawned -> subagent_progress -> subagent_progress -> subagent_completed`.

---

## Failure Boundaries

The runtime keeps turns alive when possible.

- malformed tool args are normalized or rejected at the boundary
- tool failures are serialized back into tool messages instead of crashing the turn
- disallowed nested spawn returns a structured spawn-tool error
- provider failures emit `error_emitted` and can be handled by the provider seam or caller

This is the intended slice 1 behavior: bounded workers, explicit runtime events, and compatibility surfaces over one shared runtime core.

---

## Reading Order

Read these files in order if you need to trace the current architecture:

1. `hannah/cli/app.py`
2. `hannah/cli/agent_command.py`
3. `hannah/cli/chat.py`
4. `hannah/agent/loop.py`
5. `hannah/runtime/core.py`
6. `hannah/agent/worker_runtime.py`
7. `hannah/agent/tool_registry.py`
8. `hannah/session/manager.py`
9. `hannah/session/event_records.py`
