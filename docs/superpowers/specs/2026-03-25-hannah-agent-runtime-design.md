# Hannah Agent Runtime Redesign

## Summary

Redesign Hannah from a command-shaped F1 CLI into an agent-first CLI runtime modeled after nanobot's control flow while preserving Hannah's F1 domain seams. The primary product surface becomes `hannah agent`, with existing commands retained as convenience wrappers over the same agent core.

## Goals

- Make `hannah agent` the primary runtime entrypoint for both interactive and one-shot use.
- Let the main agent choose among F1 tools instead of routing users through prompt-shaped commands.
- Add generic runtime-spawned subagents with custom prompts and restricted toolsets.
- Stream tool and subagent activity live in the CLI.
- Preserve current domain ownership: simulation, telemetry, prediction, training, and strategy logic remain below the tool layer.

## Non-Goals

- Rewriting F1 simulation or model code.
- Moving domain logic into the LLM loop.
- Removing existing command shortcuts like `simulate`, `predict`, or `strategy`.
- Introducing a web UI.

## Product Direction

Hannah should behave as a real CLI agent for F1 engineering work: exploring prior race data, running simulations, predicting winners, and producing strategy recommendations from tool-grounded evidence. The CLI should feel like one live runtime with multiple entrypoints, not a set of unrelated commands that happen to share an LLM.

## Runtime Architecture

The architecture centers on a single agent runtime:

- CLI entrypoints feed user intent into one agent core.
- The agent core owns session state, message history, context assembly, provider calls, tool dispatch, and streamed event emission.
- The runtime is bus-oriented rather than a plain turn loop. Typed events support streaming, worker visibility, replay, and future runtime substitution.
- The loop remains orchestration only. It does not compute telemetry, run simulations, or train models itself.

### Required Runtime Events

At minimum the bus should support:

- user message received
- provider request started
- provider response received
- tool call started
- tool call finished
- subagent spawned
- subagent progress
- subagent completed
- final answer emitted
- error emitted

These events are the source of truth for CLI streaming and persisted session traces.

Each event should include a minimal shared envelope:

- `event_type`
- `timestamp`
- `session_id`
- `message_id`
- `worker_id` when the event relates to a spawned worker
- `payload` for event-specific fields

## Tools and Skills

The main agent receives a tool surface composed of:

- Hannah F1 tools such as race data fetch, simulation, winner prediction, strategy analysis, training, and trace/replay
- Runtime tools, primarily `spawn`

The `spawn` tool becomes a first-class runtime capability rather than a hardcoded workflow path. The model uses it when decomposition is useful.

Workspace guidance and skill summaries should be loaded into runtime context so the main agent understands:

- the available tools
- the repo-specific architectural constraints
- the expectation that domain work happens in tools, not in freeform model reasoning

In the first version, this should be static runtime-provided context assembled at turn start rather than a new dynamic retrieval subsystem.

## Subagent Model

Subagents are generic workers, not fixed role classes.

Each spawned subagent is defined by:

- task description
- custom system prompt
- allowed tool list
- result contract

This lets the main agent create workers such as telemetry fetchers, strategy analysts, simulation verifiers, or rival-behavior investigators without encoding those roles as Python classes.

### Subagent Constraints

- Subagents run the same provider/tool loop shape as the main agent.
- Each subagent gets a reduced toolset.
- Recursive orchestration is disabled in the first version. Subagents cannot call `spawn`; only the main agent may spawn workers. This caps spawn depth at 1.
- Subagent output returns to the main runtime through structured events so the user can see progress and the main agent can incorporate results.

## Streaming UX

The CLI should present Hannah as a live system:

- tool selection and execution stream as they happen
- spawned workers appear with task labels and state transitions
- partial findings can be surfaced before the final synthesis

Visible worker states should include at least:

- running
- waiting on tool
- completed
- failed

This is the core product distinction between an agent runtime and a blocking command wrapper.

## Sessions and Memory

Sessions become first-class runtime records, not just chat transcripts.

Each session should persist:

- user messages
- assistant replies
- tool calls
- tool outputs
- subagent events
- final synthesized answers
- optional condensed memory summaries

This enables continuity across investigations, replayable decision history, and better debugging for engineering workflows.

## CLI Surface

`hannah agent` becomes the primary runtime surface.

Entry surface expectations:

- `hannah agent` with no prompt opens the interactive runtime
- `hannah agent "<prompt>"` runs a one-shot task through the same runtime
- `hannah ask "<prompt>"` remains as a backward-compatible one-shot alias into the same runtime
- `simulate`, `predict`, `strategy`, and related commands remain as convenience wrappers

The wrappers should submit structured intent into the agent runtime and render the same streamed event output. There should be one architecture underneath, not separate execution models.

Examples:

- `hannah agent`
- `hannah agent "Compare two-stop versus one-stop for Bahrain from lap 18"`
- `hannah ask "Who is the likely winner at Monza given qualifying pace and recent long-run data?"`

## Safety and Boundaries

The redesign must preserve the current hard boundaries:

- no fake telemetry
- no fake simulations
- no fake training outcomes
- provider seam remains thin and swappable
- domain logic remains in simulation, data, and model layers

The local fallback may remain, but it must behave like a runtime provider that emits structured tool calls and events through the same orchestration path.

## Recommended Migration Plan

Implement in bounded slices:

1. Introduce an event bus and make the current loop publish structured events without changing tool ownership.
2. Add a generic `spawn` tool and constrained worker runtime while keeping existing fixed F1 subagents as temporary compatibility shims.
3. Promote `hannah agent` to the primary surface and convert current CLI commands into wrappers over the agent runtime.
4. Retire fixed-role subagent classes once the generic worker path covers their use cases.
5. Upgrade sessions, memory, and traces so streamed worker activity is persisted cleanly.

This sequence preserves existing seams and lowers migration risk.

## Testing Strategy

Testing should mirror the migration order.

Add focused coverage for:

- event emission from the main runtime
- generic spawn restrictions
- worker result reinjection
- command-wrapper equivalence with the agent runtime

Then extend scenario and acceptance coverage so equivalent F1 questions succeed through:

- `hannah agent`
- `hannah ask`
- `hannah simulate`
- `hannah strategy`

The convergence test is that the same domain task succeeds regardless of entrypoint because all entrypoints share one core runtime.

## Recommendation

Use a hybrid refactor rather than a full rewrite:

- preserve Hannah's current F1 tools, provider seam, and domain modules
- adopt nanobot's agent-first runtime shape
- move generic subagent spawning and event streaming into the shared core

This achieves the desired product shape without discarding the existing F1-specific assets.
