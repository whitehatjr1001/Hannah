# AGENT_LOOP.md

## Purpose

This document explains how a single Hannah turn moves from CLI input to provider call to tool execution to final answer.

The short version:

1. the CLI or chat TUI collects a user message
2. `AgentLoop` builds a prompt from persona + guidance + memory + user input
3. the provider decides between hosted LiteLLM and local fallback
4. if the model asks for tools, Hannah runs them and feeds results back into the loop
5. when the model returns final text, Hannah stores it in memory and renders it back to the user

This is the core runtime path for Hannah. The simulator and data tools stay underneath the loop; the LLM does not own race state or invent telemetry.

---

## Main Files

| File | Responsibility |
| --- | --- |
| `hannah.py` | Root CLI wrapper |
| `hannah/cli/app.py` | Click commands like `ask`, `strategy`, `simulate`, `train`, `chat` |
| `hannah/cli/chat.py` | Nanobot-style TUI, slash commands, session handoff into `AgentLoop` |
| `hannah/agent/loop.py` | Main tool-using agent loop |
| `hannah/agent/persona.py` | System prompt for strategist behavior |
| `hannah/agent/tool_registry.py` | Tool discovery, schema export, argument normalization |
| `hannah/providers/registry.py` | Provider selection entrypoint |
| `hannah/providers/litellm_provider.py` | Hosted LiteLLM path plus local fallback bridge |
| `hannah/providers/local_fallback.py` | Deterministic offline planner when no external model is available |
| `hannah/agent/memory.py` | SQLite-backed memory for non-chat runs |
| `hannah/session/manager.py` | JSONL session store and `SessionMemory` adapter for TUI chat |

---

## End-To-End Flow

```text
user input
  -> CLI command or TUI chat
  -> AgentLoop.__init__()
     -> load config
     -> build tool registry
     -> build provider
     -> attach memory backend
  -> AgentLoop.run_turn(user_input)
     -> build messages
     -> select allowed tools for this turn
     -> provider.complete(messages, tools, ...)
        -> hosted LiteLLM
        -> or local fallback
     -> if tool calls:
        -> normalize args
        -> execute tools concurrently
        -> append tool messages
        -> loop again
     -> if final answer:
        -> optional retry if the model tried to defer instead of acting
        -> save user + assistant messages to memory
        -> return final text
  -> CLI/TUI renders the answer
```

---

## Step 1: Entry Into The Loop

There are two common entry paths:

- CLI command path from `hannah/cli/app.py`
- interactive chat path from `hannah/cli/chat.py`

Examples:

- `hannah ask "Should VER pit under VSC on lap 34?"`
- `hannah strategy --race bahrain --lap 18 --driver VER --type optimal`
- `hannah` and then typing a message in the TUI

For direct commands, `app.py` turns the command into a natural-language request and calls `AgentLoop().run_command(...)`.

For chat mode, `chat.py` creates a session-aware memory adapter and calls `AgentLoop(memory=SessionMemory(...)).run_turn(...)`.

Important boundary:

- slash commands like `/model`, `/providers`, `/configure`, `/sessions`, `/clear` are handled entirely inside `hannah/cli/chat.py`
- they do not enter the LLM loop at all

---

## Step 2: Loop Construction

`AgentLoop.__init__()` in `hannah/agent/loop.py` does four things:

1. loads config with `load_config()`
2. creates a memory backend
3. loads tool specs from `ToolRegistry`
4. resolves the provider from `ProviderRegistry`

That gives the loop a stable runtime surface:

- `self.memory`
- `self.tools`
- `self.provider`

The provider seam stays thin on purpose. `ProviderRegistry` currently resolves to `LiteLLMProvider`, which means Hannah can use:

- a hosted provider like OpenAI/Claude/Gemini through LiteLLM
- a local OpenAI-compatible endpoint through `HANNAH_RLM_API_BASE`
- deterministic offline fallback if no external model is usable

---

## Step 3: Message Assembly

`AgentLoop.run_turn(user_input)` builds the prompt in layers:

1. Hannah system persona from `hannah/agent/persona.py`
2. optional dynamic system guidance for this specific turn
3. recent memory
4. current user message

The dynamic guidance matters. Hannah now adjusts the turn shape before the first provider call.

Example:

- if the user asks for upcoming-race strategy analysis, Hannah hides `train_model` from the turn and injects guidance telling the model to use analysis tools instead

That prevents the hosted model from inventing bogus training flows for questions that should be solved with `race_data`, `race_sim`, or `pit_strategy`.

---

## Step 4: Tool Selection Per Turn

The full tool list comes from `ToolRegistry.get_tool_specs()`, which discovers each `hannah/tools/*/tool.py` module and exports its schema.

Current core tools are:

- `race_data`
- `race_sim`
- `pit_strategy`
- `predict_winner`
- `train_model`

Before the provider call, `AgentLoop` can reduce that list for the current turn.

Current special rule:

- if the prompt looks like race analysis rather than explicit retraining, `train_model` is removed from the available tools for that turn

This is a product guardrail, not just a prompt hint.

---

## Step 5: Provider Call

`LiteLLMProvider.complete(...)` is the provider boundary.

The decision tree is:

1. if `HANNAH_FORCE_LOCAL_PROVIDER=1`, use deterministic local fallback
2. else import `litellm`
3. if no matching hosted credentials are available, use local fallback
4. if `HANNAH_RLM_API_BASE` is set, point LiteLLM at the local OpenAI-compatible endpoint
5. otherwise call the hosted model configured in `HANNAH_MODEL`

Before the hosted call, the provider:

- suppresses LiteLLM debug noise
- sanitizes messages to the keys the provider path expects
- passes tool schemas with `tool_choice="auto"` when tools are available

If the hosted call throws, Hannah falls back locally instead of crashing the whole turn.

---

## Step 6: Response Coercion

Provider responses can come back in slightly different shapes.

`AgentLoop` normalizes them into internal adapters:

- `_MessageAdapter`
- `_ToolCallAdapter`
- `_FunctionAdapter`

That keeps the rest of the loop stable whether the response came from:

- LiteLLM hosted output
- dict-like provider payloads
- the deterministic local fallback planner

---

## Step 7: Tool Execution

If the assistant response contains tool calls, Hannah:

1. logs each tool name to the terminal
2. appends the assistant tool-call message to the running message list
3. executes all tool calls concurrently with `asyncio.gather(...)`
4. converts each tool result into a `role="tool"` message
5. sends the expanded conversation back through the provider

Tool execution is strict at the boundary:

- raw tool arguments are parsed from JSON or dict form
- `ToolRegistry.normalize_args(...)` validates and normalizes them
- tool-specific normalizers can run before schema validation
- unsupported extra args are dropped
- missing required args raise a clear error

That is the seam that prevents noisy hosted-model tool calls from corrupting the runtime.

---

## Step 8: Tool Result Serialization

Tool results are not blindly dumped back into the prompt.

`AgentLoop` serializes them with two important guards:

1. non-JSON-native values are stringified safely
2. oversized tool payloads are compacted before being sent back to the model

Special case:

- `race_data` payloads are summarized when they get too large, so the model sees session info, driver list, and telemetry counts instead of a huge raw dump

This matters because race data can be large enough to waste context or break hosted follow-up passes.

---

## Step 9: Retry On Permission-Seeking Deferrals

Some hosted models answer ambiguous prompts with a useless conversational deferral:

> I can analyze that. Let me know if you'd like me to proceed.

That is not acceptable for Hannah.

`AgentLoop` now detects that pattern for race-analysis turns. If the model tries to defer instead of acting:

1. Hannah appends the assistant deferral to the message history
2. Hannah injects a one-time system correction
3. the same turn is sent back through the provider again

The retry instruction says, in effect:

- the user already asked for the analysis
- do not ask for permission
- call the tools now and answer decisively

This is how the hosted path was hardened against vague prompts like:

- “can model the 2026 make ai models to predict the race strategy for the upcoming Japanese Grand Prix?”

---

## Step 10: Final Answer And Memory Persistence

When the assistant returns final text without tool calls:

1. the user message is written to memory
2. the assistant message is written to memory
3. the text is returned to the caller

Memory backend depends on the entry path:

- normal CLI runs use `Memory` in `hannah/agent/memory.py` with SQLite
- TUI chat uses `SessionMemory` in `hannah/session/manager.py` with JSONL session files

This means the loop code itself does not care whether the caller is:

- a one-shot CLI command
- a persistent chat session

It only depends on the shared `add(...)` and `get_recent(...)` memory interface.

---

## Example: Strategy Question In Chat Mode

A real strategy-style turn in the TUI looks like this:

1. user types a freeform question into `hannah`
2. `chat.py` loads the active JSONL session and wraps it as `SessionMemory`
3. `AgentLoop.run_turn(...)` builds the prompt
4. if the question is race analysis, `train_model` is removed from this turn
5. the provider gets the message stack and allowed tool list
6. the model calls tools such as `race_data`, `race_sim`, and `pit_strategy`
7. Hannah executes those tools and appends their outputs
8. the provider synthesizes a final strategist answer
9. the session file is updated with both the user turn and the assistant reply
10. the TUI renders the answer panel

---

## What The LLM Does Not Own

These boundaries are intentional:

- the LLM does not simulate the race directly
- the LLM does not invent telemetry as a substitute for `race_data`
- the LLM does not own artifact training logic
- the optional runtime helper is not part of the main default loop

The loop is an orchestrator. The tools own the domain work.

---

## Failure Behavior

The loop is built to keep the turn alive when possible.

### No hosted model credentials

- Hannah falls back to deterministic local planning

### Hosted provider throws

- Hannah falls back locally rather than crashing the CLI

### Tool call is malformed

- argument normalization catches it before the tool body runs

### Tool body raises

- the error is wrapped into a tool message and the model gets a chance to recover inside the same turn

### Live data is incomplete

- `race_data` and the simulation path still return shaped payloads so the turn can continue

---

## Practical Reading Order

If you want to trace the loop in code, read files in this order:

1. `hannah/cli/app.py`
2. `hannah/cli/chat.py`
3. `hannah/agent/loop.py`
4. `hannah/providers/litellm_provider.py`
5. `hannah/agent/tool_registry.py`
6. `hannah/tools/race_data/tool.py`
7. `hannah/tools/race_sim/tool.py`
8. `hannah/tools/pit_strategy/tool.py`
9. `hannah/tools/predict_winner/tool.py`
10. `hannah/tools/train_model/tool.py`

That path shows the real runtime sequence without getting lost in training or simulation internals too early.
