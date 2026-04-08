# Tools Trace — Hannah Smith F1 Agent

> Generated: 2026-04-04
> Methodology: Systematic backward tracing from CLI entry through tool execution, verified with diagnostic runs.

---

## Architecture Overview

```
CLI (click) → agent_command → bus/queue → AgentLoop → Provider (LiteLLM)
                                                    ↓ tool calls
                                             ToolRegistry.call()
                                                    ↓
                                        tools/*/tool.py::run()
                                                    ↓
                                    Data / Sim / Model / Training layers
```

The LLM is the orchestrator. Tools own all deterministic F1 work. The LLM never fabricates simulator outputs.

---

## Layer 1: CLI Ingress

### Entry Point: `hannah/cli/app.py`

```
load_dotenv(override=True)  ← critical: overrides stale shell env vars
    ↓
@click.group() cli()
    ├── hannah agent -m "..."     → run_agent_command(message, interactive=False)
    ├── hannah agent              → run_agent_command(None, interactive=True)  [TTY required]
    ├── hannah chat               → compatibility wrapper over same runtime
    ├── hannah ask "..."          → build_ask_intent() → run_agent_command()
    ├── hannah simulate           → build_simulate_intent() → run_agent_command()
    ├── hannah predict            → build_predict_intent() → run_agent_command()
    ├── hannah strategy           → build_strategy_intent() → run_agent_command()
    ├── hannah sandbox            → natural language prompt → _run_agent_command()
    ├── hannah fetch              → natural language prompt → _run_agent_command()
    ├── hannah train              → natural language prompt → _run_agent_command()
    ├── hannah tools              → ToolRegistry().list_tools()
    ├── hannah model              → render_model_status()
    ├── hannah providers          → render_provider_status_table()
    ├── hannah configure          → run_provider_configure_flow()
    ├── hannah sessions           → print_sessions()
    ├── hannah trace              → direct race_sim_tool.run() (bypasses agent loop)
    └── hannah rlm-probe          → probe_runtime_helper() (bypasses agent loop)
```

### Command Prompt Builders: `hannah/cli/command_prompts.py`

Pure string functions that translate structured CLI args into natural language:

| Function | Output |
|---|---|
| `build_ask_intent(question)` | Returns question as-is |
| `build_simulate_intent(...)` | `"Run a race simulation for {race} {year}. Driver: {driver}. Laps: {laps}. Weather: {weather}."` |
| `build_predict_intent(...)` | `"Predict the winner for the {race} Grand Prix {year}. Fetch current qualifying and historical data..."` |
| `build_strategy_intent(...)` | `"Strategy call for {driver} at {race}, lap {lap}. Strategy type: {strategy_type}..."` |

### Execution Paths: `hannah/cli/agent_command.py`

Three paths from `run_agent_command()`:

```
interactive=True     → run_interactive_chat_session()  [TUI loop with slash commands]
persist_session=False → _EphemeralMemory() + run_bus_turn()  [in-memory, no disk writes]
persist_session=True  → run_message_chat_session()  [SessionManager JSONL persistence]
```

---

## Layer 2: Bus Ingress/Egress

### `hannah/bus/queue.py` — `run_bus_turn()`

```
InboundMessage.create(channel, session_id, content)
    ↓
MessageBus.publish(inbound)
    ↓
MessageBus.receive_inbound()
    ↓
agent_loop.run_turn(content, session_id=...)  [signature-inspected]
    ↓
OutboundMessage.create(channel, session_id, response)
    ↓
MessageBus.publish(outbound)
    ↓
return OutboundMessage
```

### `hannah/bus/events.py` — Message Types

```
BusMessage (base)
├── InboundMessage   direction="inbound"  role="user"
└── OutboundMessage  direction="outbound" role="assistant"
```

All messages are immutable frozen dataclasses with auto-generated `message_id` (UUID) and `timestamp` (UTC).

---

## Layer 3: AgentLoop — The Orchestrator

### `hannah/agent/loop.py` — `AgentLoop`

#### Initialization

```python
AgentLoop(memory, registry, provider)
    ↓
load_config()                                    # from config.yaml + env
Memory() or provided                             # SQLite at data/hannah_memory.db
ToolRegistry() or provided                       # auto-discovers tools/*/tool.py
ProviderRegistry.from_config(config)             # LiteLLMProvider
AsyncEventBus()                                  # event publishing
RuntimeContextBuilder()                          # message assembly
WorkerRuntime(...)                               # if registry has with_runtime_tools
registry.with_runtime_tools({"spawn": handler})  # bind spawn tool
self.tools = registry.get_tool_specs()           # cache for LLM
```

#### Turn Flow

```
run_turn(user_input, session_id="default")
    │
    ├─ 1. _select_tools_for_turn(user_input)
    │     └─ May hide train_model if race-analysis intent detected
    │
    ├─ 2. context_builder.build_main_turn(MainAgentContext(
    │       persona=HANNAH_PERSONA,
    │       dynamic_guidance=_dynamic_turn_guidance(user_input),
    │       recent_messages=memory.get_recent(n=10),
    │       user_input=user_input,
    │   ))
    │     └─ Builds 6 system message blocks:
    │        1. Identity/runtime (with dynamic guidance if hiding train_model)
    │        2. Bootstrap docs
    │        3. Memory context
    │        4. Skills summary
    │        5. Resolved roster
    │        6. Hannah persona
    │        + recent conversation history
    │        + user input (final message)
    │
    ├─ 3. Wrap in TurnState(session_id, messages)
    │
    ├─ 4. Publish "user_message_received" event
    │
    └─ 5. while True:
          │
          ├─ Publish "provider_request_started" (message_count, tool_names)
          │
          ├─ provider.complete(messages, tools, temperature, max_tokens)
          │   ├─ LiteLLMProvider._force_local() → LocalCompletion
          │   ├─ import litellm fails → LocalCompletion
          │   ├─ _hosted_credentials_available() → LocalCompletion
          │   ├─ litellm.acompletion(...) → response object
          │   └─ Exception → LocalCompletion (silent fallback)
          │
          ├─ _coerce_first_message(response) → ProviderMessage
          │
          ├─ Publish "provider_response_received" (has_tool_calls)
          │
          ├─ IF tool_calls exist:
          │   ├─ Print "◆ calling tool: {name}" for each
          │   ├─ Append assistant message to state
          │   ├─ _execute_tool_calls(tool_calls, state)  [concurrent via asyncio.gather]
          │   │   ├─ For each tool call:
          │   │   │   ├─ Publish "tool_call_started"
          │   │   │   ├─ _call_tool() → registry.call(name, args, state)
          │   │   │   │   ├─ normalize_args(name, args)  [schema + signature validation]
          │   │   │   │   ├─ Inject state parameter if callable accepts it
          │   │   │   │   ├─ Call run_fn(**args)
          │   │   │   │   └─ Await if result is awaitable
          │   │   │   ├─ Publish "tool_call_finished"
          │   │   │   └─ Build {"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}
          │   │   │
          │   │   └─ For spawn results: append extra system message with worker result JSON
          │   │
          │   ├─ Extend state with tool result messages
          │   └─ continue (back to provider call)
          │
          ├─ IF no tool_calls AND _should_retry_analysis_turn():
          │   ├─ Append assistant message + retry system guidance
          │   ├─ retry_used = True
          │   └─ continue
          │
          └─ ELSE:
              ├─ Publish "final_answer_emitted"
              ├─ memory.add("user", user_input)
              ├─ memory.add("assistant", final_text)
              └─ return final_text
```

#### Tool Selection Logic

```python
_should_hide_train_model(user_input):
    IF any explicit training hint in input → False (show train_model)
    IF race_context_token AND analysis_intent_token → True (hide train_model)
    ELSE → False (show all tools)

Explicit training hints: "train ", "train the", "retrain", "training", "fine-tune", "fine tune"
Race context tokens: "race", "grand prix", "prix", "gp"
Analysis intent tokens: "predict", "prediction", "strategy", "pit", "simulate", "simulation",
                        "analysis", "analyze", "upcoming", "next race", "next grand prix",
                        "ai model", "ai models", "model the", "models to predict"

_should_retry_analysis_turn(final_text):
    IF retry_used → False
    IF train_model should NOT be hidden → False
    IF any deferral hint in final_text → True
    Deferral hints: "let me know if you'd like", "if you'd like me to proceed",
                    "i can analyze", "i can model"
```

#### Payload Compaction

Tool responses over 20,000 characters are compacted before reinjection:

```
race_data tool → specialized summary (session_info, drivers, resolved_roster, telemetry_counts)
other tools    → generic summary {"tool": name, "raw_payload_chars": N, "message": "compacted"}
```

---

## Layer 4: Tool Registry — Discovery and Dispatch

### `hannah/agent/tool_registry.py` — `ToolRegistry`

#### Discovery

```
__init__()
    ↓
_discover()
    ↓
tools_dir.glob("*/tool.py")
    ↓
For each file:
    ├─ Import module
    ├─ Extract SKILL dict (module-level)
    ├─ Extract run function (module-level)
    ├─ Validate both exist and are correct types
    └─ Create RegisteredTool(name, description, module_name, module, parameters, run_fn, signature)
    ↓
_discover_mcp_tools()  [if MCP enabled]
    ↓
_runtime_placeholders()  [adds "spawn" placeholder with run_fn=None]
```

#### Registered Tools (verified)

| Tool | Module | Parameters | Has run_fn |
|---|---|---|---|
| `race_data` | `hannah.tools.race_data.tool` | race, year, session, driver | Yes |
| `race_sim` | `hannah.tools.race_sim.tool` | race, year, weather, drivers, laps, n_worlds, trace, trace_checkpoints, replay | Yes |
| `pit_strategy` | `hannah.tools.pit_strategy.tool` | race, year, lap, driver | Yes |
| `predict_winner` | `hannah.tools.predict_winner.tool` | race, year, drivers | Yes |
| `train_model` | `hannah.tools.train_model.tool` | model_name, years, races | Yes |
| `spawn` | `runtime.spawn` (placeholder) | task, system_prompt, allowed_tools, result_contract | Bound at runtime |

#### Dispatch

```
call(name, args, state=None)
    ↓
Look up RegisteredTool by name
    ↓
normalize_args(name, args)
    ├─ If module defines normalize_args() → call it
    ├─ Validate against SKILL schema (required keys, types, enum, min/max)
    ├─ Coerce types ("42" → 42, "true" → True)
    ├─ Strip _CONTEXT_ONLY_PARAMS ({"state"})
    └─ Intersect schema properties with callable signature params (drop unknown keys)
    ↓
Inject state if callable signature has "state" parameter
    ↓
Call run_fn(**call_args)
    ↓
Await if result is awaitable
    ↓
Validate return type is dict
    ↓
return dict
```

#### Runtime Tool Binding

```
with_runtime_tools(handlers: dict[str, callable])
    ↓
For each (name, handler) in handlers:
    ├─ Replace placeholder RegisteredTool with real one
    ├─ Set run_fn = handler
    ├─ Set signature = inspect.signature(handler)
    └─ Return cloned registry
```

#### Subset (Worker Isolation)

```
subset(allowed_names: set[str])
    ↓
Clone registry with only tools whose names are in allowed_names
    ↓
Return cloned registry (original unchanged)
```

---

## Layer 5: Tool Implementations

### Tool 1: `race_data` — `hannah/tools/race_data/tool.py`

**SKILL:**
```python
{
    "name": "race_data",
    "description": "Fetches F1 race data from FastF1 and OpenF1.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string"},
            "year": {"type": "integer"},
            "session": {"type": "string", "enum": ["R", "Q", "FP1", "FP2", "FP3"]},
            "driver": {"type": "string"},
        },
        "required": ["race"],
    },
}
```

**Execution flow:**
```
run(race, year=2025, session="R", driver=None)
    │
    ├─ should_enrich_from_openf1(year) → bool
    │
    ├─ IF enrichment enabled:
    │   ├─ asyncio.gather(
    │   │   fetch_session(race, year, session),     # FastF1
    │   │   client.get_sessions(year, race)          # OpenF1
    │   │ )
    │   ├─ _resolve_openf1_session_key() → session_key
    │   ├─ IF session_key found:
    │   │   └─ asyncio.gather(
    │   │       client.get_stints(session_key),
    │   │       client.get_weather(session_key),
    │   │       client.get_drivers(session_key),
    │   │   )
    │
    ├─ IF enrichment disabled:
    │   └─ fetch_session(race, year, session) only
    │
    ├─ resolve_season_roster(year, fastf1_payload, openf1_drivers)
    │
    ├─ Build session_info dict (race, year, session, circuit, etc.)
    │
    └─ Return {"laps": ..., "stints": ..., "weather": ..., "drivers": ..., "session_info": ..., "resolved_roster": ...}
```

**Dependencies:**
- `hannah.data.fastf1_loader.fetch_session()` — FastF1 with cache at `data/fastf1_cache/`
- `hannah.data.openf1_client.OpenF1Client` — HTTP client for api.openf1.org/v1
- `hannah._data_.season_roster_resolver.resolve_season_roster()` — driver roster resolution

**Fallbacks:**
- Drivers default to `["VER", "NOR", "LEC"]` if not resolved
- Weather falls back from OpenF1 to FastF1
- OpenF1 errors return empty lists (graceful)

---

### Tool 2: `race_sim` — `hannah/tools/race_sim/tool.py`

**SKILL:**
```python
{
    "name": "race_sim",
    "description": "Runs the fast Monte Carlo simulation and returns strategy outputs.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string"},
            "year": {"type": "integer"},
            "weather": {"type": "string"},
            "drivers": {"type": "array", "items": {"type": "string"}},
            "laps": {"type": "integer"},
            "n_worlds": {"type": "integer"},
            "trace": {"type": "boolean"},
            "trace_checkpoints": {"type": "array", "items": {"type": "integer"}},
            "replay": {"type": "object"},
        },
        "required": ["race"],
    },
}
```

**Execution flow:**
```
run(race, year=2025, weather="dry", drivers=None, laps=57, n_worlds=1000,
    trace=False, trace_checkpoints=None, replay=None)
    │
    ├─ IF drivers is None:
    │   └─ race_data_tool.run(race, year, session="R") → resolve roster
    │
    ├─ _resolved_roster(race_data, drivers or ["VER", "NOR", "LEC"])
    │
    ├─ _stable_seed() → int  [MD5 hash of race params for determinism]
    │
    ├─ RaceContext(race, year, laps, weather, drivers, race_data, resolved_roster)
    │
    ├─ RaceState.from_context(context)
    │   └─ state.seed = stable_seed
    │
    ├─ run_fast(race_state, n_worlds=n_worlds)  # Monte Carlo simulation
    │   └─ Returns SimResult with winner_probs, p50_race_time_s, final_positions, etc.
    │
    ├─ StrategyEngine().analyse(race_state, sim_result)
    │   └─ Returns {recommended_pit_lap, recommended_compound, strategy_type, confidence, ...}
    │
    ├─ IF trace=True:
    │   └─ build_replay_trace() → add to result
    │
    └─ Return {"simulation": sim_result.to_dict(), "strategy": strategy, "trace": ...}
```

**Dependencies:**
- `hannah.simulation.monte_carlo.run_fast()` — vectorized Monte Carlo engine
- `hannah.simulation.strategy_engine.StrategyEngine.analyse()` — strategy analysis
- `hannah.simulation.sandbox.RaceState.from_context()` — race state builder
- `hannah.simulation.replay_trace.build_replay_trace()` — deterministic trace (optional)

---

### Tool 3: `pit_strategy` — `hannah/tools/pit_strategy/tool.py`

**SKILL:**
```python
{
    "name": "pit_strategy",
    "description": "Returns a pit stop recommendation and confidence score.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string"},
            "year": {"type": "integer"},
            "lap": {"type": "integer"},
            "driver": {"type": "string"},
        },
        "required": ["race", "driver"],
    },
}
```

**Execution flow:**
```
run(race, driver, year=2025, lap=1)
    │
    ├─ race_data_tool.run(race, year, session="R") → session_state
    │
    ├─ RaceState.from_race_data(session_state)
    │
    ├─ IF driver in race_state.drivers:
    │   ├─ Reorder all state arrays to put driver at index 0:
    │   │   drivers, compounds, positions, gaps, tyre_ages, base_lap_times
    │   └─ Set current_lap = requested lap
    │
    ├─ IF driver NOT in race_state.drivers:
    │   ├─ Create synthetic RaceState:
    │   │   drivers=[driver, "NOR", "LEC"]
    │   │   compounds=["SOFT", "SOFT", "SOFT"]
    │   │   positions=[1, 2, 3]
    │   │   gaps=[0.0, 1.5, 3.2]
    │   │   tyre_ages=[current_lap, current_lap, current_lap]
    │   │   base_lap_times=[90.0, 90.5, 91.0]
    │   ├─ run_fast(state, n_worlds=500)
    │   └─ StrategyEngine().analyse(state, sim_result)
    │
    ├─ run_fast(race_state, n_worlds=500)  # 500 worlds (faster than default 1000)
    │
    └─ Return StrategyEngine().analyse(race_state, sim_result)
       → {recommended_pit_lap, recommended_compound, strategy_type, confidence,
          undercut_window, rival_threats, reasoning}
```

**Dependencies:**
- `race_data` tool (fetches session data)
- `hannah.simulation.sandbox.RaceState` — race state container
- `hannah.simulation.monte_carlo.run_fast()` — simulation (500 worlds)
- `hannah.simulation.strategy_engine.StrategyEngine.analyse()` — strategy analysis

---

### Tool 4: `predict_winner` — `hannah/tools/predict_winner/tool.py`

**SKILL:**
```python
{
    "name": "predict_winner",
    "description": "Predicts winner probabilities for the requested race.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string"},
            "year": {"type": "integer"},
            "drivers": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["race"],
    },
}
```

**Execution flow:**
```
run(race, year=2025, drivers=None)
    │
    ├─ IF drivers is None:
    │   └─ race_data_tool.run(race, year, session="R") → resolve roster
    │
    ├─ _resolved_roster(race_data, ["VER", "NOR", "LEC"])
    │
    ├─ Build payload: {"race": race, "year": year, "drivers": resolved_drivers}
    │
    ├─ load_and_predict(payload)  # from hannah.models.train_winner
    │   ├─ IF models/saved/winner_ensemble_v1.pkl exists:
    │   │   └─ Load XGBClassifier + RandomForestClassifier ensemble → predict
    │   └─ IF no saved model:
    │       └─ Return graceful fallback message
    │
    └─ Return {"winner_probs": {driver: probability, ...}}
```

**Dependencies:**
- `race_data` tool (fetches session data for roster resolution)
- `hannah.models.train_winner.load_and_predict()` — XGBoost + RF ensemble

---

### Tool 5: `train_model` — `hannah/tools/train_model/tool.py`

**SKILL:**
```python
{
    "name": "train_model",
    "description": "Launches offline retraining jobs for Hannah's backend artifacts...",
    "parameters": {
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "enum": ["tyre_model", "laptime_model", "pit_rl", "pit_policy_q", "winner_ensemble", "all"]},
            "years": {"type": "array", "items": {"type": "integer"}},
            "races": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["model_name"],
    },
}
```

**Special: `normalize_args()` — alias normalization**
```python
_MODEL_NAME_ALIASES = {
    "lap_time_model": "laptime_model",
    "laptime": "laptime_model",
    "pit_policy": "pit_policy_q",
    "pit_strategy_model": "pit_policy_q",
    "strategy": "pit_policy_q",
    "strategy_backend": "pit_policy_q",
    "tyre_deg": "tyre_model",
    "tire_model": "tyre_model",
    "winner_model": "winner_ensemble",
    # ... 18 total aliases
}
```

**Execution flow:**
```
run(model_name, years=None, races=None)
    │
    ├─ years = years or [2024]
    ├─ model_name = _normalize_model_name(model_name)
    │
    ├─ trainers = {
    │   "tyre_model":      train_tyre_deg.train,
    │   "laptime_model":   train_laptime.train,
    │   "pit_rl":          train_pit_rl.train,
    │   "pit_policy_q":    train_pit_q.train,
    │   "winner_ensemble": train_winner.train,
    │ }
    │
    ├─ IF model_name == "all":
    │   └─ Run all 5 trainers → return {"saved": {name: path, ...}}
    │
    ├─ ELIF model_name in trainers:
    │   └─ Run single trainer → return {"saved": path}
    │
    └─ ELSE:
        └─ raise ValueError(f"Unknown model. Supported: {list(trainers)}")
```

**Training model details:**

| Model | Algorithm | Saved to | Features |
|---|---|---|---|
| `tyre_model` | GradientBoostingRegressor | `models/saved/tyre_deg_v1.pkl` | tyre_age, compound, track_temp, air_temp, rainfall |
| `laptime_model` | PyTorch LSTM (128h, 2 layers, 0.2 dropout) | `models/saved/laptime_v1.pt` | lap_number, tyre_age, compound, fuel_load, gap_to_leader |
| `pit_rl` | Stable-Baselines3 PPO + custom Gym env | `models/saved/pit_rl_v1.zip` | lap, tyre_age, compound, gap_ahead, gap_behind, deg_rate, fuel |
| `pit_policy_q` | Q-learning | `models/saved/pit_policy_q_v1.pkl` | lap, tyre_age, compound, gap_ahead, gap_behind, deg_rate |
| `winner_ensemble` | XGBoost + RandomForest soft-vote | `models/saved/winner_ensemble_v1.pkl` | grid_position, q3_time, team, track_type, tyre_strategy, avg_pace, sc_prob |

---

### Tool 6: `spawn` — `hannah/agent/worker_runtime.py`

**SKILL (SPAWN_TOOL_SPEC):**
```python
{
    "name": "spawn",
    "description": "Spawn a bounded worker with an allowlisted tool surface.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "system_prompt": {"type": "string", "minLength": 1},
            "allowed_tools": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "result_contract": {"type": "object"},
        },
        "required": ["task", "system_prompt", "allowed_tools", "result_contract"],
        "additionalProperties": False,
    },
}
```

**Execution flow:**
```
_handle_spawn_tool(task, system_prompt, allowed_tools, result_contract, state)
    │
    ├─ Validate worker spec:
    │   ├─ worker_id, task, system_prompt must not be empty
    │   ├─ allowed_tools must not be empty
    │   ├─ "spawn" NOT in allowed_tools  [depth-1 spawn enforcement]
    │   └─ result_contract must be non-empty dict
    │
    ├─ Resolve allowed tools (check they exist in registry)
    │
    ├─ Publish "subagent_spawned" event
    │
    ├─ Create RuntimeCore with:
    │   ├─ registry.subset(allowed_tools)  [restricted tool surface]
    │   ├─ allow_spawn_tool=False  [no nested spawn]
    │   └─ Same provider, event_bus, temperature, max_tokens
    │
    ├─ Run single turn: [system_prompt, task] → provider → tools → response
    │
    ├─ Coerce result to match result_contract
    │   ├─ Try json.loads()
    │   ├─ If fails, wrap in {"summary": raw_string} or {"content": raw_string}
    │   └─ If JSON is non-dict, wrap similarly
    │
    ├─ Validate result against contract
    │   ├─ Check all contract keys exist in result
    │   └─ Check types (string, integer, number, boolean, list, object)
    │
    ├─ Publish "subagent_progress" and "subagent_completed" events
    │
    └─ Return WorkerResult(worker_id, status, result, error)
```

**Policy rules (Slice 1):**
- Unknown `allowed_tools` → rejected
- Empty tool surfaces → rejected
- Nested `spawn` → rejected
- Worker output must satisfy declared `result_contract`
- Worker activity emits `subagent_*` runtime events

---

## Layer 6: Provider — LLM Abstraction

### `hannah/providers/litellm_provider.py` — `LiteLLMProvider`

```
complete(messages, tools, temperature, max_tokens)
    │
    ├─ _force_local() → True? → _local_complete()
    │   └─ HANNAH_FORCE_LOCAL_PROVIDER in {"1", "true", "yes", "on"}
    │
    ├─ import litellm fails? → _local_complete()
    │
    ├─ litellm.suppress_debug_info = True
    │
    ├─ _hosted_credentials_available() → False? → _local_complete()
    │   └─ Checks: RLM enabled OR provider API key env var is set
    │
    ├─ IF RLM enabled:
    │   └─ Set litellm.api_base and litellm.api_key from config/env
    │
    ├─ litellm.acompletion(model, _sanitize_messages(messages), tools, ...)
    │   └─ On exception → _local_complete()
    │
    └─ Return response object (with .choices[0].message)
```

### `hannah/providers/local_fallback.py` — `DeterministicFallbackPlanner`

```
complete(messages, tools)
    │
    ├─ Extract latest user text from messages
    ├─ Collect tool outputs from messages
    ├─ Determine available tool names
    │
    ├─ IF tools available AND no tool outputs yet:
    │   └─ _plan_tool_calls(user_text, available_tools)
    │       ├─ "train"/"training" → train_model
    │       ├─ "predict"/"winner"/"podium"/"probability" → race_data(Q) + predict_winner
    │       ├─ "simulate"/"simulation"/"sandbox"/"strategy"/"pit" → race_data(R) + race_sim [+ pit_strategy]
    │       ├─ "fetch"/"data"/"telemetry"/"session" → race_data(R)
    │       └─ else → [] (no tool calls)
    │
    └─ _synthesize(user_text, tool_outputs)
        ├─ train_model → "Training complete. Saved artifacts: {artifact}."
        ├─ pit_strategy → "Recommendation: pit around lap {lap} for {compound}..."
        ├─ predict_winner → "Prediction complete. Top probabilities: {top}."
        ├─ race_sim → "Simulation complete. Recommended pit lap {lap} on {compound}..."
        ├─ race_data → "Data fetched for {race} {year}. Drivers: {drivers}."
        └─ default → "No external model was available, but local planning is active..."
```

### Provider Selection: `hannah/providers/registry.py`

```
ProviderRegistry.from_config(config) → LiteLLMProvider(config)
```

Currently single-provider. The registry is a factory that always returns LiteLLMProvider.

---

## Layer 7: Session Persistence

### `hannah/session/manager.py` — `SessionManager`

```
Sessions stored as JSONL files in data/sessions/

File format:
  Line 1: {"_type": "metadata", "key": "...", "created_at": "...", "updated_at": "...", "metadata": {...}}
  Line 2-N: {"role": "user"/"assistant", "content": "...", "timestamp": "...", ...}
  Line N+1-M: {"record_type": "event", "session_id": "...", "created_at": "...", "payload": {...}}

SessionManager:
    get_or_create(key) → Session  [cache → disk → new]
    save(session) → None          [write JSONL, update cache]
    list_sessions() → list[dict]  [scan *.jsonl, read metadata]
    append_event(session_id, event) → None  [serialize + save]

SessionMemory (adapts Session to legacy Memory interface):
    add(role, content) → auto-saves
    get_recent(n=10) → last N messages
    clear() → wipes + saves
```

### `hannah/session/event_records.py`

```
serialize_event_record(event: EventEnvelope, session_id) → dict
    {
        "record_type": "event",
        "session_id": session_id,
        "created_at": event.timestamp.isoformat(),
        "payload": {
            "event_type": event.event_type,
            "message_id": event.message_id,
            "worker_id": event.worker_id,
            "payload": {json_safe_dict},
        },
    }

Handles: MappingProxyType, datetime, frozenset, non-string keys
```

---

## Layer 8: Runtime Events

### `hannah/runtime/events.py` — `EventEnvelope`

```
Allowed event types (10):
    user_message_received
    provider_request_started
    provider_response_received
    tool_call_started
    tool_call_finished
    subagent_spawned
    subagent_progress
    subagent_completed
    final_answer_emitted
    error_emitted

EventEnvelope is immutable (frozen dataclass) with deep-frozen payload.
```

### `hannah/runtime/bus.py` — `AsyncEventBus`

```
subscribe(handler, event_type=None)
    └─ event_type=None → global subscriber (receives ALL events)

publish(envelope: EventEnvelope)
    ├─ Acquire _publish_lock
    ├─ Find matching subscribers (global + event-specific)
    ├─ asyncio.gather(_dispatch(handler, envelope) for each)
    └─ Per-handler lock prevents reentrant calls

_dispatch(handler, envelope)
    ├─ Acquire per-handler asyncio.Lock
    ├─ await handler(envelope)
    └─ On error: log via Rich, do NOT crash bus
```

---

## Complete Data Flow: End-to-End Example

### Example: `hannah simulate --race bahrain --year 2025`

```
1. CLI: app.py::simulate()
   ↓
2. Prompt: build_simulate_intent(race="bahrain", year=2025, driver=None, laps=57, weather="dry")
   → "Run a race simulation for bahrain 2025. Driver: all. Laps: 57. Weather: dry."
   ↓
3. Agent command: run_agent_command(message, interactive=False, persist_session=False)
   ↓
4. Bus: run_bus_turn(agent_loop=AgentLoop(memory=_EphemeralMemory()), message=..., session_id="cli:direct", channel="cli")
   ↓
5. AgentLoop.run_turn("Run a race simulation...", session_id="cli:direct")
   ↓
6. Tool selection: _select_tools_for_turn() → all tools (no train_model hiding needed)
   ↓
7. Context build: build_main_turn() → 6 system blocks + user message
   ↓
8. Provider call: LiteLLMProvider.complete(messages, tools, temperature=0.2, max_tokens=2048)
   → Model returns tool calls for: race_data, race_sim
   ↓
9. Tool execution (concurrent):
   ├─ race_data(race="bahrain", year=2025, session="R")
   │   ├─ FastF1: fetch_session("bahrain", 2025, "R") → laps, weather, car_data, results
   │   ├─ OpenF1: get_sessions(2025, "bahrain") → session_key
   │   ├─ OpenF1: get_stints, get_weather, get_drivers (parallel)
   │   ├─ resolve_season_roster() → driver roster
   │   └─ Return {laps, stints, weather, drivers, session_info, resolved_roster}
   │
   └─ race_sim(race="bahrain", year=2025, weather="dry", drivers=["VER","NOR","LEC"], laps=57, n_worlds=1000)
       ├─ RaceContext → RaceState.from_context()
       ├─ _stable_seed() → deterministic seed
       ├─ run_fast(race_state, n_worlds=1000) → SimResult
       ├─ StrategyEngine.analyse(race_state, sim_result) → strategy
       └─ Return {simulation: {...}, strategy: {...}}
   ↓
10. Tool results serialized and appended to message state
    ↓
11. Provider call #2: messages now include tool results
    → Model returns final assistant text (no more tool calls)
    ↓
12. Publish "final_answer_emitted"
    ↓
13. Return final_text
    ↓
14. CLI: make_hannah_panel(final_text) → Rich Panel output
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| LLM is orchestrator, not simulator | Deterministic F1 work stays in tools/sim layers |
| Tool payloads compacted at 20K chars | Prevents context overflow on large race data |
| train_model hidden on analysis queries | Prevents hosted models from suggesting training instead of doing analysis |
| Deferral retry (one-shot) | Handles models that ask for permission instead of proceeding |
| Depth-1 spawn only | Prevents infinite worker recursion |
| Result contract enforcement | Workers must return structured data, not free text |
| Silent fallback to local planner | Agent stays functional even without hosted model |
| load_dotenv(override=True) | .env file takes precedence over stale shell env vars |
| JSONL session format | Append-friendly, line-oriented, easy to parse |
| Frozen event payloads | Thread-safe, hashable, safe for concurrent access |

---

## Failure Boundaries

| Failure point | Handling |
|---|---|
| Provider import fails | Silent fallback to local planner |
| Provider API key missing | Silent fallback to local planner |
| Provider call throws exception | Silent fallback to local planner |
| Tool import fails | Rich warning, tool skipped |
| Tool throws exception | Error serialized as tool message, loop continues |
| Worker result contract violation | Error status returned, not silently accepted |
| Unknown worker tool requested | WorkerPolicyError raised, spawn rejected |
| Nested spawn attempted | WorkerPolicyError raised, spawn rejected |
| Session file corrupt | Silently returns None, new session created |
| Bus subscriber throws | Logged via Rich, does not crash bus |
| Oversized tool payload | Compacted (race_data: specialized summary, others: generic) |
