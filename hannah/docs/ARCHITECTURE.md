# ARCHITECTURE.md — Hannah Smith v1
> Nanobot-style agent kernel + FormulaGPT race semantics + pit-stop-simulator ML assets

## What This Architecture Is Optimizing For

For the concrete runtime sequence, see [AGENT_LOOP.md](AGENT_LOOP.md).

Hannah v1 should be built as a **small, readable, CLI-native Python package** with:

- a thin command-line entrypoint,
- a reusable agent kernel,
- a domain-specific F1 strategy layer,
- an offline-trainable simulation and modeling stack,
- a clean path to swap hosted LLMs for a local RLM later.

The right approach is **not** to merge three repos blindly.
The right approach is to use each repo for the part it is already good at:

- `nanobot` for agent packaging, tool boundaries, config, CLI ergonomics
- `FormulaGPT` for race-state concepts, command vocabulary, team/driver strategy semantics
- `pit-stop-simulator` for environment design, RL training, lap-time and pit-decision modeling

---

## My Take On Your Plan

The plan is directionally correct.

Using those codebases is faster than building Hannah from zero, but only if you keep the reuse disciplined:

- **Use nanobot as the architecture reference, not as a code dump.**
- **Use FormulaGPT for domain logic ideas, not for frontend structure.**
- **Use pit-stop-simulator for training and environment logic, not for Streamlit app design.**

If you mix UI code, agent code, and simulation code directly, Hannah will turn into a stitched prototype instead of a maintainable product.

---

## Reuse Matrix

### 1. Reuse from `nanobot`

Adopt the style and some patterns from:

- package-oriented layout
- CLI command module separation
- provider abstraction
- agent loop separation from transport/UI
- skill/tool registry
- session and memory boundaries
- small composable modules

Good targets to emulate conceptually:

- `nanobot/cli/`
- `nanobot/agent/`
- `nanobot/providers/`
- `nanobot/config/`
- `nanobot/session/`

Do **not** bring over:

- multi-channel chat integrations
- general-purpose messaging bus unless Hannah actually needs it
- extra platform adapters

Hannah is simpler: one CLI, one domain, one main operator.

### 2. Reuse from `FormulaGPT`

Use it for:

- F1 team and driver domain modeling
- command grammar like `pit`, `push`, `conserve`
- race-state snapshot structure for strategist prompts
- rival-team reasoning patterns
- tire and pace semantics

Do **not** reuse:

- React component structure
- animation loop
- browser state management
- frontend-only data plumbing

FormulaGPT proves an important design idea:
the LLM should propose strategy commands, not become the simulator.

That idea should be central to Hannah.

### 3. Reuse from `pit-stop-simulator`

Use it for:

- Gym-style environment concepts
- PPO training path
- Q-learning baseline if you want a cheap benchmark
- lap-time model features
- pit-stop reward shaping ideas
- scenario variables: weather, traffic, safety car, fuel load, tire wear

Do **not** reuse as-is:

- Streamlit app
- dashboard/reporting code
- runtime assumptions tied to the app UI

For Hannah, this repo should become the source for:

- offline model training scripts
- evaluation harnesses
- environment abstractions behind CLI tools

---

## Architectural Rule

**The agent does not simulate the race.**

The agent:

- reads user intent,
- fetches data,
- calls simulation/model tools,
- coordinates specialist sub-agents,
- returns a recommendation.

The simulation and ML stack:

- models race dynamics,
- predicts outcomes,
- scores strategy options,
- returns structured outputs.

This separation is the single most important thing to preserve from your source repos.

---

## Proposed Repository Shape

The cleanup target is now straightforward:

- `hannah/` is the single package root
- project docs live under `hannah/docs/`
- the repo root stays thin

```text
files/
├── pyproject.toml
├── README.md
├── .env.example
├── config.yaml
├── hannah.py                     # thin CLI launcher only
├── tests/
│
├── hannah/
│   ├── __init__.py
│   ├── docs/
│   │   ├── GOAL.md
│   │   ├── PRD.md
│   │   ├── ARCHITECTURE.md
│   │   └── AGENT_LOOP.md
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── app.py                # click/typer entrypoint
│   │   └── format.py             # rich rendering, panels, colors
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── schema.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── loop.py               # main tool-using loop
│   │   ├── context.py            # race/session command context
│   │   ├── persona.py
│   │   ├── memory.py
│   │   ├── subagents.py
│   │   ├── prompts.py
│   │   └── tool_registry.py
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── litellm_provider.py
│   │   └── registry.py
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── teams.py              # drivers, teams, aliases
│   │   ├── commands.py           # pit/push/conserve schema
│   │   ├── prompts.py            # strategist prompt builders
│   │   ├── race_state.py         # shared typed race state
│   │   └── tracks.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fastf1_loader.py
│   │   ├── openf1_client.py
│   │   ├── openpitwall_loader.py
│   │   ├── cache.py
│   │   └── preprocess.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── sandbox.py
│   │   ├── monte_carlo.py
│   │   ├── tyre_model.py
│   │   ├── gap_engine.py
│   │   ├── strategy_engine.py
│   │   ├── environment.py        # optional gym-style wrapper
│   │   └── competitor_agents.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── race_data/
│   │   │   ├── SKILL.md
│   │   │   └── tool.py
│   │   ├── race_sim/
│   │   │   ├── SKILL.md
│   │   │   └── tool.py
│   │   ├── pit_strategy/
│   │   │   ├── SKILL.md
│   │   │   └── tool.py
│   │   ├── predict_winner/
│   │   │   ├── SKILL.md
│   │   │   └── tool.py
│   │   └── train_model/
│   │       ├── SKILL.md
│   │       └── tool.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_tyre_deg.py
│   │   ├── train_laptime.py
│   │   ├── train_pit_rl.py
│   │   ├── train_winner.py
│   │   ├── evaluate.py
│   │   └── saved/
│   │
│   ├── rlm/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── time.py
│       └── logging.py
│
└── data/
    └── fastf1_cache/
```

---

## Why This Structure Works

### `hannah/agent`

This is the nanobot-style kernel.
It should contain only orchestration concerns:

- prompt assembly
- memory
- tool dispatch
- sub-agent execution
- final synthesis

It should not contain telemetry parsing, lap-time physics, or model training logic.

### `hannah/domain`

This is where FormulaGPT concepts belong after being ported into Python.
Put shared F1 language here:

- teams
- drivers
- command schema
- track metadata
- prompt fragments
- race state DTOs

This prevents simulation code and prompts from duplicating the same team/tire logic.

### `hannah/simulation`

This is where pit-stop-simulator ideas should land, but rewritten around Hannah’s needs:

- fast batch simulation
- strategy scoring
- undercut/overcut analysis
- RL environment wrappers
- competitor policy evaluation

This module should be callable directly from tools and tests.

### `hannah/tools`

Each tool should be thin.
It should:

- validate inputs,
- call domain/data/simulation/model code,
- return structured output.

Do not let tool files become business-logic dumping grounds.

### `hannah/models`

All training scripts and evaluators live here.
This keeps offline experimentation separate from runtime inference.

### `hannah/providers`

This is the LiteLLM boundary.
The rest of the agent should not care whether the underlying model is:

- Claude
- OpenAI
- OpenRouter
- local RLM

That is exactly the v1 to v2 seam you want.

---

## Nanobot-Style Principles Hannah Should Copy

Copy these principles, not necessarily the exact code:

1. **Package-first layout**
   Root scripts should be tiny wrappers.

2. **One clear agent loop**
   The main loop should be easy to read in one file.

3. **Explicit tool registry**
   Tools should be discoverable and inspectable.

4. **Provider abstraction**
   Keep model routing behind one boundary.

5. **Config loading in one place**
   Avoid `os.getenv()` scattered everywhere.

6. **Session and memory separation**
   Memory should be a service, not mixed into CLI code.

7. **Small modules**
   Split by responsibility, not by feature marketing.

---

## Suggested Runtime Flow

```text
CLI command
  -> parse command/options
  -> build AgentCommandContext
  -> agent loop
      -> fetch memory + persona
      -> call LiteLLM provider
      -> dispatch tools
          -> data tool
          -> simulation tool
          -> prediction tool
          -> training tool
      -> optionally spawn sub-agents
      -> synthesize response
  -> rich terminal renderer
```

For a simulation request:

```text
user asks for strategy/simulation
  -> race_data tool builds structured race context
  -> simulation tool runs monte carlo and strategy engine
  -> rival sub-agents inspect the same context
  -> winner model adds finishing probability
  -> Hannah returns one decisive recommendation
```

---

## Import Strategy From Source Repos

Do the import in this order.

### Phase 0 — Repo Shape

This cleanup is now the baseline:

- `hannah/` is the package root
- `hannah.py` stays a thin wrapper
- docs live under `hannah/docs/`
- duplicate root modules are removed instead of maintained in parallel

### Phase 1 — Port nanobot-style agent kernel

Build first:

- `hannah/cli/`
- `hannah/config/`
- `hannah/providers/`
- `hannah/agent/loop.py`
- `hannah/agent/tool_registry.py`
- `hannah/agent/memory.py`

At the end of this phase, Hannah should already be able to:

- accept CLI commands
- call LiteLLM
- list tools
- persist memory

### Phase 2 — Port FormulaGPT domain semantics

Build:

- `hannah/domain/teams.py`
- `hannah/domain/commands.py`
- `hannah/domain/prompts.py`
- `hannah/domain/race_state.py`

Use FormulaGPT to define:

- valid strategy commands
- team/driver mappings
- prompt formats for rival strategy agents
- race snapshot layout

### Phase 3 — Port pit-stop-simulator environment ideas

Build:

- `hannah/simulation/environment.py`
- `hannah/simulation/tyre_model.py`
- `hannah/simulation/strategy_engine.py`
- `hannah/models/train_pit_rl.py`
- `hannah/models/train_laptime.py`

Use the repo as a modeling reference, not as final package structure.

### Phase 4 — Integrate fast prediction engine

Build:

- `hannah/simulation/monte_carlo.py`
- `hannah/tools/race_sim/tool.py`
- `hannah/tools/predict_winner/tool.py`

This is the step where Hannah becomes product-like.

### Phase 5 — Add async rival sub-agents

Build:

- `hannah/agent/subagents.py`
- `hannah/simulation/competitor_agents.py`

Each rival should produce structured strategy opinions, not free-form noise.

### Phase 6 — Lock the v2 seam

Build:

- `hannah/rlm/server.py`
- `hannah/providers/litellm_provider.py` support for local base URL
- a smoke test that swaps hosted model to local RLM config

---

## What I Would Not Do

I would not:

- copy the entire `nanobot` package into this repo
- port React code from FormulaGPT into Python
- keep Streamlit inside Hannah v1
- let RL training code sit inside runtime tool files
- let the LLM own the race state directly
- maintain both root-level modules and packaged modules long term

Those are the shortcuts that will slow you down later.

---

## Current Cleanup Baseline

Keep the repo in this shape:

1. `hannah/` remains the single source package root.
2. `hannah/docs/` holds the product and handoff markdown.
3. `hannah.py` stays a thin launcher only.
4. Do not reintroduce duplicate root modules like `core.py` or `server.py`.
5. Keep the repo root focused on launcher/config/test metadata, not parallel runtime code.

If you skip this, every future import boundary will get harder.

---

## Recommended v1 Milestone Order

1. **Kernel**
   CLI, config, provider, memory, tool registry

2. **Data**
   FastF1, OpenF1, caching, preprocessing

3. **Simulation**
   typed race state, tyre model, strategy engine, Monte Carlo

4. **Prediction**
   winner model, pit model, evaluation harness

5. **Sub-agents**
   rival team strategist agents and streamed output

6. **Training**
   tyre, lap-time, pit RL, winner ensemble

7. **RLM seam**
   local OpenAI-compatible backend swap

---

## Bottom Line

Your plan is good if you make Hannah a **nanobot-style Python agent kernel with F1-specific domain and simulation modules**.

The correct blend is:

- `nanobot` for structure
- `FormulaGPT` for race semantics
- `pit-stop-simulator` for modeling and training

The wrong blend is:

- copying all three repos into one directory tree and hoping it becomes a product.

If we do this properly, Hannah will stay small, readable, and migration-ready.
