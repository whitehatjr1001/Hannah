# GOAL.md — Hannah Smith: Red Bull Race Director
> F1 Strategy Simulation CLI Agent

---

## 🎯 North Star

Build **Hannah Smith** — a CLI-first, agentic AI system that acts as a virtual Red Bull Race Director. Hannah ingests real F1 telemetry, runs fast async race simulations, makes data-backed strategy calls (pit windows, tyre compounds, undercut/overcut), and predicts upcoming race outcomes — all from a single terminal interface inspired by Claude Code and NanoChat.

---

## 🏁 Core Goals

### G1 — Real F1 Data, Live and Historical
- Pull lap times, tyre data, sector splits, weather, and car telemetry from **FastF1** and **OpenF1**
- Cache data locally for offline training and simulation
- Support any race from 2018 → present + upcoming race weekends

### G2 — Fast Async Race Simulation (Prediction Mode)
- Simulate a full race in **under 2 seconds** using vectorised Monte Carlo (not real-time lap-by-lap rendering)
- Run 1000 parallel race worlds simultaneously using `numpy` + `asyncio`
- Output: winner probabilities, optimal pit windows, tyre strategy, gap deltas

### G3 — NanoChat-Style Sub-Agent Architecture
- Spawn independent async sub-agents for each concern: simulation, strategy, prediction, rival agents
- All agents run concurrently via `asyncio.gather()` — results stream back as each finishes
- Each agent is a lightweight NanoChat-style unit with its own system prompt and tool set
- Agents are scalable — add more rival agents, more simulation workers, without changing core

### G4 — Train Models From the CLI
- `hannah --train tyre_model --year 2024` streams training loss live to terminal
- Uses **OpenPitWall** data + FastF1 telemetry as training corpus
- Supports: tyre degradation model, lap time predictor, pit stop RL agent (PPO), winner ensemble
- Models saved to `models/saved/` as `.pkl` / `.pt` checkpoints

### G5 — LiteLLM Unified Gateway (v1 → v2 Migration-Ready)
- All LLM/RLM calls route through **LiteLLM** — one unified `completion()` interface
- v1: `HANNAH_MODEL=claude-sonnet-4-6` — routes to Claude
- v2: `HANNAH_MODEL=openai/rlm-local` — routes to local RLM with zero agent code changes
- Sub-agents (rival teams) also use LiteLLM — swappable independently per agent

### G6 — RLM as Drop-In API (v2)
- Build `rlm/` module in parallel from day one — FastAPI server exposing OpenAI-compatible `/v1/chat/completions`
- Train the RLM on F1 race strategy data accumulated during v1 operation
- v2 migration = change one `.env` line — no rewrites, no agent changes

### G7 — Claude Code-Style CLI Experience
- Dark terminal UI, monospaced output, colour-coded race data (green = good, amber = warning, red = critical)
- Streaming output — results appear as sub-agents complete, not all at once
- Shortcut buttons + freeform natural language commands
- Hannah speaks in character: direct, data-backed, decisive — like a real race director on the pit wall

---

## 📐 Success Metrics

| Metric | Target |
|---|---|
| Simulation speed (prediction mode) | < 2 seconds for 1000-world Monte Carlo |
| Sub-agent parallelism | All agents fire concurrently, no sequential blocking |
| CLI response latency | First token streamed < 500ms |
| Winner prediction accuracy | > 60% top-3 accuracy on 2024 test set |
| Tyre deg model RMSE | < 0.8 laps on held-out test data |
| v1 → v2 migration effort | Change 1 `.env` line, zero code changes |
| Training CLI | Full tyre model trained in < 5 min on 3 seasons of data |

---

## 🚫 Non-Goals (v1)

- No real-time race broadcast integration (v2+)
- No web UI or dashboard (CLI only in v1)
- No multi-user / cloud deployment (local only in v1)
- No full lap-by-lap animated replay (prediction mode only — replay is v2)
- RLM training pipeline is v2 — v1 only builds the server scaffold

---

## 📦 Deliverables (v1)

1. `hannah.py` — CLI entrypoint
2. `agent/` — NanoChat-style agent core with LiteLLM + sub-agent spawner
3. `simulation/` — fast async Monte Carlo engine
4. `tools/` — SKILL.md registered tools (race_data, pit_strategy, race_sim, predict_winner, train_model)
5. `models/` — training scripts for all four models
6. `rlm/` — FastAPI scaffold for v2 (server only, no weights yet)
7. `data/` — OpenF1 client, FastF1 cache, OpenPitWall loader
8. `GOAL.md` + `PRD.md` + `ARCHITECTURE.md`
