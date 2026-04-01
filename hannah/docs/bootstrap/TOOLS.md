# Hannah Tools

Hannah exposes a small F1-focused tool surface. The bootstrap loader should only describe these tools so the agent can assemble context later.

- `race_data` fetches race telemetry and session data.
- `pit_strategy` evaluates pit windows and strategy calls.
- `race_sim` runs the fast Monte Carlo simulation.
- `predict_winner` returns winner probabilities.
- `train_model` launches the model-training workflows.
- `spawn` creates bounded subagents when the runtime needs decomposition.

The rule is simple: tools own the deterministic work, not the model.

