# Hannah v1 Public Scenario Contracts

This directory defines the **public** scenario contracts for Hannah v1.

These tests are intentionally broad:
- they lock expected tool paths,
- they lock required output shapes,
- they keep deterministic execution for CI stability,
- they avoid encoding hidden acceptance answers.

## Scope

The scenario matrix covers ~20 public contracts across:
- strategy calls,
- prediction flows,
- safety-car and weather crossover patterns,
- tyre-cliff and undercut/overcut pressure,
- training smoke paths.

## Contract Model

Each public scenario defines:
- `input_context`: race/lap/weather/rival setup used for the scenario
- `available_telemetry`: which telemetry families are assumed available
- `expected_tool_path`: ordered tool sequence expected for the scenario
- `tool_inputs`: concrete inputs per tool in the expected path
- `expected_recommendation_shape`: required keys for strategy recommendations
- `expected_sim_output_shape`: required keys for simulation payloads
- `expected_prediction_shape`: required keys for winner prediction payloads
- `expected_training_shape`: required keys for training payloads
- `pass_fail_criteria`: human-readable criteria for broad contract success

## Determinism

Public tests patch external/network/simulation dependencies so results are deterministic:
- `race_data` FastF1/OpenF1 calls are replaced with stable fixtures
- Monte Carlo simulation calls are replaced with deterministic `SimResult`
- training smoke assertions validate artifact path contracts, not model quality

## Notes

- These tests are **not** hidden acceptance tests.
- Hidden acceptance scenarios can be stricter and assert richer behavior.
- Public tests here are designed to keep v1 implementation aligned to architecture boundaries.
