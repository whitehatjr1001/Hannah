# Prompt and Skill Card Guide

## Purpose

This guide defines how to write better prompts and skill cards for Hannah's agent-first runtime. It is intentionally short and implementation-oriented so prompt authors can create reliable runtime instructions, tool descriptions, and spawned worker definitions without over-engineering.

## Prompt Writing Principles

### 1. State the job clearly

Start with a direct statement of role and objective. A good prompt makes it obvious:

- who the model is for this task
- what outcome it must produce
- what constraints it must respect

Avoid broad personas that do not change behavior. Prefer operational roles such as:

- "You are an F1 strategy analyst validating pit-window recommendations."
- "You are a telemetry worker fetching and summarizing race data."

### 2. Separate instructions, context, and input

Do not mix everything into one paragraph. Keep these layers distinct:

- instructions: what the model must do
- context: rules, constraints, background
- input: the current task or payload

For structured prompts, prefer explicit sections or tags so the runtime can compose them safely.

### 3. Be specific about outputs

Tell the model exactly what shape the answer should take:

- final answer only
- JSON object with named fields
- bullet summary plus evidence
- recommendation with confidence and assumptions

If the output will be parsed or post-processed, specify the schema directly.

### 4. Give the model the decision boundary

A strong prompt makes the boundary between reasoning and action explicit. For Hannah prompts, say when the model should:

- answer directly
- call a tool
- spawn a worker
- refuse to guess and ask for more data

This matters more than style language.

### 5. Use examples when behavior matters

If format, tone, or task decomposition matters, include a few small examples that match the real use case. Examples should be:

- relevant to the task
- diverse enough to avoid overfitting to one pattern
- short enough that the actual task remains primary

### 6. Encode constraints positively

Do not rely only on vague bans like "don't hallucinate." Replace them with clear operational rules such as:

- "Use race_data before making claims about prior-session telemetry."
- "Do not invent lap times or tyre compounds; cite tool results or state that data is unavailable."

Positive operating rules are easier for the model to follow than abstract warnings.

### 7. Keep prompts narrow

One prompt should own one job. If a prompt tries to plan, fetch data, simulate, compare rivals, and write a report all at once, split the work:

- main agent decides
- tools gather evidence
- spawned workers handle bounded subtasks

That keeps both prompts and runtime behavior understandable.

## Prompt Template for Main-Agent Tasks

Use this shape for runtime-composed task prompts:

```text
Role: <operational role>

Objective:
- <what the agent must accomplish>

Constraints:
- <architectural or domain rules>
- <when to use tools>
- <what not to guess>

Available actions:
- answer directly
- call tools
- spawn bounded workers when decomposition is useful

Required output:
- <format>

Task input:
<current user request or structured intent>
```

## Prompt Template for Spawned Workers

Spawned workers should be even narrower:

```text
Role: <worker role>

You are a bounded worker inside the Hannah runtime.
Complete only the assigned task.
Use only the allowed tools.
Do not call spawn.
Return only the requested result contract.

Task:
<specific subtask>

Allowed tools:
- <tool name>
- <tool name>

Result contract:
- summary
- evidence
- unresolved risks
```

Prompt text is not the enforcement layer. The runtime should pass `allowed_tools` as structured worker configuration and the prompt should mirror that boundary for clarity.

## Skill Card Principles

In this redesign, a skill card means an authoring and reference artifact for developers and prompt authors. It does not imply a new runtime skill execution subsystem unless a later spec explicitly adds one.

A skill card should tell authors:

- when to use a skill
- what problem it solves
- what boundaries it must obey
- what output or workflow it should produce

The best skill cards are invocation guides, not essays.

### 1. Define a narrow purpose

A skill card should have one primary job. Good examples:

- brainstorm a design before implementation
- review a spec for planning readiness
- write a deterministic worker prompt

Bad examples combine unrelated work such as planning, coding, reviewing, and release management in one card.

### 2. Make invocation criteria explicit

The skill card should answer:

- when should this skill be used
- when should it not be used
- what signals or trigger phrases suggest it applies

This prevents both overuse and missed activation.

### 3. State boundaries and failure modes

If the skill is rigid, say so. If it has hard gates, make them visible near the top. If it must not perform implementation work, say that explicitly.

Every good skill card should make misuse obvious.

### 4. Define the workflow as ordered steps

Use a short ordered checklist when sequence matters. That makes the skill easier to execute consistently and easier to test.

### 5. Define output expectations

Say what the user should get at the end:

- a design
- a reviewed spec
- a plan
- a concise recommendation

If the output has a path or schema, include it.

### 6. Keep examples grounded

Include short examples of:

- a good invocation
- a bad invocation
- a minimal expected output

Examples should reflect the actual repo style and task patterns.

## Skill Card Template

```md
---
name: <skill-name>
description: <one-line trigger-oriented summary>
---

## Purpose
<one job this skill does>

## Use When
- <trigger 1>
- <trigger 2>

## Do Not Use When
- <boundary 1>
- <boundary 2>

## Inputs
- <required context>

## Workflow
1. <step one>
2. <step two>
3. <step three>

## Output
- <what this skill must produce>

## Constraints
- <hard rule>
- <hard rule>
```

## Hannah-Specific Authoring Rules

For Hannah's runtime, prompts and skill cards should preserve these boundaries:

- tools own F1 domain work
- the main loop is an orchestrator
- workers are bounded and cannot recursively spawn in v1 of the redesign
- streaming events should align with real runtime actions
- prompts should prefer evidence-backed outputs over stylistic flourish
- claims about telemetry, simulation, prediction, or training outcomes must be grounded in tool outputs or explicit data absence, not inferred as facts from prior context alone

## Review Checklist

Before accepting a new prompt or skill card, check:

- Is the job narrow and unambiguous?
- Does it say when to use tools versus answer directly?
- Does it define output shape clearly?
- Does it avoid hidden assumptions about unavailable data?
- Does it preserve Hannah's architectural seams?
- Could a new contributor use it correctly without reading implementation internals?

## Recommendation

Treat prompt and skill-card authoring as part of runtime design, not an afterthought. If prompts and skill cards are vague, the agent runtime will behave like a wrapper with heuristics. If they are precise, the runtime can stay autonomous without becoming sloppy.
