# SGLang Diffusion Single-Agent Flow Design

## Goal

Replace the SGLang Diffusion optimizer's Master/executor agent hierarchy with
one interactive root agent: the agent in the user's current conversation.

The Python package remains as a deterministic experiment tool. It locks source
and workload state, runs or records benchmarks, verifies durable evidence,
tracks the search ledger, integrates accepted candidates, and packages the
patch. It must never launch Codex, Claude, or any other AI process.

## User Contract

One optimization campaign has exactly one AI owner:

- the root agent in the current conversation chooses hypotheses;
- the same root agent edits one assigned worktree at a time;
- the same root agent launches candidate measurements serially;
- the same root agent performs the code/method or visual review;
- deterministic Python recomputes performance and checks artifact provenance;
- no Master, executor, reviewer, or integration sub-agent is started.

The user supplies the same four inputs as before: machine, model, frozen
baseline command, and target measured end-to-end speedup. The skill owns the
campaign until a reviewed terminal condition is reached.

## Approaches Considered

### 1. Interactive root agent plus a thin deterministic campaign tool

Keep the source lock, baseline runner, profiler, state store, verifier,
integrator, progress renderer, and patch packager. Replace agent spawning with
an explicit work-order protocol between the CLI and the current conversation.

This is the chosen approach. It preserves reproducibility and durable recovery
without creating competing agents or GPU jobs.

### 2. Delete the Python package and implement the workflow only in `SKILL.md`

The root agent would run shell commands and maintain Markdown notes manually.
This removes the controller completely, but loses idempotency, command
freezing, machine-readable evidence, deterministic speedup recomputation,
structured progress, and safe patch packaging.

### 3. Keep one detached executor agent and serialize all techniques inside it

This would eliminate sibling contention but still create an AI process outside
the user's conversation. It violates the one-agent requirement and leaves two
contexts that can diverge.

## Architecture

The system is split into an AI control plane and a deterministic evidence
plane.

### AI control plane

The installed `sglang-diffusion-auto-optimize` skill governs the current root
agent. It tells that agent how to:

1. resolve the remote machine and fixed GPU set;
2. create or resume one campaign;
3. inspect the frozen baseline and profile;
4. claim one technique work order;
5. implement and measure hypotheses serially;
6. write a structured self-review bound to real artifacts;
7. submit the delivery for deterministic verification;
8. integrate only verified latency-positive candidates; and
9. continue until the target or reviewed search boundary is reached.

The skill must explicitly prohibit `spawn_agent`, nested `codex exec`, Claude
sessions, background AI workers, and parallel candidate GPU runs.

### Deterministic evidence plane

`sgl-engine-sglang-diffusion` remains a normal Python CLI. It may launch the
frozen benchmark command and other non-AI validation processes. It may not
launch an agent command.

Its responsibilities are:

- source and workload locking;
- one authoritative baseline;
- native-backend profiling;
- technique routing suggestions;
- serial worktree and work-order creation;
- run-directory and artifact validation;
- raw latency and speedup recomputation;
- deterministic LPIPS and structural checks;
- search-budget accounting;
- selected-subset integration;
- clean-room final validation;
- progress projection and patch packaging.

The following components leave the campaign execution path:

- `AgentRunner`;
- `ExecutorManager`;
- executor process receipts and leases;
- Master-agent method auditing;
- AI-based quality evaluator subprocesses;
- executor resume prompts;
- per-role executor/Master token accounting.

Files that have no remaining non-agent consumer are removed rather than kept as
dormant orchestration code.

## Campaign State and CLI Protocol

Add `AWAITING_AGENT` as a first-class, non-terminal state. It means the
deterministic tool has finished all currently possible work and is waiting for
the current conversation to choose or complete one action.

The lifecycle is:

```text
NEW
  -> BASELINE_LOCKED
  -> PROFILED
  -> AWAITING_AGENT
  -> SEARCHING
  -> INTEGRATING
  -> FINAL_VERIFYING
  -> AWAITING_AGENT | TARGET_REACHED | SEARCH_SPACE_EXHAUSTED
```

`WAITING_RESOURCE`, `INFRA_BLOCKED`, and `PAUSED_BUDGET` remain recoverable.
They never become scientific search exhaustion automatically.

The root agent uses these commands:

```text
sgl-diffusion-engine launch --request <request> --detach
sgl-diffusion-engine work --campaign <campaign> --json
sgl-diffusion-engine claim --campaign <campaign> --technique <name>
sgl-diffusion-engine submit --campaign <campaign> --delivery <DELIVERY.json>
sgl-diffusion-engine skip --campaign <campaign> --technique <name> \
  --classification <unsupported|no_gain|blocked> --reason <text>
sgl-diffusion-engine progress --campaign <campaign> --json
```

`launch --detach` may use the existing watchdog to run deterministic setup,
baseline, and profiling. The watchdog stops at `AWAITING_AGENT`; it never
starts AI work.

`work` reports:

- the current status and active work order, if any;
- routed techniques and the profile evidence for each;
- per-technique used and remaining scientific rounds;
- prior failure signatures and classifications;
- verified frontier points;
- the current integrated result; and
- the exact next CLI actions that are legal.

`claim` creates exactly one campaign-owned worktree and `AGENT-WORK.json`.
Claiming a second technique while one is active fails closed. The work order
contains the frozen baseline, profile digest, technique contract, source lock,
worktree, delivery path, and remaining round budget.

`submit` does not trust the root agent's reported performance. It recomputes
latency, speedup, hashes, command equivalence, backend/fallback status,
engagement, and deterministic correctness metrics from the referenced run
bundle. Accepted points enter the verified frontier. Rejected points return
structured findings to the same conversation and return the campaign to
`AWAITING_AGENT`.

`skip` requires an explicit classification and reason. `unsupported` records a
hardware or runtime capability boundary. `no_gain` closes a reviewed technique
that has no useful frontier. `blocked` records an external dependency or
resource boundary and is recoverable unless the root agent explicitly closes
it after review.

## Serial Search and Resource Ownership

There is at most one active technique and one candidate benchmark at a time.
The fixed GPU set remains identical to the baseline for every run.

The work-order protocol, not a per-technique process lease, is the admission
boundary. Baseline, profile, candidate, integration, and final verification
must not overlap. Every run receives a unique run directory and distributed
rendezvous port.

Before launching a candidate, the root agent checks the recorded GPU UUIDs,
process ownership, memory, and utilization. A busy fixed GPU produces
`WAITING_RESOURCE`; it is not a failed hypothesis.

## Routing and Budget Semantics

The profiler produces suggestions, not mandatory lanes. The root agent chooses
which technique to claim based on hardware capability, model support, measured
hotspots, and prior evidence.

Routing must exclude known-inapplicable methods. For example, NVFP4 cannot be a
Hopper candidate. Quality-gated routing does not automatically enable every
quality-changing technique.

A scientific round is counted only when one distinct candidate completes a
full frozen-workload measurement and reaches a gate decision.

These events do not consume a scientific round:

- GPU or port contention;
- remote disconnect;
- agent interruption;
- missing external dependency discovered during preflight;
- benchmark process launch failure before model execution;
- malformed delivery metadata that references no candidate run.

A repeated failure signature rejects that hypothesis. It does not terminate
the technique or campaign. `SEARCH_SPACE_EXHAUSTED` is valid only after all
routed techniques are explicitly `unsupported`, `no_gain`, or out of
scientific rounds and the best verified integrated subset remains below target.

## Verification Model

Removing the Master agent changes the trust statement. The flow must no longer
claim independent AI review.

The root agent writes `AGENT-REVIEW.json` for every submitted candidate:

- candidate and run identifiers;
- reviewed source commit and diff hash;
- authenticity decision;
- engagement evidence;
- lossless method-equivalence argument, or lossy visual-artifact verdict;
- inspected frame paths for visual review;
- known limitations.

The deterministic verifier binds that review to source and artifact hashes and
performs every check that does not require AI judgment:

- frozen command equivalence;
- source and implementation provenance;
- native backend and zero forbidden fallback;
- request success counts and timing scope;
- raw latency and speedup recomputation;
- global denoising-step and DiT-call counts;
- declared engagement counters;
- aligned LPIPS and other configured numeric quality checks;
- real video/frame/run-bundle existence.

For lossless methods, the same root agent's code review replaces the former
Master method auditor. For quality-gated methods, the same root agent's
multimodal inspection replaces the former evaluator subprocess. Final
reporting calls these `same_agent_method_review` and
`same_agent_visual_review`, never `independent_master_review`.

Clean-room final execution remains deterministic and uses the packaged patch
against the locked base commit.

## Integration

Integration operates on a selected verified subset, not on every routed
technique.

A candidate is eligible for the default latency stack only when:

- its full-workload latency is better than the frozen baseline or current
  integrated stack;
- its required correctness or quality gates pass;
- its activation is compatible with the already selected stack; and
- it has no forbidden fallback.

Memory-only Pareto points and latency regressions remain reportable but are not
forced into the latency stack. A skipped, unsupported, blocked, or exhausted
technique does not prevent integration.

Every new selected point triggers a fresh composed full-workload run. Isolated
speedups are never added or multiplied.

## Persistence and Compatibility

New campaigns use the single-agent protocol by default and contain no agent
command in their normalized goal.

For input compatibility, version-1 requests containing `agent.command` remain
parseable, but the field is deprecated and ignored. The frozen campaign
records `execution_mode: interactive_single_agent`.

Legacy campaigns already in `SEARCHING` with live or recorded executor
processes are not silently migrated. Status and artifacts remain readable, but
resume returns an actionable incompatibility explaining that a new
single-agent campaign must be created from the same locked source and workload.

## Documentation Changes

Update the package README, skill, request template, progress contract, and
examples to:

- describe one interactive root agent;
- remove Executor/Master terminology and token-role examples;
- document `work`, `claim`, `submit`, and `skip`;
- explain that the conversation must remain available while agent judgment is
  required, while deterministic baseline/profile work may remain detached;
- distinguish deterministic verification from same-agent judgment;
- state that no AI subprocess is launched.

## Testing

Add focused tests proving:

1. no campaign command calls `AgentRunner`, `subprocess` with `codex`, or
   `ExecutorManager`;
2. launch reaches `AWAITING_AGENT` with one route ledger and no executor
   receipts;
3. only one technique can be claimed;
4. `submit` recomputes evidence and rejects fabricated speedup;
5. rejected delivery returns findings without starting an agent;
6. infrastructure failures do not consume scientific rounds;
7. a repeated failure signature does not end the campaign;
8. skipped or unsupported routes do not block integration;
9. a latency-regressive memory point is not forced into the latency stack;
10. same-agent review terminology replaces independent-Master claims;
11. legacy multi-agent campaigns fail closed on resume without deleting
    artifacts; and
12. the package test suite, repository tests, Ruff, pre-commit, and
    `git diff --check` pass.

## Non-Goals

- Running multiple root conversations against one campaign.
- Preserving automatic background AI progress after the conversation ends.
- Claiming independent AI review with only one agent.
- Migrating a partially executed legacy multi-agent search in place.
- Changing Sol-derived numerical quality thresholds.
- Weakening frozen-workload, native-backend, engagement, or clean-room gates.

## Acceptance Criteria

The change is complete when a new campaign can freeze and profile
deterministically, wait for the current root agent, accept exactly one serial
work order, verify its real delivery without launching AI, integrate a selected
verified subset, and produce honest progress and terminal artifacts.

A repository-wide search of the new campaign path must show no reachable
`codex exec`, Claude session, Master agent, executor agent, or reviewer-agent
launch.
