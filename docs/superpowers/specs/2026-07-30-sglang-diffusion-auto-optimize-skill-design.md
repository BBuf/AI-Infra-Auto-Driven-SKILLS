# SGLang Diffusion Auto-Optimize Skill Design

**Date:** 2026-07-30

**Status:** Approved by standing user instruction to finish without further
clarification

## Goal

Turn `sgl-engine-sglang-diffusion` into a conversational, installable skill that
can own a complete SGLang Diffusion optimization campaign. A user supplies only:

- the target machine;
- the diffusion model;
- the exact command that establishes the baseline; and
- the desired end-to-end speedup.

The agent must enter the target environment, freeze and run the baseline, start
a durable optimization controller, monitor it across terminal/chat disconnects,
and stop only at an existing terminal state: target reached, search space
exhausted, or independently certified unreachable. The result remains a patch
against the SGLang commit locked from the requested checkout.

## Non-Goals

- Do not replace the Sol-Engine-derived correctness and quality gates.
- Do not attribute additive speedups to techniques without measurements.
- Do not build a network service, web dashboard, scheduler, or shared
  multi-tenant control plane.
- Do not assume that a requested speedup is physically achievable.
- Do not permit a natural-language wrapper to weaken immutable workload,
  source-lock, engagement, clean-room, or final verification requirements.

## Selected Approach

Use a thin conversational skill above a richer local controller:

```text
natural-language request
        |
        v
sglang-diffusion-auto-optimize Skill
  - resolve machine/host skill
  - connect to host and enter container
  - normalize baseline command
  - bootstrap controller
        |
        v
sgl-diffusion-engine launch --request REQUEST.yaml --detach
        |
        +--> frozen campaign + watchdog
        +--> Codex executor/master processes
        +--> token and optimization ledgers
        +--> PROGRESS.json
        |
        v
sgl-diffusion-engine progress --campaign ... --watch
```

This keeps SSH/container policy in the host skill, scientific state in the
controller, and user intent in the orchestration skill. A chat session may end
without ending the remote campaign.

## User Experience

The installed skill is named `sglang-diffusion-auto-optimize`. A sufficient
request is:

```text
Use sglang-diffusion-auto-optimize.
Machine: ion-b200
Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers
SGLang checkout: /home/sglang-omni/bbuf/repos/sglang
Baseline command:
CUDA_VISIBLE_DEVICES=0 python python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py ...
Target: 2x end-to-end speedup
```

The skill must not ask the user to create `goal.yaml`, run `init`, remember a
campaign path, or maintain a watchdog terminal. It may ask only when a required
value cannot be inferred, the machine is unreachable, or continuing would be
unsafe.

The skill performs these phases:

1. Read the named host skill when available; otherwise use the supplied SSH
   alias and environment facts.
2. Inspect the target checkout, container, GPU state, and benchmark help.
3. Normalize the baseline command and run a non-mutating preflight.
4. Bootstrap the Python controller into an isolated environment on the host.
5. Create and launch one detached campaign.
6. Show the initial progress view and durable campaign path.
7. Monitor periodically while the conversation remains active.
8. On a later invocation, discover and resume the same campaign by ID or by the
   target checkout/model index.
9. Report the terminal result and patch bundle.

## Request and Baseline Command Contract

Add a strict `LaunchRequest` schema distinct from the frozen scientific
`CampaignGoal`. The request contains:

- model ID;
- target SGLang checkout or repository/ref;
- hardware environment and GPU count;
- target speedup and quality-gated policy;
- baseline `argv`, explicit environment overrides, and working directory;
- optional user-facing original command for audit;
- agent command/model and optional token budget;
- run root and detach preference.

The skill parses the user's command with shell-aware tokenization. Leading
environment assignments are moved into an environment mapping. The persisted
controller executes only an argv vector with `shell=False`. Pipes, redirects,
command substitutions, compound commands, and unresolved shell expansions are
rejected unless the user explicitly provides a reviewed wrapper script. Common
secret-looking environment values and argv options are redacted from receipts.

For the native SGLang Diffusion offline benchmark, normalization must:

- recognize the benchmark entrypoint independently of whether it is invoked
  with `python`, an absolute Python path, or `python -m`;
- extract model, prompt/dataset, seed, resolution, frames, FPS, steps, guidance,
  dtype, prompt count, and timing scope;
- require exactly five validation prompts for the Sol contract;
- replace the checkout-relative executable and output/media paths for every
  isolated worktree;
- preserve all other user flags in their original order;
- inject candidate activation arguments and environment overrides;
- reject ambiguous duplicates or flags that change the frozen workload.

`BASELINE-COMMAND.json` stores the normalized template, its SHA-256, the
redacted original form, the parser adapter, and the frozen workload fields.
Every baseline, candidate, integrated, and clean-room command is derived from
this template. This proves that the optimization loop measures the command the
user supplied, rather than a separately reconstructed approximation.

## One-Shot Launch and Remote Ownership

Add:

```bash
sgl-diffusion-engine launch \
  --request campaign-request.yaml \
  --detach
```

`launch` performs request validation, source/workload discovery, campaign
initialization, and watchdog startup. It prints a machine-readable launch
receipt containing the campaign ID, path, watchdog PID, status command, and
progress command.

Detach uses a new session and campaign-owned logs. The campaign manifest freezes
the exact watchdog/controller argv. The existing lease, heartbeat, SQLite WAL,
idempotency, process receipt, and resume behavior remain authoritative. Repeated
`launch` with the same idempotency key returns the existing campaign rather than
starting another baseline.

Machine selection remains outside the Python CLI. The skill invokes `launch`
inside the remote machine/container selected by the matching host skill. This
avoids embedding private SSH endpoints and container conventions in a public
package.

## Token Telemetry

Codex is invoked with JSONL output when the configured command is recognized as
`codex exec`. The process runner keeps the raw stream and normalizes usage into
an append-only `TOKEN-USAGE.jsonl` ledger.

Each normalized record includes:

- schema version and timestamp;
- campaign ID, epoch, invocation ID, PID, and agent role;
- technique when applicable;
- provider/runtime and model;
- input, cached-input, output, reasoning, and total tokens when emitted;
- source event digest and whether the measurement is exact.

The ledger includes executor, executor resume, independent method audit, and
master visual review invocations. Duplicate final events are deduplicated by
invocation ID plus source-event digest. A runtime without exact usage emits
`available: false`; the controller never estimates token counts from bytes or
characters.

An optional request-level `token_budget` controls a budget bar and may transition
the campaign to the existing recoverable `PAUSED_BUDGET` status. Absence of a
budget displays only exact cumulative usage.

## Optimization and Progress Ledger

Build progress as a projection of durable truth, not a second state machine.
The projection reads:

- campaign status, epoch, transitions, executor receipts, and failures;
- technique round budgets;
- verified candidate deliveries;
- integrated deliveries;
- baseline and final performance records; and
- token usage records.

Write the projection atomically to `PROGRESS.json`. Add:

```bash
sgl-diffusion-engine progress --campaign <path>
sgl-diffusion-engine progress --campaign <path> --watch
sgl-diffusion-engine progress --campaign <path> --json
```

The human view contains:

- model, machine, target, phase, epoch, elapsed time, and last update;
- a measured-performance bar;
- a search-round bar;
- token totals and optional budget bar;
- current active work;
- a row for each technique with attempts, state, best isolated measured
  end-to-end speedup, gate result, and integration state;
- current integrated-stack end-to-end speedup; and
- terminal patch/certificate paths.

Performance progress is:

```text
clamp((best_verified_speedup - 1) / (target_speedup - 1), 0, 1)
```

Search progress is consumed technique rounds divided by the sum of routed
technique round budgets. It is labeled as search-budget consumption, not
estimated time remaining.

Technique gains use isolated full-workload measurements already independently
verified by the Master. The integrated stack is reported separately. No
technique's gains are added together. Optional final leave-one-out runs may
provide marginal attribution, but an absent ablation is displayed as
`not_measured`.

## Skill Package

Create:

```text
skills/sglang-diffusion-auto-optimize/
├── SKILL.md
├── agents/openai.yaml
└── references/
    ├── request-template.yaml
    ├── progress-contract.md
    └── remote-ownership.md
```

The `SKILL.md` frontmatter contains only `name` and `description`. Its
description must trigger on requests to autonomously optimize an SGLang
Diffusion model on a named GPU machine from a baseline command, or to monitor or
resume such a campaign.

The skill instructs the agent to read:

- the matching installed machine skill;
- SGLang Diffusion benchmark/performance skills;
- KDA/KernelWiki/NCU skills only when the active route needs them; and
- the controller's checked-in Sol and SGLang contracts.

The skill owns remote setup and monitoring but does not reimplement controller
state transitions in prose.

## Safety and Correctness Invariants

- The supplied baseline command is frozen before optimization and reused via a
  verified template.
- The baseline cannot be silently refreshed on resume.
- Candidate source paths must resolve inside their assigned worktree.
- The controller never uses a shell for persisted benchmark or Agent commands.
- The skill checks GPU/process ownership before starting or cleaning processes.
- Detached process cleanup is scoped to the campaign's recorded process groups.
- Token and command receipts redact secret-looking values.
- `TARGET_REACHED` still requires integrated full-workload revalidation and a
  clean-room patch.
- `UNREACHABLE_CERTIFIED` still requires the existing independent lower-bound
  certificate; a plateau remains `SEARCH_SPACE_EXHAUSTED`.

## Validation

Add unit and integration tests for:

- natural command parsing and rejection of unsafe shell forms;
- native SGLang command discovery and immutable-workload extraction;
- candidate command derivation from the baseline template;
- one-shot launch idempotency and detached watchdog receipts;
- Codex JSONL usage parsing, deduplication, unavailable telemetry, and role/
  technique attribution;
- progress projection for new, running, rejected, integrated, target-reached,
  exhausted, and budget-paused campaigns;
- human and JSON progress output;
- skill folder validation and root metadata/discoverability;
- an end-to-end fake campaign that launches, records tokens, reports isolated
  and integrated speedups, survives a controller restart, and packages a patch.

Run the complete existing package suite, root metadata tests, formatting,
linting, and a fresh-environment install/test before updating the PR.
