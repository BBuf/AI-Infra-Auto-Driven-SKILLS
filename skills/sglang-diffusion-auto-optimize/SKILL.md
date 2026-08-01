---
name: sglang-diffusion-auto-optimize
description: Autonomously optimize an SGLang Diffusion model on a named GPU machine from a user-supplied baseline command until a requested measured end-to-end speedup is reached or the reviewed search terminates. Use when Codex should take remote ownership of baseline freezing, profiling, SGLang/kernel changes, Sol-Engine-compatible correctness checks, persistent recovery, token accounting, progress reporting, patch packaging, or resume/monitoring for an existing SGLang Diffusion optimization campaign.
---

# SGLang Diffusion Auto-Optimize

## Contract

Turn one natural-language request into one durable remote campaign. Require only:

- target machine name or SSH alias;
- model ID or checkpoint;
- exact SGLang Diffusion baseline command; and
- target measured end-to-end speedup.

Infer the SGLang checkout, container, GPU IDs, and artifact root from the
matching host skill and the command when possible. Do not ask the user to write
`goal.yaml`, run controller subcommands, keep a terminal alive, or manually
resume executors.

Keep Sol-Engine correctness and quality gates binding. Additional SGLang,
FastVideo, KDA-Pilot, KernelWiki, profiler, or model-history knowledge may
generate hypotheses but may not relax those gates.

The new `sgl-diffusion-engine` is the sole outer controller. It may reuse
pinned Sol-Engine components, including Executor/Master conventions, quality
evaluators, and kernel utilities, but must not launch Sol-Engine's full
campaign flow. The controller owns baseline/profile state, serial GPU
measurement scheduling, candidate composition, final quality, and termination.

## Read Before Acting

Read [references/remote-ownership.md](references/remote-ownership.md) and
[references/progress-contract.md](references/progress-contract.md).

Read the installed skill whose name exactly matches the requested machine when
one exists. Follow its SSH, container, repository, GPU-allocation, and cleanup
rules. If no host skill exists, use only a real SSH alias/configuration supplied
by the user or discoverable in the active environment; never invent an
endpoint.

Read these installed skills when available:

- `sglang-diffusion-benchmark-profile`;
- `sglang-diffusion-performance`;
- `model-pr-history-knowledge`.

For every routed kernel lane, read and use the pinned `KernelWiki`,
`ncu-report-skill`, and `warp-specialization-report-skill` resources. Use
`kernel-knowledge`, `ncu-report`, `add-jit-kernel`, or `add-sgl-kernel` when the
candidate needs their upstream evidence or implementation workflow. The
controller snapshots these resources and enforces `KERNEL-EVIDENCE.json`; do
not replace immutable inputs with memory. Warp timeline instrumentation is
conditional on an actually warp-specialized CUDA/CuTe candidate, but the
applicability audit is always required.

## Phase 1: Resolve The Remote Environment

Connect using the matching host skill. Determine:

- the container or isolated runtime;
- the SGLang checkout;
- usable GPU IDs without killing or preempting another user's processes;
- Python 3.11+ and the active CUDA/PyTorch environment;
- the exact SGLang origin URL and latest requested main commit; and
- a campaign root on persistent storage.

Run read-only GPU and process checks before launch. Fetch the target SGLang main
ref without resetting, cleaning, or overwriting the user's worktree. Resolve the
fetched ref to a full commit and place the origin URL plus that commit in the
launch request. A dirty user checkout is not an optimization worktree; the
controller creates detached worktrees from its own bare cache.

Do not run the complete baseline as a separate preflight. Check paths, prompt
count, command syntax, benchmark `--help`, and imports only. The controller owns
the single authoritative baseline run.

## Phase 2: Bootstrap The Controller

Prefer the controller shipped beside this skill:

```text
<plugin-or-repository-root>/sgl-engine-sglang-diffusion
```

Install it into an isolated virtual environment on the target machine:

```bash
python3 -m venv <persistent-tool-root>/.venv
<persistent-tool-root>/.venv/bin/python -m pip install \
  <plugin-or-repository-root>/sgl-engine-sglang-diffusion
```

If only this skill directory was installed and the controller source is absent,
install the package from the same immutable repository revision as the skill:

```text
git+https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS.git@<revision>#subdirectory=sgl-engine-sglang-diffusion
```

Do not silently install from an unrelated branch or copy.

## Phase 3: Create The Request

Start from
[references/request-template.yaml](references/request-template.yaml). Write it
under a campaign-owned request directory, not inside the SGLang checkout.

Translate the baseline command to the request without changing its workload:

- preserve the user's flags and ordering;
- preserve leading non-secret environment assignments;
- use an absolute working directory;
- keep exactly five non-empty validation prompts for the Sol contract;
- pass credentials through the inherited process environment, never YAML;
- set the source SGLang repository to the verified origin URL;
- set the source SGLang ref to the fetched full main commit;
- default the agent command to `codex exec`;
- use the user's requested Agent model when supplied;
- add a token budget only when the user gave one.

The supplied baseline also selects the fixed parallel topology. Do not add
`--performance-mode speed`. Do not add or change TP/PP/CP/SP/Ulysses/ring/CFG
degrees after the baseline is frozen; the controller rejects such candidate
activation drift.

The launcher rejects shell pipelines, redirects, substitutions, ambiguous
duplicate workload flags, and baseline/model mismatch. Do not work around those
failures by weakening validation. Convert a safe command into explicit
`baseline.argv` and `baseline.env`, or obtain a reviewed wrapper when shell
behavior is genuinely required.

## Phase 4: Launch And Hand Off

Run inside the selected remote container/environment:

```bash
sgl-diffusion-engine launch \
  --request <absolute-request.yaml> \
  --detach
```

Capture the returned campaign ID, campaign path, watchdog PID,
`progress_command`, and `status_command`. Verify `WATCHDOG.json`,
`controller-heartbeat.json` after the first controller tick, and
`PROGRESS.json`. The detached watchdog owns recovery after the interactive
agent or SSH session exits.

Do not start a second campaign for the same request. The launch request is
idempotent; reuse the returned campaign when `reused: true`.

## Phase 5: Monitor Without Micromanaging

Use the returned progress command. For machine-readable polling:

```bash
sgl-diffusion-engine progress --campaign <campaign> --json
```

For an attached terminal:

```bash
sgl-diffusion-engine progress --campaign <campaign> --watch
```

While the conversation remains active, poll periodically and send concise
updates at meaningful transitions or at least once per long-running interval.
Do not reinterpret an executor's microbenchmark claim as progress. Report only:

- frozen-baseline and integrated full-workload end-to-end speedup;
- best independently verified isolated end-to-end speedup by technique;
- correctness/quality gate result and failure reason;
- integrated-stack speedup separately from isolated gains;
- exact emitted token totals and unavailable runtimes;
- current phase, epoch, round-budget consumption, and active technique.

The detached controller launches and resumes Executor/Master agents itself.
Only one GPU-capable Executor is active at a time. A scientific round means one
authenticated complete frozen-workload candidate measurement—not a process
launch, retry, microbenchmark, malformed delivery, or infrastructure failure.
A rejected hypothesis feeds the same lane; it does not close the lane or the
campaign. Retain every independently verified positive candidate for combined
remeasurement.

Never add technique speedups together. `marginal_attribution: not_measured`
means no leave-one-out measurement exists.

If the watchdog is stale, inspect its receipt, heartbeat, logs, recorded PID,
and process ownership. Restart only the exact campaign-owned watchdog command.
Use `resume` only for an existing recoverable campaign. Do not refresh
`BASELINE.json`.

## Phase 6: Finish

Continue until the campaign reaches one of:

- `TARGET_REACHED`;
- `UNREACHABLE_CERTIFIED`; or
- `SEARCH_SPACE_EXHAUSTED`.

Do not describe search exhaustion as theoretical impossibility.
`UNREACHABLE_CERTIFIED` is valid only with the controller's independently
checkable lower-bound certificate.

For `TARGET_REACHED`, report:

- locked SGLang commit and exact workload;
- baseline, final latency, and measured speedup;
- each technique's isolated measurement and gate result;
- integrated stack result;
- total exact tokens by role/technique when available;
- patch path, SHA-256, application command, and GPU revalidation command.

Before accepting `TARGET_REACHED`, verify that the same integrated commit has
exactly five prompt records for LPIPS, VBench, visual review, media contract,
and, when baseline media has audio, audio quality and AV synchronization.
Missing tools or evidence are blocking failures, never implicit passes.

Leave the remote artifacts and detached worktrees available for audit. Clean up
only processes and temporary paths recorded as owned by this campaign.
