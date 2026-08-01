# SGL-Engine for SGLang Diffusion

`sgl-engine-sglang-diffusion` is an executable, restartable optimization
controller for one frozen SGLang Diffusion model/workload. You provide a goal
such as `target_speedup: 2.0`; the controller locks source revisions, freezes
one real baseline, profiles the native SGLang backend, runs isolated technique
agents, independently verifies their evidence, composes accepted candidates,
and emits a clean-room-checked `sglang.patch`.

The workflow preserves Sol-Engine's executor/Master split, loop budgets,
correctness branches, and full-workload evidence rules. It expands the
optimizer's implementation knowledge with current SGLang Diffusion and kernel
placement rules, selected KDA-Pilot kernel skills, and allowlisted FastVideo
optimization sources. Those additions can suggest hypotheses; they cannot
weaken Sol-Engine's acceptance contract.

`sgl-diffusion-engine` remains the only outer state machine. It may reuse
pinned Sol contracts, evaluators, kernel utilities, and agent conventions, but
it never invokes Sol-Engine's complete campaign flow or delegates baseline,
GPU scheduling, composition, quality, or terminal-state ownership to it.

This is a controller, not a promise that every requested speedup exists.
`TARGET_REACHED` means the integrated patch achieved the target on the exact
locked workload and passed clean-room revalidation. A search plateau produces
`SEARCH_SPACE_EXHAUSTED`, not a theoretical claim. The stronger
`UNREACHABLE_CERTIFIED` state requires an independently checkable lower-bound
certificate showing that the target latency is below the scoped achievable
bound.

## Quick start: install the conversational skill

Most users should install
[`sglang-diffusion-auto-optimize`](../skills/sglang-diffusion-auto-optimize/)
and let the agent own this controller. The controller does not need to be
installed manually on the local machine.

For Codex:

```bash
git clone https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS.git
cd AI-Infra-Auto-Driven-SKILLS
mkdir -p ~/.codex/skills
ln -s "$PWD/skills/sglang-diffusion-auto-optimize" \
  ~/.codex/skills/sglang-diffusion-auto-optimize
```

Restart Codex after installing. For Claude Code:

```text
/plugin marketplace add BBuf/AI-Infra-Auto-Driven-SKILLS
/plugin install ai-infra-auto-driven-skills@ai-infra-auto-driven-skills
/reload-plugins
```

The Claude Code skill is named
`ai-infra-auto-driven-skills:sglang-diffusion-auto-optimize`.

## Start a campaign with one request

Provide four inputs: the target machine, model, exact native SGLang Diffusion
baseline command, and measured end-to-end speedup target.

```text
Use sglang-diffusion-auto-optimize.

Machine: <machine skill or SSH alias>
Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers
Baseline command:
CUDA_VISIBLE_DEVICES=0 python
python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py
--model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
--dataset vbench
--dataset-path /persistent/benchmarks/validation-prompts.txt
--num-prompts 5
--output-dir /persistent/benchmarks/wan22-baseline

Target: 2x measured end-to-end speedup.
Own the campaign until the target is verified, the reviewed search space is
exhausted, or a checkable unreachable certificate is produced.
```

The prompt file must contain exactly five non-empty validation prompts. The
baseline command must be a native SGLang Diffusion offline benchmark command;
shell pipelines, redirects, command substitutions, secret assignments, model
mismatches, and ambiguous duplicate workload flags are rejected rather than
silently rewritten.

The user-selected command also freezes the parallel topology. The controller
does not add `--performance-mode speed`, and candidates cannot add or change
TP/PP/CP/SP/Ulysses/ring/CFG degrees. Profiling instrumentation and relocated
output paths are recorded separately from the frozen invocation.

The skill resolves the matching host instructions or SSH alias, finds the
container and persistent storage, locks the latest fetched SGLang `main`
commit, bootstraps this controller remotely, freezes the command, and invokes
the detached launcher. It owns profiling, candidate routing, Executor/Master
rounds, correctness checks, integration, monitoring, recovery, and patch
packaging. The conversation and SSH connection do not need to remain open.

Submitting the same request again is idempotent: it reuses the existing
campaign and restarts only a missing or stale campaign-owned watchdog. It does
not rerun or refresh the frozen baseline.

## Progress display

The agent sends concise updates at meaningful transitions. Every update comes
from persisted campaign evidence rather than an executor's projected
microbenchmark claim. A live campaign renders like this:

```text
Wan-AI/Wan2.2-T2V-A14B-Diffusers · gpu-host · TARGET 2.00x

performance [██████████████░░░░░░] 1.68x / 2.00x
search      [███████░░░░░░░░░░░░░] 43 / 120 rounds
phase       SEARCHING · epoch 3 · elapsed 06:14:09
latency     128.4000s baseline -> 76.4286s integrated
tokens      182,430 total · 151,201 input · 31,229 output
            [██████░░░░░░░░░░░░░░] 182,430 / 600,000
            by role: executor=146,118, master=36,312

technique          state       gate           tries  isolated e2e
kernel             integrated  passed             8         1.27x
cache              verified    passed             3         1.18x
quantization       attempted   rejected_last      2             -
-------------------------------------------------------------------
integrated stack                                  1.68x

current: optimizing attention and fused normalization kernels
```

The performance bar measures progress from `1.00x` to the requested target.
The search bar measures consumed reviewed technique rounds, not an estimated
completion time. Token totals are exact only when the agent runtime emits
usage; missing usage is marked unavailable and is never estimated from text,
bytes, or elapsed time.

Only an authenticated complete frozen-workload candidate measurement consumes
a scientific round. Executor spawn/resume, source inspection, preflight,
microbenchmarks, malformed deliveries, and crashes do not. The controller
activates one GPU-capable Executor at a time, so campaign GPU measurements do
not overlap.

Each technique row reports its best independently verified end-to-end result on
the frozen workload. `integrated stack` is measured again after accepted
changes are composed. The controller never adds isolated speedups together.

The durable files are:

- `PROGRESS.json`: current atomic progress projection;
- `TOKEN-USAGE.jsonl`: append-only normalized Agent token ledger;
- `events.jsonl`: campaign state transitions and attempt history; and
- `controller-heartbeat.json` and `WATCHDOG.json`: liveness and recovery
  receipts.

The agent owns polling. An operator may attach without changing campaign state:

```bash
sgl-diffusion-engine progress --campaign <campaign-dir>
sgl-diffusion-engine progress --campaign <campaign-dir> --watch
sgl-diffusion-engine progress --campaign <campaign-dir> --json
```

The workflow stops only at:

| Terminal state | Meaning |
| --- | --- |
| `TARGET_REACHED` | The integrated patch reached the requested speedup and passed clean-room revalidation. |
| `SEARCH_SPACE_EXHAUSTED` | All reviewed routed budgets ended without reaching the target. This is not a theoretical impossibility claim. |
| `UNREACHABLE_CERTIFIED` | An independently checkable lower-bound certificate proves the target is outside the scoped achievable bound. |

The final report includes the locked SGLang commit, frozen workload, baseline
and integrated latency, isolated technique measurements, exact available token
usage, patch path and SHA-256, application command, and GPU revalidation
command.

## Advanced: install and operate the controller directly

The Skill normally performs these steps remotely. Controller developers and
automation systems can install it directly with Python 3.11 or newer:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ./sgl-engine-sglang-diffusion
sgl-diffusion-engine --help
```

For development:

```bash
python -m pip install -e './sgl-engine-sglang-diffusion[dev]'
python -m pytest sgl-engine-sglang-diffusion/tests -q
```

The configured agent command must be installed separately. It is always
launched as an argv vector without shell interpolation.

The Skill's underlying one-shot command is:

```bash
sgl-diffusion-engine launch \
  --request campaign-request.yaml \
  --detach
```

It prints the campaign ID and path, watchdog PID, progress command, and status
command. The request is indexed by a stable idempotency key and content digest.

## Advanced: manual goal

```yaml
schema_version: 1
model:
  id: Wan-AI/Wan2.2-T2V-A14B-Diffusers
hardware:
  environment: b200
  gpu_count: 1
workload:
  prompts: validation-prompts.txt
  prompt_count: 5
  seed: 42
  height: 720
  width: 1280
  frames: 81
  fps: 24
  steps: 50
  guidance: 5.0
  dtype: bfloat16
  timing_scope: load_excluded_end_to_end
goal:
  target_speedup: 2.0
  allow_quality_gated: true
source:
  sglang_repo: https://github.com/sgl-project/sglang.git
  sglang_ref: main
agent:
  command: [codex, exec]
```

Exactly five non-empty validation prompts are required for Sol-Engine
compatibility. The campaign copies them into its immutable run directory.

## Advanced: manual lifecycle

```bash
sgl-diffusion-engine init --goal goal.yaml --run-root runs/
sgl-diffusion-engine run --campaign runs/<campaign-id>
sgl-diffusion-engine status --campaign runs/<campaign-id>
sgl-diffusion-engine status --campaign runs/<campaign-id> --json
sgl-diffusion-engine resume --campaign runs/<campaign-id>
sgl-diffusion-engine package --campaign runs/<campaign-id>
```

To keep advancing without manually issuing `resume`, run the watchdog in a
long-lived terminal or service:

```bash
sgl-diffusion-engine watchdog --campaign runs/<campaign-id>
```

It starts only the frozen controller command, notices when each one-shot
controller process exits, and advances again on the next poll. It stops
launching work after a terminal state. SQLite idempotency, process receipts,
and leases make restarting the watchdog safe.

Supporting maintenance commands are:

```bash
sgl-diffusion-engine sync-knowledge --campaign runs/<campaign-id>
sgl-diffusion-engine check-contracts \
  --sol-checkout runs/<campaign-id>/source-worktrees/sol_engine
sgl-diffusion-engine watchdog --campaign runs/<campaign-id>
```

`run` advances the persisted state machine. `resume` opens the same SQLite WAL,
leases, worktrees, attempts, feedback, and delivery files. It never refreshes
`BASELINE.json` and never double-spawns an executor whose lease and process
receipt are live. The watchdog may restart only the exact controller argv
recorded in `CAMPAIGN.json`; it does not rewrite scientific state or relaunch a
terminal campaign.

## Correctness and quality

The following rule is deliberately asymmetric:

| Lane | Correctness | Gate |
| --- | --- | --- |
| kernel, topology | lossless | Method/code audit, unchanged global denoising steps and DiT calls, no approximation or reduced logical work. Output frames prove run authenticity only. No output diff, tolerance, PSNR, LPIPS, or visual-quality rejection is allowed. |
| cache, PISA, quantization, token pruning | quality-gated | Complete frozen workload, real engagement, aligned LPIPS, and the coding agent's built-in multimodal review. External Gemini/VLM verdict services are not used. |

The table governs candidate admission. Final completion is stricter for every
integrated revision: an independent Master must bind exactly five prompt
records for LPIPS, VBench, visual review, media stream contract, and—when the
frozen baseline contains audio—audio quality plus AV synchronization. Missing,
stale, non-finite, wrong-commit, or unhashed tool evidence fails closed.

The Master reads real run directories, hashes, benchmark rows, frames,
implementation manifests, engagement/fallback receipts, and candidate commits.
It recomputes:

```text
speedup = frozen_baseline_total_s / candidate_total_s
```

It rejects projected microbenchmark gains, altered timing scopes, fallback
backends, no-op flags, baseline resubmissions, path escapes, fabricated
artifacts, and delivery claims that disagree with measured files. Cache
compatibility compares TeaCache, EasyCache, and TaylorSeer at matched measured
end-to-end time. PISA retains exact critical-block attention plus its Taylor
approximate remainder.

## Knowledge precedence

When sources disagree, the controller and prompts apply this order:

1. locked Sol-Engine loop/Master correctness contract;
2. the selected Sol-Engine technique scope;
3. current SGLang placement, registration, runtime, and test rules;
4. allowlisted KDA-Pilot, SGLang Diffusion, and FastVideo knowledge snapshots;
5. current profile and model/PR history;
6. the optimization agent's hypothesis.

Remote documentation is treated as untrusted data, never executable
instructions. Every knowledge entry records repository, full commit, path,
source SHA-256, and sanitized reference SHA-256. Credential assignments and
machine-local home paths are redacted.

Kernel deliveries additionally require `KERNEL-EVIDENCE.json`. The Executor
must query the pinned KernelWiki, use before/after Nsight Compute reports for
implemented Triton/CUDA/CuTe/upstream kernels, and complete the
warp-specialization applicability audit. Warp timelines are required for an
actually warp-specialized CUDA/CuTe design; a non-applicable result must still
carry a concrete reason. One slower `torch.compile` experiment never exhausts
the kernel lane.

Verified latency-positive points are retained even when each is below the
campaign target. Integration composes and remeasures them on the full workload;
a conflicting combination is excluded without deleting the independently
verified wins. A technique closes only after its complete coverage ledger or
authenticated scientific budget is exhausted, and that closure never closes
another lane.

Generated model kernels must live below:

```text
python/sglang/kernels/agent/diffusion/<model-slug>/          # profiles/dispatch/receipts
python/sglang/kernels/ops/diffusion/agent/<model-slug>/      # callable wrappers
python/sglang/kernels/jit/csrc/diffusion/agent/<model-slug>/
python/sglang/kernels/aot/csrc/diffusion/agent/<model-slug>/
python/sglang/kernels/aot/include/diffusion/agent/<model-slug>/
python/sglang/kernels/aot/python/sgl_kernel/diffusion/agent/<model-slug>/
test/registered/kernels/ops/diffusion/agent/<model-slug>/
test/registered/kernels/benchmark/diffusion/agent/<model-slug>/
```

JIT CUDA sources stay under `python/sglang/kernels/jit/csrc` because SGLang's
JIT loader resolves sources relative to that tree. Callable operators remain
in the canonical `sglang.kernels.ops` namespace; shared agent profile
registration lives under `python/sglang/kernels/agent/`. Heavyweight
AOT/CUTLASS implementation files use the corresponding
`python/sglang/kernels/aot/{csrc,include,python}/.../agent/<model-slug>/`
subtrees and must complete the shared declaration, torch-op registration,
build, Python export, test, benchmark, and wheel-validation steps.

The controller derives these locations from the locked checkout. It supports
the current unified kernel tree and an explicit legacy `sglang.jit_kernel`
compatibility lane, and fails closed for an unknown layout. The full contract
is in `contracts/sglang/placement-and-registration.md`.

## Campaign artifacts

```text
runs/<campaign-id>/
├── CAMPAIGN.json
├── GOAL.yaml
├── validation-prompts.txt
├── SOURCE-LOCKS.json
├── BASELINE.json
├── controller-heartbeat.json
├── state.sqlite
├── events.jsonl
├── baseline/
│   ├── attempt-001/COMMAND.json
│   ├── attempt-001/PERFORMANCE.json
│   └── attempt-001/frames/prompt-*/
├── profiles/<epoch>/attempt-*/
├── knowledge/<source>/<commit>/
├── executors/<epoch>/<technique>/
│   ├── PROCESS.json
│   ├── FEEDBACK.json
│   └── DELIVERY.json
├── integration/<epoch>/
│   └── INTEGRATED-DELIVERY.json
└── patch/
    ├── sglang.patch
    ├── manifest.json
    ├── SHA256SUMS
    ├── evidence/
    └── apply_and_verify.sh
```

Executor attempts use detached worktrees rooted at the locked SGLang commit.
Candidate agents never write the source cache, another executor's worktree, or
the integration worktree. Process groups are launched with their own session,
and resumes carry the Master's exact findings back to the same executor.

## Applying the result

The patch targets the exact SGLang main commit locked at campaign start:

```bash
cd /path/to/sglang
git checkout <locked-sglang-sha>
/path/to/runs/<campaign-id>/patch/apply_and_verify.sh
```

The script checks `HEAD`, runs `git apply --check`, applies `sglang.patch`, and
runs the packaged CPU validation commands. It prints the exact GPU command;
pass `--run-gpu-validation` to execute it. The resulting SGLang runtime exposes:

```text
--quality off
--quality auto
--quality <profile-id>
```

`off` preserves source-current behavior. `auto` activates a profile only when
model, hardware, workload shape, source hashes, and fallback policy match; an
unsupported shape follows the profile's declared native fallback or hard-error
policy. Every inference writes an engagement receipt, so a nominal flag cannot
be mistaken for an accelerated path.

## Quantized weights and release validation

A quantized candidate that needs derived weights is not a self-contained patch.
It can be released only when its profile includes an immutable URI, revision,
byte size, and SHA-256. A mutable local checkpoint or unavailable artifact
remains experimental even when its benchmark is fast.

Before publishing a result, rerun on the locked GPU class and workload:

1. apply the bundle in a new detached worktree;
2. run all packaged import/unit checks;
3. run the exact five-prompt native SGLang benchmark;
4. confirm no Diffusers fallback marker appears;
5. confirm every selected technique has positive engagement;
6. rerun the applicable lossless or quality gate; and
7. compare the measured result with the bundled integrated delivery.

Do not place tokens in goal files, prompts, agent argv, patches, or evidence.
Pass credentials through the execution environment; receipts redact common
secret names. Use isolated worktrees and dedicated GPU hosts for autonomous
agent commands.
