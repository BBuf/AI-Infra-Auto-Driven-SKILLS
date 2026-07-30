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

This is a controller, not a promise that every requested speedup exists.
`TARGET_REACHED` means the integrated patch achieved the target on the exact
locked workload and passed clean-room revalidation. A search plateau produces
`SEARCH_SPACE_EXHAUSTED`, not a theoretical claim. The stronger
`UNREACHABLE_CERTIFIED` state requires an independently checkable lower-bound
certificate showing that the target latency is below the scoped achievable
bound.

## Install

Python 3.11 or newer is required:

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

## Recommended: install the conversational skill

Install `skills/sglang-diffusion-auto-optimize` in Codex, Claude Code, or
another compatible skill runtime. Then make one request:

```text
Use sglang-diffusion-auto-optimize.
Machine: <machine skill or SSH alias>
Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers
Baseline command: CUDA_VISIBLE_DEVICES=0 python
  python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py ...
Target: 2x measured end-to-end speedup
```

The skill resolves the matching machine instructions, enters the remote
container, freezes the supplied command, and invokes the one-shot detached
launcher. It owns monitoring and recovery; you do not create the YAML or run
the commands below yourself.

The underlying reproducible command is:

```bash
sgl-diffusion-engine launch \
  --request campaign-request.yaml \
  --detach
```

Inspect a running campaign with:

```bash
sgl-diffusion-engine progress --campaign runs/<campaign-id>
sgl-diffusion-engine progress --campaign runs/<campaign-id> --watch
sgl-diffusion-engine progress --campaign runs/<campaign-id> --json
```

`PROGRESS.json` and `TOKEN-USAGE.jsonl` persist exact emitted token totals,
technique attempts and gate results, each technique's best isolated measured
end-to-end speedup, and the combined integrated-stack speedup. Isolated gains
are never added together.

## Minimal goal

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

## Run and resume

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
--agent-optimization off
--agent-optimization auto
--agent-optimization <profile-id>
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
