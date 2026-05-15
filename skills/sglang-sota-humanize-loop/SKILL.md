---
name: sglang-sota-humanize-loop
description: "Run an autonomous Humanize-governed SGLang SOTA performance loop for one LLM model: first perform the fixed fair SGLang/vLLM/TensorRT-LLM deployment search and benchmark, then profile every required leading framework, compare kernel/overlap/fusion evidence, patch SGLang code, optionally invoke KernelPilot for slow kernels, and iterate until SGLang matches or beats the best observed framework under the same workload and SLA."
---

# SGLang SOTA Humanize Loop

## Overview

Use this skill when the user names a model and wants the SGLang serving path to
autonomously keep improving until it matches or beats the best reproducible
vLLM or TensorRT-LLM result in the same target environment.

This is a Humanize wrapper around the existing SOTA workflow. It separates the
work into two phases:

1. A fixed baseline phase that must be completed once before any code patching.
2. A Humanize RLCR phase that iterates on SGLang patches using benchmark and
   profiler evidence until the stop criteria are met.

Run only one Humanize RLCR loop at a time in one Codex session. KernelPilot is a
serialized handoff, not a nested live loop inside an active SGLang RLCR round.
Do not rely on resuming the exact same `.humanize/rlcr/<timestamp>` session
after a KernelPilot loop. Preserve model-loop continuity with checkpoint files
and start a fresh SGLang continuation RLCR after KernelPilot completes.

## Runtime Roots

The installer hydrates these local paths:

```text
Humanize runtime: /Users/bbuf/.codex/skills/humanize
KernelPilot root: /Users/bbuf/工作目录/Common/kernel-pilot
KernelPilot loop skill: /Users/bbuf/.codex/skills/humanize-kernel-agent-loop/SKILL.md
```

If the Humanize runtime is missing, locate a skill directory containing
`scripts/setup-rlcr-loop.sh`. If KernelPilot is missing and a kernel-level
optimization is needed, locate a repository containing `knowledge/index.json`
and `references/kernel-source-catalog.md`.

## Companion Skills

Read these before a real run:

- `../sglang-sota-performance/SKILL.md`
- `../llm-serving-auto-benchmark/SKILL.md`
- `../llm-torch-profiler-analysis/SKILL.md`
- the matching host or operator skill for SSH, container, GPU, and artifact
  conventions

Read the KernelPilot loop skill only after profiler evidence identifies a
specific slow kernel family or candidate kernel path that should be optimized
outside the framework checkout.

## Contract

Given a model-level SGLang SOTA request, do not ask the user to run separate
benchmark, profiler, gen-plan, refine-plan, or Humanize setup commands. Do the
setup yourself.

Ask the user only if the model, target GPU environment, or precision/quantization
policy is missing and cannot be inferred from local configs or the active host
skill.

Keep the fixed benchmark phase outside the RLCR patch loop. Humanize starts only
after the first fair cross-framework search, winner table, required profiles,
and initial root-cause report exist.

Treat the model optimization campaign as the durable unit, not one Humanize
session directory. The campaign is recoverable from the run artifact root,
checkpoint files, benchmark/profile artifacts, ledgers, and continuation plan.

## Phase 0: Inputs And Run Directory

Collect or infer:

- model id or checkpoint path, tokenizer, precision, quantization, trust policy,
  and max context length
- target SGLang checkout to patch
- GPU type/count, visible GPU ids, container or remote shell, CUDA/NCCL versions,
  and whether multi-node is allowed
- framework set, defaulting to SGLang, vLLM, and TensorRT-LLM when available
- artifact root

Create one run directory:

```text
runs/YYYYMMDD_<model_slug>_sota_humanize/
  manifest.md
  help/
  benchmark/
  profiles/
  analysis/
  patches/
  humanize/
    model-loop-checkpoint.md
    sglang-continuation-plan.md
  final_report.md
```

Never save Hugging Face tokens or other secrets in artifacts.

## Phase 1: Fixed Fair Benchmark Gate

This phase is mandatory and happens exactly once before Humanize starts.

Use `llm-serving-auto-benchmark` as the source of truth for candidate generation,
result schema, workload, and comparison.

Hard requirements:

- Search SGLang, vLLM, and TensorRT-LLM best deployment commands when each
  framework is supported in the target environment.
- Do not compare tuned SGLang against competitor defaults. Every framework gets
  its own bounded search.
- Use the same model weights, tokenizer, precision, quantization, GPU type/count,
  GPU ids, endpoint path, sampling settings, and SLA.
- Use the default two dataset scenarios from `llm-serving-auto-benchmark` unless
  the user explicitly provides a production workload:
  - dataset kind `random`, `num_prompts: 80`
  - `chat`: random input `1000`, output `1000`
  - `summarization`: random input `8000`, output `1000`
  - treat the two input/output pairs as aligned scenarios, not a cartesian
    product
- Do not replace those scenarios with an easier smoke dataset for the real SOTA
  decision. Smoke runs are allowed only when labeled as flow checks.
- For TensorRT-LLM, keep `trtllm-serve serve --backend pytorch`; reject
  non-PyTorch TensorRT-LLM server backends for this skill.
- Keep failed, skipped, and SLA-failing candidates in the benchmark artifact.

Write:

- `benchmark/candidates.jsonl`
- `benchmark/summary.md`
- `benchmark/winning-commands.md`
- framework help outputs under `help/`
- the exact launch and benchmark commands for every winner

## Phase 2: Gap Decision

Compute SGLang's gap against the best SLA-passing framework for each scenario.

Use `1%` as the default stable noise threshold. If the initial result is within
`+/-1%`, rerun the winning commands enough times to decide whether the gap is
stable before starting Humanize.

Start the Humanize patch loop only when SGLang is slower than the best framework
by more than `1%`, fails SLA while another framework passes, or uses materially
more memory for the same workload.

If SGLang is already best or tied within the stable threshold, write the final
report and do not start RLCR.

## Phase 3: Required Profiling Before RLCR

Before patching, profile the best SGLang command and the leading competitor
command with `llm-torch-profiler-analysis`.

Rules:

- Always profile SGLang when it is behind.
- Always profile at least the current best framework.
- If both vLLM and TensorRT-LLM are more than `1%` ahead of SGLang in a stable
  result, profile both.
- Use the slow benchmark scenario lengths, not the profiler defaults:
  - prefill profile: slow input length -> `1` output token
  - decode profile: `1` input token -> slow output length
- For mixed or production datasets, use the slowest representative p50 or p95
  bucket already recorded by the benchmark artifact.
- Capture or analyze separate prefill and decode evidence when the framework
  supports it.

For every profiled framework, save the same three tables:

- kernel table
- overlap-opportunity table
- fuse-pattern table

Then write `analysis/root-cause.md` with the initial cross-framework comparison:
which stage is slower, which table rows explain it, and which SGLang source
paths or kernel families are plausible patch targets.

Do not patch SGLang until this report exists.

## Phase 4: Build The Humanize Plan

Create a Humanize plan inside the SGLang checkout that will be patched:

```text
.humanize/sglang-sota-agent/refined-plan.md
```

Use [references/refined-plan-template.md](references/refined-plan-template.md)
as the skeleton and fill it with the actual model, workload, benchmark winners,
profile paths, root-cause rows, and target artifact root.

The plan must require:

- preserving the fixed benchmark workload and SLA throughout the loop
- patching SGLang code, not just benchmark parameters
- re-running real model benchmark/profile after each accepted patch
- continuing through multiple minimal patches when one patch only closes part
  of the gap
- using KernelPilot when the profiler root cause is an optimizable CUDA, Triton,
  CuTe, CUTLASS, TileLang, or torch.compile kernel path
- never running a KernelPilot `setup-rlcr-loop.sh` while the SGLang RLCR loop is
  still active in the same Codex session
- writing a model-loop checkpoint and SGLang continuation plan before leaving a
  SGLang RLCR for a KernelPilot handoff
- limiting the KernelPilot optimization loop to at most 10 Humanize rounds, then
  returning the best correct candidate or a no-kernel-change result to the model
  loop
- recording every attempt, failed idea, partial win, rejected source idea, and
  final selected patch in artifacts

## Phase 5: Start RLCR

From the SGLang checkout, run:

```bash
"/Users/bbuf/.codex/skills/humanize/scripts/setup-rlcr-loop.sh" \
  .humanize/sglang-sota-agent/refined-plan.md --yolo
```

If setup exits non-zero, stop and report the error. Do not bypass the gate.

After setup succeeds:

1. Read `.humanize/rlcr/<timestamp>/round-0-prompt.md`.
2. Execute the current round.
3. Commit SGLang changes.
4. Write the required Humanize round summary.
5. Stop normally so the native Humanize Stop hook can review.

If the hook blocks exit, follow the generated next-round prompt exactly.

## KernelPilot Handoff

Use KernelPilot only when the cross-framework profiler evidence points at a
specific kernel or small kernel family, not for generic scheduling or benchmark
knob tuning.

### Eligibility Gate

KernelPilot is allowed only when all of these are true:

- SGLang is still more than `1%` behind the best framework for the fixed
  benchmark scenario after the required repeat/profiler checks.
- The slow stage has a concrete SGLang kernel or tightly scoped kernel family
  in the kernel table with at least `1%` cumulative GPU-time share. Do not use
  KernelPilot for a lone kernel below `1%` share unless a single shared
  implementation affects an aggregated family above `1%`.
- The profiler comparison shows that kernel or family is plausibly part of the
  SGLang gap: the winning framework has a faster equivalent path, SGLang runs
  extra kernel work for the same stage, or SGLang's own kernel evidence shows a
  local inefficiency that cannot be explained by scheduling, overlap, launch
  overhead, data movement, or a missing framework fast path.
- The proposed KernelPilot target has a clear standalone correctness reference,
  representative shapes/dtypes/layouts from the model run, and a path to port
  the selected candidate back into the active SGLang serving code.

If any condition fails, keep the work in the SGLang RLCR loop and patch the
appropriate non-kernel issue first. Do not send sub-`1%` profiler rows to
KernelPilot just because they look locally optimizable.

Examples:

- vLLM or TensorRT-LLM has a faster fused RMSNorm, activation, quantization,
  attention, cache update, MoE routing, sampling, or GEMM path.
- SGLang spends a dominant share in one custom CUDA/Triton/CuTe kernel.
- The gap survives SGLang scheduling and overlap patches and the remaining hot
  row is kernel-local.

### No Simultaneous RLCR Loops

Do not start KernelPilot's `setup-rlcr-loop.sh` while an SGLang SOTA RLCR loop is
active in the same Codex session. A live nested loop can confuse Humanize's
project-root and Stop-hook state because both loops maintain `.humanize/rlcr`
state, round summaries, goal trackers, and exit gates.

Use one of these serialized patterns:

- If Phase 3 already proves the root cause is kernel-local, run the KernelPilot
  loop first in its standalone optimization repo, wait for it to complete, port
  the selected candidate into SGLang, then start the SGLang SOTA RLCR loop for
  integration and real-model revalidation.
- If a kernel-local bottleneck is discovered after the SGLang RLCR loop has
  started, do not launch KernelPilot from inside that round. Write
  `analysis/kernelpilot-handoff.md` with the target kernel, shapes, dtype,
  profiler rows, source paths, and acceptance criteria. Also write
  `humanize/model-loop-checkpoint.md` and
  `humanize/sglang-continuation-plan.md`; finish the current SGLang round as a
  handoff checkpoint, not as final success; then start the standalone
  KernelPilot loop only after the SGLang RLCR loop is no longer active.
- After KernelPilot completes, return to SGLang with the selected candidate,
  provenance, correctness results, benchmark result, and profile digest, then
  start a fresh non-nested SGLang continuation RLCR from
  `humanize/sglang-continuation-plan.md`.

If the Humanize Stop hook rejects the handoff round because the original SOTA ACs
are not met, update the active plan/round summary to make the handoff checkpoint
explicitly scoped: the current loop is completing the safe suspension artifact,
not claiming final SOTA. Do not bypass the hook or edit Humanize state files.

### Continuation Contract

Before leaving SGLang RLCR for KernelPilot, `humanize/model-loop-checkpoint.md`
must record:

- original model, tokenizer, precision, quantization, hardware, workload, SLA,
  artifact root, and benchmark winner commands
- current SGLang branch, commit, patches already applied, tests already run,
  and current best SGLang benchmark row
- remaining gap, profiler rows, cumulative GPU-time share for the target kernel
  or kernel family, source paths, and why the next step is KernelPilot rather
  than another SGLang-local patch
- exact KernelPilot standalone repo path to use or create

`humanize/sglang-continuation-plan.md` must be a ready-to-use Humanize plan
draft for the post-KernelPilot integration loop. It must say to import the
KernelPilot selected candidate, make it active in the SGLang model-serving path,
re-run correctness, re-run the same real-model benchmark/profile, update
ledgers, and continue until the original SOTA stop criteria are met.

After KernelPilot finishes, do not try to resurrect the old `.humanize/rlcr`
session. Copy or adapt `humanize/sglang-continuation-plan.md` into
`.humanize/sglang-sota-agent/refined-plan.md`, fill in the KernelPilot result
paths, and start a new SGLang RLCR with `setup-rlcr-loop.sh`.

When using KernelPilot:

- Create or enter the clean standalone optimization repo required by
  `humanize-kernel-agent-loop`.
- Cap the KernelPilot Humanize loop at 10 rounds. When launching the standalone
  KernelPilot Humanize loop, pass `--max 10` to the Humanize setup command if
  that setup path supports normal Humanize options; otherwise stop after the
  tenth KernelPilot round and return the best correct candidate found so far.
- Keep the SGLang checkout protected from experimental kernel mutations until a
  correct candidate wins in the standalone harness.
- Record source provenance, license/notice, copied files, and deltas before
  adapting competitor or upstream kernel code.
- Port only the selected candidate or minimal integration change back to SGLang,
  and wire it into the active model-serving path that produced the original
  profiler row.
- Re-run the same model-level benchmark and profiler after porting. Do not
  continue the SGLang model loop from a standalone microbench win alone.

KernelPilot owns its own standalone repo and Humanize state, but it still must
not be live at the same time as the parent SGLang RLCR loop in the same Codex
session.

## Loop Ledgers

Keep these files under the run artifact root or the SGLang checkout, depending
on the host convention:

```text
humanize/attempt-ledger.md
humanize/optimization-ledger.md
humanize/source-idea-ledger.md
humanize/lineage.jsonl
humanize/profile-digests/
```

Every patch attempt gets an attempt row. Only correct patches with measured
improvement get optimization rows. Source ideas must include both profiler rows
and code provenance so later rounds can avoid re-reading the same source.

After two consecutive rounds with less than `1%` geomean improvement over the
prior best SGLang result, expand research before editing again. Prefer code and
PR evidence from SGLang, vLLM, TensorRT-LLM, KernelPilot knowledge, and relevant
kernel source guides before prose-only articles.

## Stop Conditions

Stop only when one of these is true:

- SGLang beats the best SLA-passing vLLM/TensorRT-LLM result on the fixed
  workload.
- SGLang is tied within the stable `1%` threshold after repeat runs.
- The remaining gap is proven external to SGLang, such as unavailable hardware
  support, missing framework dependency, unsupported TensorRT-LLM PyTorch
  backend, or model weights that cannot be loaded fairly.
- Profile evidence shows the remaining hot path is already near the relevant
  hardware or algorithmic limit and no low-risk SGLang patch remains.

The final report must include the pre-loop benchmark table, post-patch benchmark
table, all winner commands, profile paths, SGLang changed files, tests, and
whether SGLang reached target-environment SOTA.
