---
name: sglang-sota-humanize-loop
description: "Run an autonomous Humanize-governed SGLang SOTA performance loop for one LLM model: first perform the fixed fair SGLang/vLLM/TensorRT-LLM deployment search and benchmark, then profile every required leading framework, compare kernel/overlap/fusion evidence, patch SGLang code, optionally use KernelPilot knowledge and ncu-report evidence for kernel-local fixes, and iterate until SGLang matches or beats the best observed framework under the same workload and SLA."
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

Run exactly one Humanize RLCR loop for the model campaign. Do not start a
KernelPilot Humanize loop, `humanize-kernel-agent-loop`, or any second
`setup-rlcr-loop.sh` session from this skill. KernelPilot is used only as a
knowledge and source-evidence repository, while `ncu-report` supplies measured
kernel diagnostics. All SGLang patches, including kernel patches, stay inside
the same model-level RLCR loop and are accepted only after real-model
benchmark/profile revalidation.

## Runtime Roots

The installer hydrates these local paths:

```text
Humanize runtime: /Users/bbuf/.codex/skills/humanize
KernelPilot root: /Users/bbuf/工作目录/Common/kernel-pilot
ncu-report skill: /Users/bbuf/.codex/skills/ncu-report/SKILL.md
Model PR history knowledge: /Users/bbuf/工作目录/Common/AI-Infra-Auto-Driven-SKILLS/model-pr-optimization-history
```

If the Humanize runtime is missing, locate a skill directory containing
`scripts/setup-rlcr-loop.sh`. If KernelPilot is missing, continue with
SGLang/vLLM/TensorRT-LLM source and PR evidence; do not block the model loop
only because the optional knowledge repository is unavailable.

## Companion Skills

Read these before a real run:

- `../sglang-sota-performance/SKILL.md`
- `../llm-serving-auto-benchmark/SKILL.md`
- `../llm-torch-profiler-analysis/SKILL.md`
- `../../model-pr-optimization-history/SKILL.md`
- `/Users/bbuf/.codex/skills/ncu-report/SKILL.md` when a kernel edit needs
  Nsight Compute evidence
- the matching host or operator skill for SSH, container, GPU, and artifact
  conventions

Read KernelPilot knowledge files only after profiler evidence identifies a
specific slow kernel family or candidate kernel path. Use them for source ideas,
PR references, implementation patterns, and provenance checks, not for starting a
separate optimization loop.

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
checkpoint files, benchmark/profile artifacts, NCU digests, and ledgers.

## Phase 0: Inputs And Run Directory

Collect or infer:

- model id or checkpoint path, tokenizer, precision, quantization, trust policy,
  and max context length
- target SGLang checkout to patch
- GPU type/count, visible GPU ids, container or remote shell, CUDA/NCCL versions,
  and whether multi-node is allowed
- framework set, defaulting to SGLang, vLLM, and TensorRT-LLM when available
- model-family history slug inferred from the model id, checkpoint, or hot
  SGLang/vLLM source path when possible
- artifact root

Create one run directory:

```text
runs/YYYYMMDD_<model_slug>_sota_humanize/
  manifest.md
  help/
  benchmark/
  profiles/
  analysis/
  history/
    model-pr-history-notes.md
  kernel/
    kernelpilot-knowledge-notes.md
    ncu-digests/
  patches/
  humanize/
    model-loop-checkpoint.md
  final_report.md
```

Never save Hugging Face tokens or other secrets in artifacts.

## Phase 0.5: Model PR History Knowledge Gate

Before the fixed benchmark and before any patch planning, query and read
`model-pr-optimization-history` for the target model family.

Rules:

- If the slug is unclear, run `scripts/query.py "<model id or family>"` from
  the knowledge root and choose the closest model-family history.
- Read the SGLang history for that family whenever it exists.
- Read the vLLM history too when vLLM is in the comparison set, later becomes
  the leading competitor, or its source/trace suggests a missing SGLang fast
  path.
- Write `history/model-pr-history-notes.md` with the paths read, PR numbers,
  source files, symbols, validation risks, and the concrete decision each item
  influences.
- Treat these notes like KernelPilot knowledge: source and PR memory that helps
  choose a better patch, not measured proof by itself.

If the knowledge root is unavailable, record the blocker in the same notes file
and continue with benchmark/profile evidence.

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
- preserving and consulting `history/model-pr-history-notes.md` before choosing
  model-specific SGLang source paths
- patching SGLang code, not just benchmark parameters
- re-running real model benchmark/profile after each accepted patch
- continuing through multiple minimal patches when one patch only closes part
  of the gap
- using KernelPilot knowledge and `ncu-report` only as assists when the profiler
  root cause is an optimizable CUDA, Triton, CuTe, CUTLASS, TileLang, or
  torch.compile kernel path
- never starting a KernelPilot RLCR, `humanize-kernel-agent-loop`, or second
  `.humanize/rlcr` tree for kernel work
- keeping kernel edits, microbench harnesses, NCU digests, integration, and
  real-model revalidation inside the single SGLang model loop
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

## Kernel Evidence Assist

Use KernelPilot knowledge and `ncu-report` only when the cross-framework
profiler evidence points at a specific kernel or small kernel family. Keep
generic scheduling, overlap, batching, memory residency, and benchmark-command
issues in the normal SGLang patch path.

### Eligibility Gate

Kernel-level assistance is allowed only when all of these are true:

- SGLang is still more than `1%` behind the best framework for the fixed
  benchmark scenario after the required repeat/profiler checks.
- The slow stage has a concrete SGLang kernel or tightly scoped kernel family
  in the kernel table with at least `1%` cumulative GPU-time share. Do not spend
  kernel-specialist effort on a lone kernel below `1%` share unless a shared
  implementation affects an aggregated family above `1%`.
- The profiler comparison shows that kernel or family is plausibly part of the
  SGLang gap: the winning framework has a faster equivalent path, SGLang runs
  extra kernel work for the same stage, or SGLang's own kernel evidence shows a
  local inefficiency that cannot be explained by scheduling, overlap, launch
  overhead, data movement, or a missing framework fast path.
- The proposed kernel target has a clear correctness reference,
  representative shapes/dtypes/layouts from the model run, and a path to wire
  the candidate into the active SGLang serving code.

If any condition fails, keep the work in the SGLang RLCR loop and patch the
appropriate non-kernel issue first. Do not send sub-`1%` profiler rows to
kernel-specialist research just because they look locally optimizable.

Examples:

- vLLM or TensorRT-LLM has a faster fused RMSNorm, activation, quantization,
  attention, cache update, MoE routing, sampling, or GEMM path.
- SGLang spends a dominant share in one custom CUDA/Triton/CuTe kernel.
- The gap survives SGLang scheduling and overlap patches and the remaining hot
  row is kernel-local.

### Single-Loop Kernel Workflow

Do not start KernelPilot's `setup-rlcr-loop.sh`, `humanize-kernel-agent-loop`, or
any standalone `.humanize/rlcr` session. A kernel candidate is just another
SGLang patch candidate in the active model RLCR round.

For each eligible kernel target:

1. Record `kernel/kernelpilot-knowledge-notes.md` with the target kernel,
   model-derived shapes, dtype/layout, profiler rows, source paths, and the
   reason kernel-local work is more promising than scheduling or overlap work.
2. Read only the relevant KernelPilot knowledge entries, source catalog rows,
   PR notes, and referenced upstream code for that operator/architecture.
3. Record provenance, license/notice requirements, copied snippets, and rejected
   ideas before adapting any implementation pattern.
4. Use `ncu-report` when the next kernel edit is not obvious, a candidate is
   within `+/-2%`, a candidate regresses, or reviewer feedback asks for measured
   counter evidence.
5. Store NCU outputs under `kernel/ncu-digests/<version>/` or the host's
   equivalent artifact root. Each digest must compare baseline vs candidate and
   end with exactly one concrete next edit.
6. Patch the SGLang kernel or call path directly in the SGLang checkout, with
   focused correctness and microbench coverage when available.
7. Wire the candidate into the active model-serving path that produced the
   original profiler row.
8. Re-run the same real-model benchmark and profiler after the candidate is
   correct. A microbench or NCU win alone is not success.

If no focused harness exists, build the smallest harness that preserves the
model-derived shapes/dtypes/layouts. If NCU cannot run on the host, record the
blocker in the digest path and keep the next edit grounded in the available
torch-profiler and source evidence.

### Model-Loop Checkpoint

After every accepted round, update `humanize/model-loop-checkpoint.md` with:

- original model, tokenizer, precision, quantization, hardware, workload, SLA,
  artifact root, and benchmark winner commands
- current SGLang branch, commit, patches applied, tests run, and current best
  SGLang benchmark row
- remaining gap, profiler rows, model PR history notes, kernel-assist notes, NCU
  digest paths, rejected source ideas, and the next planned SGLang patch

This checkpoint is for campaign recovery inside the same model-level workflow.
It is not a handoff to another RLCR loop.

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
Model PR history evidence should be recorded beside SGLang, vLLM,
TensorRT-LLM, KernelPilot, and NCU source ideas when it influenced the patch.

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
table, all winner commands, model PR history paths, profile paths, SGLang
changed files, tests, and whether SGLang reached target-environment SOTA.
