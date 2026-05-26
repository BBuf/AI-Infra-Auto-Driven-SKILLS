---
name: vllm-sota-humanize-loop
description: "Run an autonomous Humanize-governed vLLM SOTA performance loop for one LLM model: first perform the fixed fair vLLM/SGLang/TensorRT-LLM deployment search and benchmark, then start one RLCR loop that repeatedly decides the gap, profiles the current bottleneck, runs layer/kernel pipeline analysis, patches vLLM code, optionally uses ncu-report-skill for kernel evidence, and revalidates until vLLM matches or beats the best observed framework under the same workload and SLA."
---

# vLLM SOTA Humanize Loop

## Overview

Use this skill when the user names a model and wants the vLLM serving path to
autonomously keep improving until it matches or beats the best reproducible
SGLang or TensorRT-LLM result in the same target environment.

This workflow has two durable parts:

1. A fixed baseline phase that must be completed once before any code patching.
2. One Humanize RLCR loop that owns gap decision, profiling, required
   layer/kernel deep dive, vLLM patching, optional NCU evidence, and real-model
   revalidation.

Do not split the campaign into a pre-loop profiling phase plus a later patch
loop. After the fixed benchmark exists, Phase 2 gap decisions, Phase 3 profiling,
`llm-pipeline-analysis`, kernel evidence, and code changes all belong inside the
same model-level RLCR loop.

## Runtime Roots

This skill can run from Claude Code, Codex, or another compatible skill runtime.
Resolve companion roots in this order:

1. Prefer installed Claude Code skills under `~/.claude/skills` when running in
   Claude Code.
2. Prefer installed Codex skills under `${CODEX_HOME:-~/.codex}/skills` when
   running in Codex.
3. Fall back to checked-out repositories when the skills are symlinked or kept
   local for development.

Example local paths:

```text
Humanize runtime: ${CODEX_HOME:-~/.codex}/skills/humanize
ncu-report-skill: ${CODEX_HOME:-~/.codex}/skills/ncu-report-skill/SKILL.md
Model PR history knowledge: <repo>/model-pr-optimization-history
```

For Claude Code installs, the equivalent defaults are typically:

```text
Humanize runtime: ~/.claude/skills/humanize
ncu-report-skill: ~/.claude/skills/ncu-report-skill/SKILL.md
Model PR history knowledge: ~/.claude/skills/model-pr-history-knowledge
```

If the Humanize runtime is missing, locate a plugin or skill directory
containing `scripts/setup-rlcr-loop.sh`. If `ncu-report-skill` is unavailable,
kernel edits may still proceed from torch-profiler/source evidence, but record
the missing NCU evidence path as a blocker when a kernel change would normally
need Nsight Compute diagnostics.

## Companion Skills

Read these before a real run:

- `../llm-serving-auto-benchmark/SKILL.md`
- `../llm-torch-profiler-analysis/SKILL.md`
- `../llm-pipeline-analysis/SKILL.md`
- `../../model-pr-optimization-history/SKILL.md`
- the matching host or operator skill for SSH, container, GPU, and artifact
  conventions

Read `ncu-report-skill/SKILL.md` only when the active RLCR round is writing or
evaluating a CUDA, Triton, CuTe, CUTLASS, TileLang, or torch.compile kernel path
and Nsight Compute evidence is needed.

## Contract

Given a model-level vLLM SOTA request, do not ask the user to run separate
benchmark, profiler, gen-plan, refine-plan, or Humanize setup commands. Do the
setup yourself.

Ask the user only if the model, target GPU environment, or precision/quantization
policy is missing and cannot be inferred from local configs or the active host
skill.

Keep only the fixed benchmark phase outside the RLCR patch loop. Once the fixed
cross-framework benchmark and model PR history notes exist, start Humanize. The
RLCR loop itself must decide whether a gap still exists, collect current
profiler evidence, run layer pipeline analysis, patch vLLM, and revalidate.

Treat the model optimization campaign as the durable unit, not one terminal
session. The campaign is recoverable from the run artifact root, checkpoint
files, benchmark/profile artifacts, NCU digests, and ledgers.

## Phase 0: Inputs And Run Directory

Collect or infer:

- model id or checkpoint path, tokenizer, precision, quantization, trust policy,
  and max context length
- target vLLM checkout to patch
- GPU type/count, visible GPU ids, container or remote shell, CUDA/NCCL versions,
  and whether multi-node is allowed
- framework set, defaulting to vLLM, SGLang, and TensorRT-LLM when available
- model-family history slug inferred from the model id, checkpoint, or hot
  vLLM/SGLang source path when possible
- artifact root

Create one run directory:

```text
runs/YYYYMMDD_<model_slug>_sota_humanize/
  manifest.md
  help/
  benchmark/
  profiles/
  analysis/
    root-cause.md
    layer-pipeline.md
  history/
    model-pr-history-notes.md
  kernel/
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
- Read the vLLM history for that family whenever it exists.
- Read the SGLang history too when SGLang is in the comparison set, later
  becomes the leading competitor, or its source/trace suggests a missing vLLM
  fast path.
- Write `history/model-pr-history-notes.md` with the paths read, PR numbers,
  source files, symbols, validation risks, and the concrete decision each item
  influences.
- Treat these notes as source and PR memory that helps choose a better vLLM
  patch, not measured proof by itself.

If the knowledge root is unavailable, record the blocker in the same notes file
and continue with benchmark/profile evidence.

## Phase 1: Fixed Fair Benchmark Gate

This phase is mandatory and happens exactly once before Humanize starts.

Use `llm-serving-auto-benchmark` as the source of truth for candidate generation,
result schema, workload, and comparison.

Hard requirements:

- Search vLLM, SGLang, and TensorRT-LLM best deployment commands when each
  framework is supported in the target environment.
- Do not compare tuned vLLM against competitor defaults. Every framework gets
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

Do not choose a code patch outside RLCR. The fixed winner table is the baseline
input to the model loop.

## Phase 2: Build The Humanize Plan

Create a Humanize plan inside the vLLM checkout that will be patched:

```text
.humanize/vllm-sota-agent/refined-plan.md
```

Use [references/refined-plan-template.md](references/refined-plan-template.md)
as the skeleton and fill it with the actual model, workload, benchmark winners,
artifact root, model PR history notes, and target vLLM checkout.

The plan must require every RLCR round to:

- preserve the fixed benchmark workload and SLA
- preserve and consult `history/model-pr-history-notes.md` before choosing
  model-specific vLLM source paths
- run the gap decision inside the loop before patching
- run `llm-torch-profiler-analysis` inside the loop when vLLM is behind or
  when the previous patch changed the profiled path
- run `llm-pipeline-analysis` inside the loop after profiler triage and before
  choosing a source path, representative layer, or kernel target
- patch vLLM code, not just benchmark parameters
- use `ncu-report-skill` inside the same loop when a kernel edit needs Nsight
  Compute evidence
- re-run real model benchmark/profile after each accepted patch
- continue through multiple minimal patches when one patch only closes part
  of the gap
- record every attempt, failed idea, partial win, rejected source idea, and
  final selected patch in artifacts

## Phase 3: Start RLCR

Before starting Humanize from the vLLM checkout:

- Ensure the vLLM checkout is a git repository with at least one commit and a
  clean working tree, excluding only gitignored Humanize runtime state.
- Ensure `.humanize*` is gitignored so RLCR state, round summaries, and local
  checkpoints cannot be staged accidentally.
- Ensure the intended review base branch is present locally. Pass
  `--base-branch <branch>` if Humanize's auto-detection would be ambiguous.
- Do not start a new loop if any existing `.humanize/rlcr/*/state.md` is active
  in the vLLM checkout. Resume, finish, or cancel the old model loop first.

From the vLLM checkout, run:

```bash
"$HUMANIZE_RUNTIME_ROOT/scripts/setup-rlcr-loop.sh" \
  .humanize/vllm-sota-agent/refined-plan.md --yolo --strict-success
```

If `HUMANIZE_RUNTIME_ROOT` is not already set by the client/plugin environment,
resolve it to the installed Humanize runtime first. In Codex, this is often
`${CODEX_HOME:-~/.codex}/skills/humanize`; in Claude Code it is often
`~/.claude/skills/humanize` or a plugin-provided Humanize runtime. If setup
exits non-zero, stop and report the error. Do not bypass the gate.

After setup succeeds:

1. Find the active state file with
   `find .humanize/rlcr -maxdepth 2 -name state.md -print`.
2. Verify the state file exists and contains `strict_success: true`.
3. Read `.humanize/rlcr/<timestamp>/round-0-prompt.md`.
4. Execute the current round.
5. Commit vLLM changes.
6. Write the required Humanize round summary.
7. Stop normally so the native Humanize Stop hook can review.

If no active state file exists, or if `strict_success: true` is missing, stop
and report that RLCR did not start correctly. Do not continue into vLLM patch
work outside the Humanize loop. If the hook blocks exit, follow the generated
next-round prompt exactly.

## Inside Each RLCR Round

### Gap Decision

At the start of every round, compute current vLLM's gap against the best
SLA-passing framework for each fixed scenario.

Use `1%` as the default stable noise threshold. If the current result is within
`+/-1%`, rerun the winning commands enough times to decide whether the gap is
stable before choosing a patch.

Patch only when vLLM is slower than the best framework by more than `1%`, fails
SLA while another framework passes, or has a profiled bottleneck that explains
the remaining gap under the fixed workload.

If vLLM is already best or tied within the stable threshold, write the final
report and stop under the normal Humanize review path.

### Required Profiling

When vLLM is behind, profile the current best vLLM command and the leading
competitor command with `llm-torch-profiler-analysis`.

Rules:

- Always profile vLLM when it is behind.
- Always profile at least the current best framework.
- If both SGLang and TensorRT-LLM are more than `1%` ahead of vLLM in a stable
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

Then write or update `analysis/root-cause.md` with the current cross-framework
comparison: which stage is slower, which table rows explain it, and which vLLM
source paths or kernel families are plausible patch targets.

Do not patch vLLM until this report exists for the current gap.

### Layer Pipeline Deep Dive

Run `llm-pipeline-analysis` inside every RLCR round after profiler triage and
before choosing a patch target.

The report must identify:

- the chosen forward pass and why it is representative
- the relevant layer types, especially for heterogeneous layers such as MoE,
  hash layers, or `compress_ratios`
- representative layers for the patch target
- top hot kernels in those representative layers
- any Perfetto time ranges needed for inspection

Use the profiled vLLM trace and the served model config. Write
`analysis/layer-pipeline.md` with the chosen forward pass, layer-type timing
table, representative layers, top hot kernels, and any Perfetto ranges used for
inspection. Do not choose a vLLM patch before this report exists for the current
round.

## Kernel Evidence Assist

Use `ncu-report-skill` only when the active RLCR round is writing a concrete
kernel or small kernel-family patch and torch-profiler evidence is not enough to
choose or validate the next edit.

Kernel-level assistance is allowed only when all of these are true:

- vLLM is still more than `1%` behind the best framework for the fixed
  benchmark scenario after the required repeat/profiler checks.
- The slow stage has a concrete vLLM kernel or tightly scoped kernel family
  in the kernel table with at least `1%` cumulative GPU-time share. Do not spend
  kernel-specialist effort on a lone kernel below `1%` share unless a shared
  implementation affects an aggregated family above `1%`.
- `llm-pipeline-analysis` has identified the representative layer/forward pass
  and top hot kernels for the current round.
- The proposed kernel target has a clear correctness reference,
  representative shapes/dtypes/layouts from the model run, and a path to wire
  the candidate into the active vLLM serving code.

For each eligible kernel target:

1. Read `ncu-report-skill/SKILL.md` and follow its Nsight Compute workflow for
   harness construction, `ncu` collection, report parsing, stall diagnosis, and
   evidence-backed next-edit selection.
2. Store NCU outputs under `kernel/ncu-digests/<version>/` or the host's
   equivalent artifact root. Each digest must compare baseline vs candidate and
   end with exactly one concrete next edit.
3. Patch the vLLM kernel or call path directly in the vLLM checkout, with
   focused correctness and microbench coverage when available. CUDA and C++
   kernel code lives under `csrc/`, Triton kernels and attention/quantization
   wrappers live under `vllm/`, and torch.compile-driven paths live under
   `vllm/compilation/`.
4. Wire the candidate into the active model-serving path that produced the
   original profiler row.
5. Re-run the same real-model benchmark and profiler after the candidate is
   correct. A microbench or NCU win alone is not success.

If no focused harness exists, build the smallest harness that preserves the
model-derived shapes/dtypes/layouts. If NCU cannot run on the host, record the
blocker in the digest path and keep the next edit grounded in the available
torch-profiler, layer-pipeline, and source evidence.

Do not start any standalone `.humanize/rlcr` session for a kernel target. Kernel
work stays inside the active model RLCR loop.

### Model-Loop Checkpoint

After every accepted round, update `humanize/model-loop-checkpoint.md` with:

- original model, tokenizer, precision, quantization, hardware, workload, SLA,
  artifact root, and benchmark winner commands
- current vLLM branch, commit, patches applied, tests run, and current best
  vLLM benchmark row
- remaining gap, profiler rows, model PR history notes, layer-pipeline notes, NCU
  digest paths, rejected source ideas, and the next planned vLLM patch

This checkpoint is for campaign recovery inside the same model-level workflow.
It records enough context to resume the campaign without losing
benchmark/profile lineage.

## Loop Ledgers

Keep these files under the run artifact root or the vLLM checkout, depending
on the host convention:

```text
humanize/attempt-ledger.md
humanize/optimization-ledger.md
humanize/source-idea-ledger.md
humanize/lineage.jsonl
humanize/profile-digests/
```

Every patch attempt gets an attempt row. Only correct patches with measured
improvement get optimization rows. Source ideas must include profiler rows,
layer-pipeline evidence, NCU report paths when used, and code provenance so
later rounds can avoid re-reading the same source. Model PR history evidence
should be recorded beside vLLM, SGLang, TensorRT-LLM, and NCU source ideas when
it influenced the patch.

After two consecutive rounds with less than `1%` geomean improvement over the
prior best vLLM result, expand code-first research before editing again. Prefer
code and PR evidence from vLLM, SGLang, TensorRT-LLM, and relevant kernel source
guides before prose-only articles.

## Stop Conditions

Stop only when one of these is true:

- vLLM beats the best SLA-passing SGLang/TensorRT-LLM result on the fixed
  workload.
- vLLM is tied within the stable `1%` threshold after repeat runs.
- The remaining gap is proven external to vLLM, such as unavailable hardware
  support, missing framework dependency, unsupported TensorRT-LLM PyTorch
  backend, or model weights that cannot be loaded fairly.
- Profile evidence shows the remaining hot path is already near the relevant
  hardware or algorithmic limit and no low-risk vLLM patch remains.

The final report must include the fixed benchmark table, post-patch benchmark
table, all winner commands, model PR history paths, profile paths,
layer-pipeline paths when used, NCU digest paths when used, vLLM changed files,
tests, and whether vLLM reached target-environment SOTA.
