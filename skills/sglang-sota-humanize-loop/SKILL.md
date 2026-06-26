---
name: sglang-sota-humanize-loop
description: "Run an autonomous Humanize-governed SGLang SOTA performance loop for one LLM model: first perform a fixed fair SGLang benchmark against the requested comparison framework set, then start one RLCR loop that repeatedly decides the gap, profiles the current bottleneck, runs layer/kernel pipeline analysis, patches SGLang code, optionally uses ncu-report-skill for kernel evidence, and revalidates until SGLang matches or beats the best observed requested framework under the same workload and SLA."
---

# SGLang SOTA Humanize Loop

## Overview

Use this skill when the user names a model and wants the SGLang serving path to
autonomously keep improving until it matches or beats the best reproducible
result from the requested comparison framework set in the same target
environment.

This workflow has two durable parts:

1. A fixed baseline phase that must be completed once before any code patching.
2. One Humanize RLCR loop that owns gap decision, profiling, required
   layer/kernel deep dive, SGLang patching, optional NCU evidence, and
   real-model revalidation.

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

Given a model-level SGLang SOTA request, do not ask the user to run separate
benchmark, profiler, gen-plan, refine-plan, or Humanize setup commands. Do the
setup yourself.

Ask the user only if the model, target GPU environment, or precision/quantization
policy is missing and cannot be inferred from local configs or the active host
skill.

Keep only the fixed benchmark phase outside the RLCR patch loop. Once the fixed
cross-framework benchmark and model PR history notes exist, start Humanize. The
RLCR loop itself must decide whether a gap still exists, collect current
profiler evidence, run layer pipeline analysis, patch SGLang, and revalidate.

Treat the model optimization campaign as the durable unit, not one terminal
session. The campaign is recoverable from the run artifact root, checkpoint
files, benchmark/profile artifacts, NCU digests, and ledgers.

## Phase 0: Inputs And Run Directory

Collect or infer:

- model id or checkpoint path, tokenizer, precision, quantization, trust policy,
  and max context length
- target SGLang checkout to patch
- GPU type/count, visible GPU ids, container or remote shell, CUDA/NCCL versions,
  and whether multi-node is allowed
- target framework: always SGLang
- comparison framework set: if the user explicitly names or excludes comparison
  frameworks, honor that exact set; otherwise default to vLLM, TensorRT-LLM, and
  TokenSpeed when available
- model-family history slug inferred from the model id, checkpoint, or hot
  framework source path when possible
- artifact root

Create one run directory:

```text
runs/YYYYMMDD_<model_slug>_sota_humanize/
  manifest.md
  help/
  benchmark/
    fairness/
  profiles/
  analysis/
    root-cause.md
    layer-pipeline.md
    fairness-diagnostics.md
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

## Framework Selection

Treat SGLang and the comparison frameworks as separate concepts:

- SGLang is always the target framework because this loop patches SGLang.
- `comparison_frameworks` contains only competitors, using canonical names
  `vllm`, `tensorrt_llm`, and `tokenspeed`.
- If the user says "only compare with vLLM", "comparison_frameworks: [vllm]",
  or "do not consider TensorRT-LLM/TokenSpeed", run only SGLang plus vLLM.
- Do not silently add TensorRT-LLM or TokenSpeed because a config or doc exists.
- Record selected comparison frameworks and user-excluded frameworks in
  `manifest.md` and `benchmark/fairness/diagnostics.md`.
- Only read competitor history, launch competitor benchmarks, and profile
  competitor traces for the selected comparison framework set.

## Phase 0.5: Model PR History Knowledge Gate

Before the fixed benchmark and before any patch planning, query and read
`model-pr-optimization-history` for the target model family.

Rules:

- If the slug is unclear, run `scripts/query.py "<model id or family>"` from
  the knowledge root and choose the closest model-family history.
- Read the SGLang history for that family whenever it exists.
- Read competitor history too when a selected comparison framework later becomes
  the leading competitor or its source/trace suggests a missing SGLang fast path.
- Write `history/model-pr-history-notes.md` with the paths read, PR numbers,
  source files, symbols, validation risks, and the concrete decision each item
  influences.
- Treat these notes as source and PR memory that helps choose a better SGLang
  patch, not measured proof by itself.

If the knowledge root is unavailable, record the blocker in the same notes file
and continue with benchmark/profile evidence.

## Phase 1: Fixed Fair Benchmark Gate

This phase is mandatory and happens exactly once before Humanize starts.

Use `llm-serving-auto-benchmark` as the source of truth for candidate generation,
result schema, workload, and comparison.

Hard requirements:

- Search SGLang and every selected comparison framework's best deployment
  command when that framework is supported in the target environment.
- Do not compare tuned SGLang against competitor defaults. Every framework gets
  its own bounded search.
- Use the same model weights, tokenizer, precision, quantization, GPU type/count,
  GPU ids, endpoint path, sampling settings, and SLA.
- Record package version or git commit plus server/benchmark `--help` snapshots
  for SGLang and every selected comparison framework. The 2026-06-26
  source-refresh anchors are SGLang
  `8524678889485801e7a4a12d62015be0c68f7a90`, vLLM
  `37ce34922f7f5e58241369511130cd99c1c50bfe`, TensorRT-LLM
  `0722c5f47d2cae69ac1a237da51e550dd214532c`, and TokenSpeed
  `5aedf69d6b476baa65571011de6ea60fd5a238a8`; still prefer target-image
  `--help` over these source notes.
- Use the default two dataset scenarios from `llm-serving-auto-benchmark` unless
  the user explicitly provides a production workload:
  - dataset kind `random`, `num_prompts: 80`
  - `chat`: random input `1000`, output `1000`
  - `summarization`: random input `8000`, output `1000`
  - treat the two input/output pairs as aligned scenarios, not a cartesian
    product
- Do not replace those scenarios with an easier smoke dataset for the real SOTA
  decision. Smoke runs are allowed only when labeled as flow checks.
- For selected TensorRT-LLM comparisons, keep
  `trtllm-serve serve --backend pytorch`; reject non-PyTorch TensorRT-LLM server
  backends for this skill.
- For selected TokenSpeed comparisons, use `tokenspeed serve <model>` or the
  target image's exact alias such as `ts serve`; record the exact binary spelling
  and validate `tokenspeed serve --help` plus `tokenspeed bench serve --help`.
- If TokenSpeed is selected and a TokenSpeed-native agentic workload exists for
  the model family, run it as an additional lane beside the common
  cross-framework workload. Do not let it replace the common scenario unless the
  user explicitly chose that workload as the SOTA target before benchmarking.
- Keep failed, skipped, and SLA-failing candidates in the benchmark artifact.
- Record optional fairness fields in normalized rows whenever available:
  speculative accept length, pre-scheduler/scheduler time, cache hit rate,
  offload or memory residency, endpoint, and request-shape notes.

Write:

- `benchmark/candidates.jsonl`
- `benchmark/summary.md`
- `benchmark/winning-commands.md`
- `benchmark/fairness/diagnostics.md`
- framework help outputs under `help/`
- the exact launch and benchmark commands for every winner

Do not choose a code patch outside RLCR. The fixed winner table is the baseline
input to the model loop.

## Phase 2: Build The Humanize Plan

Create a Humanize plan inside the SGLang checkout that will be patched:

```text
.humanize/sglang-sota-agent/refined-plan.md
```

Use [references/refined-plan-template.md](references/refined-plan-template.md)
as the skeleton and fill it with the actual model, workload, benchmark winners,
artifact root, model PR history notes, and target SGLang checkout.

The plan must require every RLCR round to:

- preserve the fixed benchmark workload and SLA
- preserve and consult `history/model-pr-history-notes.md` before choosing
  model-specific SGLang source paths
- preserve and consult `benchmark/fairness/diagnostics.md` before treating a
  gap as a code bottleneck
- run the gap decision inside the loop before patching
- run `llm-torch-profiler-analysis` inside the loop when SGLang is behind or
  when the previous patch changed the profiled path
- run `llm-pipeline-analysis` inside the loop after profiler triage and before
  choosing a source path, representative layer, or kernel target
- patch SGLang code, not just benchmark parameters
- use `ncu-report-skill` inside the same loop when a kernel edit needs Nsight
  Compute evidence
- re-run real model benchmark/profile after each accepted patch
- continue through multiple minimal patches when one patch only closes part of
  the gap
- record every attempt, failed idea, partial win, rejected source idea, and final
  selected patch in artifacts

## Phase 3: Start RLCR

Before starting Humanize from the SGLang checkout:

- Ensure the SGLang checkout is a git repository with at least one commit and a
  clean working tree, excluding only gitignored Humanize runtime state.
- Ensure `.humanize*` is gitignored so RLCR state, round summaries, and local
  checkpoints cannot be staged accidentally.
- Ensure the intended review base branch is present locally. Pass
  `--base-branch <branch>` if Humanize's auto-detection would be ambiguous.
- Do not start a new loop if any existing `.humanize/rlcr/*/state.md` is active
  in the SGLang checkout. Resume, finish, or cancel the old model loop first.

From the SGLang checkout, run:

```bash
"$HUMANIZE_RUNTIME_ROOT/scripts/setup-rlcr-loop.sh" \
  .humanize/sglang-sota-agent/refined-plan.md --yolo --strict-success
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
5. Commit SGLang changes.
6. Write the required Humanize round summary.
7. Stop normally so the native Humanize Stop hook can review.

If no active state file exists, or if `strict_success: true` is missing, stop
and report that RLCR did not start correctly. Do not continue into SGLang patch
work outside the Humanize loop. If the hook blocks exit, follow the generated
next-round prompt exactly.

## Inside Each RLCR Round

### Gap Decision

At the start of every round, compute current SGLang's gap against the best
SLA-passing selected comparison framework for each fixed scenario.

Use `1%` as the default stable noise threshold. If the current result is within
`+/-1%`, rerun the winning commands enough times to decide whether the gap is
stable before choosing a patch.

Patch only when SGLang is slower than the best selected comparison framework by
more than `1%`, fails SLA while a selected competitor passes, or has a profiled
bottleneck that explains the remaining gap under the fixed workload.

Before patching, update `analysis/fairness-diagnostics.md` from the current
benchmark rows and the fixed `benchmark/fairness/diagnostics.md`. Check:

- same GPU count, visible GPU ids, model weights, tokenizer, precision,
  quantization, endpoint, sampling settings, and request shape
- speculative decoding status, algorithm, number of steps, and mean accept
  length for every framework that enables a drafter/MTP/EAGLE-style path
- pre-scheduler, scheduler, queue, tokenizer, and detokenizer time when exposed
  by server logs or benchmark output
- prefix/KV cache status and hit rate, especially for repeated prompts or
  multi-turn workloads
- GPU memory utilization, KV dtype, offload, and model weight residency
- TokenSpeed-specific serving knobs, when TokenSpeed is selected, such as
  `attn_tp_size`, `moe_tp_size`, `enable_expert_parallel`, backend choices, and
  communication-fusion flags

If the current leader wins because it used a non-equivalent workload,
non-equivalent cache/speculative setting, different resident weights, or a
framework-specific endpoint shortcut, fix the benchmark normalization first and
rerun. Do not patch SGLang code until the gap survives this fairness gate.

If SGLang is already best or tied within the stable threshold, write the final
report and stop under the normal Humanize review path.

### Required Profiling

When SGLang is behind, profile the current best SGLang command and the leading
competitor command with `llm-torch-profiler-analysis`.

Rules:

- Always profile SGLang when it is behind.
- Always profile at least the current best selected comparison framework.
- If multiple competitors are more than `1%` ahead of SGLang in a stable
  result, profile each ahead competitor that can produce a trace. For
  TokenSpeed, analyze an existing torch-profiler trace or run-directory trace
  emitted by the deployment; do not fabricate a live profiler endpoint when the
  target image only supports offline traces.
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
comparison: which stage is slower, which table rows explain it, and which SGLang
source paths or kernel families are plausible patch targets.

Do not patch SGLang until this report exists for the current gap.

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

Use the profiled SGLang trace and the served model config. Write
`analysis/layer-pipeline.md` with the chosen forward pass, layer-type timing
table, representative layers, top hot kernels, and any Perfetto ranges used for
inspection. Do not choose a SGLang patch before this report exists for the
current round.

### Kernel Evidence Assist

Use `ncu-report-skill` only when the active RLCR round is writing a concrete
kernel or small kernel-family patch and torch-profiler evidence is not enough to
choose or validate the next edit.

Eligibility gate:

- SGLang is still more than `1%` behind the best selected comparison framework
  for the fixed benchmark scenario after the required repeat/profiler checks.
- The slow stage has a concrete SGLang kernel or tightly scoped kernel family
  in the kernel table with at least `1%` cumulative GPU-time share. Do not spend
  kernel-specialist effort on a lone kernel below `1%` share unless a shared
  implementation affects an aggregated family above `1%`.
- `llm-pipeline-analysis` has identified the representative layer/forward pass
  and top hot kernels for the current round.
- The proposed kernel target has a clear correctness reference,
  representative shapes/dtypes/layouts from the model run, and a path to wire
  the candidate into the active SGLang serving code.

For each eligible kernel target:

1. Read `ncu-report-skill/SKILL.md` and follow its Nsight Compute workflow for
   harness construction, `ncu` collection, report parsing, stall diagnosis, and
   evidence-backed next-edit selection.
2. Store NCU outputs under `kernel/ncu-digests/<version>/` or the host's
   equivalent artifact root. Each digest must compare baseline vs candidate and
   end with exactly one concrete next edit.
3. Patch the SGLang kernel or call path directly in the SGLang checkout, with
   focused correctness and microbench coverage when available.
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
- current SGLang branch, commit, patches applied, tests run, and current best
  SGLang benchmark row
- remaining gap, profiler rows, model PR history notes, layer-pipeline notes,
  NCU digest paths, rejected source ideas, and the next planned SGLang patch

This checkpoint is for campaign recovery inside the same model-level workflow.
It records enough context to resume the campaign without losing
benchmark/profile lineage.

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
improvement get optimization rows. Source ideas must include profiler rows,
layer-pipeline evidence, NCU report paths when used, and code
provenance so later rounds can avoid re-reading the same source. Model PR
history evidence should be recorded beside SGLang, selected comparison
frameworks, and NCU source ideas when it influenced the patch.

After two consecutive rounds with less than `1%` geomean improvement over the
prior best SGLang result, expand code-first research before editing again.
Prefer code and PR evidence from SGLang, selected comparison frameworks, and
relevant kernel source guides before prose-only articles.

## Stop Conditions

Stop only when one of these is true:

- SGLang beats the best SLA-passing selected comparison framework result on the
  fixed workload.
- SGLang is tied within the stable `1%` threshold after repeat runs.
- The remaining gap is proven external to SGLang, such as unavailable hardware
  support, missing framework dependency, unsupported TensorRT-LLM PyTorch
  backend, unsupported TokenSpeed build, or model weights that cannot be loaded
  fairly.
- Profile evidence shows the remaining hot path is already near the relevant
  hardware or algorithmic limit and no low-risk SGLang patch remains.

The final report must include the fixed benchmark table, post-patch benchmark
table, all winner commands, model PR history paths, profile paths, layer-pipeline
paths when used, NCU digest paths when used, SGLang changed files, tests, and
whether SGLang reached target-environment SOTA.
