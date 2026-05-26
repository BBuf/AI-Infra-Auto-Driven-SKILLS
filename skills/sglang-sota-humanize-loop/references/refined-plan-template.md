# Humanize Plan Template For SGLang SOTA Loops

Use this template for `.humanize/sglang-sota-agent/refined-plan.md` after the
fixed fair benchmark and model PR history notes have already been captured.

```markdown
# SGLang SOTA Humanize Plan: <model>

## Goal Description

Make SGLang match or beat the best observed vLLM/TensorRT-LLM serving
performance for `<model>` on `<hardware>` under the fixed workload, precision,
quantization, and SLA captured in `<artifact-root>`.

The fixed benchmark phase is complete. The RLCR loop must perform the current
gap decision, collect profiler evidence, run layer-pipeline deep dives, patch
SGLang code, optionally use `ncu-report-skill` for kernel evidence, re-run the
same model-level benchmark/profile, and continue through minimal patches until
SGLang reaches the stop criteria.

The matching PR-driven model history has been read from
`model-pr-optimization-history`, and `history/model-pr-history-notes.md`
records the SGLang/vLLM PR evidence that influenced source-path selection.

## Acceptance Criteria

- AC-1: Fixed benchmark evidence is preserved
  - Positive Tests (expected to PASS):
    - `benchmark/candidates.jsonl`, `benchmark/summary.md`, and
      `benchmark/winning-commands.md` exist under `<artifact-root>`.
    - The workload uses the fixed scenario set or a user-provided production
      workload recorded before RLCR began.
  - Negative Tests (expected to FAIL):
    - A patch changes only benchmark workload, request count, SLA, or competitor
      commands to make SGLang look faster.

- AC-2: Required model PR history evidence exists before Humanize starts
  - Positive Tests (expected to PASS):
    - `history/model-pr-history-notes.md` exists under `<artifact-root>`.
    - The notes cite the matching SGLang model history when available.
    - The notes cite matching vLLM history when vLLM is the leading competitor
      or when vLLM evidence influenced a suspected missing SGLang fast path.
    - The notes include docs read, PR numbers, source files, symbols,
      validation risks, and the decision each item influenced.
  - Negative Tests (expected to FAIL):
    - A model-specific source patch is proposed without checking matching model
      PR history for prior SGLang changes and relevant competitor evidence.

- AC-3: Gap decision and profiler triage happen inside each RLCR round
  - Positive Tests (expected to PASS):
    - The round computes the current SGLang gap against the fixed benchmark
      winner table before choosing a patch.
    - SGLang profile analysis exists for the current slow scenario when SGLang
      is more than 1% behind.
    - At least the best framework profile analysis exists when SGLang is behind.
    - If both vLLM and TensorRT-LLM are more than 1% ahead, both competitor
      analyses exist.
    - Every analysis contains kernel, overlap-opportunity, and fuse-pattern
      tables with prefill/decode evidence when available.
    - `analysis/root-cause.md` maps the current gap to profiler rows and SGLang
      source paths or kernel families before patching.
  - Negative Tests (expected to FAIL):
    - A code patch is proposed without citing a profiler table row and source
      path or kernel family for the current gap.
    - Profiling is treated as a one-time pre-loop gate and not refreshed after
      a patch changes the profiled path or leaves a gap.

- AC-4: Layer-pipeline evidence is required inside each RLCR round
  - Positive Tests (expected to PASS):
    - `llm-pipeline-analysis` is run after profiler triage and before choosing
      a patch target.
    - `analysis/layer-pipeline.md` records the chosen forward pass,
      representative layers, layer-type timing, top hot kernels, and Perfetto
      ranges when used.
  - Negative Tests (expected to FAIL):
    - The loop chooses a SGLang patch without a current
      `analysis/layer-pipeline.md` report.

- AC-5: SGLang patches are evidence-driven and minimal
  - Positive Tests (expected to PASS):
    - Each accepted patch cites the benchmark symptom, profiler row,
      layer-pipeline row, source path, and expected impact.
    - Changes are local to the SGLang bottleneck path unless a broader change
      is required and justified.
  - Negative Tests (expected to FAIL):
    - A patch disables correctness checks, weakens output quality, or changes
      only launch parameters after the winner table is known.

- AC-6: Kernel-level work uses ncu-report-skill inside the same model loop
  - Positive Tests (expected to PASS):
    - For a specific slow CUDA/Triton/CuTe/CUTLASS/TileLang/torch.compile
      kernel, the profiler evidence shows SGLang is more than 1% behind and the
      target kernel or tightly scoped kernel family has at least 1% cumulative
      GPU-time share in the slow stage.
    - `ncu-report-skill` is used when the next kernel edit is unclear, a
      candidate is within +/-2%, a candidate regresses, or review asks for
      counter evidence; each digest under `kernel/ncu-digests/<version>/`
      compares baseline vs candidate and ends with exactly one concrete next
      edit.
    - The kernel candidate is patched directly in the SGLang checkout, wired
      into the active model-serving path, and validated with focused correctness
      checks plus the same real-model benchmark/profile.
  - Negative Tests (expected to FAIL):
    - Kernel-specialist effort is spent on a sub-1% lone SGLang kernel with no
      aggregated family above 1%.
    - Upstream or competitor kernel code is copied without recording
      provenance, license/notice obligations, and the local delta.
    - The loop declares success from a microbench or NCU win without rerunning
      the same model-level benchmark/profile.

- AC-7: Real-model revalidation is run after each accepted patch
  - Positive Tests (expected to PASS):
    - The SGLang winner command or a re-searched SGLang command is benchmarked
      on the same workload after the patch.
    - Profiler triage is rerun when the original diagnosis was profile-derived
      or when the gap remains.
  - Negative Tests (expected to FAIL):
    - The loop declares success from a microbench only.

- AC-8: Iteration ledgers and single-loop continuity are preserved
  - Positive Tests (expected to PASS):
    - Attempt, optimization, source-idea, lineage, and profile-digest artifacts
      are updated after each round.
    - `humanize/model-loop-checkpoint.md` records the original benchmark
      winners, workload/SLA, SGLang commit, applied patches, current best
      SGLang result, remaining gap, model PR history notes, profiler rows,
      layer-pipeline notes, NCU digest paths, rejected source ideas, and the
      next planned SGLang patch.
    - The campaign can resume from the same model-loop artifacts with benchmark,
      profile, source-evidence, and patch lineage intact.
  - Negative Tests (expected to FAIL):
    - A second `.humanize/rlcr` session is launched for kernel work instead of
      keeping the work in the model loop.

- AC-9: Stop criteria are satisfied
  - Positive Tests (expected to PASS):
    - SGLang beats the best framework, or is within the stable 1% threshold
      after repeat runs, or the remaining gap is proven external/not patchable.
  - Negative Tests (expected to FAIL):
    - The loop stops while SGLang remains more than 1% behind and there is an
      uninvestigated profiler table row with plausible SGLang source impact.

## Path Boundaries

### Upper Bound (Maximum Scope)

Multiple minimal SGLang patches, kernel-local SGLang edits assisted by
`ncu-report-skill`, repeated `llm-torch-profiler-analysis`,
`llm-pipeline-analysis`, and real-model benchmark/profile runs are allowed when
needed to close the measured gap.

### Lower Bound (Minimum Scope)

One profiler-backed SGLang patch plus real-model benchmark/profile
revalidation, unless the current in-loop evidence proves no patch is needed.

### Allowed Choices

- Can use: SGLang source patches, guarded heuristics, existing fast-path
  selection, fusion or overlap fixes, model-specific runtime fixes, PR-driven
  model history knowledge for SGLang/vLLM source-path selection,
  `llm-torch-profiler-analysis`, `llm-pipeline-analysis`, `ncu-report-skill`
  digests, focused tests, microbenchmarks, torch-profiler, Nsight Compute, and
  Nsight Systems.
- Cannot use: changing the fixed workload/SLA after seeing results, removing a
  competitor from comparison without a recorded unsupported reason, disabling
  correctness or tokenizer behavior, spending kernel-specialist effort on
  sub-1% lone kernels, or claiming SOTA from smoke-only runs.

## Dependencies and Sequence

### Milestones

1. Preserve fixed baseline artifacts
   - Confirm `history/model-pr-history-notes.md` has matching SGLang history
     and, when applicable, leading-competitor vLLM history.
   - Confirm winner commands, workload, and SLA.
2. Run in-loop gap decision and profiling
   - Compare current SGLang to the fixed winner table.
   - Run `llm-torch-profiler-analysis` for current SGLang and the leading
     competitor when SGLang is behind.
   - Run `llm-pipeline-analysis` after profiler triage and before patch
     selection.
3. Patch the highest-confidence SGLang bottleneck
   - Choose one minimal code change from the current evidence.
   - Use `ncu-report-skill` inside the same model loop when writing a kernel
     patch needs Nsight Compute evidence.
   - Add or update focused tests when behavior changes.
4. Revalidate with the same real model workload
   - Re-run the SGLang benchmark.
   - Re-run profiler triage when the gap remains or the patch changes the
     profiled path.
5. Continue or stop
   - If SGLang is still more than 1% behind, pick the next profiler-backed
     patch.
   - After two weak rounds below 1% geomean improvement over the prior best,
     expand code-first research before editing again.
   - Stop only under AC-9.

## Implementation Notes

- Keep Humanize local state under `.humanize/`.
- Keep benchmark/profile artifacts under `<artifact-root>`.
- Keep model PR history notes under `<artifact-root>/history/`.
- Keep layer-pipeline reports under `<artifact-root>/analysis/`.
- Keep NCU digests under `<artifact-root>/kernel/ncu-digests/`.
- Commit SGLang changes after each round summary.
- Mention exact changed files, commands, result deltas, and remaining risk in
  each Humanize round summary.
```
