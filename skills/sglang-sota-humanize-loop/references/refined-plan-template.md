# Humanize Plan Template For SGLang SOTA Loops

Use this template for `.humanize/sglang-sota-agent/refined-plan.md` after the
fixed benchmark and required profiles have already been captured.

```markdown
# SGLang SOTA Humanize Plan: <model>

## Goal Description

Make SGLang match or beat the best observed vLLM/TensorRT-LLM serving
performance for `<model>` on `<hardware>` under the fixed workload, precision,
quantization, and SLA captured in `<artifact-root>`.

The fixed benchmark phase is complete. The RLCR loop must patch SGLang code
using profiler evidence, re-run the same model-level benchmark/profile, and
continue through minimal patches until SGLang reaches the stop criteria.
The matching PR-driven model history has been read from
`model-pr-optimization-history`, and `history/model-pr-history-notes.md`
records the SGLang/vLLM PR evidence that influenced source-path selection.
Kernel-local work follows the model-level RLCR evidence path: KernelPilot may
provide knowledge and source evidence, and `ncu-report` may provide measured
counter digests. Kernel candidates are accepted through the same real-model
benchmark/profile revalidation path as other SGLang patches.

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

- AC-2: Required model PR history and profiler evidence exists before patching
  - Positive Tests (expected to PASS):
    - `history/model-pr-history-notes.md` exists under `<artifact-root>`.
    - The notes cite the matching SGLang model history when available.
    - The notes cite matching vLLM history when vLLM is the leading competitor
      or when vLLM evidence influenced a suspected missing SGLang fast path.
    - The notes include docs read, PR numbers, source files, symbols,
      validation risks, and the decision each item influenced.
    - SGLang profile analysis exists for the slow scenario.
    - At least the best framework profile analysis exists.
    - If both vLLM and TensorRT-LLM are more than 1% ahead, both competitor
      analyses exist.
    - Every analysis contains kernel, overlap-opportunity, and fuse-pattern
      tables with prefill/decode evidence when available.
    - When optional capacity, layer-pipeline, or compute-simulation gates were
      triggered, the matching `analysis/capacity.md`,
      `analysis/layer-pipeline.md`, or `analysis/compute-simulation.md` report
      exists and is cited by the patch choice.
  - Negative Tests (expected to FAIL):
    - A code patch is proposed without citing a profiler table row and source
      path or kernel family.
    - A model-specific source patch is proposed without checking matching model
      PR history for prior SGLang changes and relevant competitor evidence.
    - The loop ignores a triggered optional analysis gate and patches without
      preserving the resulting evidence.

- AC-3: SGLang patches are evidence-driven and minimal
  - Positive Tests (expected to PASS):
    - Each accepted patch cites the benchmark symptom, profiler row, source
      path, and expected impact.
    - Changes are local to the SGLang bottleneck path unless a broader change
      is required and justified.
  - Negative Tests (expected to FAIL):
    - A patch disables correctness checks, weakens output quality, or changes
      only launch parameters after the winner table is known.

- AC-4: Kernel-level bottlenecks stay inside the model RLCR loop
  - Positive Tests (expected to PASS):
    - For a specific slow CUDA/Triton/CuTe/CUTLASS/TileLang/torch.compile
      kernel, the profiler evidence shows SGLang is more than 1% behind and the
      target kernel or tightly scoped kernel family has at least 1% cumulative
      GPU-time share in the slow stage.
    - KernelPilot is used only as a knowledge/source-evidence repository:
      relevant knowledge entries, source catalog rows, PR notes, and referenced
      upstream code are recorded in `kernel/kernelpilot-knowledge-notes.md`.
    - `ncu-report` is used when the next kernel edit is unclear, a candidate is
      within +/-2%, a candidate regresses, or review asks for counter evidence;
      each digest under `kernel/ncu-digests/<version>/` compares baseline vs
      candidate and ends with exactly one concrete next edit.
    - The kernel candidate is patched directly in the SGLang checkout, wired
      into the active model-serving path, and validated with focused correctness
      checks plus the same real-model benchmark/profile.
  - Negative Tests (expected to FAIL):
    - Kernel-specialist effort is spent on a lone SGLang kernel below 1%
      cumulative GPU-time share with no aggregated family above 1%.
    - Upstream or competitor kernel code is copied without recording
      provenance, license/notice obligations, and the local delta.
    - The loop declares success from a microbench or NCU win without rerunning
      the same model-level benchmark/profile.

- AC-5: Real-model revalidation is run after each accepted patch
  - Positive Tests (expected to PASS):
    - The SGLang winner command or a re-searched SGLang command is benchmarked
      on the same workload after the patch.
    - Profiler triage is rerun when the original diagnosis was profile-derived
      or when the gap remains.
  - Negative Tests (expected to FAIL):
    - The loop declares success from a microbench only.

- AC-6: Iteration ledgers are complete
  - Positive Tests (expected to PASS):
    - Attempt, optimization, source-idea, lineage, and profile-digest artifacts
      are updated after each round.
    - Failed, regressed, partial, and abandoned ideas are recorded.
  - Negative Tests (expected to FAIL):
    - A failed idea is retried without checking the source-idea ledger.

- AC-7: Stop criteria are satisfied
  - Positive Tests (expected to PASS):
    - SGLang beats the best framework, or is within the stable 1% threshold
      after repeat runs, or the remaining gap is proven external/not patchable.
  - Negative Tests (expected to FAIL):
    - The loop stops while SGLang remains more than 1% behind and there is an
      uninvestigated profiler table row with plausible SGLang source impact.

- AC-8: Single-loop continuity is preserved
  - Positive Tests (expected to PASS):
    - `humanize/model-loop-checkpoint.md` records the original benchmark
      winners, workload/SLA, SGLang commit, applied patches, current best
      SGLang result, remaining gap, model PR history notes, profiler rows,
      kernel-assist notes, NCU digest paths, rejected source ideas, and the
      next planned SGLang patch.
    - The campaign can resume from the same model-loop artifacts with benchmark,
      profile, source-evidence, and patch lineage intact.
  - Negative Tests (expected to FAIL):
    - Kernel changes are accepted without updating the checkpoint, ledgers, NCU
      digest links when applicable, and real-model benchmark/profile results.
    - A second `.humanize/rlcr` session is launched for KernelPilot or any
      kernel-local subloop instead of keeping the work in the model loop.

## Path Boundaries

### Upper Bound (Maximum Scope)

Multiple minimal SGLang patches, kernel-local SGLang edits assisted by
KernelPilot knowledge and `ncu-report`, and repeated real-model
benchmark/profile runs are allowed when needed to close the measured gap.

### Lower Bound (Minimum Scope)

One profiler-backed SGLang patch plus real-model benchmark/profile
revalidation, unless the initial evidence proves no patch is needed.

### Allowed Choices

- Can use: SGLang source patches, guarded heuristics, existing fast-path
  selection, fusion or overlap fixes, model-specific runtime fixes, PR-driven
  model history knowledge for SGLang/vLLM source-path selection,
  optional capacity/layer-pipeline/compute-simulation reports,
  KernelPilot knowledge/source evidence for eligible hot kernels,
  `ncu-report` digests, focused tests, microbenchmarks, torch-profiler, Nsight
  Compute, and Nsight Systems.
- Cannot use: changing the fixed workload/SLA after seeing results, removing a
  competitor from comparison without a recorded unsupported reason, disabling
  correctness or tokenizer behavior, spending kernel-specialist effort on
  sub-1% lone kernels, or claiming SOTA from smoke-only runs.

## Dependencies and Sequence

### Milestones

1. Preserve fixed baseline artifacts
   - Confirm `history/model-pr-history-notes.md` has matching SGLang history
     and, when applicable, leading-competitor vLLM history.
   - Confirm winner commands, workload, SLA, and gap.
   - Confirm required profile analyses and root-cause report.
2. Patch the highest-confidence SGLang bottleneck
   - Choose one minimal code change from the profiler evidence.
   - Add or update focused tests when behavior changes.
   - Re-run relevant SGLang checks.
3. Revalidate with the same real model workload
   - Re-run the SGLang benchmark.
   - Re-run profiler triage when the gap remains or the patch changes the
     profiled path.
4. Continue or stop
   - If SGLang is still more than 1% behind, pick the next profiler-backed
     patch.
   - After two weak rounds below 1% geomean improvement over the prior best,
     expand code-first research before editing again.
   - If a kernel-local bottleneck remains, use KernelPilot knowledge and
     `ncu-report` inside the same model loop, then validate the integrated
     SGLang patch with the same real-model benchmark/profile.
   - Stop only under AC-7.

## Implementation Notes

- Keep Humanize local state under `.humanize/`.
- Keep benchmark/profile artifacts under `<artifact-root>`.
- Keep model PR history notes under `<artifact-root>/history/`.
- Keep KernelPilot knowledge notes and NCU digests under `<artifact-root>/kernel/`.
- Commit SGLang changes after each round summary.
- Mention exact changed files, commands, result deltas, and remaining risk in
  each Humanize round summary.
```
