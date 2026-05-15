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

- AC-2: Required profiler evidence exists before patching
  - Positive Tests (expected to PASS):
    - SGLang profile analysis exists for the slow scenario.
    - At least the best framework profile analysis exists.
    - If both vLLM and TensorRT-LLM are more than 1% ahead, both competitor
      analyses exist.
    - Every analysis contains kernel, overlap-opportunity, and fuse-pattern
      tables with prefill/decode evidence when available.
  - Negative Tests (expected to FAIL):
    - A code patch is proposed without citing a profiler table row and source
      path or kernel family.

- AC-3: SGLang patches are evidence-driven and minimal
  - Positive Tests (expected to PASS):
    - Each accepted patch cites the benchmark symptom, profiler row, source
      path, and expected impact.
    - Changes are local to the SGLang bottleneck path unless a broader change
      is required and justified.
  - Negative Tests (expected to FAIL):
    - A patch disables correctness checks, weakens output quality, or changes
      only launch parameters after the winner table is known.

- AC-4: Kernel-level bottlenecks use KernelPilot when appropriate
  - Positive Tests (expected to PASS):
    - For a specific slow CUDA/Triton/CuTe/CUTLASS/TileLang/torch.compile
      kernel, the profiler evidence shows SGLang is more than 1% behind and the
      target kernel or tightly scoped kernel family has at least 1% cumulative
      GPU-time share in the slow stage.
    - The loop serializes the KernelPilot handoff: either run KernelPilot
      before starting SGLang RLCR, or write
      `analysis/kernelpilot-handoff.md` and wait until the SGLang RLCR loop is
      no longer active before starting KernelPilot.
    - Before leaving SGLang RLCR for KernelPilot, the loop writes
      `humanize/model-loop-checkpoint.md` and
      `humanize/sglang-continuation-plan.md`.
    - The KernelPilot Humanize loop is capped at 10 rounds, then returns the
      best correct candidate or an explicit no-kernel-change result.
    - The standalone KernelPilot optimization repo records provenance, ledgers,
      correctness tests, benchmarks, and profiler evidence there.
    - Only the selected kernel candidate is ported back to SGLang, wired into
      the active model-serving path, and validated with the same model-level
      benchmark/profile.
  - Negative Tests (expected to FAIL):
    - KernelPilot is started for a lone SGLang kernel below 1% cumulative
      GPU-time share with no aggregated family above 1%.
    - A KernelPilot `setup-rlcr-loop.sh` is launched while the SGLang RLCR loop
      is still active in the same Codex session.
    - Experimental kernel mutations are made directly in SGLang before a clean
      standalone candidate proves correctness and speed.

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

- AC-8: Model-loop continuity survives KernelPilot handoff
  - Positive Tests (expected to PASS):
    - If the SGLang RLCR is suspended for KernelPilot, the checkpoint records
      the original benchmark winners, workload/SLA, SGLang commit, applied
      patches, current best SGLang result, remaining gap, profiler rows, and
      KernelPilot target repo.
    - After KernelPilot completes, a fresh SGLang continuation RLCR can be
      started from `humanize/sglang-continuation-plan.md` without relying on the
      old `.humanize/rlcr/<timestamp>` session.
  - Negative Tests (expected to FAIL):
    - The loop assumes the exact parent `.humanize/rlcr/<timestamp>` session can
      be safely resumed after a standalone KernelPilot RLCR.
    - KernelPilot results are ported back without rerunning the same model-level
      benchmark/profile and updating the ledgers.

## Path Boundaries

### Upper Bound (Maximum Scope)

Multiple minimal SGLang patches, optional standalone KernelPilot kernel work,
and repeated real-model benchmark/profile runs are allowed when needed to close
the measured gap.

### Lower Bound (Minimum Scope)

One profiler-backed SGLang patch plus real-model benchmark/profile
revalidation, unless the initial evidence proves no patch is needed.

### Allowed Choices

- Can use: SGLang source patches, guarded heuristics, existing fast-path
  selection, fusion or overlap fixes, model-specific runtime fixes,
  KernelPilot standalone kernel optimization for eligible hot kernels, focused
  tests, microbenchmarks, torch-profiler, Nsight Compute, and Nsight Systems.
- Cannot use: changing the fixed workload/SLA after seeing results, removing a
  competitor from comparison without a recorded unsupported reason, disabling
  correctness or tokenizer behavior, launching KernelPilot for sub-1% lone
  kernels, or claiming SOTA from smoke-only runs.

## Dependencies and Sequence

### Milestones

1. Preserve fixed baseline artifacts
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
   - If a kernel-local handoff is required, write the checkpoint and
     continuation plan before leaving SGLang RLCR.
   - Stop only under AC-7.

## Implementation Notes

- Keep Humanize local state under `.humanize/`.
- Keep benchmark/profile artifacts under `<artifact-root>`.
- Commit SGLang changes after each round summary.
- Mention exact changed files, commands, result deltas, and remaining risk in
  each Humanize round summary.
```
