# DeepSeek V4 Public Day-0 Case Study

## Contents

- [Public Evidence Boundary](#public-evidence-boundary)
- [Pre-Merge Release Preparation](#pre-merge-release-preparation)
- [Mainline Support](#mainline-support)
- [Immediate Backfill](#immediate-backfill)
- [Post-Day-0 Repair Categories](#post-day-0-repair-categories)
- [Default Flips and Reverts](#default-flips-and-reverts)
- [Reusable PR Design Lessons](#reusable-pr-design-lessons)
- [Detailed Public Dossier](#detailed-public-dossier)

## Public Evidence Boundary

This case uses only the public `sgl-project/sglang` history. The two anchor PRs
were manually reviewed at immutable public heads:

- Evidence: https://github.com/sgl-project/sglang/pull/23882 | state: merged | head: 7978aa75e2c16db50f249aa25b9c5678abf6c7d2 | limitation: the 154-file mainline merge established the support spine but did not include every branch fix or later platform and feature combination
- Evidence: https://github.com/sgl-project/sglang/pull/24793 | state: merged | head: d2330b054a99218c6f348ba7f3855b40cedcafa3 | limitation: the next-day backfill closes identified omissions and expands tests; it is evidence that the first merge was not a complete terminal state

The detailed dossier contains 171 public PR cards. This case deliberately
links representative cards instead of copying the entire inventory.

## Pre-Merge Release Preparation

Public cookbook edits, hardware verification notes, and dedicated Docker
workflows appeared before the main runtime support merge. This is a useful
Day-0 pattern when handled carefully:

1. Draft the runnable recipes early so required flags, dependencies, images,
   parallel modes, and memory assumptions become explicit.
2. Mark the recipes as branch-bound or pre-release until the runtime spine
   merges.
3. Attach hardware observations to exact public revisions and commands.
4. Remove temporary release machinery once the normal mainline image covers
   the model.

The last step is visible in the
[release-Docker cleanup card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-27404---remove-deepseek-v4-release-docker-workflow).
Temporary packaging is a bridge, not a second permanent release channel.

## Mainline Support

The main PR adds far more than a model class. Its manually reviewed diff spans
154 files and introduces the core Day-0 surfaces:

| Surface | Mainline responsibility | Required proof |
| --- | --- | --- |
| Model/config | `deepseek_v4.py`, configuration detection, checkpoint mapping, and quantized expert setup | Config parse, weight coverage, missing/unexpected key audit |
| DSA attention | Dedicated attention backend, sparse indexing, metadata, prefill/decode paths | Reference parity across short/long prefill and decode |
| Compressed state | Memory pool and compressed-state structures with location and dtype rules | Allocation, update, reclaim, and cross-mode state parity |
| SWA | Sliding-window layer/cache behavior and mapping | Boundary lengths, eviction, prefix reuse, and mixed-layer transitions |
| mHC | Hyper-connection projections, normalization, and fusion hooks | Unfused/fused numerical parity |
| MTP | Next-N model and target/draft integration | Acceptance and committed-state parity |
| API protocol | Reasoning, tool calls, request protocol, and parser registration | Streaming/non-streaming and malformed/partial token tests |
| Quantization | FP4/FP8 loading and runtime dispatch | Scale-layout, dtype, hardware predicate, and fallback tests |
| Tests and launch | Sanity utilities, manual tests, CI dependencies, and recipes | Reproducible launch plus accuracy and API smoke |

The main PR also carries kernels and performance paths. Do not automatically
classify every kernel as Day-0-required. A correct fallback can move a kernel
to `performance-only`; a memory-capacity constraint may move it back into the
release cut with an explicit rationale.

## Immediate Backfill

The next-day PR is a model of why the release ledger must survive the main
merge. Its 15-file diff:

- restores missing branch changes;
- expands DeepSeek V4 function-call parsing, including self-closing zero-arg
  tool invocations;
- adjusts OpenAI protocol handling;
- completes loader behavior;
- strengthens B200/H200 registered tests; and
- adds a dedicated CI dependency installation path.

Classify these as `post-day0-fix` when they repair an advertised behavior, not
as fresh Day-0 proof. A high-quality Day-0 PR body should list known omissions
before merge and link the queued backfill node.

## Post-Day-0 Repair Categories

DeepSeek V4's public history shows which combinations need first-class cells
in the initial validation matrix.

### State and cache lifecycle

The
[HiCache card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-24691---unifiedtree-support-hicache-for-deepseek_v4)
extends hierarchical cache behavior. Compression V2, online compression with
MTP, mixed compression dtypes, SWA-location caching, and later allocation or
mapping fixes show that state must be tested through allocate, write, reuse,
speculative accept/reject, evict, and distributed transfer—not just one
forward pass.

### Parallelism combinations

The
[PP/PD card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-24704---feat-add-pipeline-parallelism-pp-and-pd-support-for-deepseek-v4)
and
[MTP context-parallel card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-24934---deepseek-v4-mtp-support-cp)
extend combinations beyond the first support cut. Later DP attention, MoE TP,
HiCache, PP, PD, CP, and idle-rank fixes demonstrate that each combination
needs ownership and synchronization invariants. Passing each mode separately
does not prove their Cartesian product.

### Quantization and platform ports

The
[NVFP4 MoE card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-25820---nvidia-support-nvfp4-moe-for-deepseek-v4)
and
[Ascend NPU card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-25144---npu-add-ascend-npu-support-for-deepseek-v4)
show that quantization and platform coverage arrive as distinct workstreams.
AMD and Intel paths in the full dossier reinforce the same rule: declare a
primary Day-0 hardware/quantization matrix and label every other cell
experimental or unsupported until its own correctness gate closes.

### Graph capture and speculative execution

The
[breakable decode-graph card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-25195---bcg-support-breakable-cuda-graph-for-deepseek-v4-dp-attention),
[breakable prefill-graph card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-30898---enable-breakable-prefill-cuda-graph-for-dp-attention),
and
[idle-rank repair card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-31705---deepseek-v4-fix-idle-rank-dummy-extend-sparse-prefill-crash-under-dp-breakable-cuda-graph)
form a dependency chain. Graph replay must refresh live metadata, represent
idle ranks safely, and cover prefill, decode, and MTP paths.

The
[speculative D2H-sync removal card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-30365---dsv4-remove-per-step-seqlen-d2h-from-speculative-to-make-overlap-scheduler-work)
shows a performance fix with correctness risk: eliminating host synchronization
changes when sequence metadata becomes visible. Validation must cover both
overlap enabled and disabled.

### Test stability versus product correctness

The
[flaky determinism-test card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-31125---disable-flaky-dsv4-flash-fp4-bcg-determinism-test-nondeterminism-from-30898-idle-rank-dummy-extend)
records a disabled test after graph changes. Disabling a flaky gate is neither
a model fix nor evidence of correctness. The ledger must keep the lost
assertion visible until a stable replacement closes the same risk.

## Default Flips and Reverts

The sparse-prefill sequence is especially reusable:

1. A merged change enables FlashMLA sparse prefill by default:
   [default-on card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-29775---deepseek-v4-enable-flashmla-sparse-prefill-by-default).
2. A revert proposal records concern but does not itself define shipped state.
3. A merged platform guard keeps sparse prefill off on ROCm while preserving
   the new default elsewhere:
   [ROCm guard card](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#pr-29982---amddeepseek-v4-fix-default-flashmla-sparse-prefill-off-on-rocmhip).
4. Dense and sparse prefill tests then need independent coverage.

Online compressed-state MTP also accumulated revert proposals after its merge.
The lesson is to describe effective final state, not merely count merged,
closed, or proposed PRs. For every default change, record:

- old and new behavior;
- supported platforms and shapes;
- explicit override behavior;
- rollback trigger;
- dense/reference fallback;
- test cells for default and override; and
- whether a revert merged, closed, or was superseded.

## Reusable PR Design Lessons

- Use a release-cut table with declared model, checkpoint, quantization,
  platform, parallelism, protocol, and speculative-mode cells.
- Land generic prerequisites early, but keep pre-merge recipes explicitly
  branch-bound.
- Keep a conflict ledger when a large support branch repeatedly rebases; every
  resolved subsystem needs a targeted regression test.
- Plan the main support PR and immediate backfill as a DAG, not as an
  expectation that one merge ends support work.
- Model compressed attention, SWA, speculative state, and graph metadata as
  lifecycles with ownership, refresh, and rollback rules.
- Audit combined parallel modes and idle-rank behavior; single-mode tests are
  insufficient.
- Treat temporary Docker workflows and disabled tests as debt with explicit
  removal or restoration conditions.
- Make default flips platform-aware and reversible.

## Detailed Public Dossier

Use the full manually reviewed public inventory for implementation-level
evidence:

- [English DeepSeek V4 public PR dossier](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md)
- [中文 DeepSeek V4 公开 PR 档案](../../../../model-pr-optimization-history/sglang/deepseek-v4/README.zh.md)

Re-check live PR and source state before applying these lessons to a new
release cut.
