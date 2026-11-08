# DeepGEMM histogram producer for LiteTopK decode

Reviewed on 2026-11-09 (Asia/Shanghai). This review covers the producer PR;
the [SGLang migration review](sglang-deepgemm-migration-review.md) is complete,
while the consumer reconciliation remains in progress.
No GPU execution, sanitizer, SASS inspection or performance rerun was done here.

## PR #94 — optional histogram of stored BF16 logits

- Link/title: [[SM100] Paged MQA logits: optional exact coarse score histogram for LiteTopK decode](https://github.com/sgl-project/DeepGEMM/pull/94).
- State: open, not merged; head `1b5f91a4e61a1a468a05afad4b7666137551df4b`,
  API base and actual parent `7c4a95f88db417500a51380b639b61137ca6196a` on
  `dev`. The API reports a merge conflict at this check.
- Coverage: all 983 lines of the complete twelve-file diff read, +617/-34;
  the full body, one issue comment, two inline comments, one empty COMMENTED
  review and the single commit message were read. Complete changed files
  were saved; surrounding host validation, scheduler, reduction and sanitizer
  paths were inspected. Current comparisons use DeepGEMM `dev`
  `909f68c19cc23c6afd9cc039c0493b00586b66b2`, inspected for this increment.
- Motivation: the paged MQA producer already holds the reduced scores in
  registers. Counting their coarse bins there lets a downstream selector
  avoid a separate histogram-building pass over the score matrix.
  **The producer still materializes BF16 logits.** This is a different
  memory contract from a score-and-top-k fusion that never writes that matrix.

Reviewed runtime files are `csrc/apis/attention.hpp`,
`csrc/jit_kernels/impls/sm100_mqa_logits.hpp`, `csrc/torch_library_api.cpp`,
both `deep_gemm/__init__.py` and `sgl_deep_gemm/__init__.py`, and
`deep_gemm/include/deep_gemm/{epilogue/coarse_histogram.cuh,impls/sm100_mqa_logits.cuh,scheduler/sm100_paged_mqa_logits.cuh}`.
The remaining files are `sgl_deep_gemm/README.md`, `run_tests.sh`,
`tests/test_attention.py` and `tests/utils.py`.

## API, numerical and lifetime contracts

The [host API](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/csrc/apis/attention.hpp)
adds `histogram=None` after `indices`. Admission requires CUDA architecture
major 10, 32 or 64 query heads, and a same-device contiguous int32 tensor of
shape `[rows, 1024]`, aligned to eight bytes. The existing paged path admits
pages of 32/64/128, MXFP8 or MXFP4, and requires varlen `indices` with
`next_n == 1`: multiple tokens per request are flattened into consecutive
rows with equal request IDs, rather than passed through a larger Q `next_n`.
Within a request run, the last row must have the longest context and the
rows must share the request's block table. Those scheduler assumptions are
not established merely by the tensor-shape checks.

The [Torch schema](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/csrc/torch_library_api.cpp)
declares the histogram mutation explicitly:

```diff
+          "Tensor(a!)? histogram) -> Tensor");
```

Counts are **added**, not overwritten. Zero the global histogram before an
independent production, or ensure its consuming selector has restored the
required state before reuse. A replayed standalone producer needs its zero
operation captured in the graph. Reusing an accumulating histogram forever
also violates the packed-counter bound; the two adjacent int32 counts are
combined into one uint64 atomic add and must not carry between words.
Ordinary sequential stream ordering or a valid PDL dependency must cover
both the scores and completed global histogram writes.

The [shared core](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh)
observes the result of the existing BF16 head reduction. It does not bin
an earlier FP32 accumulator:

```diff
                     const auto result = __hadd2_rn(__low2bfloat162(sum), __high2bfloat162(sum));
+                    if constexpr (Histogram::kEnabled)
+                        histogram_scores[i] = result;
```

Only live columns are counted, and NaNs are excluded. Tail columns remain
unwritten, so whole-allocation equality is not the contract. Bins descend
from high scores to low scores: below magnitude 16 they use FP16 round-to-
nearest exponent/top-four-mantissa encoding, then unit bins, saturating at
the ends. Negative unit-bin boundaries are lower-inclusive; signed zeros
share a bin. These are coarse bins, not a complete score/tie ordering.
The tests compare stored BF16 bits between the two producer variants;
that does not establish equality with an older FP32 producer or a complete
model's accuracy.

## Shared window and scheduler changes

The new [RowWindowHistogram](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/deep_gemm/include/deep_gemm/epilogue/coarse_histogram.cuh)
keeps two Q blocks in shared memory, with 16-bit per-row counters. Four rows
share each eight-byte bin cell; groups of four rows occupy 8 KiB. With
32/64 heads, `BLOCK_Q` is 4/2 and the window is 16/8 KiB respectively.
Warp-uniform paths handle pairs of BF16 scores when every lane's values are
in the fine range, unit range, or their union. Partial live columns, NaNs,
signed zero, tiny scores, positive infinity and the negative-16 boundary
take the scalar path. Packing pairs is an arithmetic optimization, not a
change to the reference bin definition.

All 384 math threads initialize the window. A task that leaves its row
window or would exceed its split budget synchronizes, publishes bin pairs
with global atomics, clears shared cells and synchronizes again. The final
math-thread barrier precedes the last flush. Review comments about opaque
zero tests and duplicate histogram types are addressed by the single
`RowWindowHistogram`/disabled `NoHistogram` policy and plain comparisons.

The [scheduler](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh)
now bounds a one-Q-block task when the histogram is enabled:

```diff
+        else
+            task.num_kv_splits = cute::min(remaining, current.num_q_blocks == 1 ? kMaxTaskSplits : kSplitsPerChunk);
```

`kMaxTaskSplits = 65535 / 384 = 170`; even if every live score lands in one
bin, at most 65,280 increments accrue before the next publish. Multi-Q-block
tasks still visit chunks of the shared KV ring. The
[host configuration](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/csrc/jit_kernels/impls/sm100_mqa_logits.hpp)
reduces the KV-stage count until the window fits shared memory. For MXFP4
D128 this changes eight stages to seven; FP4 chunks follow that stage count
so other Q blocks can reuse the resident KV. MXFP8 keeps three KV stages
and eight-split chunks. Histogram cost therefore includes scheduling and
pipeline changes, not just extra atomic instructions.

Without a histogram, the disabled policy removes the new hooks and leaves
the old split policy and stage counts. The paged kernel parameter block
still gains a pointer. The author's 49 identical generated-source/SASS
comparisons and 30 histogram variants with 128 registers/no spills are
reported evidence; this source review did not reproduce compiled binaries.

## Test coverage and performance limits

The [committed test](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/sgl_deep_gemm/tests/test_attention.py)
enumerates 128 cases: four format/head/page combinations, four score modes
and eight request/length shapes. It checks live-score bit identity, equality
to the FP32-definition reference histogram, and exactly doubled counts
after a second call without zeroing. This is not the full Cartesian product
of every supported head/page/format combination.

Beyond that sampled matrix, it retains 36 edge combinations (three layouts,
one through six rows per request, PDL off/on), nine forced-SM combinations
(1/7/100 SMs, including a 1M row), and twelve graph replays across two
formats. Graph capture includes `histogram.zero_()` and metadata production;
replays change lengths and request grouping while preserving compatible
page tables. The release profile selects four of the 128 matrix entries,
but still runs the separate edge/SM/graph loops. Under the
[sanitizer runner](https://github.com/sgl-project/DeepGEMM/blob/1b5f91a4e61a1a468a05afad4b7666137551df4b/sgl_deep_gemm/tests/test_sanitizer.py),
CUDA caching is disabled for this test and its graph section explicitly
skips. A sanitizer pass must not be presented as graph-replay coverage.

The author reports 128/128 matrix passes, zero memcheck/synccheck errors,
exhaustive bin arithmetic checks and a passing SGLang kernel suite on B200,
CUDA 13.0, Torch 2.14.1. Paginated workflow, check-run and status queries for
the exact final head return no records. These results therefore remain
author-reported; there is no independently inspected final-head CI log here.
The committed timing loop uses `bench_kineto` and does not zero the histogram
between timed producer calls. It is distinct from the body's external
cold-cache, interleaved single-replay experiment.

That experiment reports a 1 GiB flush and A/A spread of 0.98–1.01:

| Producer-only comparison | Reported cost of enabling the histogram |
|---|---|
| GLM-5.2 MXFP8, 383 cells | Geometric mean +2.28%; median +0.42 µs; maximum +9.84%. |
| V4.1 MXFP4, 142 cells, 1/2/3/4 tokens per request | Geometric mean +1.25/+4.18/+5.46/+5.38%. |
| V4.1, 5/6 tokens per request | Geometric mean +13.85/+14.54%; up to +21% in a power-capped cell. |

The earlier producer revision had a six-token Q tile. The current four-row
tile is reported 1.135×/1.114× slower than that revision at five/six tokens,
with 108 compared cells retaining identical logits. That tile optimization
is absent from this diff. Include producer overhead, metadata, selector and
any dtype conversion in a combined indexer comparison; neither a skipped
histogram pass nor producer-only timing establishes serving speedup.

## Current-source reconciliation

At the inspected `dev` revision, the
[Python API](https://github.com/sgl-project/DeepGEMM/blob/909f68c19cc23c6afd9cc039c0493b00586b66b2/sgl_deep_gemm/__init__.py)
has a compatibility `*args, **kwargs` wrapper routing old and new calls;
the modern private function and
[C++ API](https://github.com/sgl-project/DeepGEMM/blob/909f68c19cc23c6afd9cc039c0493b00586b66b2/csrc/apis/attention.hpp)
still do not accept `histogram`. Installing current `dev` is not sufficient
to obtain this proposal. Signature-based capability detection must also
account for that wrapper when porting the consumer.

The current [host configuration](https://github.com/sgl-project/DeepGEMM/blob/909f68c19cc23c6afd9cc039c0493b00586b66b2/csrc/jit_kernels/impls/sm100_mqa_logits.hpp)
uses four-split paged FP4 chunks on CC 10.7. The current
[shared core](https://github.com/sgl-project/DeepGEMM/blob/909f68c19cc23c6afd9cc039c0493b00586b66b2/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh)
derives the active KV ring from the chunk count, supports the 10.7 FP4 K128
shape, uses the allocated TMEM base rather than assuming zero, and changes
contiguous-mode reduction scheduling. Those changes are outside #94's base.
Preserve them when rebasing; the old B200 timings and no-histogram binary
comparison cannot be carried forward as evidence for that combined code.
