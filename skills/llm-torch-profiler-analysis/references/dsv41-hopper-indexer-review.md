# DeepSeek V4.1 Hopper indexer: current source and earlier proposals

Inspected on 2026-11-08. All three complete diffs, PR bodies and discussion
below were read; the recorded heads and bases were checked again. Current
callers were inspected at SGLang `c24285bde97c3e6424cd5d8cb35788889034d8ea`.
Upstream GPU results are author reports, not experiments rerun for this refresh.
The newer [Hopper ports](dsv41-open-pr-review.md) are separate proposals.

## Current profiling contract

The source/consumer distinction matters. `DeepseekV41Indexer` projects all
configured index heads on each TP rank. A candidate source publishes unordered,
`-1`-padded block IDs; a consumer selects within those blocks. Model dispatch
routes prefill and decode/target-verify to their respective backends.

- Hopper prefill currently gathers/dequantizes each request's visible K and
  materializes BF16 `[rows, heads, padded_K]` GEMM scores. The default scratch
  budget is **1 GiB**. K is padded to 64-row alignment for CUDA BF16 lengths
  at least 1024, and the returned scores are sliced to the logical length.
- The current 32-head CUDA BF16 score epilogue already fuses ReLU, weight
  multiplication and head reduction. Its four interleaved FP32 accumulators
  preserve the intended Torch order before the final BF16 rounding. Unsupported
  dtypes/layouts, ROCm, other head counts and widths at most one use Torch.
  The large GEMM intermediate still exists; a separate epilogue is not evidence
  that all its pointwise operations are separate launches.
- Hopper decode uses `fp4_index_logits_paged`. Up to 4096 query rows, its balanced
  grid partitions the total **visible** tiles across resident programs; larger
  batches use per-row workers. These scores have capacity-sized storage, but
  only the live prefix is initialized. Top-k must use device-side lengths.
  Do not read the unused tail as if every column were initialized to `-inf`.
- Current Hopper candidate consumers retain a context-width mask/score layout.
  Compact candidate scoring in #40860 is still pending. The current CUDA
  candidate graph-variant factory requires capability major at least 10;
  #40860 proposes admitting Hopper. DeepGEMM FP8/FP4 indexer dispatch remains
  SM100-family-specific at the frozen source.

Current source:
[model-side indexer](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py),
[model dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/deepseek_v4_backend.py),
[backend factory](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py),
[scoring and chunk budgets](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/v41_indexer/scoring.py),
[candidate publication/consumption](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/v41_indexer/dense_blocks.py),
[head reduction](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/dsv4/index_score_reduce.py),
[paged logits and finalization](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py),
[graph variants](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/graph_variants.py).

## PR #40160 — Closed grouped FP4-to-FP8 WGMMA prototype

- Link: https://github.com/sgl-project/sglang/pull/40160
- State: closed **without merge** on 2026-10-26 by the idle-PR policy bot;
  head `0041169143a5fdd29ac5361b95f459328d39b16f`,
  base `9784d5f979c9eaa5a6d9b31e596a7a73cb76dab8`.
  Closure is not a numerical rejection or evidence that the implementation shipped.
- Coverage: full 1,196-line diff, five files, +1166/-0; body and all three issue
  comments, with no reviews or inline comments. All five files are new. The body
  also claims a Triton-fallback change, which is absent from this final diff.
- Motivation: request-major speculative rows reuse K, but scoring each query
  independently repeats FP4 decoding. Hopper lacks native FP4 tensor-core MMA.
- Implementation: a strict BF16 `[B,64,128]` query/weight wrapper packs Q to FP4
  using RNE. CUDA expands E2M1 Q/K into E5M2 shared-memory tiles with a common
  per-head/per-key exponent, runs `MMA_64x64x32_F32E5M2E5M2_SS_TN`, restores
  scales and applies the BF16 dot/product/final-sum rounding boundaries.
  This fixed 64-head contract is not the 32-head compact scorer in #40860.
- The rectangular `auto` path skips wholly invisible 64-column K tiles and
  reuses decoded Q across two/four/eight tiles when shape thresholds permit.
  Explicit `persistent`/`pipeline` modes compute a visible-chunk prefix inside
  every CTA on every invocation, with no CPU length readback or global counter.
  At most 1024 groups and six rows per group use that path; larger shapes fall
  back to the single-tile kernel. The resident grid has at most twice the SM count.
- The pipeline has three K stages: packed tile `i+2` loads with `cp.async` while
  CUDA cores decode `i+1` and WGMMA consumes `i`. An unaligned table-row stride
  uses scalar loads. Invisible slots never dereference request-table sentinels;
  mask-only output writes remain necessary. Request-major grouping is a caller
  precondition: K addressing takes the first request ID of each group.

Exact kernel excerpt:

```diff
+    if (p.group_size > 6 || ceil_div(p.batch_size, p.group_size) > MAX_GROUPS) {
+      Sm90Fp4GroupedIndexerKernel<>::run(p);
+      return;
+    }
```

Reviewed files: `sm90_fp4_grouped_indexer/{entry.cuh,kernel.cuh,params.h}` under
`python/sglang/kernels/jit/csrc/`, the Python wrapper under
`python/sglang/kernels/ops/attention/dsv4/`, and registered
`test/registered/kernels/ops/attention/test_sm90_fp4_grouped_indexer.py`.
There is **no production model dispatch change**. These paths are absent from
the frozen main tree; the current Hopper caller uses paged Triton logits.

The three registered tests compare against Triton, poison invisible slot
entries, and replay all three modes with changing visibility, empty groups and
an incomplete final group. The body additionally reports 23 shape configurations,
large-group fallbacks and Compute Sanitizer checks; those broader harnesses are
not added by this diff. Random production-like inputs do not establish a contract
for arbitrary FP4 block-exponent spreads or nonfinite values.

H20 operator tables separate logits-only timing from Q/slot preparation. One
48-row, heavily capacity-padded case reports 3.77x logits-only and 11.57x including
preparation, against the earlier visibility-pruned Triton revision. Small/short
cases can favor Triton, and fully visible cases can favor `auto`; the pipeline
is not a universal winner. Its reported ptxas output has 20-byte spill loads and
stores and 105600 bytes of shared memory, unlike the spill-free simpler variants.
No model-accuracy or end-to-end TPOT result is established by this prototype.

## PR #42365 — Pending fused dense prefill scoring

- Link: https://github.com/sgl-project/sglang/pull/42365
- State: open, updated 2026-10-03;
  head `71bbd5402845f6651080339814c266d505dd9bab`,
  base `1093c501dfef0789b46fc3afe44949e35867a11a`.
- Coverage: full 361-line diff, five files, +297/-3; body read, with no issue
  comments, reviews or inline comments.
- Motivation: remove the BF16 `[rows, heads, K]` intermediate from dense prefill
  scoring, including source layers that cannot consume earlier candidate blocks.
- Implementation: one Triton program covers a 64-query by 128-key output tile,
  loads shared dequantized K, loops over heads and accumulates the weighted scores
  in FP32 registers. Each dot and weighted product is rounded through BF16;
  the final head sum is rounded again, and causal masking writes `-inf`.
  `num_warps=4`, `num_stages=2` are the H200-selected configuration.
- `_score_chunks` uses `lc * 16` bytes per row for its budgeting heuristic,
  allowing the FP32 output one quarter of the configured budget before later
  copies. This is not a whole-model peak-memory bound. Its new env flag defaults
  on in the PR, while `SGLANG_DSV41_TORCH_PREFILL_INDEXER=1` forces the oracle.
  The actual gate is `q.is_cuda`, not a capability-9 check; do not describe the
  code as enforcing Hopper-only dispatch.

Exact dispatch excerpt:

```diff
+    fused = (
+        q.is_cuda
+        and envs.SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES.get()
+        and not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
+    )
```

Reviewed files: attention kernel inventory `__init__.py`, new
`dsv4/fp4_indexer_prefill.py`, `srt/environ.py`, `v41_indexer/scoring.py`, and
`test/registered/kernels/ops/attention/dsv4/test_fp4_indexer_prefill.py`.
Neither the flag nor the new kernel is present in frozen main. A current port
must preserve the newer padded-K slicing and row-budget calculation. Main also
already has the separate fused 32-head epilogue described above, so the PR's
older multi-pointwise-launch baseline does not describe all current work.

Tests use 32 heads, non-tile-multiple shapes, a constrained 1 MiB budget and an
oracle-flag dispatch guard. They require at least 99% bit-equal visible scores
and 99% mean top-k overlap, with a normalized maximum score-error bound; they
do **not** prove identical selections for every row. The head accumulation
order differs from the current four-accumulator epilogue.

The author reports H200 scoring speedups of 4.37–4.61x for 8192 query rows and
100K–400K keys, and scratch reduction from 3.0 to 0.5 GiB for the largest case.
TP4 cold-prefill TTFT reports are 22.6→20.1 s at 190K tokens and 66.2→38.7 s at
400K. Those runs explicitly use a **2 GiB** score budget; current default is
1 GiB. These are historical branch measurements, not current-main predictions
or a model-accuracy evaluation. The PR body lists failed CI runs and provides
no successful current-head upstream-CI qualification.

## PR #40860 — Pending compact candidate scoring and selection

- Link: https://github.com/sgl-project/sglang/pull/40860
- State: open, updated 2026-10-15;
  head `b96b21b9546ba84aa26301bfa2531d2623b2ebe4`,
  base `33618d4f00cb5c0b3c2182bca2561737325dfd2a`.
- Coverage: full 1,684-line diff, ten files, +1255/-61; body and all eight issue
  comments, with no reviews or inline comments. The discussion's proposed extra
  sort was withdrawn after recognizing the existing unordered publication
  contract; it must not be presented as an accepted correctness fix.
- Motivation: source layers already publish compact block IDs, but Hopper
  consumers still traverse context-width score space. Carry compact coordinates
  through scoring and top-k without materializing per-query gathered K.
- Prefill: a guarded BF16 `[R,32,128]`, block-size-8 Triton path loads candidate
  positions directly from shared, already-dequantized per-request K. It preserves
  BF16 dot/product/sum rounding, emits `[R, blocks*8]` FP32 logits and masks
  padding/causality. Unsupported shapes retain Torch. Row chunking budgets
  `compact_width * 32` bytes for scores and selection temporaries.
- Decode/verify: `fp4_index_logits_candidates` follows block IDs through the
  request table and reuses the existing FP4 tile calculation. The source and
  FullTopK paths remain dense; compact consumers omit the full-width mask.
  Compact rows initialize all score columns, while logical `lens` remains the
  causal bound and separate `score_lens` is the complete compact scan width.
- Ragged/paged Top-K v2 selects compact columns. The shared finalizer filters
  invalid scores, restores `block*8 + offset`, sorts **logical** positions, maps
  slots and writes `-1` padding. Prefill can scatter into specified output rows
  or write raw indices only, without reading a request table.

Exact scan-length excerpt:

```diff
+            # Valid candidates may be interspersed with padding/causal holes.
+            score_lens = torch.full_like(lens, candidate_blocks.shape[1] * 8)
+            plan = plan_topk_v2(score_lens) if compute_topk_plan else None
```

The factory enables compact consumers on capability major 9 with block size 8.
Graph variants extend the existing CUDA gate from major 10 to 9: ordinary
decode can use four variants, but DSpark target verify retains causal scoring
and has only unfiltered/filtered variants. Its sequence bound includes the
captured verify width; unknown bounds choose filtered execution. This graph
policy change is distinct from replacing a score kernel.

Reviewed runtime files: `candidate_bf16_mqa.py`, `fp4_indexer.py`,
`v41_indexer/{__init__.py,dense_blocks.py,scoring.py}`, and `graph_variants.py`.
Reviewed tests: `test_candidate_bf16_mqa.py`,
`test_dsv41_triton_candidate_indexer.py`, `test_fp4_indexer.py`, and
`unit/layers/test_dsv41_candidate_graph_variants.py` under `test/registered/`.
Current main still lacks compact scoring and the Hopper graph gate. Since this
PR's base, prefill gained flattened K-slot inputs, optional CP request rows,
row sharding and padded K; integration must adapt those contracts, not copy the
older `prefill_requests(req_to_token=...)` interface blindly.

Tests cover exact dyadic cases, signed-weight cancellation bounds, strided
inputs, empty/invalid blocks, causal holes before valid candidates, ratios 1/2,
tail slicing, raw-only outputs, untouched scatter rows, output padding,
unsupported-shape fallback, and changed-input CUDA-graph replay. They also
exercise FullTopK DeepSelect versus Top-K v2 without imposing a false output
ordering requirement on DeepSelect. Approximate random-score comparisons and
tie-tolerant selection checks do not imply general bitwise model equivalence.

The author reports DeepSeek-V4.1-Flash on real 1P1D, eight H20-3e GPUs per node,
TP8/EP8/DP1/CP1 and DSpark block5. Prefill TTFT reductions range from 5.09% at
32K to 27.63% at 256K; the dedicated 16K repeat is effectively unchanged.
Decode throughput gains range from 8.11% to 29.96% at BS8 over 16K–128K.
GPQA Diamond has 180/198 correct for each branch, but truncations rise from
three to six. One stochastic sample per question is not proof of equal accuracy.
The latest discussion repeats those H20 results and asks for CI; the body still
leaves current-head upstream CI unchecked. This refresh did not reproduce them.
