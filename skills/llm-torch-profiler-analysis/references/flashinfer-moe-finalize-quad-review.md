# FlashInfer TP2 quad-route finalize and generated-source cleanup

Reviewed on 2026-11-09 (Asia/Shanghai), following the
[small-token and early two-shot reviews](flashinfer-moe-finalize-prefetch-review.md).
Current comparisons use `31c146b16789f112e9b9861bd79247d4701e437a`.
No GPU execution, SASS comparison, sanitizer or benchmark was run here.

## PR #8503 — four routes per round for eligible TP2 batches

- Link/title: [perf(cake_moe_finalize_allreduce_fusion): quad-route programs for batches of up to 142 tokens at world size 2](https://github.com/flashinfer-ai/flashinfer/pull/8503).
- State: merged 2026-11-03 at 14:09:02 UTC; head
  `1fe8a4193326d7ab46592f07773f76ec666b113f`, API base
  `791a9a8b092925481f1cbeeea6b581a4ce64e046`, actual merge-base
  `be78c804cc3c236e213d77315759290134e1d7d2`, merge
  `283300b79da41aeaad237a1c53c2db68c29a4e6c`.
- Coverage: full 8,694-line diff fetched; all unique changes reviewed across
  56 files, +4,621/-1,370. The 48 retained base bodies match #8321. Explicit
  full-file transformations account for their source cleanup without other
  byte changes. One complete BF16 quad body and each distinct new region
  were inspected; all four new bodies are covered by complete prefix,
  routing, shared-load/addition and exchange/epilogue comparisons against
  retained programs, with strict local-temporary bijections. All body prose,
  structured checks of 720 rows and confidence intervals, two issue comments,
  three reviews, no inline comments and the single commit message are covered.
- Reviewed files: four `*_ws2_o{110,111}_quad_kernel.cu` additions and 48
  regenerated device files under `csrc/cake_moe_finalize_allreduce_fusion`;
  its binding, `flashinfer/jit/cake_moe_finalize_comm.py`, and CPU/GPU dispatch
  tests. The registry grows from 48 to 52 programs per architecture.
- Motivation: use four-route gathering and early shared-expert loads for
  intermediate TP2 batches that previously selected the paired persistent
  program, while preserving its one-shot exchange and numerical order.

## Dispatch, occupancy and source cleanup

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/1fe8a4193326d7ab46592f07773f76ec666b113f/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
adds four entries to `kW2QuadKernels`, with null TP4/TP8 entries, and inserts:

```diff
+  } else if (w2_quad != nullptr && tokens <= kW2QuadMaxTokens) {
+    // Keyed on the token count and the world size alone (the table holds world size 2 only).
+    kernel = w2_quad;
```

`kW2QuadMaxTokens` is 142. Small-token selection and K4 large-token selection
remain earlier; persistent remains the final fallback. At the default
small-token limit 64 and sufficient capacity for the T71 small extension,
quad covers TP2/K4/no-shared T65–142, and T72–142 for the other classes.
A smaller limit exposes earlier K4/no-shared batches. If the small extension
fails its capacity check, other classes can also fall through to quad below
T72. TP4/TP8 do not acquire a quad program.

The new kernels request `__launch_bounds__(224, LAUNCH_MIN_BLOCKS)` with
`LAUNCH_MIN_BLOCKS=4` and four-CTA clusters. This constrains compilation;
it is not a proof of four resident CTAs on every device. The final launch
still uses `min(tokens, cluster_capacity(kernel))`, where capacity is the
minimum of per-SM residency-derived capacity and
`cudaOccupancyMaxActiveClusters`. Tokens advance by the launched cluster
count. The fixed T142 selection cutoff therefore does not guarantee
single-wave execution on every supported GPU or resource configuration.

The 48 regenerated bodies delete only definitions proven unused in each
file, simplify redundant address parentheses, change hexadecimal letter
case and a prologue comment, and join adjacent identical namespace blocks.
They replace the inline `laneId()` helper's volatile assembly with direct
non-volatile `asm("mov.u32 ... %%laneid")`. These explicit transformations
reproduce all 48 complete head files exactly; no additional temporary
renaming is needed. The author's claim of identical SASS/resource usage on
SM100/103 is not independently established by that source comparison.

## Route order and shared-expert lifetime

The [BF16 plain-output body](https://github.com/flashinfer-ai/flashinfer/blob/1fe8a4193326d7ab46592f07773f76ec666b113f/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o110_quad_kernel.cu)
retains TP2's prologue and post-publication exchange/epilogue modulo local
temporary names. Its route core matches the retained TP4 four-route core:
four initial inverse indices, four next-round indices prefetched while the
current round is processed, and sequential `qa/qb/qc/qd` accumulation.
K4 uses one route round and K8 two. Each route's scaled value and each
accumulator update retain activation-dtype rounding; gathering four routes
does not replace those additions with an unordered FP32 sum.

For each token, the new body reads and converts the shared-expert row before
gathering routes, guarded by `has_shared_expert`. That read remains after
`griddepcontrol.wait` and the sampled current/clear words. It retains the
converted values until after routed scaling, where the shared contribution
is added and rounded at the old arithmetic position. The address and
conversion match the prior shared-row load; only its lifetime is extended.
The [FP16 body](https://github.com/flashinfer-ai/flashinfer/blob/1fe8a4193326d7ab46592f07773f76ec666b113f/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_float16_ws2_o110_quad_kernel.cu)
preserves its FP16 conversions under the same region comparisons.

The TP2 exchange suffix is retained for both output profiles: publish to
both ranks, clear the dirty ring, poll the two rank payloads with lookahead,
sum in rank order, add residual and perform ordered cluster RMS exchange.
The plain and NVFP4 epilogues keep their respective rounding and scale
layout. One-shot signed-zero canonicalization is retained; this is not the
two-shot BF16 escape protocol. Workspace counters, ring rotation and PDL
completion are unchanged. Source correspondence is not a fresh GPU parity
or model-accuracy result.

## Negative routes and the test gap

Unlike the guarded ordinary loads in the plain-output program, the
[BF16 NVFP4 body](https://github.com/flashinfer-ai/flashinfer/blob/1fe8a4193326d7ab46592f07773f76ec666b113f/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o111_quad_kernel.cu)
uses predicated `ld.global.nc.v4.u32` row loads and `ld.global.nc.u16` scale
loads. Negative row indices are clamped to row zero for the address;
predicates test the route slot against `top_k`, not whether that index is
nonnegative. The first predicate uses a copied `top_k_opaque` through an
empty read/write assembly operand. Accumulation still tests each index
against zero, so these extra reads contribute nothing for negative routes.
This is a memory-access/performance distinction, not evidence of incorrect
outputs for valid allocated inputs.

The bot review identified this for BF16 o111. Inspecting the
[FP16 NVFP4 body](https://github.com/flashinfer-ai/flashinfer/blob/1fe8a4193326d7ab46592f07773f76ec666b113f/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_float16_ws2_o111_quad_kernel.cu)
confirms the same eight row/scale predicates. Their entire load blocks
match after the dtype conversions and local-temporary renaming. This
behavior persists at the current pin; the review suggestion was not
incorporated into these files.

The [CPU tests](https://github.com/flashinfer-ai/flashinfer/blob/1fe8a4193326d7ab46592f07773f76ec666b113f/tests/cpu/comm/test_cake_moe_finalize_dispatch.py)
add the quad table to registry coverage and check the translated selector
at limits 16/64. Executing their original ASTs with decorators removed and
a local source/module provider passes ten cases: seven tables, their union
and two quad-policy cases. The latter execute 80 assertions across routing
classes and T64/65/71/72/141/142/143 plus `small_token_max+1`, and exclude
quad at TP4/TP8 T100. Capacity is fixed to 71; the evaluator omits earlier
two-shot selection. This verifies the inspected predicates, not real
occupancy, all selector behavior or GPU output.

The [GPU test](https://github.com/flashinfer-ai/flashinfer/blob/1fe8a4193326d7ab46592f07773f76ec666b113f/tests/comm/test_cake_moe_finalize_allreduce.py)
adds T142/T143, growing TP2 from ten to twelve token values. TP4/TP8 counts
are unchanged; T141 is a CPU boundary assertion and reported timing shape,
not a newly added GPU test value. The test still constructs inverse indices
with `torch.arange`, including at the current pin. Thus these cases do not
exercise permuted or negative routes. The bot's non-identity-input suggestion
is valid for this test, but is not an implemented test improvement. Existing
kernel-name assertions are conditional on PDL, weight bias and the small
extension's device gate. Public APIs, the separate unified MoE test, committed
benchmark and timer are byte-identical to #8321.

Final-head [run 41637898764](https://github.com/flashinfer-ai/flashinfer/actions/runs/41637898764)
contains ten jobs including six successful AOT builds. Setup and summary
checked merge `7b1a8e9c9` of this head into its API base. H100 was disabled,
all A10G/T4 targeted scopes were empty, and no JUnit artifacts were found.
There was no targeted CPU or GPU test job. An earlier summary failed while
waiting for a cancelled run, with its own setup skipped. The empty maintainer
approval and CODEOWNERS-gap attestation do not supply numerical evidence.

## Performance evidence and current applicability

The external CUDA 13.4 experiment uses interleaved previous/new Cake runs,
rank-min CUPTI kernel times and median paired speedup ratios with 95%
bootstrap intervals. Its 720-row shape inventory matches the prior table.

| Published cohort | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| Changed 32 TP2 T128 rows | 1.041 / 1.010–1.070 | 1.043 / 1.001–1.069 |
| Unchanged 688 rows | 1.000 / 0.974–1.025 | 0.999 / 0.984–1.018 |
| All 720 | 1.002 / 0.974–1.070 | 1.001 / 0.984–1.069 |

All 32 changed ratios exceed one, but B200 FP16/K4/PDL-off/NVFP4/no-shared
has an interval of 0.995–1.017 around 1.001. The other changed B200 intervals
and all changed B300 intervals lie above one. Controls contain 278/344
below-one ratios on B300/B200, including 37/48 intervals entirely below one.
Preserve this variability rather than inferring universal improvement.

The body separately reports 176 default-limit changed measurements:
eight each at T65/T71 for K4/no-shared, and 32 each at T72/96/128/141/142.
It reports means 1.038/1.041, minima 0.996/0.985 and one/two regressions on
B300/B200. Only the 32 T128 rows are individually published; the other 144
cannot be recomputed from this table. A separate eight-shape custom-limit
cohort at T64/K4/no-shared reports 1.013/1.019. Neither cohort covers every
changed token count or capacity-limited configuration. The unchanged
committed rank-max TRT/Cake/TRT harness uses a different timing protocol;
these operator results do not establish serving throughput or accuracy.

All 52 historical head kernel files are byte-identical to their counterparts
at the current pin, closing the intervening generated-source comparison for
those programs. The quad table, T142 constant and selection branch also
match exactly. Eleven additional current device bodies remain outside this
review. When profiling quad finalize, record the selected program, actual
cluster capacity, inverse-index distribution and output profile; verify
compiled load placement before attributing time to the proposed prefetch.
