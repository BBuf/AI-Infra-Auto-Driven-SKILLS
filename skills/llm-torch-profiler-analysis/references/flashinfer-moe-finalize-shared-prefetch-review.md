# FlashInfer shared-expert prefetch in MoE finalize

Reviewed on 2026-11-09 (Asia/Shanghai), following the
[TP2 quad-route review](flashinfer-moe-finalize-quad-review.md).
Current comparisons use `31c146b16789f112e9b9861bd79247d4701e437a`.
No GPU execution, SASS inspection or new performance measurement was run here.

## PR #8697 — deferred TP4 shared-expert prefetch

- Link/title: [perf(cake_moe_finalize_allreduce_fusion): deferred two-shot programs with a shared expert at world size 4](https://github.com/flashinfer-ai/flashinfer/pull/8697).
- State: merged 2026-11-04 at 20:04:24 UTC; head
  `d397d5e368a27ae2b632b35a13c205fd7bc6f049`, API base
  `40f281b5c01a9133931c89d5f8f6d3c29367b63d`, actual merge-base
  `beeb23bc0ac24effce80b23aec4a84ae8848df56`, merge
  `7b23deef3b118479a624956c1e7f7459cdd0de43`.
- Coverage: full 5,404-line diff fetched; all unique changes read across
  eight files, +5,241/-13. Each new complete device file was compared with
  its same-head deferred counterpart; removing the exact prefetch block,
  restoring the original shared load and renaming the entry point reproduces
  that counterpart byte-for-byte. All 52 retained bodies match #8503.
  Coverage includes all body prose, structured checks of all 720 table rows
  and intervals, two issue comments, two reviews, no inline comments and the
  single commit message.
- Reviewed files: four new `*_ws4_o{110,111}_tsds_kernel.cu` files, the binding
  in `csrc/cake_moe_finalize_allreduce_fusion`,
  `flashinfer/jit/cake_moe_finalize_comm.py`, and the CPU/GPU dispatch tests.
  The loader grows from 52 to 56 programs per architecture, SM100a/SM103a.
- Motivation: overlap an L2 prefetch of the shared-expert row with routed
  expert processing in large TP4 batches, preserving the existing deferred
  two-shot exchange, arithmetic order and workspace contract.

## Selection and memory operations

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/d397d5e368a27ae2b632b35a13c205fd7bc6f049/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
adds `kTwoShotDeferSharedKernels` with four TP4 entries and null TP2/TP8
entries. Both two-shot selection sites change identically:

```diff
-    kernel = two_shot_deferred ? two_shot_defer : two_shot_now;
+    kernel = two_shot_deferred ? ((v_has_shared_expert != 0 && defer_shared != nullptr) ? defer_shared : two_shot_defer) : two_shot_now;
```

At this revision, TP4 T≥1536 with a shared expert uses `_tsds`; the same
shape without one uses `_tsd`. Both K4/K8 and both output profiles are
eligible. This branch precedes small-token selection, so changing
`small_token_max` does not change this large-batch choice. TP2 and TP8 keep
their earlier programs, including TP8's K4/no-shared/non-quantized deferred
exclusion. Rank-local shared-expert presence can differ because `_tsds` and
`_tsd` retain the same messages, workspace and cross-rank summation order.

In the [BF16 ordinary-output body](https://github.com/flashinfer-ai/flashinfer/blob/d397d5e368a27ae2b632b35a13c205fd7bc6f049/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o110_tsds_kernel.cu),
each active publish step reads the initial four inverse indices, then adds:

```diff
+            if (has_shared_expert != 0) {
+                if ((tid & 7) == 0) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(shared_expert_output + (published_token * 7168 + cluster_thread * 8)))); }
+            }
```

This occurs after `griddepcontrol.wait`, inside `local_tokens > step`, and
before the route loop. With 224 threads per CTA and four CTAs per cluster,
one in eight threads issues a hint: addresses are spaced by 128 bytes for
the 7168-element two-byte row. The hint is not a synchronization operation
or proof that the data will still reside in L2 at consumption. It retains
the existing producer-readiness and stable-input requirements.

The actual shared-row load stays after routed scaling, at its old arithmetic
position. The same file replaces the ordinary `uint4` load with:

```diff
+                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
+                            : "=r"(_vld_8[_blk].x), "=r"(_vld_8[_blk].y), "=r"(_vld_8[_blk].z), "=r"(_vld_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)));
```

The [BF16 NVFP4 body](https://github.com/flashinfer-ai/flashinfer/blob/d397d5e368a27ae2b632b35a13c205fd7bc6f049/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o111_tsds_kernel.cu)
uses the same change with temporary suffix 4. The
[FP16 ordinary](https://github.com/flashinfer-ai/flashinfer/blob/d397d5e368a27ae2b632b35a13c205fd7bc6f049/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_float16_ws4_o110_tsds_kernel.cu)
and [FP16 NVFP4](https://github.com/flashinfer-ai/flashinfer/blob/d397d5e368a27ae2b632b35a13c205fd7bc6f049/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_float16_ws4_o111_tsds_kernel.cu)
programs preserve their own conversion blocks. The full-file comparisons
prove that rounding, shared addition, deferred carry/drain, BF16 escape
handling, reduction, RMSNorm, quantization and epoch updates otherwise retain
their corresponding source. They do not establish compiled instruction
placement or fresh numerical parity. Unlike TP2 quad's early converted
shared-row load, this change does not move that converted value's lifetime
across route processing.

## Tests and actual CI coverage

The [CPU test](https://github.com/flashinfer-ai/flashinfer/blob/d397d5e368a27ae2b632b35a13c205fd7bc6f049/tests/cpu/comm/test_cake_moe_finalize_dispatch.py)
extends table/registry coverage and translates the two identical C++
selection expressions. Its new test covers limits 16/64; TP4
T1535/1536/1537/2048/4096 and early T72/128; TP8 deferred boundaries and
its exclusion; TP2 non-selection; and five mixed shared-presence rank sets.
The new test executes 344 assertions and 704 scalar program evaluations
across the two limits. These are selector checks with fixed capacity 71,
not mixed-rank communication executions.

Locally, the original AST functions, with decorators removed and the source
provider replaced, pass 13 cases: eight tables, their union, two retained
early-boundary cases and two new shared-policy cases. The
[GPU test](https://github.com/flashinfer-ai/flashinfer/blob/d397d5e368a27ae2b632b35a13c205fd7bc6f049/tests/comm/test_cake_moe_finalize_allreduce.py)
updates only expected names and comments; its token lists, input construction
and numerical checks are unchanged. TP4's retained list includes T1535/T1536
but not T1537/T2048/T4096. Public APIs, the separate unified test, benchmark
and timer are byte-identical to #8503.

Final-head [run 41886923504](https://github.com/flashinfer-ai/flashinfer/actions/runs/41886923504)
has eleven jobs, including six successful AOT builds and a successful
[targeted H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/41886923504/job/126774882084).
Setup checked merge `b79d4d617` of the reviewed head into the API base.
The job log and downloaded JUnit artifact agree: **26 CPU tests passed;
all six GPU test nodes skipped**, citing requirements for 2, 4 or 8 CUDA
devices. The summary explicitly reports skip-all for the GPU file. A10G/T4
targeted scopes were empty. The green job therefore establishes CPU dispatch
coverage, without a numerical execution of these kernels. The maintainer's
LGTM includes a readability nit about nested ternaries; the rate-limited bot
notice and CODEOWNERS-gap attestation add no kernel correctness evidence.

## Timing evidence and current scope

The external experiment uses CUDA 13.4, default small-token limit 64,
interleaved previous/new runs, rank-min CUPTI times and median paired ratios
with 95% bootstrap intervals. Recomputed from the printed ratios:

| Published cohort | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| Changed 16 TP4 T2048 shared rows | 1.028 / 1.022–1.037 | 1.030 / 1.023–1.036 |
| Unchanged 704 rows | 1.001 / 0.976–1.033 | 1.000 / 0.985–1.016 |
| All 720 | 1.002 / 0.976–1.037 | 1.001 / 0.985–1.036 |

All 16 changed intervals lie above one on both GPUs. The unchanged cohort
has 271/281 below-one ratios and 36/35 intervals entirely below one on
B300/B200. The body also reports a second independent measurement of the
same 16 changed shapes: 32 measurements, not 32 unique shapes. Its aggregate
means are 1.028/1.030 and minima 1.022/1.023, with no reported regression;
the second set lacks individual rows. No timing row measures the T1536
threshold or the additional CPU boundary sizes. The unchanged committed
rank-max TRT/Cake/TRT harness uses a different protocol. These operator
measurements do not establish serving throughput or model accuracy.

All 56 historical device files are byte-identical to their counterparts at
the current pin. Current TP4 entries and its T1536 rule retain this behavior;
the selector's later `world_size != 8 || top_k == 8` conjunct is always true
at TP4. Seven additional current bodies and their TP2/TP8 selection require
the subsequent review. For attribution, record rank-local shared presence,
the actual `_tsds` program and producer dependency; inspect load/cache stalls
and compiled code before attributing a speedup to the source-level L2 hint.
