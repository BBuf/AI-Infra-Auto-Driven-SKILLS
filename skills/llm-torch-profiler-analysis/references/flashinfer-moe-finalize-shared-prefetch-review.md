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
at TP4. The seven additional current bodies and their TP2/TP8 selection are
covered below. For attribution, record rank-local shared presence,
the actual `_tsds` program and producer dependency; inspect load/cache stalls
and compiled code before attributing a speedup to the source-level L2 hint.

## PR #8864 — TP8 deferred and TP2 large-token shared prefetch

- Link/title: [perf(cake_moe_finalize_allreduce_fusion): shared-expert prefetch programs at world sizes 8 and 2](https://github.com/flashinfer-ai/flashinfer/pull/8864).
- State: merged 2026-11-05 at 23:08:38 UTC; head
  `7be3d7fa65cf308a0df4ff712f85d03b9fa3aebc`, API base
  `f16f4ec95af5bbfcb5b2211aea9124f5f27144ef`, actual merge-base
  `6ff41626ee40fc1021504f163d7e9d8c6880e4ee`, merge
  `adf6addac9006420fda1a3ba89aa54ae1c56d220`.
- Coverage: full 8,335-line diff fetched; all unique changes reviewed across
  eleven files, +7,986/-68. All seven new complete files reproduce their
  same-head originals after the same three explicit inverse transformations
  used above. The 56 retained device files are byte-identical to #8697.
  Coverage includes all body prose, structured checks of 720 rows and
  intervals, two issue comments, three reviews, no inline comments and the
  single commit message.
- Reviewed files: four `*_ws2_o{110,111}_lts_kernel.cu` and three
  `*_ws8_o{110,111}_tsds_kernel.cu` additions, their binding and loader,
  and both dispatch test files. TP8 FP16/o110 has no `_tsds` file or registry
  slot. The registry grows from 56 to 63 programs per architecture.
- Motivation: apply the shared-row prefetch to eligible TP8 deferred and
  TP2 large-token workloads while retaining each original exchange and
  avoiding the TP8 FP16 ordinary-output variant reported to spill.

### Gates, fallback and preserved protocols

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/7be3d7fa65cf308a0df4ff712f85d03b9fa3aebc/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
extends the shared deferred table with three TP8 entries and restricts both
two-shot selection sites identically:

```diff
-    kernel = two_shot_deferred ? ((v_has_shared_expert != 0 && defer_shared != nullptr) ? defer_shared : two_shot_defer) : two_shot_now;
+    kernel = two_shot_deferred ? ((v_has_shared_expert != 0 && (world_size != 8 || top_k == 8) && defer_shared != nullptr) ? defer_shared : two_shot_defer) : two_shot_now;
```

At TP8 T≥768, only K8 with a shared expert can select `_tsds`. BF16 supports
both output profiles; FP16 supports only NVFP4. The null FP16/o110 slot
keeps `_tsd`. The source comment attributes that omission to spilling;
there is no compiler-resource report here to independently quantify it.
TP8 K4/shared stays `_tsd`, and K4/no-shared/non-quantized keeps its older
non-deferred exception. TP4's shared behavior is unchanged.

A separate `kLargeTokenSharedKernels` table has four TP2 entries. Inside the
existing K4 large-token branch, the assignment becomes:

```diff
-    kernel = large_token;
+    kernel = (v_has_shared_expert != 0 && large_token_shared != nullptr) ? large_token_shared : large_token;
```

Thus TP2/K4/shared uses `_lts` above T1280; T1280 itself does not qualify.
K4/no-shared still uses `_lt` above T512. K8 gains no large-token variant.
The [Python launcher](https://github.com/flashinfer-ai/flashinfer/blob/7be3d7fa65cf308a0df4ff712f85d03b9fa3aebc/flashinfer/jit/cake_moe_finalize_comm.py)
passes `SMALL_TOKEN_MAX=64`. The tested limits 16/64 do not affect these
large-batch choices. Do not generalize the body's independence claim to
arbitrary direct-FFI overrides: TP2's small branch still runs first, and
`small_token_max≥tokens` would select it. The binding does not clamp that
argument. TP8's first two-shot branch precedes small-token selection.

The [TP2 BF16 large-token body](https://github.com/flashinfer-ai/flashinfer/blob/7be3d7fa65cf308a0df4ff712f85d03b9fa3aebc/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o110_lts_kernel.cu)
issues the guarded L2 hint inside `step < interleaved_tokens`, after its
first two routing indices and before the two-route loop. It retains the
one-shot large-token pipeline and own-partial shared-memory ring. The
[TP8 BF16 deferred body](https://github.com/flashinfer-ai/flashinfer/blob/7be3d7fa65cf308a0df4ff712f85d03b9fa3aebc/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws8_o110_tsds_kernel.cu)
places the hint inside `local_tokens > step`, after four initial indices
and before the four-route loop. It retains two-shot reduce-scatter/gather,
the deferred RMS carry and the final drain. Both remain after PDL wait.
Each variant changes the shared-row load to `ld.global.nc.v4.b32` at its
original location, with dtype conversion and addition order unchanged.
The full-file comparisons cover both output epilogues and all seven bodies.
The shared prefetch does not convert TP2 to a two-shot protocol.

### Tests, mixed ranks and CI

The [CPU tests](https://github.com/flashinfer-ai/flashinfer/blob/7be3d7fa65cf308a0df4ff712f85d03b9fa3aebc/tests/cpu/comm/test_cake_moe_finalize_dispatch.py)
now read actual table slots by dtype/world/profile, including the intentionally
empty FP16 slot. The deferred test covers TP8 T767/768/769/1536/2048/4096,
TP4 boundaries, mixed TP4 shared presence and mixed TP8 shared presence/K.
The new large-token test covers T512/513/1280/1281/2048/4096, excludes K8
and TP4/TP8, and mixes `_lts`/`_lt` across two ranks at T1281/T2048.
Local execution of the original ASTs passes sixteen cases: nine tables,
their union, two early-boundary, two shared-deferred and two shared-large
cases. The large-token evaluator omits earlier two-shot dispatch and only
asserts those worlds never choose a large program; it is not a full dispatch
or occupancy model.

The [GPU test](https://github.com/flashinfer-ai/flashinfer/blob/7be3d7fa65cf308a0df4ff712f85d03b9fa3aebc/tests/comm/test_cake_moe_finalize_allreduce.py)
adds dtype to its name oracle and recognizes `_lts`. It adds no token values:
the existing TP2 T1280/T1281 and TP8 T767/T768 cover the immediate thresholds.
Every worker still iterates the same `_cases(world_size)`, so K and shared
presence are uniform across ranks; inverse indices remain `torch.arange`.
The bot's request for mixed-rank GPU execution is therefore an outstanding
coverage gap, not an implemented test. The retained test reuses workspaces
between cases, but that alone does not exercise mixed programs within one
call. The API wrappers, separate unified test, benchmark and timer remain
byte-identical to #8697.

Final-head [run 42116827323](https://github.com/flashinfer-ai/flashinfer/actions/runs/42116827323)
contains eleven jobs, with six successful AOT builds and a successful
[targeted H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/42116827323/job/127559592888).
Logs and JUnit agree: **29 CPU passes and all six GPU nodes skipped** for
insufficient visible device counts. Setup checked merge `00570cf51` of the
reviewed head into the API base; A10G/T4 scopes were empty. The summary
reports skip-all for the GPU file. Neither that green CI nor the maintainer's
brief LGTM proves numerical or mixed-rank execution of the new kernels.

### Measured shapes and current correspondence

The external timing protocol matches #8697's rank-min CUDA 13.4 experiment.
Fourteen published changed rows comprise six TP8/K8/shared T2048 shapes
and eight TP2/K4/shared T2048 shapes. The two TP8 FP16 ordinary-output rows
are controls because they retain `_tsd`.

| Published cohort | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| Changed 14 rows | 1.025 / 1.014–1.034 | 1.024 / 1.013–1.035 |
| Unchanged 706 rows | 1.000 / 0.981–1.021 | 1.000 / 0.974–1.020 |

All changed intervals lie above one. Controls have 315/339 below-one ratios
and 43/48 intervals entirely below one on B300/B200. From printed ratios,
the all-720 means are 1.000505/1.000155; the body reports 1.000/1.000.
Full-precision samples are not provided to reconcile that rounding-level
difference. Keep the approximately neutral whole-table result separate
from the changed-shape gains.

The body additionally reports six repeat TP8 measurements and eight new
TP2/K4/shared T1281 shapes. Those eight report means 1.021/1.021 and minima
1.012/1.013. Combined with the published fourteen, this is 28 measurements
of 22 distinct shapes, with means 1.023/1.023 and no reported regression.
The fourteen extra measurements lack individual rows. There is no timing
row at TP8 T768/T769 or a large custom small-token limit. The committed
rank-max harness remains a different experiment; these data supply no
serving throughput or model-accuracy result.

All 63 head device files, the binding, loader and both dispatch tests are
byte-identical to the current pin. This closes the current Cake indexed
finalize source lineage covered by these reviews. It does not extend the
numerical evidence or establish an automatic SGLang/vLLM caller: verify
`moe_finalize_backend="cake"`, the producer's inverse mapping, output
profile, shared presence and exact selected program in the serving trace.
