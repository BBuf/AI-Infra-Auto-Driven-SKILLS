# FlashInfer indexed finalize: token scheduling and resident clusters

Reviewed on 2026-11-09 (Asia/Shanghai). Historical evidence below is for
FlashInfer #7392. Current comparisons use
`31c146b16789f112e9b9861bd79247d4701e437a`. No GPU capture, benchmark,
sanitizer or generated SASS comparison was rerun in this audit.

Read the [indexed finalize contract and earlier lineage](flashinfer-moe-finalize-source-review.md)
for input layouts, per-addition rounding, workspace ownership and the
public-only payload bound. This remains distinct from the
[dense expert-major reduction API](flashinfer-moe-collective-source-review.md).

## PR #7392 — small/large token programs and ordered RMS exchange

- Link: https://github.com/flashinfer-ai/flashinfer/pull/7392
- State: merged 2026-10-26; head
  `682578cfa24b924a5fb9e32c1b04597ed0602e7f`, API base
  `407ca71f72f263085b510b7b97a4865638a58322`, merge-base
  `2ce15391b7552d457708597eeecb2e3bbd2fd577`, merge
  `ee376e97371400db429d03dbaaa583a820646cfd`.
- Coverage: all 38 changed files, 42,390 diff lines, +40,442/-1,533;
  body prose and all 720 timing rows through structured cohort/ratio checks;
  two issue comments, two reviews, no inline comments and five commit messages.
  Generated-source coverage combines complete canonical bodies, every
  world-size/schedule difference, and full-file comparisons of the explicitly
  inspected dtype and quantization transformations. This is source coverage,
  not a claim of 32 independent GPU runs.
- Reviewed files: all 32 new device programs, removed template, complete
  `cake_moe_finalize_allreduce_fusion_binding.cu`, `.clang-format`,
  `flashinfer/jit/cake_moe_finalize_comm.py`, and both changed tests under
  `tests/comm/test_cake_moe_finalize_*.py`. The deleted template matches the
  fully reviewed #6166 head. The indexed public function, unified finalize
  tests and committed benchmark match the reviewed #5870 versions. Relevant
  timing functions differ only in their deprecation docstrings; other
  communication APIs in their enclosing files have changed independently.
- Motivation: shorten small-batch expert gathering, hide waiting at TP2,
  overlap publication/cleaning/reduction at large TP4/TP8 batches, and keep
  more useful communication work resident without oversubscribing clusters.

### Dispatch is a resource-dependent decision

The [loader](https://github.com/flashinfer-ai/flashinfer/blob/682578cfa24b924a5fb9e32c1b04597ed0602e7f/flashinfer/jit/cake_moe_finalize_comm.py)
builds one module per SM100a/SM103a architecture from 32 device translation
units plus one binding: twelve regular programs, twelve small programs and
eight TP4/TP8 large-token programs. Each includes the FP16/BF16 and
non-quantized/FP4-output choices. `SMALL_TOKEN_MAX=64` is passed into C++.
All programs use 224 threads, four CTAs per cluster and 256 shared bytes.

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/682578cfa24b924a5fb9e32c1b04597ed0602e7f/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
raises the block-occupancy cap from four to five CTAs/SM, then separately
queries `cudaOccupancyMaxActiveClusters`. Both occupancy results are cached
by device and kernel. For a candidate program, its launched cluster count is
the smaller of the active-cluster capacity and
`floor(min(ctas_per_sm * SM_count, tokens * 4) / 4)`. Five is a cap, not a
guarantee that every program reaches five resident CTAs on every SM.
Cluster placement can constrain the launch below a block-only estimate.
These peer-polling kernels require all launched clusters to fit in one wave;
increasing the grid independently can introduce a cross-rank scheduling wait.

T<=64 selects a small program. Above that, the binding first sizes the regular
program and then applies this actual gate:

```diff
+    if (large_token != nullptr && tokens > 2 * clusters) {
+      kernel = large_token;
+      clusters = launched_clusters(kernel);
+    }
```

There is no TP2 large entry here. The selected large program gets its own
occupancy calculation.
The nearby comment says every cluster would hold at least three tokens;
`T > 2*C` only makes the average exceed two. For example, `T=2*C+1` gives
one cluster three tokens and the others two. Keep the predicate rather than
turning that comment into an all-clusters invariant. The source records
rejected boundary timings: TP8/T284 ratios 0.990/0.996 versus T352
1.047/1.038 on B200/B300; TP4/T288 reports 0.92 and T352 1.013/1.016.

### What the device schedules actually do

| Historical path | Gather and publication | Consumption and independent loads |
|---|---|---|
| Small, TP2/4/8 | Load up to eight route rows/scales before ordered accumulation; load shared-expert row early; publish all assigned tokens | Load residual before local polling; no TP2 lookahead cache |
| Regular TP2 | Gather two routes at a time; publish all assigned tokens, then clear old phase | After a successful current poll, prefetch the next token's peer words and residual; use prefetched words only after checking every halfword against the sentinel |
| Regular TP4/TP8 | Gather four routes at a time; publish all assigned tokens, then clear old phase | TP4 loads residual before polling; TP8 loads it after rank reduction |
| Large TP4/TP8 | Publish token m, clear a bounded part of the dirty phase, consume token m-1; drain the last token and remaining cleaning | Same TP4/TP8 residual distinction; `__launch_bounds__(224,4)` guides register allocation |

The [small TP2 body](https://github.com/flashinfer-ai/flashinfer/blob/682578cfa24b924a5fb9e32c1b04597ed0602e7f/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o110_small_kernel.cu)
prefetches eight indices, raw row words and scales but still consumes routes
in order. Negative inverse indices skip that route. Each contribution is
rounded to the activation type before addition, and each sum is rounded
again. Routed scaling and shared-expert addition retain their positions and
rounding. Early shared loading changes its live range, not its addition order.

In [regular TP2](https://github.com/flashinfer-ai/flashinfer/blob/682578cfa24b924a5fb9e32c1b04597ed0602e7f/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o110_kernel.cu),
`ahead_ready` means words were fetched; it does not mean peers published
them. A separate sentinel test decides whether to consume the cached words
or poll again. Publication canonicalizes zero and writes volatile 128-bit
vectors to each peer. Polling checks all 16-bit halves for negative-zero
Lamport sentinels and keeps the ordered, activation-rounded rank sum.

The [large TP4 body](https://github.com/flashinfer-ai/flashinfer/blob/682578cfa24b924a5fb9e32c1b04597ed0602e7f/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o110_lt_kernel.cu)
uses `local_tokens+1` iterations: the first publishes only, intermediate
iterations publish then consume, and the last consumes only. Dirty-phase
cleaning is apportioned across these steps, with a final remainder loop.
The initial cluster wait occurs at step 1; the zero-local-token case has
its own wait. RMS slot/phase counters advance even on the publish-only
iteration, so copying the regular loop's initial parity would be incorrect.
Quantized output and its scale row use the **consumed** token, not the
just-published token. This is a scheduling change within the same one-shot
exchange protocol, not the later two-shot protocol.

### RMS ordering, synchronization and quantization

Each thread accumulates eight residual squares with `__fmaf_rn`; each warp
uses the XOR reduction. Instead of another warp-wide reduction of seven
warp sums, thread 0 explicitly computes
`((v0 + v4) + (v2 + v6)) + ((v1 + v5) + v3)`. This preserves the meaningful
addition tree of the previous zero-padded reduction for these square sums.
The source change alone does not establish bit identity across compilers,
non-finite inputs or different builds.

The two DSM slot banks remain, but their exchange now uses mbarriers.
Thread 0 expects 16 transaction bytes on its local barrier, asynchronously
writes its partial into the corresponding slot of all four CTAs, issues the
restricted cluster release fence, then arrives on the other three barriers.
Four arrivals and four FP32 stores complete each phase. Threads wait with
cluster acquire semantics before reading their four local slots in rank
order. Preserve initialization, release/acquire scope, transaction bytes,
slot alternation and phase advancement together when reusing this pattern.

The workspace counter still counts CTA **entry**, not completion. Reading
stable workspace pointers/stride before `griddepcontrol.wait` does not make
phase state ready; the phase and cleaning size are read after the wait.
The final phase advance and launch-dependents instruction must remain
consistent with the owning stream and workspace's sequential-use contract.

FP4 is derived from rounded FP16/BF16 norm values. Neighboring lanes form a
16-element absolute maximum; E4M3 scale rounding precedes approximate
reciprocal scaling and saturating E2M1 packing. Only logical scale entries
are written in the padded `SWIZZLED_128x4` allocation. The host fixes the
global `scale_factor` argument to one. All eight BF16 quantized bodies match
their non-quantized bodies after removing exactly this inspected epilogue
and bijectively renaming generated temporaries. All sixteen FP16 programs
match explicit type/packing/unpacking substitutions, including their FP16
amax instructions. Neither comparison equates BF16 and FP16 numerical values.

### Tests and performance evidence have different coverage

The [distributed test](https://github.com/flashinfer-ai/flashinfer/blob/682578cfa24b924a5fb9e32c1b04597ed0602e7f/tests/comm/test_cake_moe_finalize_allreduce.py)
crosses T=1/64/65/128 with K4/K8, shared expert, both output profiles and
weight bias 0/1; it runs both PDL settings and both dtypes at TP2/4/8.
One workspace per backend is reused across program switches. Its reference
preserves route/rank rounding, but residual/norm comparisons use
atol=rtol=0.01, and decoded Cake FP4 uses atol=1.0/rtol=0.1. It does not
assert byte identity, scale-padding writes, negative/permuted route indices,
graph replay or downstream model accuracy. Its token ceiling does not
exercise the eight large programs on ordinary full B200/B300 devices.
The final review explicitly raises that gap; no human resolution is recorded.

The CPU tests verify argument forwarding and registered source names;
they do not execute the C++ occupancy selector. The
[final-head H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/40259026597/job/121367241127)
checks out `3aaa6b7`, merging `682578cf` into `407ca71f`: nine CPU passes
and six distributed GPU skips, using CUDA 12.9 in the container. Six AOT
build/import jobs pass. Two earlier workflows stop at the unauthorized-PR
reporting step with test setup skipped; those failures are not kernel
numerical failures. None of these jobs proves Blackwell large-path correctness.

The body reports external CUDA 13.4 CUPTI timing on B200 and B300 SXM6 AC:
minimum rank latency, median kernel times and median paired old-Cake/new-Cake
ratios from interleaved runs. All 720 rows have unique shape keys; their
rounded ratios reproduce geomeans 1.120x on B300 and 1.119x on B200.
There are 256 TP2, 224 TP4 and 240 TP8 rows. K8/T512 and K8/T1024 are absent
at TP2; all T512/T1024 rows are absent at TP4; K8/T512 and all T1024 rows
are absent at TP8. T64/T65 boundary cases are in the tests, not this table.
Keep the three B300 0.997x rows: TP4/T128/BF16/K8/non-quantized/no-shared
with either PDL setting, and TP8/T128/FP16/K8/quantized/no-shared/PDL-off.
The B200 minimum is 1.000x. A median paired ratio need not equal the ratio
of the displayed median times.

The unchanged committed harness instead compares TRT-LLM/Cake/TRT-LLM,
aggregates rank maxima and can fall back from CUPTI to events. It does not
reproduce the body's previous-Cake paired rank-min experiment. No service
throughput, accepted-token latency or model-accuracy gain follows from that
operator table, and the table is not a substitute for numerical validation
of the large-token programs.

### Applying this to the current pin

The current [host selector](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
retains the five-CTA cap and active-cluster bound but has 63 registered
programs. Twelve of #7392's 32 device paths have disappeared; all twenty
remaining files differ. Do not attach this historical proof to a current
kernel merely because its symbol is familiar.

Current TP4/TP8 two-shot selection starts at T72/T56 ahead of the small
test, with early-load forms through T128 and deferred forms from T1536/T768
(including a TP8 K4/no-shared/no-quant exclusion). Small-path extension to
T71 is conditional on shape and cluster capacity. TP2 can select quad gather
through T142 and large K4 programs above T512, or above T1280 with a shared
expert. Thus the historical universal T64 cut and `T > 2*C` large gate are
not a current dispatch recipe. Later two-shot/deferred bodies and their PRs
still require separate complete reviews.

For a new optimization, record the selected source revision, actual grid
and resident-cluster limit; preserve rounding and the whole phase protocol;
exercise both sides of each live dispatch boundary with repeated workspace
reuse. Report rank-min and rank-max timing separately and verify model
accuracy before translating an isolated operator result into serving guidance.
