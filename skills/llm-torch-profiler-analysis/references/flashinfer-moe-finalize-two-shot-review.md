# FlashInfer indexed finalize: two-shot exchange and TP2 own partials

Reviewed on 2026-11-09 (Asia/Shanghai). This card covers historical
FlashInfer #7511; current comparisons use
`31c146b16789f112e9b9861bd79247d4701e437a`. No GPU run, sanitizer,
benchmark or SASS comparison was performed in this audit.

Read the [indexed input/workspace contract](flashinfer-moe-finalize-source-review.md)
and [#7392 scheduling review](flashinfer-moe-finalize-schedule-review.md)
first. The indexed H7168 operation is distinct from the dense expert-major
reduction API. Its route, rank and residual rounding are part of the contract.

## PR #7511 — reduce-scatter/all-gather within one finalize launch

- Link: https://github.com/flashinfer-ai/flashinfer/pull/7511
- State: merged 2026-10-28; head
  `8c0a8691e9e3b9244c6dc216950dc9326ae689c6`, API base and merge-base
  `593b468bcea8553d9e1195a49ef7d7d2cd79b963`, merge
  `3cf451835d127fd1b9d2be4678c530fe3f775aef`.
- Coverage: all 41 changed files (49 paths without rename detection),
  17,788 diff lines, +7,169/-7,737; complete body prose, structured checks
  of all 720 timing rows, four issue comments, one empty approval, no inline
  comments and both commit messages. Generated-source coverage uses complete
  comparisons against previously reviewed bodies and inspection of every
  remaining difference; it does not represent independent GPU executions.
- Reviewed files: the host binding, all 32 head device programs and twelve
  retired paths under `csrc/cake_moe_finalize_allreduce_fusion`,
  `flashinfer/jit/cake_moe_finalize_comm.py`, the public-function comment in
  `flashinfer/comm/trtllm_ar.py`, `tests/comm/test_cake_moe_finalize_allreduce.py`
  and `tests/cpu/comm/test_cake_moe_finalize_dispatch.py`.
- Motivation: reduce replicated TP4/TP8 communication and reduction work at
  larger token counts, avoid TP2's own-partial global round trip, and retain
  small-token latency with an explicit, rank-compatible protocol choice.

### Dispatch and resident resources

The [loader](https://github.com/flashinfer-ai/flashinfer/blob/8c0a8691e9e3b9244c6dc216950dc9326ae689c6/flashinfer/jit/cake_moe_finalize_comm.py)
still builds one module per SM100a/SM103a architecture with 32 programs:
twelve small, eight persistent, four TP2 large and eight TP4/TP8 two-shot.
The twelve retired paths are TP8 persistent and TP4/TP8 large programs.
FP16/BF16 and ordinary/FP4 output are represented in every eligible kind.

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/8c0a8691e9e3b9244c6dc216950dc9326ae689c6/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
uses this order with `small_token_max=64`:

| Condition | Historical selected program |
|---|---|
| T<=64, any supported world size | Small |
| T65–71, TP2/TP4, K8 or shared expert, and T fits the small program's resident-cluster capacity | Small |
| TP8 above T64; TP4 from T96 | Two-shot |
| TP2 K4, T>512 without shared expert or T>1280 with it | Large, with own partials in shared memory |
| Remaining TP2/TP4 shapes | Persistent |

One-shot versus two-shot is determined by agreed launch arguments. Device-local
capacity can change small versus persistent selection at T65–71 because those
programs share the one-shot layout. Do not broaden this into a requirement
that every rank selects the identical program or identical grid. Conversely,
a local occupancy heuristic must not make one rank choose a different wire
protocol from its peers.

Every launch reserves 7,424 shared bytes per CTA: 256 for RMS exchange and
two slots of `224 * 16` bytes for TP2 own partials. Other programs use only
the first 256 bytes, but still receive the larger dynamic allocation. The
source says their residency is unchanged; preserve the actual occupancy
queries rather than assuming this for an arbitrary build/device. Grid size
is `4 * min(T, cluster_capacity(selected_program))`, with both the five-CTA/SM
cap and `cudaOccupancyMaxActiveClusters`. TP2 large uses
`__launch_bounds__(224,5)`; two-shot retains `(224,4)`. These are compiler
allocation constraints, not promises of achieved residency.

### Two communication stages, one kernel

The [TP4 BF16 body](https://github.com/flashinfer-ai/flashinfer/blob/8c0a8691e9e3b9244c6dc216950dc9326ae689c6/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o110_ts_kernel.cu)
keeps quad route gathering and activation-rounded contributions/additions.
Each token's 896 eight-element vectors are partitioned into four chunks.
The exact first-stage publication replaces the previous fan-out:

```diff
+            int chunk = cluster_thread / 224;
+            int owner = published_token + chunk & 3;
+            long long scatter_offset = ((long long)rank * (long long)tokens + (long long)published_token) * 1792 + (long long)(cluster_thread - chunk * 224) * 8;
+            __nv_bfloat16* scatter_dst = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[8 + owner]) + data_base + scatter_offset;
```

The [TP8 body](https://github.com/flashinfer-ai/flashinfer/blob/8c0a8691e9e3b9244c6dc216950dc9326ae689c6/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws8_o110_ts_kernel.cu)
uses eight chunks of 112 vectors and `(token + chunk) & 7`. Only the owner
polls that chunk's partials in its local workspace. TP4 polls four senders;
TP8 polls ranks 0–3, reduces them, then polls/reduces ranks 4–7. Each rank
addition retains the activation-type round trip. The owner then publishes
the result to every rank, including itself. All ranks consume the gathered
result before residual addition and RMSNorm. TP4 loads residual before this
second poll; TP8 loads it afterward.

For `L` locally assigned tokens, the loop has `L+2` iterations: publish token
m, reduce/broadcast token m-1, and finalize token m-2, with guards and drain
iterations. Cleaning targets the old dirty phase and is distributed over the
loop, followed by a remainder loop. The initial cluster wait moves to step 2;
the zero-local-token path retains its own wait. RMS slot/phase advances only
when a token is consumed. Quantization uses the consumed row/access, never
the current publication row. The CTA-entry counter, triple-buffer ownership
and PDL readiness contract remain essential; the entry counter is not a
completion counter.

### BF16 and FP16 have different second-stage payloads

The negative-zero halfword `0x8000` is the not-ready sentinel. Initial
publication still canonicalizes zero, but an ordered BF16 rank reduction
can require preserving a negative-zero result at the second publication.
The BF16 two-shot path substitutes positive zero in the transmitted vector
and stores an eight-bit sign mask alongside it. A 32-bit mask word contains
the mask in its upper half and its complemented lower half. The receiver
waits for both non-sentinel payload halves and a valid complementary mask,
then restores the negative-zero bits before adding the residual.

The [FP16 program](https://github.com/flashinfer-ai/flashinfer/blob/8c0a8691e9e3b9244c6dc216950dc9326ae689c6/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_float16_ws4_o110_ts_kernel.cu)
publishes the reduced words directly and has no sign-mask sideband. This
difference must survive dtype comparisons; it is not generated-name noise.
The used phase span is 16,128 halfwords per token for BF16 and 14,336 for
FP16, including scatter and gather areas. The final `control_0[4]` records
that span so a later program can clean it. These are used payload spans,
not new workspace allocation sizes. The public API still bounds the larger
one-shot layout `T * H * sizeof(dtype) * world_size` against `MAX_COMM_SIZE`;
it does not admit larger batches merely because two-shot uses fewer bytes.

### TP2 keeps its own partial on chip

The [TP2 large body](https://github.com/flashinfer-ai/flashinfer/blob/8c0a8691e9e3b9244c6dc216950dc9326ae689c6/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o110_lt_kernel.cu)
retains pair gathering, but writes its own packed partial into a two-slot,
per-thread shared-memory ring and publishes only to the other rank. The
`L+1` loop publishes m, performs bounded dirty-phase cleaning, then consumes
m-1. It reads its own ring slot and polls only the peer's local-workspace
segment. The conditional assembly of `rank_words` preserves rank 0 then
rank 1 summation even when the local rank is 1. The regular TP2 lookahead
cache is absent in this large form; residual is loaded before peer polling.

All 32 base device files are byte-identical to the reviewed #7392 head.
The twenty retained programs change only the inspected RMS barrier-address
and phase-counter rewrite. All sixteen dtype pairs were compared over their
full files: twelve match the explicit dtype conversion transform; the four
two-shot pairs have the separately reviewed sideband/gather/clear-span
differences above. The three new BF16 quantized programs match their plain
bodies after removing the inspected FP4 epilogue and bijectively renaming
generated temporaries. Their epilogues match #7392 after explicit consumed
row/access mapping. This source evidence preserves arithmetic intent; it
does not prove the body's bitwise-equivalence claim on GPU or across compilers.

### Boundary tests exist; final-head numerical CI did not run

The [distributed test](https://github.com/flashinfer-ai/flashinfer/blob/8c0a8691e9e3b9244c6dc216950dc9326ae689c6/tests/comm/test_cake_moe_finalize_allreduce.py)
adds T71/72 to T1/64/65/128; TP4 adds T95/96, and TP2 adds
T512/513/1280/1281. It crosses K4/K8, shared expert, both epilogues and
weight bias 0/1, both dtypes and both PDL settings. Each backend reuses one
workspace across cases and protocol switches. These shapes can exercise all
new program kinds, unlike #7392's large-path coverage gap.

Kernel-name assertions use Torch Profiler only with PDL off and weight bias
zero. At T65–71, an expected small program is asserted only when SM count is
at least 148; this is a test heuristic rather than the binding's occupancy
query. Other shapes retain their assertions, and numerical checks still run
when symbol inspection is omitted. The reference/tolerances are unchanged:
residual/norm atol=rtol=0.01, decoded Cake FP4 atol=1.0/rtol=0.1. No new
byte-equality, negative/permuted-route, negative-zero stress, graph-replay or
model-accuracy assertion establishes the broader bitwise claim. CPU additions
inspect program-table membership/uniqueness; they do not execute C++ dispatch.

The final-head [PR Test run](https://github.com/flashinfer-ai/flashinfer/actions/runs/40586232437)
is green, but its entire job list contains only six AOT build/import jobs
plus setup/summary jobs. [Setup](https://github.com/flashinfer-ai/flashinfer/actions/runs/40586232437/job/122478347139)
checks out `b4f1e19b4`, merging `8c0a8691` into `593b468b`: H100-on-PR
testing is disabled for runner capacity, and all A10G/T4 target intersections
are empty. The [summary](https://github.com/flashinfer-ai/flashinfer/actions/runs/40586232437/job/122552236583)
finds no JUnit artifacts. Therefore this run supplies neither the requested
CPU dispatch-test results nor Blackwell numerical results. Do not inherit
#7392's nine CPU passes/six GPU skips as #7511 validation. The discussion's
automated review and empty approval add no independent numerical evidence.

### Timing and current applicability

The external body experiment uses CUDA 13.4 CUPTI, minimum rank latency,
interleaved previous-Cake/new-Cake runs and median paired ratios. Its 720
unique rows reproduce rounded geomeans 1.121x on both B300 and B200, with
minima 0.977x and maxima 3.422x/3.468x. The maxima are TP8/T2048/FP16/K4,
PDL-on, no shared expert and ordinary output. TP8 T128/T512/T2048 cohort
means are 1.596/2.446/2.902 on B300 and 1.631/2.468/2.874 on B200.
TP4 T128/T2048 means are 1.109/1.483 and 1.120/1.461 respectively.
TP2 T1024/T2048 cohorts mix changed and unchanged paths.

Small-token group means near one do not erase individual regressions:
270 B300 rows and 256 B200 rows are below 1.0. The 0.977x B300 row is
TP8/T2/BF16/K4/PDL-off/FP4/shared; the 0.977x B200 row is
TP2/T2/BF16/K8/PDL-on/FP4/no-shared. Retain these measurements without
declaring every difference noise. The table contains 256 TP2, 224 TP4 and
240 TP8 rows, with the same missing large-token cohorts as #7392; it contains
none of the new dispatch-boundary pairs. The committed benchmark is unchanged
and still compares TRT-LLM/Cake/TRT-LLM using rank maxima with possible
CUPTI-to-event fallback. Its timing helpers are unchanged; an unrelated
sanitizer-detection helper was added to their file between the two PRs.
Neither protocol establishes serving latency, throughput or model accuracy.

At the current pin all 32 paths still exist, but every device file differs.
The current 63-program selector chooses TP4/TP8 two-shot from T72/T56,
with early/deferred variants and different shape gates documented in the
[current scheduling comparison](flashinfer-moe-finalize-schedule-review.md#applying-this-to-the-current-pin).
Use #7511 for protocol history, then inspect the actual selected successor
body before transferring its numerical or performance claims. Later device
lineage reviews remain outstanding.
