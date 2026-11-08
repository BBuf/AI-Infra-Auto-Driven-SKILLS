# FlashInfer indexed finalize: two-shot exchange and deferred RMS

Reviewed on 2026-11-09 (Asia/Shanghai). This card covers historical
FlashInfer #7511, #7650 and #7813; current comparisons use
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

## PR #7650 — encode BF16 negative zero inside the gathered vector

- Link: https://github.com/flashinfer-ai/flashinfer/pull/7650
- State: merged 2026-10-28; head
  `b05dd17f6491e243cf2f6416d2720202b96ff10a`, API base and merge-base
  `03ebbc6a3cc43668aa260588d3592988376ae824`, merge
  `1173475aef8facc7b7200c6020c859213651423d`.
- Coverage: all four changed files, 626 diff lines, +136/-176; complete
  body prose, structured checks of all 720 timing rows, two issue comments,
  one empty approval, no inline comments and the sole commit message.
  Every base body matches #7511. The inspected encode/broadcast/poll/decode
  changes reproduce each complete head file exactly, including the four/eight
  per-peer mask-store removals and final clear-length change.
- Reviewed files: BF16 `ws4_o110_ts`, `ws4_o111_ts`, `ws8_o110_ts` and
  `ws8_o111_ts` device files. The other 28 programs, host binding, loader,
  public API, two relevant tests, benchmark and timing helpers are byte-identical
  to the reviewed #7511 versions.
- Motivation: remove the four-byte mask word accompanying each sixteen-byte
  second-stage vector, its per-peer store and its readiness load/check.
  Historical admission remains BF16 TP4 from T96 and TP8 above T64.

### The escape value is part of the wire format

The [BF16 TP4 kernel](https://github.com/flashinfer-ai/flashinfer/blob/b05dd17f6491e243cf2f6416d2720202b96ff10a/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o110_ts_kernel.cu)
adds an encoder and decoder. The encoder applies this to both halfwords:

```diff
+    lo = ((lo == 32768) ? 65535 : lo);
+    hi = ((hi == 32768) ? 65535 : hi);
```

Thus a reduced BF16 `-0` travels as `0xFFFF`; the receiver still polls each
halfword against `0x8000`, then maps `0xFFFF` back to `0x8000` before BF16
unpacking and residual addition. Raw integer load/store operations preserve
the escape bits in transit. Removing the sideband reduces the used phase
span from 16,128 to 14,336 halfwords per token, and the cleanup control word
changes accordingly. It does not resize the public workspace or change the
one-shot capacity guard. The 20-to-16-byte change describes an individual
second-stage payload plus mask, not a 20% end-to-end traffic reduction.

An exhaustive CPU check of the two halfword mapping functions covers all
65,536 values: the encoded stream never contains the empty sentinel, and
all finite communicated values, including signed zero, round-trip exactly.
However, `0xFFFF` is itself a BF16 NaN bit pattern and decodes to negative
zero. The mapping is therefore not a bijection over arbitrary BF16 payloads.
This is a property of the helper, not a GPU reproduction of that NaN reaching
the gather through the preceding arithmetic. Preserve this qualification
instead of repeating unrestricted bitwise identity; finite model inputs also
do not by themselves prove that every intermediate reduction stays finite.

No arithmetic tree, route ordering, quantization epilogue, ownership mapping,
pipeline schedule or FP16 program changes in this PR. All four changed
programs retain the same protocol transformation. The unchanged distributed
tests use tolerant numerical checks and bounded random inputs; no new
negative-zero/NaN or byte-equality test establishes the body's full claim.

### Validation and timing scope

The complete ten-job list for the green
[final-head run](https://github.com/flashinfer-ai/flashinfer/actions/runs/40613179825)
again contains six successful AOT jobs and no requested CPU/GPU unit-test
jobs. [Setup](https://github.com/flashinfer-ai/flashinfer/actions/runs/40613179825/job/122569444981)
checks out `ecd85dd32` (`b05dd17f` into `03ebbc6a`): H100 is disabled and
the A10G/T4 target intersections are empty. The
[summary](https://github.com/flashinfer-ai/flashinfer/actions/runs/40613179825/job/122654667440)
reports no JUnit artifacts. Earlier run `40613095897` fails at the
unauthorized-PR reporting step with test setup skipped; that is not numerical
evidence. The discussion requests tests but records no separate hardware
result. No GPU run was performed for this review.

The body retains the external CUDA 13.4 CUPTI paired, interleaved rank-min
protocol. Separating shapes by the actual unchanged dispatch gates gives
72 changed-program rows and 648 unchanged-program rows:

| Cohort | B300 geomean / range | B200 geomean / range |
|---|---|---|
| All 720 | 1.004 / 0.977–1.162 | 1.004 / 0.980–1.150 |
| Changed 72 | 1.038 / 1.014–1.162 | 1.036 / 1.013–1.150 |
| Unchanged 648 | 1.000 / 0.977–1.020 | 1.001 / 0.980–1.020 |

The changed TP4 T128/T2048 means are 1.034/1.026 on B300 and
1.033/1.027 on B200. Changed TP8 T128/T512/T2048 means are
1.023/1.052/1.061 and 1.022/1.053/1.053. Both maxima occur at
TP8/T2048/BF16/K4/PDL-on/ordinary output/no shared expert. All changed
rows exceed one, but 281 B300 and 249 B200 unchanged rows are below one.
Retain that variation rather than treating unchanged rows as measured speedups
from this code edit. The 720-row shape inventory still omits the dispatch
boundaries and some large-token cohorts. The unchanged rank-max TRT/Cake/TRT
harness does not reproduce this external experiment; no serving gain follows.

At the current pin, all four corresponding files differ as a whole, but both
escape helpers remain text-identical, are still called, and the mask is absent;
the clear length remains 14,336 halfwords/token. The
[current TP4 body](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o110_ts_kernel.cu)
therefore does not use #7511's separate sign-mask word. This narrow comparison
does not certify every later program body. When diagnosing readiness or
transferring an optimization, inspect the selected revision's actual encoding,
poll predicate, decode point and cleanup span together.

## PR #7813 — defer RMS completion while later tokens communicate

- Link/title: [perf(cake_moe_finalize_allreduce_fusion): deferred two-shot programs, earlier two-shot switch at world sizes 4 and 8](https://github.com/flashinfer-ai/flashinfer/pull/7813).
- State: merged 2026-10-29 at 06:00:19 UTC; head
  `45b256f36fa7e728154fff22acf912bc27e0e7fe`, API base
  `4117c6aac8f9c560efbfddad0f649d0ccbef9003`, actual merge-base
  `c2f8204b458f9a94a328cbebacc537b62b55839a`, merge
  `9f39329cf25af1d132f2bfdf027104559dcf85aa`.
- Coverage: full 38,347-line diff fetched, 44 files, +16,748/-5,655;
  all unique source changes reviewed, with whole-file comparisons for generated
  variants. All body prose, all 720 timing rows through structured checks,
  both issue comments, both reviews, no inline comments and both commit
  messages are covered. The API base is not the diff's merge-base.
- Reviewed files: binding plus all 40 device programs under
  `csrc/cake_moe_finalize_allreduce_fusion`, the JIT registry, distributed
  `test_cake_moe_finalize_allreduce.py` and CPU
  `test_cake_moe_finalize_dispatch.py`. All 36 base files match #7650.
- Motivation: overlap a token's RMS completion with later communication,
  reduce selected route/residual load overhead, and lower the two-shot
  crossover while preserving compatible messages and numerical ordering.

### Dispatch changes and load contracts

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/45b256f36fa7e728154fff22acf912bc27e0e7fe/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
adds eight deferred programs: TP4/TP8 × FP16/BF16 × ordinary/NVFP4 output,
bringing the per-architecture total to 40. Two-shot selection now precedes
the small-token checks:

| Condition | Historical #7813 selection |
|---|---|
| TP4 T>=88; TP8 T>=56 | Two-shot, before checking `small_token_max` |
| TP4 T>=1536; TP8 T>=768 | Deferred two-shot, subject to the exception below |
| TP8 K4, no shared expert, ordinary output | Ordinary two-shot even above the deferred threshold |
| TP8 above `small_token_max`, if not selected earlier | Two-shot |
| Other shapes | Retain small/extended-small and TP2 large/persistent rules |

The exact exclusion is:

```diff
+  const bool defer_excluded =
+      world_size == 8 && top_k == 4 && v_has_shared_expert == 0 && !quantize;
```

Every rank must agree on one-shot versus two-shot. Ordinary and deferred
two-shot use the same workspace/message layout, so their choice may differ
across ranks. Preserve the compatible-protocol distinction when changing
shape or local-occupancy gates.

Of the 32 existing programs, 26 are exactly reproduced by bijective temporary
identifier renaming, preserving every other byte, including assembly strings,
constants and statement order. These are the 24 one-shot programs and two
TP4 ordinary-output two-shot programs. The body additionally reports identical
SM100a/SM103a compiled code; this audit did not reproduce that SASS comparison.
The remaining six whole-file differences, after explicit renames, are:

- TP4 NVFP4 two-shot: predicated route loads.
- TP8 ordinary-output two-shot: packed read-only residual load before gather
  polling, followed by unpacking after the poll.
- TP8 NVFP4 two-shot: both changes.

In the [predicated gather](https://github.com/flashinfer-ai/flashinfer/blob/45b256f36fa7e728154fff22acf912bc27e0e7fe/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o111_ts_kernel.cu),
`ld.global.nc.v4.u32` reads expert data and `ld.global.nc.u16` reads its scale.
The predicate checks the top-k slot, not whether the route index is negative.
Negative indices are clamped to row zero for loading; the subsequent
`route_index_q* >= 0` guards still suppress their arithmetic contributions.
The binding requires at least one expert row. Thus a skipped route can still
cause a valid row-zero read; do not equate predication with skipping all
negative-route memory traffic. The first slot uses `top_k_opaque` through an
empty inline assembly operand. Rounding after each contribution and rank
addition is retained. Positive index bounds are not newly checked.

All eight deferred programs use the packed residual load; their four NVFP4
variants also use the predicated gather. This is a memory-access/scheduling
change, not permission to remove an intermediate activation-type round trip.

### One pending token, explicit drain

The [TP4 deferred body](https://github.com/flashinfer-ai/flashinfer/blob/45b256f36fa7e728154fff22acf912bc27e0e7fe/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o110_tsd_kernel.cu)
retains the `local_tokens + 2` loop, owner reduction and broadcast. At step
`m`, it publishes token `m`, reduces/broadcasts `m-1`, finishes `m-3` when
present, then gathers `m-2`, adds residual and submits that token's RMS work.
These token numbers are local sequence positions mapped through
`token_begin + index * token_stride`.

After residual addition and activation rounding, it writes `residual_out`
immediately and keeps the same values in shared memory as FP32. Each of 224
threads owns eight floats, split into two 16-byte banks at offsets 256 and
`256 + 3584`: 7,168 carry bytes plus the 256-byte RMS area, within the already
allocated 7,424 bytes per CTA. There is one pending-token carry buffer, not
a two-token carry ring. The host's inherited comment saying other programs
use only 256 bytes is stale for these new deferred programs.

The deferred completion reads norm weights, waits on the pending token's
cluster mbarrier, reloads that thread's carry, sums the four CTA RMS slots in
the same order, and writes normalized output. It advances `access` and RMS
parity only after consumption. It finishes the old carry before the same
thread overwrites it with the next token. The last pending token is explicitly
drained after the loop when `local_tokens > 0`; the zero-token cluster wait
and dirty-workspace tail cleanup remain. Grid size still uses the selected
program's active-cluster capacity, five-CTA/SM cap and four-CTA clusters;
deferred device launch bounds remain `(224,4)`.

The [TP8 deferred body](https://github.com/flashinfer-ai/flashinfer/blob/45b256f36fa7e728154fff22acf912bc27e0e7fe/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws8_o110_tsd_kernel.cu)
preserves TP8's two groups of four rank additions. Both complete ordinary-output
deferred diffs were inspected. The quantized bodies match them after the
reviewed predicated-load replacement, two epilogue insertions and bijective
temporary renaming. Both epilogues retain the existing rounded-norm →
16-element maximum → E4M3 scale → E2M1 payload sequence. Their scale rows use
`deferred_token` in the loop and `last_token` in the drain, not the token being
published. Whole-file FP16 comparisons apply the explicit conversion
primitives and omit BF16's escape helpers/calls; they do not erase the
protocol difference. BF16 retains #7650's finite-value/NaN qualification.

### Tests, actual CI and measured cohorts

The [distributed test](https://github.com/flashinfer-ai/flashinfer/blob/45b256f36fa7e728154fff22acf912bc27e0e7fe/tests/comm/test_cake_moe_finalize_allreduce.py)
adds T87/88 and T1535/1536 for TP4, and T55/56 and T767/768 for TP8.
The existing T1/64/65/71/72/128 and TP2 large-boundary cases remain. The
cross-product still includes both dtypes, K4/K8, shared expert on/off,
both output profiles, weight bias 0/1 and PDL on/off. Workspace reuse crosses
program choices. Kernel-name assertions remain conditional on PDL off,
weight bias zero and the existing SM-count/expected-program conditions;
numerical assertions are not replaced by this name check.

Tests use sequential nonnegative route indices and tolerant residual/norm
comparisons (`atol=rtol=0.01`); decoded NVFP4 uses `atol=1, rtol=0.1`.
They do not establish byte identity, negative-route traffic, adversarial NaN
behavior, mixed ordinary/deferred ranks, graph replay or model accuracy.
The CPU test adds deferred-table membership/uniqueness, without executing
C++ dispatch. The second commit replaces probe-and-close TCP port selection
with a per-test `tmp_path` file rendezvous, addressing the startup port race.

Final-head [run 40819187288](https://github.com/flashinfer-ai/flashinfer/actions/runs/40819187288)
has ten jobs, including six successful AOT build/import jobs and **no targeted
CPU/GPU test jobs**. Setup checked merge `1fed70fa5` of the exact head into
the API base, disabled H100 PR testing because of runner capacity, and found
no applicable A10G/T4 scope. The summary found no JUnit artifacts. Earlier
runs 40818914497/40818888043 failed at unauthorized-status reporting with
setup skipped; they are not numerical failures. The bot approval is a
CODEOWNERS-gap attestation, not a test report or code review.

All 720 unique timing rows retain #7650's shape inventory. The external
experiment uses CUDA 13.4, interleaved old/new Cake runs, rank-min CUPTI times
and the median of paired speedup ratios. The committed benchmark, timer,
unified tests and public APIs are unchanged from #7650; its rank-max
TRT/Cake/TRT harness is not this external protocol.

| Rows | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| All 720 | 1.013 / 0.981–1.209 | 1.013 / 0.982–1.213 |
| Changed/new program, 128 | 1.073 / 0.994–1.209 | 1.078 / 1.001–1.213 |
| Unchanged program, 592 | 1.000 / 0.981–1.029 | 1.000 / 0.982–1.015 |
| TP8 T2048 shapes excluded from deferred, 4 | 1.000 / 0.994–1.005 | 1.007 / 1.001–1.012 |

The only changed-program B300 regressions are TP8/T2048/K4/no shared
expert/ordinary output: FP16 with PDL off is 0.994x, BF16 with PDL on is
0.999x. Both remain ordinary two-shot. Deferred TP8 T2048 NVFP4 means are
1.144x/1.154x; its ordinary-output means are 1.088x/1.097x. The maxima are
TP8/T2048/FP16/K8/NVFP4/shared expert, PDL on for B300 and off for B200.
There are no measurements in the newly admitted TP4 T88–95 or TP8 T56–64
ranges, nor at the deferred boundaries. Preserve these omissions and the
270/281 below-one unchanged-program rows; a table-wide mean is not a serving
speedup or validation of each dispatch threshold.

At the pinned current source, all 40 device paths exist but every whole file
differs. Current TP4 two-shot starts at 72, not 88, and the selector has
additional early/shared-specialized programs. TP8's 56 and deferred
1536/768 thresholds and exclusion remain in the inspected binding. Use
this historical review for data-lifetime and validation constraints; it does
not certify every subsequent generated body or current performance.
