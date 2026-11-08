# FlashInfer QK RMSNorm, RoPE and paged append

Reviewed on 2026-11-09 (Asia/Shanghai). The complete final #5405, #5952 and
#5956 diffs, bodies and all discussion were reviewed. Current-source comparisons use FlashInfer main
`31c146b16789f112e9b9861bd79247d4701e437a`. No GPU experiment was run.

## PR #5405 — Experimental packed-QKV fusion

- Link: https://github.com/flashinfer-ai/flashinfer/pull/5405
- Title: `feat(rope): fuse QK norm, RoPE, and paged KV append`.
- State: closed without merge on 2026-10-10; head
  `e7efd08d0ac3af9c90e1e49a4cbf2e0dea6e4ebd`, base
  `621fd46e1497a609693d6532ed56325a4d2ce8b2`.
- Coverage: all 3,144 diff lines, 15 files, +3,039/-1; body, three issue
  comments, three reviews, three inline comments and both commits.
- Reviewed runtime: `flashinfer/rope.py`;
  `flashinfer/experimental/fused_qk_rope_append/{__init__,backend,jit}.py`;
  its `csrc/{hpc_rope,hpc_rope_jit_binding}.cu` and
  `csrc/include/flashinfer/{hpc_rope.h,hpc_utils.cuh}`; source packaging in
  `pyproject.toml`. Reviewed validation/docs:
  `tests/experimental/test_fused_qk_rope_append.py`,
  `benchmarks/bench_fused_qk_rope_append.py`,
  `examples/experimental/fused_qk_rope_append.py`, the experimental
  `README.md`/`RESULTS.md` and `licenses/LICENSE.hpc-ops.txt`.
- Motivation: remove separate Q/K RMSNorm, rotary and paged-store launches by
  importing Tencent hpc-ops' packed-QKV kernel into an opt-in JIT backend.
  The patch does not register an automatic serving-framework dispatch.

The closing comment says the feature was covered by CAKE
[#5952](https://github.com/flashinfer-ai/flashinfer/pull/5952),
[#5956](https://github.com/flashinfer-ai/flashinfer/pull/5956) and
[#6008](https://github.com/flashinfer-ai/flashinfer/pull/6008). Fresh metadata
confirms these merged on October 2, October 3 and October 3 respectively.
The #5952 and #5956 source reviews follow below; the complete historical
review of #6008 remains pending. Current availability is established from the frozen
mainline implementation, not by treating #5405 as merged.

### Data and launch contracts

The imported kernel accepts contiguous packed BF16 QKV, head dimension 128,
Q/KV head pairs 8/1 or 64/8, contiguous NHD caches, FP32 full-width NeoX
cos/sin tables and FP32 norm weights. Norm policy 0 skips normalization, 1
normalizes after RoPE and 2 normalizes before it; epsilon is fixed at `1e-6`.
The JIT supports device major versions 9 and 10. The final test-guard change
excludes 11.x/12.x instead of assuming every device newer than SM90 works.
This admission check is not evidence of validation on every admitted GPU.

`q_indptr` describes packed query rows; `seq_lens` includes the appended
tokens. Row `r` of request `b` rotates at
`r + seq_lens[b] - q_indptr[b + 1]`. The dense request/page table supplies
physical page IDs. The four-warp CTA processes four rows of one KV head and
its eight Q heads, stages Q/K through shared memory, and copies V in vector
loads/stores. Additional CTAs clear the unused tail of each last cache page.
Optional `out_k` and `out_v` independently redirect the corresponding token
writes away from the cache; they do not request a second copy alongside an
append. Last-page clearing still occurs. Callers must own those pages and
provide valid sequence, position and page bounds.

The normalized and rotated values stay FP32 until the final BF16/FP8 store.
A separate BF16 RMSNorm primitive materializes a rounding boundary before
RoPE; matching input weights does not make the fused and separate pipelines
bitwise equivalent. Check the intended numerical baseline before porting.

FP8 output/cache use E4M3. Dynamic Q quantization computes
`max(max_abs, 1e-6) / upper_max`, with finite `0 < upper_max <= 448`.
Decode scales are `[T, Hq]`; prefill scales are
`[B, Hq, ceil(max_seqlen / 128) * 128]`. Only valid token entries are written,
so padding is not initialized output. Static Q multiplies by scalar
`q_scale_inv` and returns an empty scale tensor. K/V multiply by reciprocals
of their scalar FP32 dequantization scales. Shape/dtype checks do not prove
that device-resident scale values are finite and positive.

The FP8 Python wrapper zeroes `split_k_flag` on each call. Dynamic prefill
also launches an indptr validator before the compute kernel. This exact
fragment is from `csrc/hpc_rope.cu` in the final diff:

```diff
+  if (quant_policy == 1 && is_prefill) {
+    kernels::validate_dynamic_prefill_indptr_kernel<<<1, 256, 0, stream>>>(
+        q_index_ptr, num_batch, num_rows, max_seqlens, split_k_flag_ptr, num_kv_heads);
+  }
```

That validator checks monotonic/in-range segments, first/last boundaries and
each request's `max_seqlen`. Failure writes flags `-1` and causes the later
kernel to return before output or cache clearing. Other quantization modes
do not take that validator. Count flag initialization, validation and compute
when profiling this FP8 path; the complete invocation is not one launch.

### Validation and measurement limits

The final test source has 14 parameterized cases: three BF16 norm policies,
two FP8 prefill scale modes, BF16 64/8 optional outputs, five invalid
`upper_max` values, output-shape rejection, an empty request's status reset
and a per-request max-length overflow. All numerical calls set
`is_prefill=True`; there is no FP8 decode, FP8 64/8, graph-replay or attention
consumer test. Q/K BF16 comparisons use tolerance 0.02; FP8 comparisons
allow 0.15 relative / 0.03 absolute error. These are not bitwise FP8 tests.
The author reports 14 experimental tests and eight existing RoPE/FP8-append
trace tests passed on B200. The architecture-review reply repeats the
14-test result after its fix; the bot explicitly did not rerun it. Hopper
and CUDA Graph graduation checks remain unreported for this proposal.

The new benchmark measures **BF16 only**, norm policy 2, Hq/Hkv=8/1, D=128,
context 2048, page size 64. It compares fused execution with two RMSNorms,
RoPE and append, using the same BF16-rounded norm-weight values in both
arms. The historical B200/CUDA 13.3/Torch 2.11-cu130 report gives:

| Batch / query rows per request | Separate pipeline | Fused | Reported ratio |
|---|---:|---:|---:|
| 32 / 1 | 30.368 µs | 5.280 µs | 5.752× |
| 8 / 16 | 34.015 µs | 5.344 µs | 6.365× |

The harness uses 10 warmups, 30 measured calls and medians. At the reviewed
head, [the CUPTI helper](https://github.com/flashinfer-ai/flashinfer/blob/e7efd08d0ac3af9c90e1e49a4cbf2e0dea6e4ebd/flashinfer/testing/utils.py)
flushes L2 before each iteration and synchronizes outside the measured call.
It takes the earliest device start to latest device end of the call's
correlated activities, including gaps between kernels, rather than summing
kernel durations. CUDA Graph replay is off by default; missing CUPTI falls
back to CUDA events. The fused path also clears the last-page tail while the
baseline only appends; the benchmark checks written token rows, not equivalent
full-cache side effects. These numbers do not measure FP8, current CAKE,
attention latency, service throughput or model accuracy.

## PR #5952 — Initial CAKE BF16 implementation

- Link: https://github.com/flashinfer-ai/flashinfer/pull/5952
- Title: `feat(cake_fused_qk_rope_append): fused BF16 QK RMSNorm + NeoX RoPE + paged KV append for SM90 / SM100 / SM103`.
- State: merged on 2026-10-02; head
  `84f55b0592efeed999238dc3b4fe1b73c50c6546`, base
  `1b578de52924c973bd229fe4fd147499ead2781d`, merge
  `a973cfc9245355ce4d036d51cba043c842d2fe1e`.
- Coverage: complete 7,433-line diff, 19 files, +7,308/-1; body, three issue
  comments, two reviews, one inline comment and four commits. Generated
  sources were reviewed by reading the complete SM100 64/8 kernel/binding
  and comparing every line of all five other pairs: bindings differ only
  in generated identifiers; kernels additionally change head/warp constants.
  All remaining differences and the complete six-record manifest were read.
- Reviewed runtime: `flashinfer/cake_fused_qk_rope_append.py`, its loader in
  `flashinfer/jit/`, the `flashinfer/rope.py` export, and all twelve CUDA files
  plus manifest under `csrc/cake_fused_qk_rope_append/`. Reviewed validation:
  `tests/attention/test_cake_fused_qk_rope_append.py`,
  `benchmarks/bench_cake_fused_qk_rope_append.py` and the pre-commit exclusion.
- Motivation: provide an explicit, independent BF16 API that combines Q/K
  normalization, NeoX rotation, Q output, paged K/V append and tail clearing
  in one kernel. The comparison with #5405 does not make the two interfaces
  interchangeable or establish automatic serving-framework adoption.

### Original dispatch and numerical contract

The [original API](https://github.com/flashinfer-ai/flashinfer/blob/84f55b0592efeed999238dc3b4fe1b73c50c6546/flashinfer/cake_fused_qk_rope_append.py)
accepts packed contiguous BF16 QKV, D=128, heads 8/1 or 64/8, FP32 NeoX
cos/sin and norm weights, I32 request/page metadata, and contiguous NHD
BF16 caches. Norm policies are 0/1/2 as above, with configurable epsilon.
K/V outputs independently redirect writes; clearing is still enabled by
default, even for requests that append zero rows. Integrators must own the
affected page tails or disable clearing. Shared prefix pages need particular
care because clearing beyond one request's length can overwrite another's KV.

The [original loader](https://github.com/flashinfer-ai/flashinfer/blob/84f55b0592efeed999238dc3b4fe1b73c50c6546/flashinfer/jit/cake_fused_qk_rope_append.py)
chooses one of six JIT modules by head pair and exact architecture 9.0, 10.0
or 10.3. It validates the manifest and generated-source hashes. There is no
row-count tuning or PDL variant in this PR. The six records share the same
algorithm across architectures; 8/1 uses four warps per row, 64/8 eight.
Each CTA has four warps, giving one or two CTAs per token row respectively.
The exact launch-plan diff is:

```diff
+    rows_per_cta = max(1, 4 // warps_per_row)
+    ctas_per_row = max(1, warps_per_row // 4)
+    num_row_ctas = ((num_rows + rows_per_cta - 1) // rows_per_cta) * ctas_per_row
```

The [64/8 device source](https://github.com/flashinfer-ai/flashinfer/blob/84f55b0592efeed999238dc3b4fe1b73c50c6546/csrc/cake_fused_qk_rope_append/sm_100a/cake_fused_qk_rope_append_1a5e46c8842b05e59fbb_kernel.cu)
loads Q/K into FP32 registers and V as packed bits before finding the request.
Eight lanes cooperate on a 128-element head. Each lane owns 16 elements;
RMSNorm reduces lane sums with XOR shuffles 4, 2, 1, then applies `rsqrt`
and weights. NeoX rotation uses FP32 FMA and rounds at the final BF16 store.
It does not materialize the intermediate BF16 RMSNorm result of a separate
pipeline. V is copied without arithmetic. Per-warp request lookup preloads
up to 256 requests and scans further batches of 32 using ballot/shuffle.
Additional CTAs clear last-page tails in K/V. This is ordinary CUDA launch,
not the metadata-before-wait PDL schedule described for current main below.

Metadata contents remain caller preconditions. A row outside all request
segments, or with negative computed position, leaves its output unwritten;
the kernel does not validate physical page indices. The final Python API
does not reject zero page size, and a fully empty batch produces a zero grid
that the binding rejects. The inline review identified both cases; the
reviewed final diff does not fix them. These historical edge cases must not
be mistaken for tested no-op behavior.

### What the tests and timings establish

The [test file](https://github.com/flashinfer-ai/flashinfer/blob/84f55b0592efeed999238dc3b4fe1b73c50c6546/tests/attention/test_cake_fused_qk_rope_append.py)
defines 21 cases: 12 decode head/policy/page combinations, four cross-page
prefill cases, ragged requests with empty segments and optional K/V outputs,
disabled clearing, 300-request lookup, graph replay and argument rejection.
The reference performs FP32 norm/rotation and casts only at the end; Q/K
comparisons allow 0.01 tolerance, while V and untouched poison regions are
checked exactly. Optional-output coverage supplies both K and V, not each
independently. The graph test replays unchanged inputs/metadata; it does not
establish correctness after updating sequence lengths or page tables. Fully
empty batches, zero-sized pages and an attention consumer are not tested.
The author reports all 21 passed on H100, H200 and B200; the discussion has
a maintainer approval and test-trigger requests, not new B300 results.

The [committed benchmark](https://github.com/flashinfer-ai/flashinfer/blob/84f55b0592efeed999238dc3b4fe1b73c50c6546/benchmarks/bench_cake_fused_qk_rope_append.py)
contains 17 named shapes and separates eager calls from manually captured
one-call CUDA Graph replays. The timing helper is byte-identical to #5405's
reviewed helper: cold L2, CUPTI device-activity span, CUDA-event fallback.
Defaults are ten warmups and three groups of 30 calls. Arm order reverses on
alternate groups, leaving two forward groups and one reversed; every arm
still runs graph before eager. Saved JSON contains medians of group medians,
device name and geometry, without raw samples, executed order, baseline SHA
or confidence intervals. The optional #5405 baseline is loaded by module
name, not revision. Preserve the actual installed version when reproducing.

Selected historical **graph-mode** numbers from the PR body, in microseconds:

| Shape | GPU | CAKE | #5405 | Separate pipeline |
|---|---|---:|---:|---:|
| 8/1, B32, decode | B200 | 2.85 | 5.33 | 11.98 |
| 64/8, B128, decode | B200 | 8.51 | 10.56 | 21.82 |
| 64/8, B128, decode | H200 | 11.07 | 12.05 | 20.59 |
| 64/8, B128, decode | H100 | 14.18 | 16.77 | 21.06 |

The body's 28-row claim and minimum 95% confidence bound of 1.058 are author
reports; the committed 17-row harness and median-only JSON do not reproduce
that analysis. Its stable norm-before-RoPE arm runs two RMSNorm operations,
RoPE and append, despite the body's “3-op” label. Stable append does not clear
the cache tail. The benchmark itself does not compare correctness, and
both arms use BF16-rounded norm weights. These are historical BF16 pipeline
measurements, not FP8, current shape-policy, end-to-end attention or serving
measurements. Use the independent numerical tests and an actual consuming
model when evaluating a port.

## PR #5956 — Standalone CAKE FP8 append

- Link: https://github.com/flashinfer-ai/flashinfer/pull/5956
- Title: `feat(cake_fused_qk_rope_append): Cake backend for the fused QK RMSNorm, RoPE, FP8 quantize and paged KV append entry (SM90, SM100)`.
- State: merged on 2026-10-03; head
  `14ceb190ff848e62c02a43485252eb97f4b91252`, base
  `46340689a5ab5665ab39f6281dac4d021d7ae94b`, merge
  `335c4eb4e90c5c5069abb0340ba66635ad2679e1`.
- Coverage: complete 5,461-line diff, 12 files, +5,384/-0; body, all 24 issue
  comments, 15 reviews, 22 inline comments and the final squashed commit.
  The complete SM100 device source and binding were read; every line of the
  SM90/SM103 pairs was compared, with only generated identifiers differing.
  The full three-record manifest was read and all six source hashes verified.
- Reviewed runtime: `flashinfer/cake_fused_qk_rope_fp8_append.py`, its loader
  in `flashinfer/jit/`, `flashinfer/rope.py` and all generated CUDA/manifest
  files in `csrc/cake_fused_qk_rope_fp8_append/`. Reviewed validation:
  `tests/attention/test_cake_fused_qk_rope_fp8_append.py`,
  `benchmarks/bench_cake_fused_qk_rope_fp8_append.py` and the timing helper.
- Motivation: fuse normalization, rotation, FP8 conversion, paged append,
  indptr validation and status/tail writes into a single launch. The final
  API is standalone. Early review replies describe a different version
  stacked on #5405 under `experimental`; those paths and its empty-grid
  handling are not the final implementation.

### Fused work and caller obligations

The [final wrapper](https://github.com/flashinfer-ai/flashinfer/blob/14ceb190ff848e62c02a43485252eb97f4b91252/flashinfer/cake_fused_qk_rope_fp8_append.py)
infers heads 8/1 or 64/8 from packed BF16 QKV and the NHD cache. It accepts
D=128, FP32 weights/cos-sin/scales, I32 metadata and E4M3 Q/K/V output.
K/V output buffers must be supplied together. Dynamic Q scale layouts and
unwritten padding match the contracts above; static Q returns an empty
scale tensor. The three architectures share one runtime-parameterized
`fp8_e4m3` stage. There is no shape-specific stage selection or PDL in this
initial PR. The grid is `(T+B, Hkv, 1)` with 96 threads per CTA, 16 bytes of
static metadata storage and no dynamic shared memory.

In the [device source](https://github.com/flashinfer-ai/flashinfer/blob/14ceb190ff848e62c02a43485252eb97f4b91252/csrc/cake_fused_qk_rope_fp8_append/sm_100a/cake_fused_qk_rope_fp8_append_caf9bdb5445726a94bed_kernel.cu),
eight lanes handle each head: eight Q groups, one K and one V. Inactive lanes
still participate in the block-wide validation vote. Every CTA scans all
request segments, checks indptr endpoints and combines errors with
`__syncthreads_or`; dynamic prefill additionally checks request length.
This is part of the compute launch, not a separate validator. The final diff
contains the following limit on what is validated:

```diff
+    bool bad = begin < 0 || end < begin || end > num_rows;
+    if (dyn_prefill && end - begin > max_seqlen) {
+      bad = 1;
+    }
```

QKV, norm weights and scalar scales can be read before that vote. Invalid
indptrs suppress Q/K/V, scale and cache writes; the request/head clearing
CTAs still write every status flag to -1. Valid calls write flags to 0,
including empty request segments. Flags describe indptr validation, not
completion of every CTA or validation of sequence lengths and physical pages.
Consume them with normal stream ordering. Sequence lengths must include the
appended rows and keep derived positions within the cos/sin and page-table
capacities. Unlike #5952's negative-position suppression, this initial FP8
kernel uses the derived position directly. Its final wrapper also lacks a
positive-page-size check and a whole-empty-batch no-op; the binding rejects
a zero grid. Earlier discussion saying the Cake binding returned early
belongs to the retired stacked version.

Norm math stays FP32 through rotation, with fixed epsilon `1e-6`. All nine
sum/max shuffle reductions use an eight-lane member mask; the earlier
full-warp mask would name V/inactive lanes that do not execute the reduction.
Dynamic scale uses rounded FP32 division of `max(max_abs, 1e-6)` by
`upper_max`, followed by reciprocal multiplication of Q. K/V also use
rounded FP32 reciprocals. FP8 stores use RN satfinite E4M3 pairs. This is not
a promise of byte equality with eager division or BF16 intermediate norms.
Scale tensor contents remain caller-supplied; host shape checks do not verify
their finiteness or positivity.

Request/head CTAs clear tails using raw 16-byte zero stores, including when
K/V token writes go to caller-owned outputs. The author attributes this
choice to a ptxas 12.9 SM90 miscompile of an all-zero FP8 conversion pack;
the final source indeed avoids that conversion in its clearing loop.
Writable-page ownership remains necessary. Current shape/PDL policies below
are later implementations, not properties of this original three-architecture build.

### Actual validation and measurement coverage

The [final test file](https://github.com/flashinfer-ai/flashinfer/blob/14ceb190ff848e62c02a43485252eb97f4b91252/tests/attention/test_cake_fused_qk_rope_fp8_append.py)
defines 51 cases: 24 decode, 12 cross-page prefill, six scale-padding boundary
cases, two ragged/optional-output cases, one smaller upper bound/page case,
four invalid-metadata cases, graph replay and argument rejection. Its oracle
uses FP32 unrounded payloads, with FP8 `atol=rtol=0.1`; Q scales use
`rtol=1e-5, atol=1e-6`. Untouched poison bytes and status flags are checked
exactly. Contrary to the PR's “tails bitwise” description, expected tail
zeros pass through the same 0.1 absolute-tolerance comparison. The invalid
metadata tests cover dynamic prefill 8/1 only. The graph test poisons outputs
between three replays but leaves QKV and metadata unchanged. It compares
FP8 bytes with eager execution; floating scale NaNs are normalized first.
There is no attention consumer, changing-metadata replay or whole-empty
batch/zero-page numerical test.

The [H100 targeted CI job](https://github.com/flashinfer-ai/flashinfer/actions/runs/37079854496/job/111077995514)
is attached to final head `14ceb190f`. Its log checks out the PR merge of
that head into `46340689`, runs this exact file in CUDA 12.9 with
Torch 2.13-cu129, and reports **51 passed, zero failed/skipped**. This is
stronger evidence than a test-trigger comment. The body's B200 results,
73-case external CAKE harness and sanitizer claims remain author reports;
this refresh did not rerun them or establish H200/B300 coverage.

The [committed benchmark](https://github.com/flashinfer-ai/flashinfer/blob/14ceb190ff848e62c02a43485252eb97f4b91252/benchmarks/bench_cake_fused_qk_rope_fp8_append.py)
has 20 named shapes, ten warmups and three groups of 30 calls by default.
It always runs CAKE before stable, recreates buffers per arm/group, and saves
group medians in `samples_us`, not individual samples. It does not capture
CUDA Graphs. Its helper is byte-identical to #5405/#5952 and measures the
earliest-to-latest correlated GPU activity span, including gaps; the
docstring's “union” wording does not describe that implementation.

The body instead describes an external 44-row paired harness with three
sessions of 2,000 samples per arm, reports B200/H100 decode/prefill cohorts,
and compares against a modeled latency/bandwidth floor. Those raw samples,
floor calculation and internal design document are not supplied by this
diff. Reported stable/CAKE geometric means of 18.2× on B200 and 14.9× on H100
for the 24-row cohort are explicitly **report-only**: the stable path has
BF16 intermediates that fail the stated FP8 contract tolerance and omits tail
clearing. They are not equivalent-output replacement gains or service
speedups. Keep external paired results, the committed fixed-order harness,
kernel correctness and downstream model accuracy separate when profiling.

## What is available at the frozen mainline

There are dedicated public modules, outside `experimental`:
[BF16 API](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_append.py)
and [FP8 API](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_fp8_append.py).
Their functions are `cake_fused_qk_rmsnorm_rope_append_paged_kv_cache` and
`cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache`. Thus the old
catalog statement that mainline lacks the preceding Q/K RMSNorm is obsolete.
Availability of an explicit API does not establish that a deployed SGLang or
vLLM model invokes it.

The [BF16 backend](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_append_backend/__init__.py)
and [FP8 backend](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_fp8_append_backend/__init__.py)
retain packed BF16 input, D=128, 8/1 or 64/8 heads and contiguous NHD storage,
with exact capability targets SM90/100/103. BF16 adds configurable epsilon
and `clear_unused_last_page_rows`; optional K/V outputs remain independent redirects.
FP8 requires both K/V outputs or neither, retains fixed epsilon, and allocates
uninitialized status storage rather than issuing the old Python zero-fill.
Preallocate outputs when comparing capture/launch overhead.

The [BF16 stage policy](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_append_backend/cake_jit.py)
selects lookup, warps per row and clearing granularity from architecture,
SM count, request/row counts, pages and norm policy. Its decode heuristic is
`num_rows == num_requests`; the accepted `is_prefill` argument is ignored by
that policy. The [FP8 policy](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_fp8_append_backend/cake_jit.py)
also uses explicit prefill mode, page-table width and caller-owned K/V. It can
select folded clearing, swapped grid order or persistent work scheduling.
A missing selected build raises instead of silently falling back. Current
stage tables have evolved beyond the three October closure references.

The selected BF16 8/1 direct-lookup PDL
[device implementation](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_append_backend/csrc/cake_fused_qk_rope_append_5bc2fa0cac8e05b05678_kernel.cu)
uses eight lanes per head and FP32 norm/rotation before the final store.
The FP8 prefetching PDL
[device implementation](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_fp8_append_backend/csrc/cake_fused_qk_rope_fp8_append_371b6c27760107c6590c_kernel.cu)
uses three warps per row/KV-head CTA. Every CTA checks indptr validity in all
quantization modes and combines the verdict with a block vote; clearing CTAs
write every request/head status to 0 or -1, including empty requests. The
same kernel computes outputs and clears tails only for valid indptrs, removing
the experimental path's separate validator and flag fill. Sequence/page/scale
contents still have caller preconditions; the indptr vote is not a general
bounds check for every device pointer. Padded Q-scale entries remain unwritten.

`enable_pdl` defaults off and only selects a PDL build on SM100/103 when the
shape policy admits it. In the inspected kernels, metadata, weights, cos/sin
and FP8 scalar scales can be read **before** `griddepcontrol.wait`; QKV loads
and all writes follow the wait. Those earlier inputs must already be ready.
The [FP8 binding](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/cake_fused_qk_rope_fp8_append_backend/csrc/cake_fused_qk_rope_fp8_append_371b6c27760107c6590c_binding.cu)
can use dynamic shared memory to cap resident CTAs for a prefetching PDL
build; it is a scheduling cap, not additional algorithmic workspace.

Current-source review covered both public APIs/backends, both executable
stage/PDL policies and launch dispatch, selected build records, and the full
two device sources and bindings linked above. It did not review every generated
variant. The two `tests/attention/test_cake_fused_qk_rope_*_api.py` facade tests
compare caller-buffer identity and backend bytes; they are not independent
numerical or model-accuracy oracles. The eager/replay tests in
`test_cake_fused_qk_rope_append_pdl.py` change the QKV producer and compare
with a plain launch. These are source-level validation paths to run on the
target GPU, not newly executed results from this refresh.
