# FlashInfer QK RMSNorm, RoPE and paged append

Reviewed on 2026-11-09 (Asia/Shanghai). The complete final #5405, #5952,
#5956, #6008 and #4890 diffs, bodies and all discussion were reviewed. Current-source comparisons use FlashInfer main
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
The three successor source reviews follow below. Current availability is established from the frozen
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

## PR #6008 — Shape-selected BF16 programs

- Link: https://github.com/flashinfer-ai/flashinfer/pull/6008
- Title: `perf(cake_fused_qk_rope_append): per-architecture / per-shape program selection for the fused BF16 QK RMSNorm + NeoX RoPE + paged KV append (SM90 / SM100 / SM103)`.
- State: merged 2026-10-03; head
  `8e137c135f3abc9bce887111f8a35d7574244bae`, base
  `1fc76f71b6e6c1928ffa9acbfe721566e23593ce`, merge
  `0557a54ee631a93cc8e27eaf48d7df4882cba963`.
- Coverage: complete 25,166-line, 52-file diff, +20,120/-4,198; body, all
  seven comments, one review, zero inline comments and the final commit.
  All 20 generated kernel/binding pairs were compared line by line after
  normalizing only generated symbols/namespaces: eight device bodies and one
  binding remain. The common 680-line kernel, 160-line binding and every
  variant difference were read. All manifest fields, 40 source hashes and
  specialization constants were checked; the 12 replaced source files match
  the already reviewed #5952 versions. The wrapper, JIT policy and complete
  tests were read; benchmark and timing-helper blobs are unchanged from #5952.
- Motivation: improve request lookup, row parallelism, V ownership and tail
  clearing without changing the public BF16 API or adding a precision mode.
  The single approval names earlier head `99e10f39`, not the final head.

### What selects each program

The [host wrapper](https://github.com/flashinfer-ai/flashinfer/blob/8e137c135f3abc9bce887111f8a35d7574244bae/flashinfer/cake_fused_qk_rope_append.py)
passes architecture, SM count, total query rows `T`, request count `B` and
page size to the [JIT policy](https://github.com/flashinfer-ai/flashinfer/blob/8e137c135f3abc9bce887111f8a35d7574244bae/flashinfer/jit/cake_fused_qk_rope_append.py).
The [manifest](https://github.com/flashinfer-ai/flashinfer/blob/8e137c135f3abc9bce887111f8a35d7574244bae/csrc/cake_fused_qk_rope_append/cake_fused_qk_rope_append_manifest.json)
ships four SM90 stages and eight each for SM100/103. These are 20
architecture/stage records, not 20 different numerical algorithms. All use
128-thread CTAs, zero dynamic shared memory and ordinary launches without PDL.

Let `U = 2 * ceil(page_size * Hkv * 128 / 8 / 256)` be the number of 4 KiB
clearing units per request, and `C = ceil(B * U / 4)` the original clearing
CTA count. The historical policy is:

| Choice | Exact condition / resulting work |
| --- | --- |
| Decode heuristic | `T == B`; it does not inspect every indptr interval. |
| Twenty warps per 64/8 row | `T <= SMs` and `C <= 16 * (SMs - T)`; five CTAs per row, with 18 Q/K head groups and two V groups. This is a scheduling heuristic, not a measured occupancy guarantee. |
| Direct request guess | `B >= 257`; also all SM90 8/1 or twenty-warp rows; also SM100/103 64/8 decode and 8/1 decode with `B >= 128`. |
| V placement | 8/1 direct builds put V on warp 3; twenty-warp builds put it on logical warps 18/19. Other builds retain the original placement. |
| Two clearing units per warp | SM100/103, 64/8 and `C >= 2048`; one warp clears corresponding K and V chunks, halving clearing CTAs for these even-sized unit counts. |

The decisive added policy excerpt is:

```diff
+        free_sms = num_sms - num_rows
+        w20 = free_sms >= 0 and clear_ctas <= free_sms * c["clear_ctas_per_free_sm"]
+        if w20:
+            out["WARPS_PER_ROW"] = c["w20"]
+            out["V_SUB_BASE"] = c["w20_v_sub_base"]
```

The [direct variant](https://github.com/flashinfer-ai/flashinfer/blob/8e137c135f3abc9bce887111f8a35d7574244bae/csrc/cake_fused_qk_rope_append/sm_100a/cake_fused_qk_rope_append_d374e1c215a49809c71a_kernel.cu)
guesses `min(row, B - 1)`, accepts it only if its indptr interval contains
the row, then retains the original ballot/scan fallback. Its emitted source
still preloads the first 256 request records before the ballot tests; direct
lookup does not mean all metadata loads disappear. The
[twenty-warp variant](https://github.com/flashinfer-ai/flashinfer/blob/8e137c135f3abc9bce887111f8a35d7574244bae/csrc/cake_fused_qk_rope_append/sm_100a/cake_fused_qk_rope_append_878eafa5087d58ce2289_kernel.cu)
changes head/V ownership while retaining eight-lane FP32 norm reductions,
FP32 NeoX rotation and final BF16 rounding. Inactive head groups still load
clamped source heads; moving V does not remove every redundant load. The
[two-unit variant](https://github.com/flashinfer-ai/flashinfer/blob/8e137c135f3abc9bce887111f8a35d7574244bae/csrc/cake_fused_qk_rope_append/sm_100a/cake_fused_qk_rope_append_d3c33d4c6b33e645e252_kernel.cu)
shares sequence/page lookup between its K and V clear units. Disabling tail
clearing does not remove clearing CTAs from this host plan or alter stage
selection; they launch and skip their writes.

The public contract remains packed BF16, D=128, 8/1 or 64/8 heads, NHD cache,
independent optional K/V redirects and caller-owned writable pages. The
direct path reads guessed metadata before testing `req >= 0`: `B=0, T>0`
passes the wrapper's structural checks but produces `g_req=-1` on Hopper
8/1. Do not pass it as an “ignored row” case. This is a source-derived
precondition gap, not a reproduced GPU fault. The review bot's two summaries
disagree about reachability; the wrapper and device reads establish the gap.
Whole-empty input still reaches the positive-grid binding check, and zero
page size is not validated by this wrapper.

There is also a reproducible **CPU policy** limit: SM100, 148 SMs, 64/8,
`T=B=8`, page size 1024 selects `hq64_hkv8_w20_vsub18_d_u2`, which the
manifest does not ship, so `stage_for` raises. Running the extracted pure
selector against its exact manifest reproduced that exception. The CPU
inventory test stops at page size 128. Frozen main explicitly separates the
eight-warp two-unit rule from `w20` and supplies later twenty-warp schedules;
do not carry this historical missing-build conclusion onto current policy.
Similarly, the legacy no-shape `stage_for(8, 1)` returns a base name absent
from SM90's inventory; that helper result alone is not a loadability check.

### Tests and performance evidence

The [test file](https://github.com/flashinfer-ai/flashinfer/blob/8e137c135f3abc9bce887111f8a35d7574244bae/tests/attention/test_cake_fused_qk_rope_append.py)
contains the unchanged 21 CUDA-dependent #5952 cases plus **21 CPU policy
cases**: 17 table entries, three inventory sweeps and one legacy-name test.
The added sweep checks membership, not kernel outputs. Applying the exact
selector to the numerical/replay fixtures reaches three of four SM90
programs at 132 SMs and four of eight SM100 programs at 148 SMs. It misses
SM90's 64/8 direct base and SM100's 64/8 direct, direct/u2, u2 and non-direct
twenty-warp programs. Thus “42 passed” does not prove every generated program
was numerically exercised. The 300-request decode test named
`test_large_batch_lookup_fallback` now gets correct direct guesses for every
row; it does not establish the beyond-256 fallback path. Replay still uses
unchanged input/metadata and the original 8/1 shape.

The [H100 targeted job](https://github.com/flashinfer-ai/flashinfer/actions/runs/37099576934/job/111136473402)
checks out merge `69be132d` of final head `8e137c135` into `1fc76f71`, runs
this file under CUDA 12.9 and reports **42 passed, zero failed/skipped**.
The B200 42-case result, three-card bit equality and H200/B200 sanitizer
results with negative controls remain author reports. This refresh did not
run GPU tests or establish B300 numerical coverage.

The body reports graph-mode paired CUPTI measurements versus #5952, with
23 gate shapes and 28 reporting shapes. Reported geometric mean ratios are
1.009×/1.065×/1.066× on B200/H200/H100 for the gate set and
1.023×/1.071×/1.073× for all 28; one B200 large-decode row reaches 1.30×.
Its “no regression” statement is tied to the reported paired 95% confidence
intervals. The committed 17-shape benchmark and timing helper were not
changed to reproduce that protocol or retain its raw samples. The base
sources were regenerated; old/new cubins or disassembly are not supplied to
verify the separate “byte-identical builds” claim. These historical results
do not establish current-policy or serving-level gains. Profile the selected
symbol and clear-grid work for the actual shape, then validate its numerical
and cache-ownership contract before treating the change as a reusable gain.

## PR #4890 — HY3 uniform-decode scheduling

- Link: https://github.com/flashinfer-ai/flashinfer/pull/4890
- Title: `feat(rope): add HY3 fused QK RMSNorm/RoPE KV-store`.
- State: open, currently unmergeable; head
  `7f12c326b3774554599f34920c54b72b64a5c1f7`, API base
  `21ec84517587c22f2d6fc622dbe8ad47a478b272`, actual diff merge base
  `71d31b5a23a3c0394edb36330dec1ce2a0def365`.
- Coverage: complete 2,792-line, eleven-file diff, +2,663/-0; body, two issue
  comments, one review and the sole commit; no inline comments. The bot's
  skipped-review comment counts 393 base-to-head files, while the PR's actual
  merge-base diff and changed-file API agree on eleven. That bot inventory
  is not the patch scope or a code review.
- Reviewed runtime: `include/flashinfer/rope_norm_store_kv_hy3.cuh`,
  `csrc/{rope_hy3,flashinfer_rope_hy3_binding}.cu`,
  `flashinfer/{rope,__init__,aot}.py` and `flashinfer/jit/rope_hy3.py`.
  Reviewed validation: `benchmarks/bench_rope_hy3.py`,
  `tests/attention/test_rope.py`, `flashinfer/trace/templates/rope.py` and
  `tests/trace/example.py`; also the actual timing helper and FFI check macros.
- Motivation: expose the packed HY3 QKV contract as an opt-in API and optimize
  its large uniform decode batches by eliminating segment lookup and separate
  tail-clearing CTAs. The header retains Tencent's MIT provenance and source
  commit `1cd332980ed46bd0172091c1c35d55338fcae47a`.

The public function `qk_rmsnorm_rope_append_paged_kv_cache_hy3` accepts
contiguous BF16 packed QKV, head pairs 8/1 or 64/8, D=128, FP32 full NeoX
cos/sin and norm weights, and contiguous NHD BF16 or E4M3 caches. Epsilon
is fixed at `1e-6`; policies 0/1/2 select no norm, RoPE-before-norm or
norm-before-RoPE. Its custom op declares all Q/scale/flag/K/V/cache mutations,
has a fake implementation and uses cached JIT/optional-argument placeholders.
AOT adds the module to the ordinary build list. The patch adds an explicit
API; it does not wire a SGLang/vLLM model to it.

The general kernel launches `(ceil(T/4) + B, Hkv)` CTAs of 128 threads, with
9 KiB dynamic shared memory for four rows of eight Q heads plus one K head.
Each warp owns one token/KV-head group, processes its eight Q heads in turn,
then K and V. Q/K are staged with vector loads; a parallel scan of indptr
segments assigns rows to requests. Norm/rotation remain FP32 until the final
store, with full-warp XOR reductions. V uses vector BF16 or packed FP8 stores.
The extra B CTAs per KV head clear each request's last-page tail even when
optional K/V outputs redirect the token writes away from the cache.

The Python flag `uniform_one_token_decode=True` is an explicit caller promise
that `q_indptr == arange(B + 1)`, not a device-data check. Python additionally
checks exact SM100. The C++ dispatcher selects only the measured specialization:

```diff
+  // ragged or CUDA-graph-padded batches, so this remains an explicit flag.
+  const bool use_sm100_fast_path = enable_sm100_uniform_decode && !is_prefill && is_fp8 &&
+                                   quant_policy == 1 && norm_policy == 2 && num_q_heads == 64 &&
+                                   num_kv_heads == 8 && batch_size >= 256 && num_rows == batch_size;
```

That specialization has grid `(ceil(T/4), Hkv)`, directly uses row as request,
keeps rotary coefficients in registers, and clears the owned request/head
tail after its token writes. Eight lanes cover one 128-byte FP8 row, so a warp
clears four tail rows per iteration. The reduced CTA count and omitted lookup
are the optimization; arithmetic remains the same. BF16, static Q, other norm
policies, smaller batches, 8/1 heads, prefill and non-SM100 use the general
path even if the flag is true. Matching T and B alone is insufficient for a
ragged or padded batch. The launch is ordinary current-stream CUDA; it has no
PDL handshake.

Dynamic FP8 Q uses `max(max_abs, 1e-6) / fp8_upper_bound` and reciprocal
multiplication. Decode scales are `[T, Hq]`; prefill scales are
`[B, Hq, round_up(max_sequence_length, 128)]`. Only live entries are written.
Static Q uses a scalar inverse scale; K/V use reciprocal dequantization scales.
BF16 and static-Q calls return `None` for inapplicable scales. K and V output
redirects are independently optional and still leave tail-clearing side effects.

The FFI validates positive B/T/page size, contiguous same-device tensors,
shapes/dtypes and scalar-policy compatibility. It does **not** validate device
indptr contents, per-request length versus the prefill scale allocation,
sequence/rotary/page bounds or scale values. FP8 flags are set to zero by the
first valid row of each request. An empty request has no such write, and
new flags are allocated with `torch.empty`; they are not a complete metadata
verdict or an inter-kernel completion signal. Negative positions return early.
Callers must establish these preconditions, including safe writable cache-page
ownership, before replay. The uniform scheduling promise cannot repair invalid
metadata.

### Test and benchmark coverage

The 59 source cases comprise 54 combinations of two head layouts, three
prefill/decode/MTP modes, three norm policies and three cache/quant policies;
one multipage reversed-page-table/redirected-output case; two output-dtype
rejections; one delayed-producer non-default-stream test; and one SM100
B=256 fast/general parity test. The stream test warms JIT first and places a
delayed QKV copy before the operator, so it meaningfully detects an accidental
default-stream launch. The fast-path test compares Q, scales, flags and full
caches with `torch.equal` and checks caller-buffer identity.

The 54 ordinary cases use small B=3 and a float reference. BF16 Q/K permit
0.08 absolute error, FP8 Q permits 1.0 and dequantized caches permit 0.5;
dynamic scales permit `rtol=2e-4, atol=1e-7`. Cache-tail checks inherit those
tolerances, except the fast/general parity comparison. There are no empty
request, malformed-indptr, undersized prefill-scale, CUDA Graph replay or
attention-consumer tests. The author reports 59 passes on B200 and four trace
passes/three skips. These are attributed results, not GPU reruns here.

The new trace template only describes the uniform 64/8 dynamic-FP8 norm2
case and has no numerical reference. Its dispatcher filters logical policies
and redirects, but does not check SM100 or B>=256. A matching trace-template
name/tag therefore does not establish that the C++ fast branch was selected.
The example suppresses exceptions; successful execution of the example script
alone does not prove this operator ran.

The benchmark fixes 64/8, D128, page64 and B256/512, preallocates outputs and
prepares separate contiguous Q/K/V outside timing. Its static comparator is
two BF16 RMSNorm calls followed by the existing FP8 RoPE/append API, with PDL
disabled. The BF16 intermediate introduces rounding absent from HY3's FP32
chain. The harness compares Q and written K at `rtol=0.2, atol=0.5`, V exactly,
and does not require identical full-cache/flag side effects: the composition
does not clear tails or write flags. Those omitted operations reduce the
baseline's work, but the comparison still is not an equivalent numerical
contract.

Dynamic fast/general arms do check Q, scales, flags and full-cache equality
before timing. They run in fixed provider order, using unchanged inputs and
cache locations, 20 warmups and 100 samples, and print median/p20/p80. The
reported B200 fallback uses CUDA events without graphs; the helper flushes
L2 before each timed interval, and intervals can include inter-launch gaps.
No randomized paired order, raw-sample archive or confidence interval is
provided. Historical static ratios are 1.183×/1.135×; dynamic scheduling ratios
are 1.115×/1.128× for B256/512. They are operator comparisons, with allocation,
input splitting, metadata and JIT excluded, not serving throughput or current
CAKE performance.

Final-head [PR Test](https://github.com/flashinfer-ai/flashinfer/actions/runs/33583869655)
has skipped setup/test jobs. Its [summary log](https://github.com/flashinfer-ai/flashinfer/actions/runs/33583869655/job/100103866762)
names head `7f12c326b377` and reports pending CI authorization. The red result
is not a failed GPU numerical test. The documentation bot separately reports
the new public API missing from `docs/api/*.rst`.

### Relationship to current CAKE

On November 3, the [maintainer review](https://github.com/flashinfer-ai/flashinfer/pull/4890#pullrequestreview-5756710557)
suggests closing this proposal in favor of merged #5952/#5956 and transferring
the large-batch schedule to CAKE FP8. It reports an approximately 1.4× advantage
at B512, without an attached exact comparison build/protocol. Treat that as a
maintainer observation, not a reproduced benchmark or evidence that the
schedule was merged. The PR remains open; its HY3 entry point/header are
absent from frozen main.

The current CAKE modules below cover the same broad packed-QKV work, but
are not drop-in aliases. FP8 requires paired K/V redirects, returns three
tensors, validates indptr in-kernel and writes every request's status,
including empty requests. BF16 permits optional clearing and configurable
epsilon. Their later shape/PDL policies differ from HY3's explicit uniform
flag. Transfer a schedule only after matching these interfaces, output and
metadata obligations; the old statement that no existing API covers dynamic
Q scales/tail flags is no longer current.

Immutable proposal sources: [kernel](https://github.com/flashinfer-ai/flashinfer/blob/7f12c326b3774554599f34920c54b72b64a5c1f7/include/flashinfer/rope_norm_store_kv_hy3.cuh),
[FFI admission/dispatch](https://github.com/flashinfer-ai/flashinfer/blob/7f12c326b3774554599f34920c54b72b64a5c1f7/csrc/rope_hy3.cu),
[Python API](https://github.com/flashinfer-ai/flashinfer/blob/7f12c326b3774554599f34920c54b72b64a5c1f7/flashinfer/rope.py),
[tests](https://github.com/flashinfer-ai/flashinfer/blob/7f12c326b3774554599f34920c54b72b64a5c1f7/tests/attention/test_rope.py),
[benchmark](https://github.com/flashinfer-ai/flashinfer/blob/7f12c326b3774554599f34920c54b72b64a5c1f7/benchmarks/bench_rope_hy3.py),
[timing helper](https://github.com/flashinfer-ai/flashinfer/blob/7f12c326b3774554599f34920c54b72b64a5c1f7/flashinfer/testing/utils.py)
and [trace dispatch](https://github.com/flashinfer-ai/flashinfer/blob/7f12c326b3774554599f34920c54b72b64a5c1f7/flashinfer/trace/templates/rope.py).

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
