# FlashInfer QK RMSNorm, RoPE and paged append

Reviewed on 2026-11-09 (Asia/Shanghai). The complete final #5405 diff, body
and all discussion were read. Current-source comparisons use FlashInfer main
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
They are closure context here; their complete historical diffs remain a
separate pending review. Current availability below is established from the
frozen mainline implementation, not by treating #5405 as merged.

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
and `clear_last_page`; optional K/V outputs remain independent redirects.
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
