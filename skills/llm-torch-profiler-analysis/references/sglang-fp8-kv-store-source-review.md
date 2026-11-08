# SGLang FP8 KV writes: prefix commits, saturation and ordinary stores

Reviewed on 2026-11-09 (Asia/Shanghai). All three complete final diffs, bodies
and all discussion were read. Current-source comparisons use frozen SGLang
main `c24285bde97c3e6424cd5d8cb35788889034d8ea`. The prefix writer and its
saturation fix are present there; the ordinary-store proposal is not. No GPU
experiment was run for this source review.

## PR #40785 — Fuse FP8 conversion into prefix-valid commits

- Link: https://github.com/sgl-project/sglang/pull/40785
- State: merged 2026-10-05; head `f97b5fb849aa9d0d459f2c42474dd05bd9e3b68e`,
  base `b1bbd74f287f13ed1276b0403a01ebb55c597e93`, merge
  `cafb37a10dc6994587c98253049efe30a6f3995f`.
- Coverage: all 802 diff lines, six files, +715/-15; body, eight issue
  comments, three reviews, two inline comments; 21 commits.
- Reviewed files: `python/sglang/kernels/ops/kvcache/cache_move.py`,
  `python/sglang/srt/mem_cache/memory_pool.py`,
  `python/sglang/srt/models/dflash.py`, shared oracle
  `python/sglang/test/kernels/prefix_valid.py`, and
  `test/registered/{kernels/ops/kvcache/test_prefix_valid_fp8.py,unit/mem_cache/test_prefix_valid_fp8_pool.py}`.
- Motivation: committing accepted DFlash rows previously launched two scale
  divisions, two casts and a byte scatter. Intermediate K/V tensors crossed
  HBM between these operations.

The new Triton writer keeps the `(batch, row, tile)` grid. Rows beyond each
request's `commit_lens` return before reading their destination, so rejected
rows may contain sentinel locations. Accepted rows use `loc_2d` indirection
and independent K/V element strides. Typed FP8 views of byte-backed buffers
make conversion part of the scatter. The wrapper still materializes
contiguous sources and metadata when needed; include those copies in timing.
Tiles contain 64, 128 or 256 elements for row widths below 2048, below 4096,
or at least 4096, respectively. Both K and V use the same row width.

Scale resolution is deliberately narrower than arbitrary tensor division.
Python numbers and CPU scalar tensors become host floats. A GPU scalar is
accepted only when it is the layer's exact parameter object and has an
existing numeric float shadow:

```diff
+        isinstance(scale, torch.Tensor)
+        and scale.ndim == 0
+        and scale.is_cuda
+        and scale is layer_scale
+        and isinstance(layer_scale_float, (float, int))
+    ):
+        return float(layer_scale_float)
```

Both resolved scales, E4M3 cache storage, and a differing source/cache dtype
are required. Missing scales and unresolved GPU tensors take eager division
using the original objects. On CUDA/HIP, asymmetric K/V widths raise
`NotImplementedError` before fusion dispatch; the body incorrectly describes
this as a fallback. Non-CUDA/non-HIP execution first selects accepted rows
and delegates to its ordinary writer.

Numerical equivalence includes the scale's *kind*. For a GPU tensor scale,
the kernel rounds the divisor to the source dtype, divides with `tl.div_rn`
in FP32, then rounds the quotient back to the source dtype before FP8
conversion. Host/CPU scalar scales use multiplication by a rounded FP32
reciprocal instead. Removing either rounding boundary can change FP8 bytes.
The September 30 review's BF16 `0.0966796875 / 0.1` example exposed the
missing divisor rounding in an earlier revision; the final diff fixes it.
Float shadows avoid a GPU-to-host synchronization, but they are not live
device loads during graph replay. Changing scales requires keeping parameter
and shadow state consistent and updating any captured host-valued arguments.

The same review found that the draft's `RadixAttention` never received
`quant_config`, leaving the scale-dependent fast path unreachable. The final
diff passes it through. `Fp8Config.get_quant_method` then selects the existing
`Fp8KVCacheMethod`: loading establishes scalar parameters and float shadows,
using 1.0 when scales are absent and adjusting loaded scales for FNUZ. This
does not make every unquantized draft eligible just by selecting FP8 storage.

Current DFlash uses `layer_router.layer(...).store_prefix(...)` with physical
locations, both in the per-layer path and the fused-materialization callback.
The router reaches `_set_mha_kv_buffer_prefix_valid`; ordinary MHA pools and
the unified allocator's `MHAView`/`DraftMHAView` share that helper. DSpark's
per-layer context path also calls it. Current tests exercise interleaved
draft views and verify that host bytes, other draft lanes and rejected rows
remain unchanged. MLA uses its separate prefix writer.

The final changed files contain three kernel test methods and six pool/
initialization/scale-resolution methods, with many subcases. They cover whole
cache bytes, NaNs and signed zero, untouched slots, all tile branches,
noncontiguous inputs, mixed commit lengths, all BF16/FP16 bit patterns,
FP32 values around rounding boundaries, scale-kind combinations, and real
DFlash scale initialization. Fusion leaves input K/V unchanged; the fallback
may mutate them. The body reports seven B200 tests under Torch 2.13.0+cu130
and Triton 3.7.1; that count is not the final nine-method file inventory.
Its original ROCm validation was incomplete: #42982 below supplies the
necessary follow-up rather than assuming CUDA parity transfers.

The B200 report uses Llama-3.1-8B with a DFlash draft, FP8 draft quantization
and KV, Triton attention, batch four, 48 output tokens, fixed acceptance
length four, and CUDA graphs. Both variants recorded 65 prefix commits;
the baseline retained the same quantization/scales and forced eager commit.
The reported per-layer cluster falls from 14.528 to 4.864 microseconds
(2.99x), **including the required source-layout copy**. The fused kernel's
1.951 microseconds alone is not a comparable cluster baseline. Ten
alternating process-level A/B pairs, each with ten warmup and 100 measured
batches, report a mean paired request-p50 saving of 0.25 ms with 95% bootstrap
interval [-0.96, +1.11] ms. That interval does not establish an E2E gain.
These are PR reports, not a rerun of the current caller or saturation fix.
The body's stale rebase/benchmark checklist and an October comment linking
another PR's CI are not evidence of this final head's validation.

Source anchors: [merged implementation's kernel](https://github.com/sgl-project/sglang/blob/f97b5fb849aa9d0d459f2c42474dd05bd9e3b68e/python/sglang/kernels/ops/kvcache/cache_move.py),
[final pool guards](https://github.com/sgl-project/sglang/blob/f97b5fb849aa9d0d459f2c42474dd05bd9e3b68e/python/sglang/srt/mem_cache/memory_pool.py),
[current shared prefix writer](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/mem_cache/memory_pool.py),
[current DFlash callers](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/speculative/dflash_worker_v2.py),
[layer router](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/mem_cache/kv/router.py),
[unified MHA views](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/mem_cache/kv/layout/mha.py),
and [scale initialization](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/quantization/kv_cache.py).

## PR #42982 — Define saturation consistently for prefix commits

- Link: https://github.com/sgl-project/sglang/pull/42982
- State: merged 2026-10-10; head `21ae7868170d4940fde5ff05c6faf1dfe53f5de7`,
  base `d9a0f1cb340cdbf1842bf9526015d846eb72f25d`, merge
  `37114afe206575d70b372e534f9826a6f5dd4cae`.
- Coverage: all 224 diff lines, five files, +114/-6; body, two issue comments,
  three reviews, one inline comment; three commits.
- Reviewed files: `python/sglang/kernels/ops/{kvcache/cache_move.py,quantization/fp8_kernel.py}`,
  `python/sglang/srt/mem_cache/memory_pool.py`,
  `python/sglang/test/kernels/prefix_valid.py`, and
  `test/registered/unit/mem_cache/test_prefix_valid_fp8_pool.py`.
- Motivation: after #40785, the AMD E4M3FNUZ eager and Triton conversions
  disagreed on overflow, producing 30,732 mismatches out of 65,536 values in
  the reported regression. Different dispatches could write different bytes
  into the same committed cache slot.

The fix defines saturation explicitly in the fused writer, eager prefix
fallback and shared test oracle. Clamp to `torch.finfo(cache_dtype).max`,
preserve input NaNs, and retain the source dtype through the clamp. In
particular, FNUZ's representable limit is 240, not the separate conservative
quantization constant 224. The kernel's crucial two operations are:

```diff
+    val = tl.where(val > FP8_MAX, FP8_MAX, val).to(DTYPE)
+    return tl.where(val < -FP8_MAX, -FP8_MAX, val).to(DTYPE)
```

Clamping in FP32 and converting directly to FP8 changes the lowering from
BF16/FP16-to-FP8. An earlier patch still differed by one FP8 subnormal step
for two FP16 inputs; the final source-dtype casts retain the required
conversion boundary. Treat the explicit clamp as the contract rather than
assuming every raw PyTorch/backend FP8 cast has identical overflow behavior.
This patch only changes the prefix path; current ordinary stores still have
their own conversion, addressed by the next proposal.

The initial oracle shared the production clamp, so a bug could move both
implementation and expectation together. Review requested an independent
pool test. The final head adds one that forces unresolved GPU-scale fallback
and checks literal outcomes: positive/negative overflow reaches the dtype
limits, NaN remains NaN, 1.0 divided by 0.5 stays 2.0, and uncommitted slots
remain byte-identical. Eager clamp was also simplified to `value.clamp`,
removing a redundant `isnan`/`where` pair. This is useful behavioral coverage
to retain when cleaning tests.

The body and review report all 42 MI300/ROCm-10 JIT test files passing on
earlier head `eaebedfca946d819e476cfeca9727c2883c05953`, including both
prefix FP8 files. The linked [run](https://github.com/sgl-project/sglang/actions/runs/37648241557)
has an overall failure despite that focused result. Fresh job metadata also
confirms the final head's [MI300 JIT unit job](https://github.com/sgl-project/sglang/actions/runs/37948936130/job/113882557175)
and JIT benchmark job succeeded. Its [AMD workflow](https://github.com/sgl-project/sglang/actions/runs/37948936130)
still has an overall failure. Preserve both the focused pass and the failed
aggregate rather than inferring one from the other. The body attributes other failures to separate
paths, but this review does not certify every such diagnosis. No timing
evidence establishes that the added clamp has zero runtime cost.

Source anchors: [final clamp and conversion](https://github.com/sgl-project/sglang/blob/21ae7868170d4940fde5ff05c6faf1dfe53f5de7/python/sglang/kernels/ops/kvcache/cache_move.py),
[eager saturation helper](https://github.com/sgl-project/sglang/blob/21ae7868170d4940fde5ff05c6faf1dfe53f5de7/python/sglang/kernels/ops/quantization/fp8_kernel.py),
and [independent fallback regression](https://github.com/sgl-project/sglang/blob/21ae7868170d4940fde5ff05c6faf1dfe53f5de7/test/registered/unit/mem_cache/test_prefix_valid_fp8_pool.py).

## PR #31652 — Fuse ordinary dense NHD FP8 stores

- Link: https://github.com/sgl-project/sglang/pull/31652
- State: open, unmerged, with merge conflicts at the check; head
  `10f5f9f732ddd5d12debd0aee1a11a111e459e30`, base
  `4eb606b2aee8a9165e16352fd4c08d998ba37990`.
- Coverage: all 908 diff lines, seven files, +742/-24; body and all six issue
  comments; no reviews or inline comments; one commit.
- Reviewed files: `python/sglang/kernels/ops/kvcache/{__init__,cache_move}.py`,
  `python/sglang/srt/mem_cache/memory_pool.py`, new
  `test/registered/kernels/{benchmark/kvcache/bench_store_cache_fp8.py,ops/kvcache/test_store_cache_fp8.py}`,
  `test/registered/unit/kernels/test_kernels_namespace.py`, and
  `test/registered/unit/mem_cache/test_asymmetric_mha_pool.py`.
- Motivation: ordinary `MHATokenToKVPool.set_kv_buffer` still performs two
  divisions, two casts and a byte store when scaled, or two casts plus store
  without scales. Prefix fusion does not change that entry point.

The final proposal extracts `_quantize_kv_fp8` from the saturated prefix
writer and uses it in a separate `(token, element tile)` ordinary scatter.
The two writers share conversion semantics, not their indexing schedules.
The new `kvcache.store_cache_fp8` registration requires CUDA SM89+, and pool
construction independently checks CUDA device, E4M3 dtype and NHD layout.
Admission also requires matching BF16/FP16/FP32 K/V dtypes, three-dimensional
sources with contiguous head/dimension axes, no DCP mask, and supported
scales. K/V widths and token strides may differ. CPU scalar tensors,
non-scalar tensors, inner-strided sources, other layouts and unsupported
devices retain the existing writers.

Host numbers and absent scales are supported; `None` becomes 1.0. Scalar
FP32 device tensors are loaded inside the kernel, so their values can change
between CUDA graph replays without host shadows. Source-rounded tensor
division and host-reciprocal multiplication still differ, followed by source
rounding, saturation and FP8 conversion. The ordinary eager E4M3 fallback
also gains explicit saturation. That aligns an overflow policy; it is not
proof that every backend's old raw conversion was identical.

Unlike prefix commits, ordinary padding rows target reserved slot zero and
must not be written. Signed 64-bit slot arithmetic and always-enabled Triton
debug assertions preserve invalid-slot failure behavior:

```diff
+    slot = tl.load(indices + row * index_stride).to(tl.int64)
+    # Slot zero is the ordinary writer's reserved CUDA-graph padding slot.
+    valid = (slot > 0) & (slot < SIZE_LIMIT)
+    tl.device_assert((slot >= 0) & (slot < SIZE_LIMIT), "KV slot out of bounds")
```

The launch uses 256-element tiles, independent K/V destination strides and
masks, and skips empty index lists. Pool/caller contracts still own input
sizes, layout, device consistency and valid write locations. Do not expose
the low-level helper as an arbitrary-layout checked API. The CUDA JIT
baseline asserts bounds and skips slot zero too.

Current caller coverage needs a fresh check when porting: the diff only
changes `MHATokenToKVPool.set_kv_buffer`. Current `MHAView.set_kv_buffer` in
`mem_cache/kv/layout/mha.py` is an independent implementation, reached through
the layer router for unified arena views; it still divides/casts/stores
separately. `DraftMHAView` inherits it. Do not claim the proposal accelerates
those paths without adapting and validating that writer. In contrast, their
prefix writes already share the merged helper. Triton prefill also keeps
defensive K/V copies because fallback stores can mutate them; fusion's
nonmutation does not authorize deleting those copies unconditionally.

The new six-method kernel suite checks whole-cache bytes, untouched slots,
BF16/FP16/FP32, padded rows, int32/int64 indices, asymmetric widths, NaNs,
overflow, scale kinds, every BF16/FP16 bit pattern under selected scales,
reserved-slot skipping and empty input. Pool tests additionally cover actual
dispatch, HND/DCP fallback, independent storage offsets, a mocked SM80 gate,
and graph replay after changing both device scales and locations. Namespace
tests check registration capability, not execution on each hardware family.

The current-head body reports 108 focused tests passing with no skips on
RTX PRO 6000 Blackwell Max-Q (SM120), Torch 2.14.1+cu130 and Triton 3.8.0.
Its linked [base CI run](https://github.com/sgl-project/sglang/actions/runs/39048453007)
is for that exact head: the PR gate failed and the GPU test jobs were skipped.
That is neither an observed kernel-test failure nor all-CI success; keep the
author's local validation report separate.
Three serial Qwen3-0.6B prompts with 24 generated tokens reportedly matched
tokens/text/logprobs in eager and graph modes on **earlier head `e42753125`**.
Negative and upper-bound assertion checks are also reported for that earlier
head. Physical SM89, Hopper and ROCm execution are not established by these
reports or the mocked capability test.

The current-head benchmark measures the conversion/store cluster with BF16
sources, E4M3 caches of 1M slots, tensor or absent scales, five head geometries
and powers-of-two batch sizes 1–16384. Reported fused/eager geometric means
are 0.274 with scales and 0.534 without (75 cases each). For 8x128 heads,
scaled B1 is 11.22 to 2.54 microseconds; unscaled B16384 is 64.18 to 55.13.
These are author measurements from an unreserved device.

Read the harness when interpreting its “cold L2” description. `marker.do_bench`
defaults to CUDA graphs, flushes L2 **before each graph replay**, and rotates
clones of K, V and indices inside the graph. The two cache buffers are not
cloned, and cloned indices have the same values, so writes revisit the same
slots within that graph. This is not proof of cold destination lines for
every timed call. The eager baseline divides out of place and omits the new
explicit clamp; it represents the old ordinary conversion/store sequence,
not the updated fallback's exact operation count. The timing harness has no
numerical assertion; correctness comes from the separate test suite.

The earlier `e42753125` serving report uses unprofiled, serial Qwen3-0.6B
`Engine.generate`, 48 generated tokens, eager mode, disabled overlap/radix
cache, and ABBA source order with two boots per source. Median latency falls
from 629.35 to 593.82 ms and 617.65 to 597.39 ms for two prompts (3.3–5.6%).
It does not measure concurrent throughput, graph-mode serving speed, this
final head, or current unified-memory dispatch. Old catalog measurements
from `97d99726b143` must not be substituted for these newer scoped reports.

Source anchors: [final proposed kernel](https://github.com/sgl-project/sglang/blob/10f5f9f732ddd5d12debd0aee1a11a111e459e30/python/sglang/kernels/ops/kvcache/cache_move.py),
[pool admission](https://github.com/sgl-project/sglang/blob/10f5f9f732ddd5d12debd0aee1a11a111e459e30/python/sglang/srt/mem_cache/memory_pool.py),
[benchmark](https://github.com/sgl-project/sglang/blob/10f5f9f732ddd5d12debd0aee1a11a111e459e30/test/registered/kernels/benchmark/kvcache/bench_store_cache_fp8.py),
[graph timing implementation](https://github.com/sgl-project/sglang/blob/10f5f9f732ddd5d12debd0aee1a11a111e459e30/python/sglang/kernels/jit/benchmark/marker.py),
[pool regression cases](https://github.com/sgl-project/sglang/blob/10f5f9f732ddd5d12debd0aee1a11a111e459e30/test/registered/unit/mem_cache/test_asymmetric_mha_pool.py),
and [current Triton caller copies](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/triton_backend.py).
