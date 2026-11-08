# vLLM CUDA QK normalization, RoPE and cache fusion

Reviewed on 2026-11-09 (Asia/Shanghai). Current-source comparisons use frozen
vLLM main `b022bb7a4873a47cdcdf474edf3bd499c221f40d`. The full PR diff, body,
all discussion and the relevant current callers, compiler gates and numerical
implementations were read. No GPU experiment was run for this review.

## PR #57682 — Extend the generic QK-norm/cache pass to CUDA

- Link: https://github.com/vllm-project/vllm/pull/57682
- State: open; head `6f11616a53a64aeee73d210e786434677b38c799`, base
  `852989f580820849113d67d13fd8053e279d0444`, checked on November 9.
- Coverage: all 1,539 diff lines, nine files, +1,059/-85; four issue comments,
  one review submission, no inline comments and both commits. The review
  submission says automated review was disabled; it is not an approval.
- Reviewed files: `csrc/{fused_qknorm_rope_kernel.cu,ops.h,torch_bindings.cpp}`;
  `vllm/_custom_ops.py`; `vllm/config/compilation.py`;
  `vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py`;
  `vllm/v1/attention/backends/flash_attn.py`; and the new
  `tests/{compile/passes/test_qk_norm_rope_kvcache_fusion.py,kernels/core/test_fused_qk_norm_rope_kvcache.py}`.
- Motivation: after the existing CUDA QK-norm/RoPE fusion, a separate
  `reshape_and_cache_flash` launch rereads rotated K and raw V. Extend that
  numerical implementation with a cache epilogue and connect it to the
  existing combined compiler pass.

The new `WRITE_CACHE` specialization treats packed QKV as read-only. It writes
contiguous Q and K outputs, writes the same rounded K values directly into
the paged cache, and assigns V heads to copy their input into the value cache.
V remains a view of the original QKV in the replacement graph. The change
removes a launch and K's reread; it does not eliminate K output materialization
or the cache's K/V stores. The original entry point uses `WRITE_CACHE=false`.

The one-head kernel assigns one warp per token/head. The multi-head variant
uses `cp.async` to stage QKV and the rotary table, reuses weights and cos/sin
across heads of the same token, and supports forced groups of 1/2/4/8 heads.
Automatic grouping retains the original SM90 thresholds based on the number
of **Q+K** token-heads, even though cache mode adds V work. Other architectures
default to one head per warp. Do not transfer those tuning thresholds to a
new GPU without measurement. The existing small-token PDL launch threshold
remains 16; the kernel waits before reading the projection output.

Cache addresses use separate 64-bit K/V block, slot and head strides.
Negative slots and tokens beyond the slot-map length skip cache writes while
still producing Q/K. Host checks require contiguous packed QKV, contiguous
Q/K outputs, matching FP16/BF16 activation/norm-weight/output/cache dtypes,
equal K/V head counts, four-dimensional cache views with contiguous head
elements, and a contiguous int64 slot vector no longer than QKV. The normal
FlashAttention cache supplies aligned vector-store addresses; these checks
do not prove arbitrary outer-stride alignment or validate slot bounds.

### Compiler admission and numerical contract

The FlashAttention hook requires CUDA, decoder attention, `kv_cache_dtype=auto`,
head width 64/128/256, no shared KV owner and DCP size one. The pass also checks
BF16/FP16 model dtype and equal Q/K/V widths, registers epsilon 1e-5/1e-6 and
both rotary styles, and leaves FP8-query patterns on ROCm only. CUDA MRoPE
and FP8-cache fusion are not added here. With opaque `LayerName` inputs,
**every** attention layer must be supported and have the same geometry;
otherwise the constructor returns without registering CUDA patterns.

```diff
+            if len(geometries) > 1:
+                logger.warning_once(
+                    "QK Norm+RoPE+KVCache fusion not enabled: attention layers "
+                    "have different geometries %s.",
+                    sorted(geometries),
+                )
+                return
```

The outer fused op resolves the cache and slots from forward context, mutates
graph-owned Q/K outputs and returns the empty dependency used to order the
attention consumer after hidden cache writes. Its no-slot profiling branch
zeros Q/K instead of invoking the cache writer. The C++ op separately declares
Q/K outputs and both caches mutable. FlashAttention's inherited
`set_interleaved_v_cache` remains a no-op, preserving its ordinary layout.

The CUDA flag is opt-in. The PR body says ROCm defaults start at O1, but the
PR-head configuration table enables the conditional resolver at **O2/O3**;
O1 is false. Normal default splitting without Inductor partition still clears
the requested combined flag. The default maximum compile-range end is 256.
The direct test backend invokes passes without `is_applicable_for_range`, so
its 2,048-token match does not establish that the normal default pipeline
fuses that range. See the [compiler review](vllm-compiler-source-review.md)
for custom-op/default ordering, graph visibility and capture-policy effects.

This proposal preserves the existing fused QK-norm/RoPE arithmetic: FP32 norm
results feed rotation without an intermediate activation-dtype store, then
Q/K are rounded to FP16/BF16. Its bitwise baseline is that fused kernel plus
a separate cache write. It is not bitwise parity with separate RMSNorm and
RoPE kernels, which have an additional rounding boundary.

The first stacked commit comes from [#57681](https://github.com/vllm-project/vllm/pull/57681).
It moves NeoX shuffle exchanges out of a lane-divergent partial-rotary branch
in **both** the original and cache-writing kernels. Full-warp synchronization
inside a branch entered only by rotary lanes was invalid; the frozen main
still contains that earlier branch. Keep this dependency when evaluating the
cache fusion. The checked-in partial case is rotary width 64 in a 128-wide
head. Host validation only checks even, bounded rotary width; that case is
not evidence for every partial width or every XOR-shuffle geometry.

### Validation and current-main comparison

The final kernel test source defines 99 parameter combinations: 80 geometry/
dtype/style/token cases, 16 forced-group cases and three padded/short-slot
cases. It compares Q, K and the entire initialized cache bitwise against the
fused baseline pair and verifies QKV is unchanged. The compile test defines
80 combinations across five shapes/styles, 5/2,048 tokens, two cache layouts,
block sizes 16/64 and two custom-op settings, all BF16. It checks one matched
pattern and uses `atol=rtol=0.01` for Q/K/cache, exact V. These are source
coverage counts, not executed-test results. Neither new test runs an attention
consumer or tests changed-buffer CUDA-graph replay.

The author's separate H200 report used 74 standalone cases, a kernel compiled
against the Torch environment of vLLM 0.28.0, and its shipped baseline pair.
Reported pair-time reductions are 31%, 20%, 6% and 16% at 64/256/2,048/8,192
tokens for Q/K heads 32/8 and head width 128. These 74 cases are not a passing
result for the final checked-in 99-case suite.

Serving used Qwen3-8B BF16 **dummy weights**, with both arms enabling the
existing QK-norm/RoPE fusion and Inductor graph partition. Two ABBA runs per
arm report 0.6–1.0% latency reductions at batch 1/8/32, effectively unchanged
prefill throughput and +0.1% decode throughput. Twelve greedy sequences match
for 32–64 generated tokens; dummy-weight agreement is not pretrained-model
accuracy evidence. The decode profile reports 5.27 + 5.05 → 7.29 ms for the
kernel pair across 64 steps × 36 layers. That aggregate saving is not the
percentage improvement in request latency. Raising the range threshold did
not materially improve the reported serving sweep.

These measurements overlaid the pass/backend hooks onto 0.28.0. The author
explicitly could not build main and did not run the new unit/compile tests;
H200 revalidation after the October 19 rebase is still marked pending. A fresh
check of the exact head shows failed
[pre-run-check](https://github.com/vllm-project/vllm/actions/runs/39228505597/job/117980540542),
skipped pre-commit/format jobs and no reported GPU test pass. This is missing
final-head validation, not evidence of a GPU test failure.

Current main still ROCm-gates this generic combined pass, and FlashAttention
inherits `fused_qk_norm_rope_kvcache_supported=False`. Current Qwen3 exposes
the split/norm/rotary/attention chain; its standalone QK-norm/RoPE pass, when
enabled, is separate from the cache write. But a generic-pass gap does not
mean all CUDA models lack such fusion: current Step5 explicitly calls
`fused_qknorm_rope_cache` when a cache is bound, with a separate small-token
DSA preparation path. Its kernel preserves activation-dtype norm rounding,
uses a different reduction order and supports a different API. Current
Muse Spark has another explicit `fused_muse_spark_qknorm_rope_kv_insert` path,
including a phase-aware FP8-Q choice for FlashInfer. Neither is a drop-in
numerical replacement for this compiler proposal. Inspect the actual model
producer and cache owner before diagnosing a missing fusion from its name.

Source anchors: [proposed CUDA kernel and validation](https://github.com/vllm-project/vllm/blob/6f11616a53a64aeee73d210e786434677b38c799/csrc/fused_qknorm_rope_kernel.cu),
[proposed pass](https://github.com/vllm-project/vllm/blob/6f11616a53a64aeee73d210e786434677b38c799/vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py),
[PR-head defaults](https://github.com/vllm-project/vllm/blob/6f11616a53a64aeee73d210e786434677b38c799/vllm/config/vllm.py),
[test backend](https://github.com/vllm-project/vllm/blob/6f11616a53a64aeee73d210e786434677b38c799/tests/compile/backend.py),
[current Qwen3 caller](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/models/qwen3.py),
[current Step5 dispatch](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/models/step5/nvidia/model.py),
[Step5 rounding](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/csrc/fused_qknorm_rope_cache_kernel.cu),
and [current Muse Spark wrapper](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/models/muse_spark/common/ops/fused_qk_norm_rope_kv.py).
