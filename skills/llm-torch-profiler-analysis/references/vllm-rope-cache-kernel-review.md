# CUDA RoPE/cache proposals: ordinary GQA and MiMo DiffKV

Reviewed on 2026-11-09 (Asia/Shanghai). Both complete final diffs, bodies and
all discussion were read, with fresh head/base checks. Current-source
comparisons use frozen vLLM main
`b022bb7a4873a47cdcdf474edf3bd499c221f40d`; neither proposed kernel is present
there. The [compiler review](vllm-compiler-source-review.md) separately covers
graph visibility, custom-op resolution and retained Q quantization. No GPU
experiment was run for this source review.

## PR #57271 — Enable CUDA BF16 GQA RoPE KV-cache fusion

- Link: https://github.com/vllm-project/vllm/pull/57271
- State: open, unmerged; head `94410fb7777a07dda69e721683f8e0275014c4d3`,
  base `f700178f90d21f21fea2b495475e83b267290c01`.
- Coverage: all 465 diff lines, six files, +306/-31; body, four issue comments,
  one review and zero inline comments; two commits.
- Reviewed files: runtime `vllm/config/{compilation,vllm}.py`,
  `vllm/v1/attention/backends/triton_attn.py`, and
  `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`; tests
  `tests/compile/test_config.py` and
  `tests/compile/passes/test_rope_kvcache_fusion.py`.
- Motivation: ordinary CUDA GQA executes rotary embedding and cache writes
  separately; the existing compiler replacement and backend admission were
  restricted to ROCm/AITER.

The patch admits CUDA in the pass configuration and default resolver, while
retaining custom-rotary and cache-op visibility requirements. Its backend
rejects per-token/head quantization, otherwise advertises support on CUDA or
AITER. The CUDA branch uses a module constant
`CUDA_FUSED_ROPE_KVCACHE_ENABLED` and a four-dimensional K-cache check; this
constant is not a user-facing environment variable. Other CUDA layouts take
separate rotary/cache calls; ROCm keeps AITER.

The Triton kernel rotates Q and K in place, loads V unchanged, and writes K/V
to the paged cache. The launch grid spans tokens, the larger head count, and
rotary-pair tiles. Both NeoX and interleaved rotary styles are implemented.
Its actual rotary extent is stricter than the backend's admission check:

```diff
+    assert key_cache.ndim == 4 and value_cache.ndim == 4
+    num_tokens, num_query_heads, head_size = query.shape
+    num_kv_heads = key.shape[1]
+    rotary_dim = cos_sin_cache.shape[-1]
+    assert rotary_dim == head_size and rotary_dim % 2 == 0
+    tile_size = min(128, triton.next_power_of_2(rotary_dim // 2))
```

The backend constructs equal-width K/V views from its packed allocation.
The kernel uses K-cache strides for both views and assumes contiguous input
head dimensions. Do not extend this to arbitrary independent K/V strides,
unequal head sizes or partial RoPE. The backend fallback does not convert a
failed rotary-width assertion into separate execution. The generic matcher
traces full-width rotary sample inputs; partial-width matching and fallback
need an explicit regression case before claiming support or a reproduced
model failure.

Negative slots suppress cache stores while Q/K still rotate. Slot loads are
unmasked for each launched KV-head token, unlike the MiMo proposal below.
The ordinary cache writer launches over the slot-map length. A shorter map
with padded Q rows therefore needs an admission check or masked load before
this fusion can preserve that behavior.

There is also a numerical boundary to verify for FP8 caches. `_fused_rope`
stores rotated K into its input dtype but returns the unrounded FP32
intermediate; the fused writer scales and converts that intermediate directly
to FP8. Separate rotary/cache calls read K after its BF16/FP16 store. The
patch has no explicit cast preserving that intermediate rounding boundary.
This is a source-level contract difference, not a measured accuracy failure.
The new wrapper also bypasses the ordinary writer's architecture/dtype
admission assertion; CUDA platform admission alone is not FP8 hardware proof.

The configuration test explicitly enables custom rotary, Inductor partitioning
and the fusion flag. It does not establish that stock O2 resolves correctly;
the splitting/default-order issues covered by #57900 remain separate. The
ordinary numerical test uses five rows, Q/KV heads 64/8, head size 64, BF16
inputs, auto/FP8 caches and both rotary styles. It checks one compiled call,
with Q/K tolerance 0.01 and cache tolerance 0.1, without an attention consumer,
changed-buffer replay or long decode. The static-Q test remains AITER-only
and verifies that the separate Q-quant operation survives.

The body reports eight selected Triton cases and one configuration case
passing. Its H100 80 GB Qwen2.5-0.5B-Instruct report uses 128 input/32 output
tokens, 100 requests, ten warmup requests and ten runs per configuration:
median output throughput rises from 16,105.90 to 21,444.51 tokens/s (33.1%).
The full launch/arrival/concurrency configuration, tested build revision,
benchmark harness and task-accuracy results are absent. The discussion's CI
job-selection preview is not a completed CI result, and the current head has
merge conflicts. Preserve this as a bounded author report, not current-main
or all-model performance evidence.

Source anchors: [proposed kernel and wrapper](https://github.com/vllm-project/vllm/blob/94410fb7777a07dda69e721683f8e0275014c4d3/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py),
[proposed backend dispatch](https://github.com/vllm-project/vllm/blob/94410fb7777a07dda69e721683f8e0275014c4d3/vllm/v1/attention/backends/triton_attn.py),
[PR-head numerical tests](https://github.com/vllm-project/vllm/blob/94410fb7777a07dda69e721683f8e0275014c4d3/tests/compile/passes/test_rope_kvcache_fusion.py),
[current cache writers](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py),
and [current fusion registration](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/fusion/rope_kvcache_fusion.py).

## PR #58798 — Fuse MiMo partial RoPE, value scaling, and packed DiffKV writes

- Link: https://github.com/vllm-project/vllm/pull/58798
- State: open, unmerged; head `d309838bbdf941c78b7af767d6229e4326444251`,
  base `b6761e8ded57ef85b708f34af8cab1649eae1069`.
- Coverage: all 770 diff lines, five files, +667/-12; body, six issue comments,
  one review and zero inline comments; three commits.
- Reviewed files: runtime `vllm/model_executor/models/mimo_v2.py`,
  `vllm/model_executor/layers/attention/attention.py`, and new
  `vllm/v1/attention/ops/triton_rope_and_cache_diffkv.py`; regression cases in
  `tests/kernels/attention/test_cache.py`; new
  `benchmarks/kernels/benchmark_mimo_rope_cache.py`.
- Motivation: MiMo Q/K width 192, V width 128, rotary width 64 and V scaling
  require different preparation from ordinary equal-width/full-RoPE GQA.
  Combining those operations avoids intermediate writes and launches.

This is a model-selected custom op, not an extension of the generic compiler
pattern. It requires CUDA, FA-DiffKV or Triton-DiffKV, FP16/BF16 activations
and cache, no query quantizer, decoder attention, DCP world size one, no KV
sharing, the exact stock `RotaryEmbedding` type without FlashInfer rotary,
and one-dimensional positions. Unsupported cases keep the existing path.

The op allocates fresh rotated Q, preserves the QKV input and writes the
hidden layer cache. It then calls the ordinary attention wrapper with
`q, None, None`, preventing a duplicate cache write while preserving wrapper
hooks and the output projection. Current decoder/DCP=1 implementations read
K/V from the cache; encoder and multi-rank DCP paths cannot be assumed to do
so. The metadata-free profiling branch returns native rotated Q without a
cache write; the fake implementation returns the corresponding fresh shape.

The kernel copies the nonrotary Q/K tail, applies V scaling and stores packed
K/V using actual strides and 64-bit token/head offsets. Missing or negative
slots skip stores while Q is still produced. It explicitly preserves the
rotary storage dtype before subsequent writes:

```diff
+    rotated_x = (x * cos - y * sin).to(source.dtype.element_ty)
+    rotated_y = (y * cos + x * sin).to(source.dtype.element_ty)
+    if head >= NUM_Q_HEADS:
+        kv_head = head - NUM_Q_HEADS
+        tail_mask = (offsets >= ROTARY_DIM) & (offsets < HEAD_SIZE_K)
+        tail = tl.load(source + offsets, tail_mask, other=0)
```

The wrapper checks contiguous last dimensions, compatible head/cache sizes,
even rotary extent at most K width, and slot-map length at most token count.
Positive slot and position bounds remain caller responsibilities. Its optional
in-place mode is distinct from model integration, which always supplies a
fresh Q output. Quantized caches are excluded.

**Current-main integration has changed since this patch.** MiMo now uses
`_fused_qkv_kv_chunk_rows` and `_split_qkv`. With block-quantized fused-QKV
checkpoints, a rank can own several checkpoint chunks whose K and V share
scale blocks. Main preserves `[Q... | K1 V1 pad | K2 V2 pad ...]`, then
extracts K/V through the helper. Both branches in this PR still use a plain
`qkv.split([q_size, k_size, v_size])`; nonzero `kv_chunk_rows` can also add
padding that makes those split sizes invalid. A port must retain the current
split helper and either adapt the custom-op inputs or exclude chunked layouts.
The existing activation/cache dtype gates do not exclude quantized weights.
The current kernel path is absent and GitHub reports merge conflicts; the
historical model checks do not validate this later layout.

The four added tests cover BF16/FP16, NeoX/interleaved styles, two head/batch
geometries, offset/strided QKV, cache guard regions, shortened maps and negative
slots. They require exact Q/cache equality and input preservation against
**C++ rotary plus the existing cache writer**. The body reports these passing
with `--noconftest`, plus external stride/large-offset/replay, six actual
random-weight attention cases and FakeTensor/AOT checks. Those external
harness results do not constitute a checked-in distributed or full-model test.

The checked-in benchmark instead uses **compiled `forward_static` RoPE/V
scaling** with Inductor combo kernels as its baseline, then the same cache
writer and attention consumer. It captures eight calls per graph, alternates
A/B/A controls for eight rounds, and has warm intra-sample L2. Projections,
metadata construction and serving are excluded. On B200, the body reports
FA4 speedups 1.060–1.423x; the faster A control reduces the batch-eight SWA
claim to 1.394x. FA4 A/A ratios span 0.988–1.021. The Triton full-attention
changes of 1–2% need comparison with its reported 4.65% maximum A/A variation.
The environment used prebuilt `0.30.1rc1.dev143+g29468dde8` binaries with Python
overlays, not a fresh current-main build.

Discussion adds a CMP 170HX/SM80 report on an old vLLM fork. Its initial
2.76–5.17x figures time preparation against an uncompiled native reference;
the later unmodified attention benchmark reports 1.009–1.328x. Full-verify
1.058x has a 1.045 A/A ratio and bimodal samples. The commenter explicitly
retracted a claimed rotary-convention mismatch after finding an incorrect
handwritten reference. A later comment describes the benchmark baseline as
C++ rotary, but the checked-in benchmark uses compiled `forward_static`;
keep it distinct from the repository test. None of these reports establishes
full-weight model accuracy, service throughput or multi-rank DCP behavior.

Source anchors: [PR-head model integration](https://github.com/vllm-project/vllm/blob/d309838bbdf941c78b7af767d6229e4326444251/vllm/model_executor/models/mimo_v2.py),
[proposed DiffKV kernel](https://github.com/vllm-project/vllm/blob/d309838bbdf941c78b7af767d6229e4326444251/vllm/v1/attention/ops/triton_rope_and_cache_diffkv.py),
[controlled benchmark](https://github.com/vllm-project/vllm/blob/d309838bbdf941c78b7af767d6229e4326444251/benchmarks/kernels/benchmark_mimo_rope_cache.py),
[current MiMo chunked QKV helpers and caller](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/models/mimo_v2.py),
[current attention wrapper](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/layers/attention/attention.py),
[current FA-DiffKV](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/attention/backends/flash_attn_diffkv.py),
and [current Triton-DiffKV](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/attention/backends/triton_attn_diffkv.py).
