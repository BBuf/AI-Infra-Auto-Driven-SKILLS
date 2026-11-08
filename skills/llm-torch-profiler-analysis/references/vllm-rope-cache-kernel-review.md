# CUDA RoPE/cache proposals: GQA, Llama and MiMo DiffKV

Reviewed on 2026-11-09 (Asia/Shanghai). All three complete final diffs, bodies and
all discussion were read, with fresh head/base checks. Current-source
comparisons use frozen vLLM main
`b022bb7a4873a47cdcdf474edf3bd499c221f40d`; none of these proposed kernels is present
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

## PR #52363 — Add manual CUDA RoPE KV-cache fusion for Llama

- Link: https://github.com/vllm-project/vllm/pull/52363
- State: open, unmerged; head `5c1405ecb51286ccdc0f34a5c6630f9ed02ecf48`,
  base `15a9677306b19201c21f54f668334d1a2531d2d7`.
- Coverage: all 2,410 final diff lines, 21 files, +1,992/-35; body, all
  23 issue comments, nine reviews and six inline comments; 29 commits.
- Reviewed files: CUDA `csrc/rope_kvcache_fusion_kernels.cu`, `ops.h`,
  `torch_bindings.cpp`, `CMakeLists.txt` and `_custom_ops.py`; Attention,
  Llama, rotary `base.py`/`fope.py`, attention backend interface and
  `flash_attn.py`; compilation configuration and pass manager; kernel,
  rotary, functional-attention and configuration tests; benchmark,
  Buildkite kernel dependencies and both design documents.
- Motivation: CUDA Llama can combine rotary and cache writes at the model
  call site without relying on a compiler pattern spanning those operations.
  It must still rotate padded Q rows when the cache slot map is shorter.

The final design passes `positions` and the rotary module from Llama into
`Attention.forward`. Rotary modules expose a `RopeRotation` descriptor:
positions, the `[cos | sin]` table and rotary style. Ordinary rotary and
variants that only change the table can provide it. Fourier explicitly
returns none; MRoPE derives from the base that returns none; LongRoPE has
no interface. FlashInfer rotary and non-1D positions also retain their original
forward. The cache conversion helper preserves AITER's cached BF16 path and
avoids buffer mutation while compiling. Future subclasses that change the
rotation must override the descriptor, not inherit it blindly.

Attention requires both `supports_kv_cache_update_bypass` and the new
implementation capability. Only CUDA FlashAttention implements the fused
writer in this patch: decoder, DCP=1, unquantized cache. The layer additionally
requires equal Q/K/V head width, activation/cache dtype equality, no KV sharing
and no attention-output quantization fusion. Normal FlashAttention selection
admits FP16/BF16. FP8 **weights** in the reported model do not imply FP8 cache
support. FlashInfer's existing bypass capability alone does not select this
writer. Unsupported backend configurations receive a one-time warning.

The graph owns a contiguous Q output. The custom op declares that output
mutable, reads the current layer's cache/slot context, and writes rotated K
plus V to the cache without changing input Q/K/V or materializing rotated K.
Attention then receives Q and `None, None`, preserving its wrapper while
avoiding another cache update. The Q dependency orders attention after the
hidden cache writes. Current FlashAttention decoder/DCP=1 consumes cached
K/V; encoder and DCP branches consume the supplied K/V, explaining their
exclusion. Current Llama still calls rotary and attention separately; the new
descriptor and writer are absent at the frozen main revision.

The token threshold is a runtime contract. In ordinary vLLM compile mode,
shape guards are skipped by default (`evaluate_guards=False`); a traced
Python comparison must not freeze dispatch for all later token counts.
The compiled graph therefore always calls the custom op, which checks the
default 256-token bound itself. Eager execution can choose the ordinary path
before entering the op. Above the bound, the compiled op uses:

```diff
+    query_out.copy_(query)
+    if slot_mapping is None:
+        # Profiling runs have no cache to update.
+        assert attn_metadata is None
+        ops.rotary_embedding(
+            positions, query_out, None, layer.head_size, cos_sin_cache, is_neox
+        )
+        return
+    key = key.clone()
```

With a slot map, fallback then applies ordinary rotary and the backend cache
writer. With no map, it only rotates Q and requires absent attention metadata.
These Q/K copies are real additional work. The patch leaves CUDA opt-in and
the bound unchanged, keeps the compiler fusion pass ROCm-only, and removes
CUDA's dependence on custom-rotary registration/default cache-op splitting.
Thus #57271's compiler admission and this manual route are different designs.

The CUDA kernel uses one block per Q token, rotates its partial or full
extent, copies the tail, then exits before K/V work for missing or negative
slots. It uses separate 64-bit token/head/page strides for inputs and K/V
caches. Large V copies use eight-element vector loads/stores only when width,
head strides and pointer alignment permit; small or unaligned cases use scalar
copies. Host checks enforce dimensions, equal storage dtype, device, positive
strides, contiguous last dimensions and contiguous int64 positions/slots.
The model supplies distinct Q output storage and valid positive slot/position
bounds; those ownership/indexing obligations are not relaxed by fusion.

The eight shape cases cover all six allocator cache layouts, partial rotary,
NeoX/interleaved styles, short/negative maps, padded cross-layer pages and
head sizes through 256 including an unaligned 82-wide case. Four additional
storage tests cover independent K/V strides, offsets and guard preservation;
one test rejects incompatible cache/table dtypes. Kernel comparisons require
exact Q/cache equality against C++ rotary and the incumbent writer (or explicit
strided reference stores where that writer cannot represent the layout).
Only the layer-compact representative invokes opcheck.

Functional tests check backend/dtype admission, real compiled Attention
graphs, disabled/LongRoPE fallback and threshold crossings in both orders
with eager and compiled calls. They inspect actual native-op events and
compare output/cache at tolerance 0.002. TinyLlama engine tests compare greedy
logprobs with fusion off/on and breakable graphs off/on; these are stronger
than asserting a flag or counting Python calls during capture. Zero Python
calls during graph replay does not mean the captured GPU kernel did not run.

The final body reports RTX 5090 D/SM120, CUDA 13.2 and PyTorch 2.14.0 validation
at the reviewed head: 13 kernel, 12 functional, 197 rotary, 26 compilation and
four configuration cases passed, with 170 platform/backend skips; three
compute-sanitizer modes report no errors/hazards. This review did not rerun
those experiments or establish a complete upstream CI pass.

The benchmark compares one fused call with separate rotary/cache calls using
independent allocations, balanced/shuffled A/B order, exact pre-timing checks
and CUDA events. CUDA Graph timing is optional, off by default; ordinary
host-submitted repetitions can include submission gaps. It excludes attention
and serving. Final-head 5090 D reports 1.53–1.59x across 24 shape/layout rows.
Earlier H200 evidence pins `ccde2be37d03` and reports 1.660–1.730x plus a
single FA3 token-parity case. H100/B200 comments report roughly 1.6–1.7x and
matching GSM8K aggregate scores for two small models, before the October
descriptor/bypass refactor; they do not pin validation to the final head.

Keep the limiting results: TinyLlama BF16 batch-one end-to-end latency at
`d93c77f73` is effectively equal (unfused/fused median ratio 1.006 over six
interleaved rounds), superseding an earlier two-round 1.0371 report. The
September discussion reports 2–2.5% regressions for 257/512/1024-token inputs
from compiled fallback copies. Raising the limit to 2048 restored those
particular cases near baseline, but head geometry/cache residency changed
larger-shape behavior, so this is not a universal tuning recommendation.
The September approval also predates the later refactor. Do not promote these
operator gains to current-main service-throughput or all-model accuracy claims.

Source anchors: [final CUDA kernel](https://github.com/vllm-project/vllm/blob/5c1405ecb51286ccdc0f34a5c6630f9ed02ecf48/csrc/rope_kvcache_fusion_kernels.cu),
[final Attention selection and runtime fallback](https://github.com/vllm-project/vllm/blob/5c1405ecb51286ccdc0f34a5c6630f9ed02ecf48/vllm/model_executor/layers/attention/attention.py),
[rotation descriptor](https://github.com/vllm-project/vllm/blob/5c1405ecb51286ccdc0f34a5c6630f9ed02ecf48/vllm/model_executor/layers/rotary_embedding/base.py),
[functional and engine tests](https://github.com/vllm-project/vllm/blob/5c1405ecb51286ccdc0f34a5c6630f9ed02ecf48/tests/v1/attention/test_functional_rope_kvcache.py),
[paired benchmark](https://github.com/vllm-project/vllm/blob/5c1405ecb51286ccdc0f34a5c6630f9ed02ecf48/benchmarks/kernels/benchmark_fused_rope_kvcache.py),
[current Llama caller](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/model_executor/models/llama.py),
[current FlashAttention](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/attention/backends/flash_attn.py),
and [current compile guard policy](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/wrapper.py).
