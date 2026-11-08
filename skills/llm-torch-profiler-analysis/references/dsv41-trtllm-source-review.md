# DeepSeek V4.1 TRT-LLM attention: merged source review

Inspected on 2026-11-08. All three complete PR diffs, bodies, issue comments,
reviews and inline discussions below were read. Current callers were checked at
SGLang `c24285bde97c3e6424cd5d8cb35788889034d8ea`; FlashInfer's counter sizing
and sparse-attention call were checked at `31c146b16789f112e9b9861bd79247d4701e437a`.
The reported upstream experiments were not rerun for this skill refresh.

## Current profiling contract

`--dsv4-attn-backend trtllm` is a mainline opt-in on SM100/SM103. It uses
uniform 512-byte E4M3 KV rows with external unit scale, including compressed
ratio-1/2 pools. Packed FlashMLA V4.1 FP8/FP4 layouts are different formats.
The current argument hook requires positive chunked prefill and rejects CP,
PD, HiSparse and encoder SWA bounded replay. The backend rejects online
compression. Prefill can be eager or breakable; decoder bounded replay is
supported. Full prefill graphs remain outside this contract.

In a trace, distinguish these operations:

- KV RMSNorm, RoPE and uniform-FP8 cache storage share one Triton launch.
  The fused writer preserves the old BF16/FP16 rounding boundary before FP8.
- Q RoPE/norm still writes a BF16 temporary before `fp8_output.copy_` converts
  it. A Q conversion kernel is expected; the API's name does not prove that
  FP8 conversion was fused into its CUDA kernel.
- `pack_sparse_tail` copies compressed indices and forms the total table span
  in one launch. That span includes all 128 reserved SWA columns, even for a
  short sequence. Empty compressed histories must have length zero.
- Breakable prefill runs attention eagerly at a graph break. Its output bridge
  uses the activation dtype, even when Q is FP8. Tail-relative SWA masking and
  full-context compressed indices must remain distinct.
- Counter allocation is once per query-head count, outside capture. Repeated
  zero-allocation kernels around attention can indicate an older dependency
  adapter or an unqualified path; do not patch FlashInfer's private namespace.

Current source:
[argument gates](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/arg_groups/deepseek_v4_hook.py),
[backend and persistent counters](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/deepseek_v4_trtllm_backend.py),
[Q preparation](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/dsv4/norm_rope.py),
[KV writer](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/dsv4/kv_norm_rope_store.py),
[sparse metadata](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/dsv4/trtllm_metadata.py).

## PR #41603 — Add opt-in TRT-LLM sparse attention support

- Link: https://github.com/sgl-project/sglang/pull/41603
- State: merged 2026-10-06; head `96f7c8d201336f9984caf20f9fc1cfb5118b8505`,
  base `734cf3cf3b9201a5beb104649acd29f05edcd1e5`,
  merge `620c76f25db14f1599aa7c7ccb141a91e4944b63`.
- Diff coverage: full diff read, 1,182 lines, 16 files, +684/-70.
- Motivation: V4.1's ratio-1/2 compressed caches and DSpark's noncausal draft
  window were not represented by the existing V4 TRT-LLM adapter. Splitting
  KV normalization, RoPE, conversion and cache writes also added preparation
  work around the sparse attention kernel.
- Implementation: `KVLayout.UNIFORM_FP8` adds a 512-byte, scale-free row to the
  C1/C2 stores. `fused_k_norm_rope_uniform_fp8` uses 64-bit row/slot offsets,
  preserves intermediate rounding and writes directly into the cache without
  modifying the input. Metadata reserves inert 64-row tile padding, restores
  true zero compressed lengths and refreshes ratio-1/2/4 tails with one kernel.
  DSpark uses the entire visible draft block and SWA storage for that tail.
  Q and sinks use native TP head widths; the final diff reuses BF16 Q
  preparation followed by conversion instead of adding a separate fused Q
  implementation. The initial integration required eager prefill; #44326 below
  subsequently relaxed that gate.
- Key excerpt from the added metadata helper:

```diff
+    tl.store(TABLE + row * TABLE_STRIDE + SWA_WIDTH + col, values, col < WIDTH)
+    length = tl.load(LENGTHS + row * LENGTH_STRIDE)
+    tl.store(TOTAL_LENGTHS + row * TOTAL_STRIDE, SWA_WIDTH + length)
```

- Reviewed files: C1/C2 CUDA stores and Python/C++ KV layout traits; the new
  `kv_norm_rope_store.py` and `trtllm_metadata.py`; Q preparation in the former
  `elementwise.py` (now `norm_rope.py`); argument validation; both attention
  backends; `deepseek_v4_memory_pool.py`; V4 and DSpark model callers; and all
  three added test files. The shared V4/DSpark callers make this broader than
  a V4.1-only microkernel change.
- Validation implications: registered `test_trtllm_kv_norm_rope_store.py` checks
  byte-level parity, strided inputs, empty rows, negative slots, FP8 overflow,
  mutable graph replay and offsets beyond 2 GiB. Registered
  `test_dsv4_trtllm_q_metadata.py` covers Q input preservation, padding, graph
  replay, 128-head offsets beyond 2^31 elements and sparse metadata dtype/device
  guards. `test_dsv4_trtllm_prefill.py` is manual SM100/SM103 coverage for
  rows 1/16/17/8192, FP32-reference attention and the long-query kernel tile.
  Inline review drove the offset, reuse and metadata-guard changes. The body
  still says tests were not added; the final diff supersedes that checklist.
  The author's earlier 4×GB300 measurements reported 124/128 GSM8K for both
  backends and 564.6→575.4 output tok/s (+1.9%) at C64, 32K input/1K output,
  DSPARK five draft tokens, Engram offload. Attention-only timings exclude
  Q/KV preparation, and those experiments do not certify the final rebased head.

## PR #44326 — Decoder bounded replay and coordinated breakable prefill

- Link: https://github.com/sgl-project/sglang/pull/44326
- State: merged 2026-10-15; head `d39074847614d77d922581234c90b9ea8685350b`,
  base `97d3150a6a3c412f9445d33c7a91238e590f8b10`,
  merge `64a0a08f0a5d5166cf78bf44c569fd349a67d8f4`.
- Diff coverage: full diff read, 360 lines, 11 files, +118/-23.
- Motivation: the initial adapter rejected bounded replay and BCG. When a
  request exceeded the captured sequence limit, active and idle DP ranks could
  also disagree about replay versus eager execution and enter mismatched
  collective sequences.
- Implementation: permit decoder bounded replay with eager/BCG prefill while
  retaining encoder/full-graph restrictions. Varlen metadata comes from the
  late-layer tail when present; SWA lengths become tail-relative while compressed
  indices retain full-context addressing. The graph output bridge follows
  `x.dtype`, not FP8 Q. The shared scheduler vote and runner check both context
  limits before the DP consensus, affecting other prefill-graph models too.
- Key excerpt:

```diff
+        # Vote on both bounds so DP ranks agree on graph versus eager.
+        for context_limit in (self.max_context_size, self.max_seq_len):
+            if context_limit is not None and (
+                batch_max_context_len is None or batch_max_context_len > context_limit
             ):
                 return False
```

- Reviewed files: argument hook, TRT-LLM backend, scheduler `dp_attn.py`,
  `prefill_cuda_graph_runner.py`, V4 model bridge; registered GB300 E2E recipe;
  and five unit-test files for graph padding, DP voting, multimodal eligibility,
  hidden-state recapture and runner setup.
- Validation implications: the new regression makes a 17K active request and
  an idle rank both honor eager fallback beyond the 16,384 limit; another case
  checks both limits independently. Keep this behavioral coverage. Review
  removed redundant standalone config/numerical additions while retaining the
  full-model test. The final-head discussion reports 72 CPU, 20 H100 metadata,
  eight B200 E2E, six B200 indexer and six GB300 GSM8K/MMLU tests passed across
  11 selected files. This is upstream targeted coverage, not full CI or our
  reproduction. The body reports 22/23 eager/BCG prompts with identical 32-token
  outputs; one differed in two tokens. Its three-run TTFT smoke does not prove
  a general throughput improvement or bitwise equivalence.

## PR #48467 — Pass the persistent multi-CTA counter through the public API

- Link: https://github.com/sgl-project/sglang/pull/48467
- State: merged 2026-11-05; head `bb9d6e372104047f5ec7cb96a6f84193924ffff4`,
  base `c67efab427ac9d7965123a3cea50bf1d3d899563`,
  merge `69fda8040cbe8312ee1777c9a4c61feb77ca51ec`.
- Diff coverage: full diff read, 214 lines, 3 files, +88/-38.
- Motivation: monkeypatching `flashinfer.mla._core` missed the allocator in
  `flashinfer.utils` on 0.7.2 and raised `AttributeError` on 0.7.3. More
  fundamentally, request count is not the semaphore bound when the kernel
  indexes flattened query rows in speculative decoding or varlen prefill.
- Implementation: reserve a query-row capacity from prefill bounds and
  requests × draft rows; obtain zeroed persistent storage per head count through
  runtime-context `get_buffer`; size it with the public byte-count helper;
  pass it explicitly at both decode and prefill calls. Allocation is forbidden
  inside capture and the existing over-capacity exception remains active.
- Key excerpt:

```diff
+            multi_ctas_kv_counter_buffer=_get_trtllm_kv_counter_buffer(
+                num_heads, q.device
+            ),
```

- Reviewed files: `deepseek_v4_trtllm_backend.py`, manual
  `test_dsv4_trtllm_prefill.py`, registered
  `attention/unittests/dsv4/test_deepseek_v4.py`. The current FlashInfer
  [sizer and resolver](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/utils.py)
  use `round_up(max(rows * heads, SMs), 8) * 4` bytes when passed the reserved
  row count, and validate device, contiguity, alignment and available bytes.
- Validation implications: the added test verifies zero initialization,
  query-row sizing, reuse and separate head-count buffers; the manual kernel
  test exercises rows 1/16/17/8192. The author reports GB200 tests with
  FlashInfer 0.7.2/0.7.3 and full 1,319-question GSM8K on DeepSeek-V4-Flash:
  96.66% baseline, 96.74% patched 0.7.2, 96.59% patched 0.7.3, no request
  errors. The 0.7.3 run included another sampling compatibility fix. These
  results cover V4, not a new V4.1 accuracy certification; preserve dependency
  versions and the complete patch set when reproducing them.
