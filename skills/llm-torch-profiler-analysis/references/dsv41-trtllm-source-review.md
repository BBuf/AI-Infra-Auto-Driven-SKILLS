# DeepSeek V4.1 TRT-LLM attention: source and PR review

Inspected on 2026-11-08. All four complete PR diffs, bodies, issue comments,
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

## PR #36652 — Pending per-token attention and persistent metadata rewrite

- Link: https://github.com/sgl-project/sglang/pull/36652
- Title: `[Perf] DSv4/4.1-Flash: Improve TRTLLM-Gen Attention Accuracy and Performance`.
- State: open, checked again November 8; head
  `9a9212b1e7054bc4492c1a80837383a769d27854`, base
  `e3ec98b15e6a82da7e58efff9f8d4448b9f0ae62`; both match the downloaded diff.
- Diff coverage: full diff read, 3,500 lines, 28 files, +1,774/-618. The body,
  46 issue comments, 26 reviews and 33 inline comments were read, including
  superseded findings. Current Q preparation, draft metadata capture, indexer
  dispatch, counter helper and FlashInfer backend resolver were cross-checked
  at the source pins above. No GPU tests were run for this review.

The motivation is to remove preparation and scheduling work around the sparse
attention kernel while retaining the V4.1, DSpark, DP-padding and replay
contracts added by the merged PRs above. The current mainline counter fix
already exists; this PR is not the only way to avoid per-call counter allocation.
Its base predates that merged fix, so the diff still removes the older private
monkeypatch. Compare the complete heads when porting it.

The implementation changes six related boundaries:

1. **Dense per-token calls.** Decode, target verification, draft extend and
   prefill pass `[query_rows, 1, heads, 512]`, with a causal length per row.
   The prefill call no longer supplies `cum_seq_lens_q` or `max_q_len`. Tail
   and encoder-window rows use their floored SWA lengths, while compressed
   slots stay full-context. DSpark's noncausal block still reads SWA storage.
   DP output padding is zeroed in reusable BF16 buffers.
2. **Shared sparse tables.** `TrtllmSparseTablePool` owns flat int32 parents
   with stable views. Live columns come from their writer; only required pad
   rows are made inert again. Separate narrow and wide SWA-length roles avoid
   a DSpark block leaving stale lengths for the next narrow step. Allocation
   growth is rejected during capture. The large `p_c128` table stays lazy and
   follows the largest served chunk's live width, rather than preallocating
   the maximum context width for every prefill row.
3. **Indexer output becomes attention input.** A row-stride argument lets
   `topk_v2` write into the shared table's tail. V4.1's BF16-logit decode
   writers also use that tail directly. Raw-index output and skipped-indexer
   variants keep their separate buffers. V4.1 prefill still packs its
   ratio-1/2 tail; do not describe all tail-copy launches as eliminated.
   `_tail_refresh_layers` verifies that reuse follows a source of the same
   compression ratio. The late-layer switch forces an initial pack even
   when its first indexed layer is not itself an index source.
4. **Metadata joins the captured draft step.** The TRT-LLM rejection is
   removed from `glue_draft_metadata`. The shared in-graph builder regenerates
   the live tables; eager preparation defers work the graph will perform.
   Request-window and other existing glue exclusions remain. Exact tensor
   aliases skip self-copies; non-tensor metadata retains its `copy_` behavior.
5. **Fuse conversions without dropping rounding boundaries.** Q's JIT norm/
   RoPE kernel gains E4M3 output, replacing the BF16 temporary plus `copy_`
   when the caller requests that output dtype.
   The rotated values still round through the input dtype before FP8 packing.
   The existing compressor norm/RoPE epilogue gains uniform-FP8 storage and
   replaces `compressor_trtllm.py`; invalid plan rows and non-boundary decode
   rows are skipped. DSpark commit projection permits strided outputs because
   the KV writer already accepts their row stride. Encoder replay uses the
   uniform row layout in its request window and byte-wise copies.
6. **PDL and capacity remain explicit contracts.** The counter and decode
   tables account for the maximum adaptive draft width, not just the active
   width. The counter uses the shared MLA allocator, including its 8,192-row
   floor. Attention immediately following an index-tail refresh disables PDL
   because the kernel reads sparse lengths before its dependency wait; reuse
   layers retain the default. The new backend-choice flag defaults to
   `trtllm-gen`. FlashInfer's resolver knows SM107, but the unchanged SGLang
   argument gate still requires SM100/SM103: the flag is not Rubin enablement.

Key excerpts from the complete diff:

```diff
+    // DType round-trip first, so the bits match a DType store followed by .to(e4m3).
+    const auto [bx, by] = cast<fp32x2_t>(cast<DType2>(rotated));
+    tile::Memory<fp8x2_e4m3_t>{lane_id, kWarpThreads}.store(output_ptr + (kHeadDim - kRopeDim), pack_fp8(bx, by));
```

```diff
+  // Row stride of `page_indices` in elements; > topk when writing a column slice of a wider table.
```

The row-stride declaration above is described in the diff's `TopKPagedParams`
comment; its actual binding uses `TensorMatcher(...).with_strides({O, 1})`
and `.out_stride = O.unwrap()`. The launch-order boundary is:

```diff
+            # The kernel loads sparse_topk_lens before griddepcontrol.wait, so it
+            # must not overlap this layer's tail write (pack or in-place top-k).
+            enable_pdl=False if refreshed else None,
```

Reviewed files include both norm/RoPE CUDA headers, `topk_v2.cuh`, compressor
and low-ratio metadata kernels; argument schema/docs; both attention backends
and metadata-copy helpers; V4.1 pools/request windows; target, DSpark and
deferred-commit callers. All seven changed test files were read, including
the deleted manual VarSeq test and the new encoder-replay E2E recipe.

Validation should retain the pool-address/padding/growth tests, same-ratio
index-source tests, strided top-k sentinels, empty ratio-2 lengths, and actual
attention comparisons with persistent versus fresh counters. The new counter
test inspects CUDA graph dependency edge types and has a control that retains
PDL. Its metadata-poisoning case checks reads beyond the live batch. The
compressor test permits a mismatch fraction up to `1e-4` and one E4M3 ULP,
reflecting RMSNorm reduction differences; it is not a byte-identical guarantee
for every compressor result. The manual long-query VarSeq dispatch assertion
is removed because the runtime now uses dense rows.

Reported performance must keep the measurement's revision and workload:

- The October 26 head `f6916d88f3` reports GB300 TP4 AgentX per-request decode
  time about 0.95% lower at c1 and 0.76–1.25% lower at c16 than FlashMLA.
  Published recipes use simulated acceptance. Later B300 TP2/GB200 TP4 results
  also favor the PR's decode latency; B300 DEP2 c64 instead favors FlashMLA
  Mega by about 1.7% on that metric, with output throughput effectively tied.
- The subsequent metadata changes report 29.2–29.7 µs less attributable time
  per GB300 TP4 c1 decode step in 40-step Torch profiles. Their estimated 0.6%
  end-to-end effect was not measured. The attention kernel itself became
  slower in that profile, while preparation savings outweighed it.
- Real-acceptance GPQA-Diamond uses 198 questions × four repeats: 0.907 versus
  FlashMLA's 0.909, paired difference −0.25 percentage points with 95% interval
  [−1.39, +0.76]. This supports the reported comparison, not proof of identical
  accuracy. Registered GB300 bounded/encoder BCG runs report GSM8K-200 and
  MMMU-Pro-200 above their thresholds; some runs include additional patches.
  These are upstream reports on their stated trees, not a fresh qualification
  of this PR against the November mainline or installed wheel.
