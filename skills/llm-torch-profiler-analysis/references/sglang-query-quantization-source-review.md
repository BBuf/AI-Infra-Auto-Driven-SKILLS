# SGLang FA3 query quantization and RoPE

Reviewed on 2026-11-09 (Asia/Shanghai). The complete final diff, body and all
discussion were read. Current-source comparisons use frozen SGLang main
`c24285bde97c3e6424cd5d8cb35788889034d8ea`. The proposal below is absent
there. No GPU experiment was run for this review.

## PR #35142 — Make Q quantization graph-visible and fuse it with RoPE

- Link: https://github.com/sgl-project/sglang/pull/35142
- State: closed without merge on 2026-10-29; head
  `794d4d7fd422b543fa521f6ca3adc0af9c45e9ad`, base
  `baae0195d9addc0d98503a944d9c35dab56736a9`.
- Coverage: all 644 diff lines, 12 files, +387/-3; body and the sole issue
  comment; no reviews or inline comments; four commits. The closing comment
  cites the idle-PR limit, not a technical approval or rejection of the kernel.
- Reviewed files: new `python/sglang/kernels/ops/attention/rope_quant.py`;
  `python/sglang/srt/layers/attention/{base_attn_backend,flashattention_backend,hybrid_attn_backend,tbo_backend}.py`;
  `python/sglang/srt/layers/{radix_attention,rotary_embedding/base}.py`;
  `python/sglang/srt/model_executor/model_runner.py`;
  `python/sglang/srt/models/{qwen2,qwen3}.py`; and
  `test/registered/{kernels/ops/attention/test_rope.py,unit/layers/test_radix_attention.py}`.
- Motivation: FA3's FP8-KV path converted Q inside attention dispatch, hiding
  the conversion from the surrounding compiled model and leaving an extra
  launch after rotary embedding.

The proposal adds `get_query_quantization_dtype` to attention backends.
FA3 returns its cache dtype when storage is non-auto, non-MXFP8 and the
layer's head dimension is at most 256. Other backends default to no override;
hybrid dispatch selects the phase's backend, while TBO delegates to its
primary. After backend initialization, `ModelRunner` stamps decode, extend
and target-verify dtypes on every `RadixAttention` layer. The forward path
reads those attributes rather than consulting a Python backend object inside
the compiled graph; idle mode returns no override.

`RadixAttention` casts Q as a generic fallback. Qwen2/Qwen3 decode can request
the dtype earlier from rotary embedding. Support is inferred by checking that
the subclass inherits both native and CUDA rotary methods unchanged. MRoPE
and other overriding implementations keep their existing paths; Qwen3 also
excludes the RL on-policy target mode and leaves its AITER path separate.
The patch does not remove the backend's existing dtype conversion: an
already-converted Q makes that later `.to` redundant.

There are two distinct execution mechanisms. In model compile mode,
`BaseFusedOp.enter_torch_compile` selects `forward_native`; adding `.to` to
that native rotary result exposes conversion to Inductor. The new eager
Triton custom op is not the implementation of that compiled native path.
In eager CUDA, full rotary width, matching FP16/BF16 Q/K, E4M3/E5M2 Q output,
the non-fallback rotary implementation and no fused-KV-store argument select
the new kernel. Other rotary cases retain ordinary rotation followed by a
cast when requested.

The eager kernel launches one program per token and handles padded Q/K head
counts with masks. Q goes to a new FP8 tensor, K rotates in place, and the
original Q input is not overwritten. Both NeoX and interleaved rotation are
implemented; V and the KV cache are untouched. Q/K token and head strides
are independent. Its wrapper checks shapes, dtypes, device equality and
unit last-dimension strides, but indexing assumes a dense FP32 rotary table
and contiguous one-dimensional positions. Do not infer arbitrary metadata
strides or partial rotary support from the QKV-view test.

The direct output introduces a numerical boundary that needs explicit review:

```diff
+    tl.store(
+        q_out_ptr + q_out_first_offset,
+        q_first * cos - q_second * sin,
+        mask=q_mask,
+    )
```

The expression uses FP32 Q/cos/sin and stores directly to FP8. The separate
CUDA rotary kernel first stores Q in its BF16/FP16 input dtype, after which
the attention backend converts it to FP8. The proposal has no intervening
cast preserving that rounding. This is a source-level difference, not a
reproduced model-accuracy failure. Native rotary also has a different
arithmetic path: `apply_rotary_emb` casts cos/sin to the input dtype before
its operations. Check the compiled lowering and reference actually used;
the eager Triton numerical test does not prove compiled/eager equivalence.

That kernel test has 16 parameter combinations: BF16/FP16, E4M3/E5M2,
NeoX/interleaved and batch one/eight, with Q/K heads 14/2 and head width 64.
It constructs strided packed-projection views, then clones K for the call.
The CUDA reference is FlashInfer in-place RoPE followed by Q conversion;
Q comparison permits `atol=0.25, rtol=0.125`, and K permits 0.01 for both.
This does not establish exact FP8-byte parity near rounding boundaries or
arbitrary K-stride coverage. The layer-level test uses a recording backend
requesting FP16 from FP32 Q; it verifies three initialization-time capability
queries, the converted Q and unchanged MLA `q_rope`, not real FA3 accuracy.
The body reports 22 focused tests passing with 5,768 deselected. There is no
new benchmark harness, full-model accuracy result or changed-buffer graph
replay test in this diff.

The body reports H200 NVL/CUDA 13, Qwen3-0.6B, FA3, E4M3 KV, torch.compile
and CUDA graphs, disabled radix cache, fixed 256 input/256 output tokens,
and three runs per point. Output throughput improves 2.88%, 2.32% and 2.23%
at maximum concurrency 1, 8 and 32. The concurrency-eight profile has 2,184
fewer launches (28 layers × 78 measured decode steps), and summed kernel
time falls 2.22%. That sum is not itself request latency or critical-path
time. A separate eager operation report gives 27–39% for selected Qwen2.5/
Qwen3 shapes at batch 1/8/32, without a checked-in harness. Preserve these
as historical PR reports; the body does not identify a tested SHA for each
measurement, and they do not validate a current-main port.

Current main keeps conversion in
`FlashAttentionBackend.prepare_paged_mha_query`, reached by extend and decode.
Its rule depends on storage, head width, MXFP8 exclusion and the **selected
per-layer version**: prefill FA4 avoids this cast, whereas decode follows the
non-auto cache-dtype rule. `_ver_for` may upgrade a configured FA3 backend to
FA4 for a wide MHA layer or image-mask requirement. A new static capability
must preserve those phase/layer contracts and separate MLA rope inputs.

Current Qwen2 still calls rotary embedding directly, but Qwen3's
`forward_prepare_native` can bypass it through `table_qk_norm_rope_` during
BF16 decode. Admission checks the fused-norm flag, deterministic settings,
compiler setting, rotary type, head width 64/128/256, weight dtypes and
position rank. That kernel preserves BF16 rounding after normalization and
after rotation. A separate narrowly gated Hopper prefill kernel also returns
source-dtype Q. Adding an FP8 output only to `RotaryEmbedding` would miss
these producers. Any port needs to inspect the selected producer, preserve
its numerical contract, then verify the actual removed cast in a trace.
The [FP8 KV-store review](sglang-fp8-kv-store-source-review.md) covers the
separate K/V-write problem; Q conversion does not eliminate those stores.

Source anchors: [proposed kernel](https://github.com/sgl-project/sglang/blob/794d4d7fd422b543fa521f6ca3adc0af9c45e9ad/python/sglang/kernels/ops/attention/rope_quant.py),
[proposed rotary paths](https://github.com/sgl-project/sglang/blob/794d4d7fd422b543fa521f6ca3adc0af9c45e9ad/python/sglang/srt/layers/rotary_embedding/base.py),
[PR-head numerical test](https://github.com/sgl-project/sglang/blob/794d4d7fd422b543fa521f6ca3adc0af9c45e9ad/test/registered/kernels/ops/attention/test_rope.py),
[current backend conversion](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/flashattention_backend.py),
[current Qwen3 dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/qwen3.py),
[current table-norm/rotary kernel](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/table_qk_norm_rope.py),
and [native compile dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/fused_op.py).
