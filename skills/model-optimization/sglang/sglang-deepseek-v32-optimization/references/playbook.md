# DeepSeek V3.2 Playbook

Use this playbook when a V3.2 optimization or regression report is ambiguous. The goal is to determine whether the failure is DSA/NSA, inherited DeepSeek runtime, parser/template, MTP, CP, or backend-specific.

## 1. Identify The Model Shape

Collect these first:

- `hf_config.architectures`
- whether `is_deepseek_nsa(config)` is true
- model variant: V3.2-Exp, V3.2, Speciale, NVFP4, MXFP4
- quantization config
- `--attention-backend`
- `--nsa-prefill-backend`
- `--nsa-decode-backend`
- `--kv-cache-dtype`
- TP / DP / EP / PP / PD
- `--enable-dp-attention`
- `--enable-nsa-prefill-context-parallel`
- `--nsa-prefill-cp-mode`
- MTP/speculative decoding args
- parser pair and chat template
- hardware

If `is_deepseek_nsa(config)` is false, switch to `sglang-deepseek-v3-r1-optimization`.
If the report is V3.1 parser behavior, switch to `sglang-deepseek-v31-optimization`.

## 2. Server Defaults

Read `server_args.py` first.

Expected current behavior:

- DSA models set `attention_backend = "nsa"` when not specified.
- `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` defaults to model `index_topk` if not user-set.
- DSA KV cache dtype defaults to `fp8_e4m3` on SM100 and `bfloat16` otherwise.
- DSA KV cache dtype is restricted to `bfloat16` or `fp8_e4m3` in the mainline path.
- ROCm defaults to TileLang NSA prefill/decode.
- Blackwell with FP8 KV defaults to TRTLLM NSA backends.
- Hopper with FP8 KV can use FlashMLA KV.
- BF16 KV can use `flashmla_sparse` plus `trtllm` or `fa3` depending on architecture.

If a user manually sets NSA backend but leaves KV cache dtype as `auto`, ask for the explicit dtype before comparing performance.

## 3. NSA Indexer

Read `nsa_indexer.py`.

Check:

- q/k projection and `weights_proj`
- mixed-type LayerNorm path
- dual-stream decode threshold
- activation quant
- FP8 index/K-cache store
- top-k transform backend
- CP rerange and all-gather behavior
- whether `skip_topk` is active through IndexCache

Common clues:

- 128K bugs often point to `get_k_and_s_triton`
- FP8 KV bugs often point to scale buffer or fused-store handling
- AMD kernel-count regressions often point to weights_proj and K-cache store fusion
- quality regressions can come from FP32 precision loss in `weights_proj`

## 4. NSA Backend

Read `nsa_backend.py`.

Check:

- metadata fields: `nsa_cache_seqlens_int32`, `nsa_cu_seqlens_q`, `nsa_cu_seqlens_k`, `nsa_seqlens_expanded`
- paged vs ragged top-k transform
- prefill backend auto-selection
- decode backend
- FP8 K-cache quant/dequant
- FlashMLA metadata and paged MQA schedule
- MTP precompute mixin

If top-k indices have the right shape but attention is wrong, look at transform, cache seqlens, and page-table offsets before changing the indexer.

## 5. Context Parallel

Context parallel is still guarded by constraints.

Check:

- `--enable-nsa-prefill-context-parallel`
- `--nsa-prefill-cp-mode`
- `round-robin-split` vs `in-seq-split`
- DP size and EP size
- DeepEP requirement for in-seq split
- PD decode restrictions
- CP positions in NextN/MTP

Common open issues:

- AMD round-robin output corruption: [#20360](https://github.com/sgl-project/sglang/pull/20360)
- ragged gather mismatch in CP round-robin: [#20531](https://github.com/sgl-project/sglang/pull/20531)
- TP indexer weight: [#19609](https://github.com/sgl-project/sglang/pull/19609)

## 6. MTP / Spec Decoding

If `SGLANG_ENABLE_SPEC_V2=1` or EAGLE args are present:

- inspect `deepseek_nextn.py`
- inspect NSA metadata precompute
- inspect target_verify and draft_extend backend handling
- verify CP positions
- check page-table overflow fixes
- verify draft-model mapping includes V3.2 when relevant

MTP bugs often show up only under DP attention or CP.

## 7. IndexCache

IndexCache means some layers skip top-k computation and reuse previous top-k indices.

Check in `deepseek_v2.py`:

- `index_topk_freq`
- `index_topk_pattern`
- `skip_topk`
- `next_skip_topk`
- `prev_topk_indices`

Validation:

- `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`

## 8. Parser And Tool Calling

Standard V3.2 uses DSML:

```text
<｜DSML｜function_calls>
<｜DSML｜invoke name="tool_name">
...
</｜DSML｜invoke>
</｜DSML｜function_calls>
```

`DeepSeekV32Detector` supports:

- XML parameter tags
- direct JSON inside invoke
- partial JSON during streaming
- stable prefix diff streaming

Open parser risks:

- reasoning parser may strip tool markers: [#21179](https://github.com/sgl-project/sglang/pull/21179)
- malformed partial JSON should not crash parsing: [#21546](https://github.com/sgl-project/sglang/pull/21546)

V3.2-Exp cookbook docs may use `deepseekv31` with the V3.2 template, while standard V3.2 uses `deepseekv32` and no custom chat template. Check the exact model variant.

## 9. Validation Order

Pick the smallest lane:

1. parser/template only: DSML parser or `test/manual/test_deepseek_chat_templates.py`
2. DSA base: `test/registered/8-gpu-models/test_deepseek_v32.py`
3. NSA backend pair: `test_deepseek_v32_nsa_backends`
4. MTP: V3.2 TP8+MTP or DP8+MTP variants
5. IndexCache: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`
6. CP: CP-specific registered or DeepEP suite tests
7. AMD/NPU/Blackwell: the platform-specific lane that matches the backend

## 10. Common False Conclusions

- "V3.2 is only V3 with a new checkpoint." It activates DSA/NSA.
- "NSA backend means one kernel." It is indexer, top-k transform, sparse MLA, KV quant/dequant, and metadata.
- "FP8 KV is always faster." Backend, padding, scale buffer, and architecture decide.
- "MTP bugs are only NextN bugs." For V3.2 they often involve NSA metadata or CP positions.
- "IndexCache only changes speed." It changes when top-k is computed and must preserve accuracy.
- "V3.2 parser is V3.1 parser." Standard V3.2 uses DSML and `deepseekv32`.
