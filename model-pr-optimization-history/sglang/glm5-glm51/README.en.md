# SGLang GLM-5/5.1 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on `2026-04-22` and sgl-cookbook `origin/main` `816bad5` on `2026-04-21`.

Scope: GLM-5, GLM-5.1, `GlmMoeDsaForCausalLM`, DSA/NSA, FP8/MXFP4/NVFP4, NextN/MTP, tool templates, AMD, GB300, and NPU.

## Summary

GLM-5/5.1 is a shared DSA/NSA lane. Changes to `deepseek_v2.py`, `deepseek_nextn.py`, `nsa_backend.py`, or `nsa_indexer.py` can affect both DeepSeek V3.2 and GLM. Serving examples should preserve `--tool-call-parser glm47` and `--reasoning-parser glm45`.

## Code Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- `python/sglang/srt/layers/attention/nsa/`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `test/registered/8-gpu-models/test_dsa_models_basic.py`
- `test/registered/8-gpu-models/test_dsa_models_mtp.py`
- `test/registered/8-gpu-models/test_glm_51_fp8.py`
- `test/registered/gb300/test_glm5_fp8.py`
- `test/registered/gb300/test_glm5_nvfp4.py`
- `test/registered/amd/accuracy/`
- `test/registered/amd/perf/`

## Diff-Reviewed PR Cards

### PR #18521 - Support GlmMoeDsaForCausalLM

- Link: https://github.com/sgl-project/sglang/pull/18521
- State: merged at `2026-02-10T07:20:10Z`
- Diff coverage: full diff fetched, `462` lines, `3` files.
- Motivation: GLM-5 can reuse the DeepSeek V3.2 DSA/NSA stack, but needed architecture registration, RoPE compatibility, draft-model rewriting, and server-argument integration.
- Key implementation: add `GlmMoeDsaForCausalLM` to `is_deepseek_nsa()`, map GLM DSA draft models to `DeepseekV3ForCausalLMNextN`, subclass `DeepseekV2ForCausalLM`, and include GLM DSA in NSA backend/speculative/deterministic paths.

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass

EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
```

- Validation implications: GLM-5 launch should auto-select NSA and exercise DeepSeek NextN for MTP.

### PR #18804 - Fix GLM-5 fused shared expert

- Link: https://github.com/sgl-project/sglang/pull/18804
- State: merged at `2026-02-16T19:50:39Z`
- Diff coverage: full diff fetched, `131` lines, `1` file.
- Motivation: GLM-5 inherited DeepSeek DSA behavior but lacked the fused shared-expert count hook.
- Key implementation:

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

- Validation implications: test shared-expert routing/fusion, not just server startup.

### PR #18911 - AMD GLM-5 day-0 nightly

- Link: https://github.com/sgl-project/sglang/pull/18911
- State: merged at `2026-02-25T03:39:17Z`
- Diff coverage: full diff fetched, `1274` lines, `5` files.
- Motivation: GLM-5 needed ROCm coverage, and HIP RoPE had to avoid CUDA-only JIT/tvm paths.
- Key implementation:

```python
def forward_hip(self, *args, **kwargs):
    return self.forward_native(*args, **kwargs)
```

```python
GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
```

- Validation implications: AMD regressions should include HIP RoPE and 8-GPU GLM-5 accuracy.

### PR #20062 - DSA dense-attention threshold

- Link: https://github.com/sgl-project/sglang/pull/20062
- State: merged at `2026-03-09T21:36:10Z`
- Diff coverage: full diff fetched, `588` lines, `6` files.
- Motivation: the old force-MLA switch was too coarse; DSA needs a KV-length threshold for dense MHA versus sparse MLA.
- Key implementation:

```python
SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD = EnvInt(2048)
if model_arch == "GlmMoeDsaForCausalLM" and is_blackwell_supported():
    envs.SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD.set(0)
```

```python
and max_kv_len <= envs.SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD.get()
```

- Validation implications: GLM Blackwell should force sparse MLA; other DSA runs should default the threshold to `index_topk`.

### PR #21710 - AMD GLM-5-FP8 perf nightly

- Link: https://github.com/sgl-project/sglang/pull/21710
- State: merged at `2026-04-08T05:43:14Z`
- Diff coverage: full diff fetched, `537` lines, `6` files.
- Motivation: AMD GLM-5-FP8 had accuracy coverage but no MI30x/MI35x throughput tracking.
- Key implementation:

```yaml
continue-on-error: true
python3 run_suite.py --hw amd --suite nightly-perf-8-gpu-glm5 --nightly
```

```python
model_path="zai-org/GLM-5-FP8",
other_args=["--reasoning-parser", "glm45", "--tool-call-parser", "glm47"]
```

- Validation implications: keep parser and FP8 KV guidance aligned with AMD CI.

### PR #21773 - AMD GLM-5-MXFP4 MI35x

- Link: https://github.com/sgl-project/sglang/pull/21773
- State: merged at `2026-04-15T01:55:36Z`
- Diff coverage: full diff fetched, `863` lines, `4` files.
- Motivation: GLM-5 MXFP4/Quark needed a distinct MI35x accuracy/perf lane.
- Key implementation:

```yaml
nightly-8-gpu-mi35x-glm5-mxfp4-rocm720:
  runs-on: linux-mi35x-gpu-8
```

- Validation implications: track GLM-5 MXFP4 separately from GLM-5 FP8 and GLM-5.1 FP8.

### PR #22179 - DeepSeek V3.2/GLM-5 docs

- Link: https://github.com/sgl-project/sglang/pull/22179
- State: merged at `2026-04-06T06:26:43Z`
- Diff coverage: full diff fetched, `127` lines, `1` file.
- Motivation: GLM-5 shares DSA usage with DeepSeek V3.2 but has GLM-specific parsers.
- Key implementation:

```diff
-To server GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
+To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
```

- Validation implications: preserve `glm47`, `glm45`, NSA flags, and verify the dense-attention env-var spelling against current code.

### PR #22285 - GLM-5 H200 8-GPU CI

- Link: https://github.com/sgl-project/sglang/pull/22285
- State: merged at `2026-04-08T08:05:36Z`
- Diff coverage: full diff fetched, `8911` lines, `2` files.
- Motivation: GLM-5 needed the same H200 DSA regression lanes as DeepSeek V3.2.
- Key implementation:

```python
GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
self.assertGreater(metrics["score"], 0.94)
self.assertGreater(avg_spec_accept_length, 2.7)
```

- Validation implications: include TP/DP and MTP/spec-v2, and inspect speculative acceptance length.

### PR #22314 - AMD GLM-5 FP8 KV dispatch

- Link: https://github.com/sgl-project/sglang/pull/22314
- State: merged at `2026-04-08T04:16:02Z`
- Diff coverage: full diff fetched, `121` lines, `1` file.
- Motivation: MI300/ROCm GLM-5 FP8 KV should use HIP raw MLA KV layout, not NVIDIA byte/scales layout.
- Key implementation:

```python
if _is_hip and self.use_nsa and self.dtype == fp8_dtype:
    set_mla_kv_buffer_triton_fp8_quant(...)
elif self.nsa_kv_cache_store_fp8:
    cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(...)
```

- Validation implications: compare FP8 KV against BF16/no-FP8-KV baselines on AMD.

### PR #22336 - AMD GLM-5.1-FP8 nightly

- Link: https://github.com/sgl-project/sglang/pull/22336
- State: merged at `2026-04-09T05:57:43Z`
- Diff coverage: full diff fetched, `1485` lines, `6` files.
- Motivation: GLM-5.1-FP8 needs separate MI30x/MI35x coverage with TP=8 and EP=8.
- Key implementation:

```python
model_path="zai-org/GLM-5.1-FP8"
other_args=["--tp", "8", "--ep-size", "8", "--reasoning-parser=glm45", "--tool-call-parser=glm47"]
```

- Validation implications: document EP=8 and split MI30x/MI35x perf diagnosis.

### PR #22399 - GLM-5.1 H200/B200/GB300 tests

- Link: https://github.com/sgl-project/sglang/pull/22399
- State: merged at `2026-04-09T00:04:57Z`
- Diff coverage: full diff fetched, `225` lines, `3` files.
- Motivation: H200/B200/GB300 needed GLM-5.1-FP8 coverage, while GLM-5 NVFP4 must remain GLM-5 because no GLM-5.1 NVFP4 checkpoint exists.
- Key implementation:

```python
GLM_51_FP8_MODEL_PATH = "zai-org/GLM-5.1-FP8"
variant="TP8+DP8+MTP"
env={"SGLANG_ENABLE_SPEC_V2": "1"}
```

- Validation implications: do not rename NVFP4 docs/tests to GLM-5.1.

### PR #22543 - GLM-5/5.1 MXFP4 checkpoint compatibility

- Link: https://github.com/sgl-project/sglang/pull/22543
- State: merged at `2026-04-14T06:56:49Z`
- Diff coverage: full diff fetched, `122` lines, `3` files.
- Motivation: GLM MXFP4/Quark checkpoints reuse DeepSeek loader code but must avoid DeepSeek-V3-only Quark post-load transforms.
- Key implementation:

```python
if model_config.quantization == "quark":
    packed_modules_mapping.update({"gate_up_proj": ["gate_proj", "up_proj"]})
```

- Validation implications: check gate/up fused loading and ensure DeepSeek-only post-load logic does not touch GLM weights.

### PR #22595 - GLM5.1 tool message normalization

- Link: https://github.com/sgl-project/sglang/pull/22595
- State: merged at `2026-04-16T08:48:38Z`
- Diff coverage: full diff fetched, `191` lines, `2` files.
- Motivation: OpenAI clients may send tool results as content-part arrays, but GLM chat templates expect strings.
- Key implementation:

```python
def normalize_tool_content(role: str, content):
    if role != "tool" or not isinstance(content, list):
        return content
    ...
    return " ".join(text_parts)
```

- Validation implications: test tool result arrays and make sure the model stops repeating tool calls.

### PR #22712 - NPU GLM-5 guide

- Link: https://github.com/sgl-project/sglang/pull/22712
- State: merged at `2026-04-13T14:53:24Z`
- Diff coverage: full diff fetched, `33` lines, `1` file.
- Motivation: Ascend GLM-5 docs should pin transformers instead of installing main.
- Key implementation:

```diff
+pip install transformers==5.3.0
+pip install git+https://github.com/huggingface/transformers.git@v5.3.0
```

- Validation implications: NPU smoke tests should use transformers 5.3.0.

### PR #22850 - AMD NSA indexer kernel reduction

- Link: https://github.com/sgl-project/sglang/pull/22850
- State: merged at `2026-04-19T07:18:12Z`
- Diff coverage: full diff fetched, `141` lines, `1` file.
- Motivation: AMD DSA indexer still had avoidable kernels around `weights_proj` and index-K cache storage.
- Key implementation:

```python
if _use_aiter:
    kv_cache = buf.unsqueeze(1).view(fp8_dtype)
    indexer_k_quant_and_cache(key, kv_cache, out_loc, self.block_size, self.scale_fmt)
    return
```

- Validation implications: compare AITER and non-AITER GLM-5/5.1 perf and accuracy.

### PR #23219 - GLM-5-MXFP4 MTP

- Link: https://github.com/sgl-project/sglang/pull/23219
- State: merged at `2026-04-20T23:09:08Z`
- Diff coverage: full diff fetched, `121` lines, `1` file.
- Motivation: Quark GLM-5-MXFP4 needs NextN/MTP projection loading and layer exclusion to match checkpoint layout.
- Key implementation:

```python
if quant_config is not None and quant_config.get_name() == "quark":
    self.eh_proj = ReplicatedLinear(..., quant_config=quant_config)
```

```python
if should_ignore_layer(mapped_prefix, nextn_quant_config.exclude_layers):
    nextn_quant_config = None
```

- Validation implications: test GLM-5-MXFP4 MTP separately from FP8 MTP.

## Next Work

Keep GLM-5 FP8, GLM-5 MXFP4, GLM-5 NVFP4, and GLM-5.1 FP8 separate. For any GLM DSA change, explicitly state whether DeepSeek V3.2 shared NSA/NextN paths are affected.
