# SGLang GLM-5/5.1 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `bca3dd958` on `2026-04-24` and sgl-cookbook `origin/main` `816bad5` on `2026-04-21`.

Scope: GLM-5, GLM-5.1, `GlmMoeDsaForCausalLM`, DSA/NSA, FP8/MXFP4/NVFP4, NextN/MTP, tool templates, AMD, GB300, NPU, and dynamic chunking/profiling.

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

### PR #23060 - GLM-5 dynamic chunking profiling

- Link: https://github.com/sgl-project/sglang/pull/23060
- State: merged at `2026-04-23T11:30:57Z`
- Diff coverage: full diff fetched, `30` lines, `1` file.
- Motivation: pipeline-parallel profiling creates a synthetic `ForwardBatch`; GLM-5 DSA/DP-attention needs `is_extend_in_batch` to be set for that batch.
- Key implementation:

```diff
+set_is_extend_in_batch(batch.forward_mode.is_extend())
 _ = model_runner.forward(
     forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
 )
```

- Validation implications: run GLM-5 dynamic chunking/profiling with extend batches before assuming the PP path is safe.

### PR #23540 - GLM-5.1 MI300X/MI325X generator split

- Link: https://github.com/sgl-project/sglang/pull/23540
- State: merged at `2026-04-23T19:01:59Z`
- Diff coverage: full diff fetched, `154` lines, `3` files.
- Motivation: MI300X and MI325X should be separate GLM-5.1 command-generator rows rather than a combined label.
- Key implementation:

```diff
+{ id: 'mi300x', label: 'MI300X', default: false },
+{ id: 'mi325x', label: 'MI325X', default: false },
+mi325x: { bf16: { tp: 8, mem: 0.80 } },
```

- Validation implications: GLM-5.1 AMD docs and perf records should distinguish MI300X, MI325X, and MI355X.

## Next Work

Keep GLM-5 FP8, GLM-5 MXFP4, GLM-5 NVFP4, and GLM-5.1 FP8 separate. For any GLM DSA change, explicitly state whether DeepSeek V3.2 shared NSA/NextN paths are affected.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `GLM-5 / GLM-5.1` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-10 | [#18521](https://github.com/sgl-project/sglang/pull/18521) | merged | Support GlmMoeDsaForCausalLM | model wrapper, MoE/router, docs/config | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2026-02-13 | [#18804](https://github.com/sgl-project/sglang/pull/18804) | merged | Fix GLM-5 fused shared expert | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2026-02-17 | [#18911](https://github.com/sgl-project/sglang/pull/18911) | merged | [AMD] [GLM-5 Day 0] Add GLM-5 nightly test | tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-06 | [#20062](https://github.com/sgl-project/sglang/pull/20062) | merged | [V32/GLM5] Control the threshold of applying dense attention with an environ | attention/backend, quantization, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` |
| 2026-03-30 | [#21710](https://github.com/sgl-project/sglang/pull/21710) | merged | [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x | tests/benchmarks | `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-31 | [#21773](https://github.com/sgl-project/sglang/pull/21773) | merged | [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x | quantization, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`, `.github/workflows/nightly-test-amd.yml` |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22285](https://github.com/sgl-project/sglang/pull/22285) | merged | Add CI tests for GLM-5 | model wrapper, tests/benchmarks | `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py` |
| 2026-04-08 | [#22314](https://github.com/sgl-project/sglang/pull/22314) | merged | [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300 | scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-04-08 | [#22336](https://github.com/sgl-project/sglang/pull/22336) | merged | [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x | tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` |
| 2026-04-08 | [#22399](https://github.com/sgl-project/sglang/pull/22399) | merged | [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model | model wrapper, quantization, tests/benchmarks | `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py` |
| 2026-04-10 | [#22543](https://github.com/sgl-project/sglang/pull/22543) | merged | GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix | model wrapper | `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py` |
| 2026-04-11 | [#22595](https://github.com/sgl-project/sglang/pull/22595) | merged | fix: normalize tool message content for GLM5.1 chat template | tests/benchmarks | `test/registered/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2026-04-13 | [#22712](https://github.com/sgl-project/sglang/pull/22712) | merged | [NPU] update glm5 running guide | docs/config | `docs/platforms/ascend/ascend_npu_glm5_examples.md` |
| 2026-04-15 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion) | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-17 | [#23060](https://github.com/sgl-project/sglang/pull/23060) | merged | [fix] Fix dynamic chunking profiling crash on GLM-5 models | scheduler/runtime | `python/sglang/srt/managers/scheduler_pp_mixin.py` |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | [AMD] Enable MTP for GLM-5-mxfp4 model | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-04-23 | [#23540](https://github.com/sgl-project/sglang/pull/23540) | merged | docs: split MI300X and MI325X options in GLM-5.1 generator | docs/config | `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/intro.mdx` |

### File-level PR diff reading notes

### PR #18521 - Support GlmMoeDsaForCausalLM

- Link: https://github.com/sgl-project/sglang/pull/18521
- Status/date: `merged`, created 2026-02-10, merged 2026-02-10; author `JustinTong0323`.
- Diff scope read: `3` files, `+22/-7`; areas: model wrapper, MoE/router, docs/config; keywords: kv, moe, config, mla, spec, attention, cuda, eagle, flash, topk.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: def is_deepseek_nsa(config: PretrainedConfig) -> bool:; def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: def _handle_model_specific_adjustments(self):; def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode; def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: kv, moe, config, mla, spec, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18804 - Fix GLM-5 fused shared expert

- Link: https://github.com/sgl-project/sglang/pull/18804
- Status/date: `merged`, created 2026-02-13, merged 2026-02-16; author `FrankMinions`.
- Diff scope read: `1` files, `+2/-1`; areas: model wrapper, MoE/router; keywords: eagle, expert, kv, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: eagle, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18911
- Status/date: `merged`, created 2026-02-17, merged 2026-02-25; author `michaelzhang-ai`.
- Diff scope read: `5` files, `+635/-1`; areas: tests/benchmarks; keywords: test, attention, benchmark, cache, config, doc, moe.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: +"""MI35x GLM-5 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: +"""AMD GLM-5 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0 (71 lines); hunks: on:; on:
  - `.github/workflows/nightly-test-amd.yml` modified +70/-0 (70 lines); hunks: on:; jobs:
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +1/-1 (2 lines); hunks: "meta-llama/Llama-3.2-3B-Instruct": 0.55,
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; keywords observed in patches: test, attention, benchmark, cache, config, doc. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20062 - [V32/GLM5] Control the threshold of applying dense attention with an environ

- Link: https://github.com/sgl-project/sglang/pull/20062
- Status/date: `merged`, created 2026-03-06, merged 2026-03-09; author `Fridge003`.
- Diff scope read: `6` files, `+32/-59`; areas: attention/backend, quantization, tests/benchmarks, docs/config; keywords: kv, flash, mla, topk, cache, quant, attention, config, cuda, fp4.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: compute_cu_seqlens,; # Reuse this workspace buffer across all NSA backend instances; symbols: NSAFlashMLAMetadata:, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl
  - `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `python/sglang/srt/environ.py` modified +1/-2 (3 lines); hunks: class Envs:; symbols: Envs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`; keywords observed in patches: kv, flash, mla, topk, cache, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21710 - [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21710
- Status/date: `merged`, created 2026-03-30, merged 2026-04-08; author `michaelzhang-ai`.
- Diff scope read: `6` files, `+345/-5`; areas: tests/benchmarks; keywords: test, fp8, attention, config, benchmark, cache, kv, mla, quant.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: +"""MI35x Nightly performance benchmark for GLM-5.; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf
  - `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: +"""Nightly performance benchmark for GLM-5 on MI30x.; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +25/-1 (26 lines); hunks: jobs:; jobs:
  - `.github/workflows/nightly-test-amd.yml` modified +25/-0 (25 lines); hunks: jobs:; jobs:
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: def get_display_name(self) -> str:; def get_display_name(self) -> str:; symbols: get_display_name, get_display_name
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; keywords observed in patches: test, fp8, attention, config, benchmark, cache. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21773 - [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x

- Link: https://github.com/sgl-project/sglang/pull/21773
- Status/date: `merged`, created 2026-03-31, merged 2026-04-15; author `michaelzhang-ai`.
- Diff scope read: `4` files, `+528/-130`; areas: quantization, tests/benchmarks; keywords: fp4, test, benchmark, cache, config, doc, moe, quant.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: +"""MI35x GLM-5-MXFP4 GSM8K Completion Evaluation Test (8-GPU); symbols: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: +"""MI35x Nightly performance benchmark for GLM-5-MXFP4 model.; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass
  - `.github/workflows/nightly-test-amd.yml` modified +30/-66 (96 lines); hunks: on:; on:
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +30/-64 (94 lines); hunks: on:; on:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`, `.github/workflows/nightly-test-amd.yml`; keywords observed in patches: fp4, test, benchmark, cache, config, doc. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`, `.github/workflows/nightly-test-amd.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- Link: https://github.com/sgl-project/sglang/pull/22179
- Status/date: `merged`, created 2026-04-06, merged 2026-04-06; author `mmangkad`.
- Diff scope read: `1` files, `+11/-12`; areas: docs/config; keywords: attention, benchmark, cache, config, deepep, doc, eagle, flash, fp8, kv.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attent
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: attention, benchmark, cache, config, deepep, doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22285 - Add CI tests for GLM-5

- Link: https://github.com/sgl-project/sglang/pull/22285
- Status/date: `merged`, created 2026-04-07, merged 2026-04-08; author `Fridge003`.
- Diff scope read: `2` files, `+153/-30`; areas: model wrapper, tests/benchmarks; keywords: attention, config, cuda, fp8, kv, test, eagle, spec.
- Code diff details:
  - `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: write_github_step_summary,; def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: register_cuda_ci(est_time=720, suite="stage-c-test-8-gpu-h200"); def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; keywords observed in patches: attention, config, cuda, fp8, kv, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22314 - [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300

- Link: https://github.com/sgl-project/sglang/pull/22314
- Status/date: `merged`, created 2026-04-08, merged 2026-04-08; author `1am9trash`.
- Diff scope read: `1` files, `+27/-31`; areas: scheduler/runtime; keywords: attention, cache, fp8, kv, mla, quant, triton.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: quantize_k_cache,; def set_mla_kv_buffer(; symbols: set_mla_kv_buffer
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, cache, fp8, kv, mla, quant. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22336 - [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/22336
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `michaelzhang-ai`.
- Diff scope read: `6` files, `+918/-25`; areas: tests/benchmarks; keywords: test, fp8, attention, benchmark, cache, config, doc, fp4, kv, mla.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: +"""MI35x GLM-5.1 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: +"""AMD GLM-5.1 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: +"""MI35x Nightly performance benchmark for GLM-5.1.; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf
  - `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: +"""Nightly performance benchmark for GLM-5.1 on MI30x.; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51
  - `.github/workflows/nightly-test-amd.yml` modified +87/-4 (91 lines); hunks: on:; on:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; keywords observed in patches: test, fp8, attention, benchmark, cache, config. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- Link: https://github.com/sgl-project/sglang/pull/22399
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `Kangyan-Zhou`.
- Diff scope read: `3` files, `+82/-6`; areas: model wrapper, quantization, tests/benchmarks; keywords: cuda, fp8, test, attention, eagle, spec, topk.
- Code diff details:
  - `test/registered/8-gpu-models/test_glm_51_fp8.py` added +69/-0 (69 lines); hunks: +import unittest; symbols: TestGlm51Fp8, test_glm51_fp8
  - `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3 (13 lines); hunks: # Runs on both H200 and B200 via nightly-8-gpu-common suite; def test_qwen35(self):; symbols: TestQwen35, test_qwen35, test_qwen35
  - `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300", nightly=True); class TestGlm5Fp8(unittest.TestCase):; symbols: TestGlm5Fp8, test_glm5_fp8, test_glm5_fp8
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`; keywords observed in patches: cuda, fp8, test, attention, eagle, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22543 - GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix

- Link: https://github.com/sgl-project/sglang/pull/22543
- Status/date: `merged`, created 2026-04-10, merged 2026-04-14; author `ColinZ22`.
- Diff scope read: `3` files, `+8/-0`; areas: model wrapper; keywords: config, quant, cuda, fp4, kv, moe.
- Code diff details:
  - `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: def _get_quantization_config(; symbols: _get_quantization_config
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: def _handle_missing_default_values(self):; symbols: _handle_missing_default_values
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: config, quant, cuda, fp4, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22595 - fix: normalize tool message content for GLM5.1 chat template

- Link: https://github.com/sgl-project/sglang/pull/22595
- Status/date: `merged`, created 2026-04-11, merged 2026-04-16; author `whybeyoung`.
- Diff scope read: `2` files, `+67/-1`; areas: tests/benchmarks; keywords: cuda, doc, test.
- Code diff details:
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: ChatCompletionRequest,; def test_required_without_parser_invalid_json_returns_none(self):; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: logger = logging.getLogger(__name__); def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template
- Optimization/support interpretation: The concrete diff surface is `test/registered/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: cuda, doc, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22712 - [NPU] update glm5 running guide

- Link: https://github.com/sgl-project/sglang/pull/22712
- Status/date: `merged`, created 2026-04-13, merged 2026-04-13; author `zhsurpass`.
- Diff scope read: `1` files, `+8/-2`; areas: docs/config; keywords: doc.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_glm5_examples.md`; keywords observed in patches: doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_glm5_examples.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- Link: https://github.com/sgl-project/sglang/pull/22850
- Status/date: `merged`, created 2026-04-15, merged 2026-04-19; author `1am9trash`.
- Diff scope read: `1` files, `+24/-5`; areas: attention/backend; keywords: attention, cache, cuda, fp8, kv, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: from sglang.srt.environ import envs; _is_npu = is_npu(); symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cache, cuda, fp8, kv, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23060 - [fix] Fix dynamic chunking profiling crash on GLM-5 models

- Link: https://github.com/sgl-project/sglang/pull/23060
- Status/date: `merged`, created 2026-04-17, merged 2026-04-23; author `Baichuan7`.
- Diff scope read: `1` files, `+3/-0`; areas: scheduler/runtime; keywords: attention, scheduler.
- Code diff details:
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: get_attention_dp_rank,; def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/scheduler_pp_mixin.py`; keywords observed in patches: attention, scheduler. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/scheduler_pp_mixin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- Link: https://github.com/sgl-project/sglang/pull/23219
- Status/date: `merged`, created 2026-04-20, merged 2026-04-20; author `1am9trash`.
- Diff scope read: `1` files, `+41/-15`; areas: model wrapper; keywords: attention, config, fp8, processor, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: is_dp_attention_enabled,; def __init__(; symbols: __init__, forward, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`; keywords observed in patches: attention, config, fp8, processor, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23540 - docs: split MI300X and MI325X options in GLM-5.1 generator

- Link: https://github.com/sgl-project/sglang/pull/23540
- Status/date: `merged`, created 2026-04-23, merged 2026-04-23; author `zijiexia`.
- Diff scope read: `3` files, `+15/-13`; areas: docs/config; keywords: doc, flash, fp8, quant, spec.
- Code diff details:
  - `docs_new/docs.json` modified +8/-8 (16 lines); hunks: {
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: export const GLM51Deployment = () => {; export const GLM51Deployment = () => {
  - `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunks: metatags:
- Optimization/support interpretation: The concrete diff surface is `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/intro.mdx`; keywords observed in patches: doc, flash, fp8, quant, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/intro.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 18; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
