# TensorRT-LLM GLM-5 Series (5/5.1/5.2/5.3-Flash) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` | [#11990](https://github.com/NVIDIA/TensorRT-LLM/pull/11990), [#13901](https://github.com/NVIDIA/TensorRT-LLM/pull/13901), [#18388](https://github.com/NVIDIA/TensorRT-LLM/pull/18388) |
| `docs/source/deployment-guide/deployment-guide-for-glm-5.3-flash-on-trtllm.md` | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) |
| `tensorrt_llm/_torch/configs/glm5_next.py` | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) |
| `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) |
| `tensorrt_llm/_torch/models/modeling_glm5_next.py` | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) |
| `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) |
| `tests/integration/defs/accuracy/test_glm52.py` | [#18470](https://github.com/NVIDIA/TensorRT-LLM/pull/18470) |
| `tests/integration/defs/accuracy/test_glm53_flash.py` | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml` | [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/aggregated/glm5_fp4_2_nodes_grace_blackwell.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#18006](https://github.com/NVIDIA/TensorRT-LLM/pull/18006) |
| `tests/scripts/perf-sanity/aggregated/glm5_fp4_grace_blackwell.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` | [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb300_glm-5-fp4_1k1k_con1_ctx1_dep2_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb300_glm-5-fp4_1k1k_con512_ctx1_dep2_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb300_glm-5-fp4_8k1k_con1_ctx1_dep2_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf-sanity/disaggregated/gb300_glm-5-fp4_8k1k_con512_ctx1_dep2_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) |
| `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con1_ctx1_dep2_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con512_ctx1_dep2_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1_ctx1_dep2_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con512_ctx1_dep2_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) |
| `tests/unittest/_torch/modeling/test_glm5_next_contracts.py` | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) |

## PR Coverage Summary

- Git-traced PRs: 8
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 8
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-03-18 | [#11990](https://github.com/NVIDIA/TensorRT-LLM/pull/11990) | merged | [None][feat] GLM 5 support and DSA MTP fixes | `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`, `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py`, `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-05-11 | [#13901](https://github.com/NVIDIA/TensorRT-LLM/pull/13901) | merged | [None][chore] Remove glm_moe_dsa tokenizer WAR after Transformers 5.x upgrade | `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`, `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py`, `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` |
| 2026-06-11 | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) | merged | [None][test] Update K2.5 andGLM-5 into CI Perf Test | `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` |
| 2026-07-23 | [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) | merged | [None][feat] Default GLM-5 to the Python KV-cache transceiver | `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` |
| 2026-08-21 | [#18006](https://github.com/NVIDIA/TensorRT-LLM/pull/18006) | merged | [https://nvbugs/6329155][fix] Raise glm5 tep8 8k1k max_num_tokens to 8192 to fit isl=8192 prefill | `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` |
| 2026-08-31 | [#18388](https://github.com/NVIDIA/TensorRT-LLM/pull/18388) | merged | [None][doc] Update GLM-5 docs for GLM-5.3 and disaggregated serving | `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` |
| 2026-09-04 | [#18470](https://github.com/NVIDIA/TensorRT-LLM/pull/18470) | merged | [None][test] validate GLM-5.2 feature support matrix | `tests/integration/defs/accuracy/test_glm52.py` |
| 2026-09-28 | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) | merged | [None][feat] Add GLM-5.3-Flash (glm5_next) support | `tensorrt_llm/_torch/models/modeling_glm5_next.py`, `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py`, `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` |

## Per-PR Diff Audit Cards

### PR #11990 - [None][feat] GLM 5 support and DSA MTP fixes

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11990
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`; associated commits `07e2440108ec`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 22 files, +868/-63, 1187 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` added +384/-0 (384 lines); hunks: -0,0 +1,384; `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, from_pretrained, touching `_load_tokenizer_config, GlmMoeDsaTokenizer, __init__`; `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +6/-0 (6 lines); hunks: -1784,6 +1784,7 @@ def forward(; -1820,6 +1821,11 @@ def __init__(self, model_config: ModelConfig[PretrainedCo...; symbols: forward, DeepseekV3ForCausalLM, __init__, touching `forward, DeepseekV3ForCausalLM, __init__`; `tensorrt_llm/_torch/models/modeling_speculative.py` modified +3/-3 (6 lines); hunks: -786,7 +786,7 @@ def __init__(; -830,7 +830,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` added +384/-0 (384 lines); hunks: -0,0 +1,384
  - `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, from_pretrained
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +6/-0 (6 lines); hunks: -1784,6 +1784,7 @@ def forward(; -1820,6 +1821,11 @@ def __init__(self, model_config: ModelConfig[PretrainedCo...; symbols: forward, DeepseekV3ForCausalLM, __init__
  - `tensorrt_llm/_torch/models/modeling_speculative.py` modified +3/-3 (6 lines); hunks: -786,7 +786,7 @@ def __init__(; -830,7 +830,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: __init__, load_weights
  - `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` added +5/-0 (5 lines); hunks: -0,0 +1,5
- Key code excerpts:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md
@@ -0,0 +1,384 @@
+# Deployment Guide for GLM-5 on TensorRT LLM - Blackwell Hardware
+## Introduction
+This deployment guide provides step-by-step instructions for running the GLM-5 model using TensorRT LLM with FP8 and NVFP4 quantization, optimized for NVIDIA Blackwell GPUs. It co
+GLM-5 uses Multi-Latent Attention (MLA) with DeepSeek Sparse Attention (DSA). It shares the same architecture as DeepSeek V3.2 and reuses the `DeepseekV32ForCausalLM` code path in
+The guide is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA's accelerated stack.
+## Prerequisites
diff -- tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py
@@ -0,0 +1,87 @@
+"""GLM-Moe-Dsa tokenizer implementation.
+Loads tokenizer from tokenizer.json and applies tokenizer_config.json manually
+to work around incompatibilities when the checkpoint was saved with
+transformers 5.x (TokenizersBackend / list-style extra_special_tokens).
+"""
+import json
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -1784,6 +1784,7 @@ def forward(
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` added +384/-0
  - runtime: `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` added +87/-0; `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +6/-0; `tensorrt_llm/_torch/models/modeling_speculative.py` modified +3/-3; `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` added +5/-0; `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +61/-37; `tensorrt_llm/_torch/speculative/interface.py` modified +83/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13901 - [None][chore] Remove glm_moe_dsa tokenizer WAR after Transformers 5.x upgrade

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13901
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`; associated commits `d68150c83771`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +6/-121, 206 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +0/-6 (6 lines); hunks: -77,7 +77,6 @@ If you want to use the latest main branch, you can build from...; -99,7 +98,6 @@ GLM-5 natively supports Multi-Token Prediction (MTP), which en...; `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` removed +0/-98 (98 lines); hunks: -1,98 +0,0; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, vocab_size, touching `_load_tokenizer_config, GlmMoeDsaTokenizer, __init__`; `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` removed +0/-5 (5 lines); hunks: -1,5 +0,0; `tensorrt_llm/tokenizer/tokenizer.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6.
- Code diff details:
  - `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +0/-6 (6 lines); hunks: -77,7 +77,6 @@ If you want to use the latest main branch, you can build from...; -99,7 +98,6 @@ GLM-5 natively supports Multi-Token Prediction (MTP), which en...
  - `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` removed +0/-98 (98 lines); hunks: -1,98 +0,0; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, vocab_size
  - `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` removed +0/-5 (5 lines); hunks: -1,5 +0,0
  - `tensorrt_llm/tokenizer/tokenizer.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6
  - `tensorrt_llm/bench/utils/data.py` modified +3/-3 (6 lines); hunks: -37,9 +37,9 @@ def initialize_tokenizer(model_name: str,; symbols: initialize_tokenizer
- Key code excerpts:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md
@@ -77,7 +77,6 @@ If you want to use the latest main branch, you can build from source: [https://n
-custom_tokenizer: glm_moe_dsa
@@ -99,7 +98,6 @@ GLM-5 natively supports Multi-Token Prediction (MTP), which enables speculative
-custom_tokenizer: glm_moe_dsa
@@ -156,10 +154,6 @@ trtllm-serve \
-#### `custom_tokenizer`
-* **Description:** Specifies a custom tokenizer to use. GLM-5 requires the `glm_moe_dsa` tokenizer.
diff -- tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py
@@ -1,98 +0,0 @@
-"""GLM-Moe-Dsa tokenizer implementation.
-Loads tokenizer from tokenizer.json and applies tokenizer_config.json manually
-to work around incompatibilities when the checkpoint was saved with
-transformers 5.x (TokenizersBackend / list-style extra_special_tokens).
-"""
-import json
diff -- tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py
@@ -1,5 +0,0 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +0/-6
  - runtime: `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` removed +0/-98; `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` removed +0/-5; `tensorrt_llm/tokenizer/tokenizer.py` modified +0/-1; `tensorrt_llm/bench/utils/data.py` modified +3/-3; `tensorrt_llm/bench/benchmark/low_latency.py` modified +1/-1; `tensorrt_llm/bench/benchmark/throughput.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/unittest/llmapi/apps/_test_openai_chat_guided_decoding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14960 - [None][test] Update K2.5 andGLM-5 into CI Perf Test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14960
- Status/date: merged / 2026-06-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/aggregated/glm5_fp4_2_nodes_grace_blackwell.yaml`, `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml`, `tests/scripts/perf-sanity/aggregated/glm5_fp4_grace_blackwell.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` and 23 files; associated commits `835fd6115bf7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 48 files, +2666/-41, 2937 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108; `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108; `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108; `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108.
- Code diff details:
  - `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` added +105/-0 (105 lines); hunks: -0,0 +1,105
- Key code excerpts:

```diff
diff -- tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml
@@ -0,0 +1,108 @@
+metadata:
+  model_name: glm_5_nvfp4
+  precision: fp4
+  model_dir_name: GLM-5-NVFP4
+  supported_gpus:
+  - GB200
diff -- tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml
@@ -0,0 +1,108 @@
+metadata:
+  model_name: glm_5_nvfp4
+  precision: fp4
+  model_dir_name: GLM-5-NVFP4
+  supported_gpus:
+  - GB200
diff -- tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml
@@ -0,0 +1,108 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` added +105/-0; `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` added +105/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/integration/test_lists/test-db/l0_b200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node8_gpu32.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16524 - [None][feat] Default GLM-5 to the Python KV-cache transceiver

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16524
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` and 13 files; associated commits `e16dcc54fde0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +110/-3, 299 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml` modified +4/-2 (6 lines); hunks: -20,7 +20,8 @@ context_servers:; -55,5 +56,6 @@ generation_servers:; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -61,6 +61,7 @@ worker_config:; -89,5 +90,6 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ worker_config:; -93,5 +94,6 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:.
- Code diff details:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml` modified +4/-2 (6 lines); hunks: -20,7 +20,8 @@ context_servers:; -55,5 +56,6 @@ generation_servers:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -61,6 +61,7 @@ worker_config:; -89,5 +90,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ worker_config:; -93,5 +94,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ worker_config:; -93,5 +94,6 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml
@@ -20,7 +20,8 @@ context_servers:
-    backend: DEFAULT
+    backend: NIXL
+    transceiver_runtime: PYTHON
@@ -55,5 +56,6 @@ generation_servers:
-    backend: DEFAULT
+    backend: NIXL
diff -- tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml
@@ -61,6 +61,7 @@ worker_config:
+      transceiver_runtime: PYTHON
@@ -89,5 +90,6 @@ worker_config:
+      transceiver_runtime: PYTHON
diff -- tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml
@@ -66,6 +66,7 @@ worker_config:
+      transceiver_runtime: PYTHON
@@ -93,5 +94,6 @@ worker_config:
+      transceiver_runtime: PYTHON
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml` modified +4/-2; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18006 - [https://nvbugs/6329155][fix] Raise glm5 tep8 8k1k max_num_tokens to 8192 to fit isl=8192 prefill

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18006
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml`; associated commits `f6d7404652f4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +1/-2, 17 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` modified +1/-1 (2 lines); hunks: -13,7 +13,7 @@ server_configs:.
- Code diff details:
  - `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` modified +1/-1 (2 lines); hunks: -13,7 +13,7 @@ server_configs:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml
@@ -13,7 +13,7 @@ server_configs:
-    max_num_tokens: 256
+    max_num_tokens: 8192
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`, `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18388 - [None][doc] Update GLM-5 docs for GLM-5.3 and disaggregated serving

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18388
- Status/date: merged / 2026-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`; associated commits `6e6f506077cb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +63/-12, 155 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +61/-11 (72 lines); hunks: -4,29 +4,44; -53,7 +68,7 @@ docker run --rm -it \.
- Code diff details:
  - `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +61/-11 (72 lines); hunks: -4,29 +4,44; -53,7 +68,7 @@ docker run --rm -it \
- Key code excerpts:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md
@@ -4,29 +4,44 @@
-GLM-5 uses Multi-Latent Attention (MLA) with DeepSeek Sparse Attention (DSA). It shares the same architecture as DeepSeek V3.2 and reuses the `DeepseekV32ForCausalLM` code path in
+GLM-5 uses Multi-Latent Attention (MLA) with DeepSeek Sparse Attention (DSA). It shares the same architecture as DeepSeek V3.2 (with minor changes) and is served through the `GlmM
+This guide applies to the GLM-5 family, including GLM-5.2 and GLM-5.3. GLM-5.3 is a weight update over GLM-5.2 with the same architecture and code path, so the server configuratio
+### Validated Features
+The following features have been tested with GLM-5 on TensorRT LLM:
+* CUDA Graph
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +61/-11
- Risk and verification: This is mostly docs/examples in `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`, `docs/source/models/supported-models.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #18470 - [None][test] validate GLM-5.2 feature support matrix

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18470
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/accuracy/test_glm52.py`; associated commits `eeaaf59f38b3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +546/-218, 826 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/accuracy/test_glm52.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _ForceTokenLogitsProcessor, __init__, __call__, _force_token, touching `_ForceTokenLogitsProcessor, __init__, __call__`.
- Code diff details:
  - `tests/integration/defs/accuracy/test_glm52.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _ForceTokenLogitsProcessor, __init__, __call__, _force_token
- Key code excerpts:

```diff
diff -- tests/integration/defs/accuracy/test_glm52.py
@@ -0,0 +1,529 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/accuracy/test_glm52.py` added +529/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/.test_durations`, `tests/integration/defs/accuracy/test_glm52.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/perf/pytorch_model_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19136 - [None][feat] Add GLM-5.3-Flash (glm5_next) support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19136
- Status/date: merged / 2026-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-glm-5.3-flash-on-trtllm.md`, `tensorrt_llm/_torch/configs/glm5_next.py`, `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_glm5_next.py`, `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` and 7 files; associated commits `d21a8cebbd16`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 59 files, +8867/-142, 9860 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_glm5_next.py` added +2081/-0 (2081 lines); hunks: -0,0 +1,2081; symbols: Glm5NextSchedule, num_layers, attention_indices, mlp_indices, touching `Glm5NextSchedule, num_layers, attention_indices`; `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` added +1210/-0 (1210 lines); hunks: -0,0 +1,1210; symbols: _image_encoder_cuda_graph_config, _text_dtype, _require_trtllm_vision_backend, _create_linear_weights, touching `_image_encoder_cuda_graph_config, _text_dtype, _require_trtllm_vision_backend`; `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Glm5NextWeightAudit, remap_glm5_next_key, audit_glm5_next_checkpoint, glm5_next_is_quantized, touching `Glm5NextWeightAudit, remap_glm5_next_key, audit_glm5_next_checkpoint`; `tensorrt_llm/_torch/configs/glm5_next.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: Glm5NextTextConfig, __init__, Glm5NextVisionConfig, Glm5NextConfig, touching `Glm5NextTextConfig, __init__, Glm5NextVisionConfig`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_glm5_next.py` added +2081/-0 (2081 lines); hunks: -0,0 +1,2081; symbols: Glm5NextSchedule, num_layers, attention_indices, mlp_indices
  - `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` added +1210/-0 (1210 lines); hunks: -0,0 +1,1210; symbols: _image_encoder_cuda_graph_config, _text_dtype, _require_trtllm_vision_backend, _create_linear_weights
  - `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Glm5NextWeightAudit, remap_glm5_next_key, audit_glm5_next_checkpoint, glm5_next_is_quantized
  - `tensorrt_llm/_torch/configs/glm5_next.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: Glm5NextTextConfig, __init__, Glm5NextVisionConfig, Glm5NextConfig
  - `tests/unittest/_torch/modeling/test_glm5_next_contracts.py` added +872/-0 (872 lines); hunks: -0,0 +1,872; symbols: test_config_fallback_round_trip, test_config_fallback_matches_native, test_text_processing_without_native_processor, test_mixed_processor_kwargs
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_glm5_next.py
@@ -0,0 +1,2081 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/modeling_glm5_next_vision.py
@@ -0,0 +1,1210 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py
@@ -0,0 +1,172 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_glm5_next.py` added +2081/-0; `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` added +1210/-0; `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` added +172/-0; `tensorrt_llm/_torch/configs/glm5_next.py` added +156/-0
  - tests: `tests/unittest/_torch/modeling/test_glm5_next_contracts.py` added +872/-0; `tests/integration/defs/accuracy/test_glm53_flash.py` added +466/-0
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5.3-flash-on-trtllm.md` added +450/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/acceptance_length.yaml`, `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_glm53_flash.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
