# TensorRT-LLM GLM-5 Series (5/5.1/5.2/5.3-Flash) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
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

## PR 覆盖总览

- git 追溯 PR 数: 8
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 8
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-03-18 | [#11990](https://github.com/NVIDIA/TensorRT-LLM/pull/11990) | merged | [None][feat] GLM 5 support and DSA MTP fixes | `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`, `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py`, `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-05-11 | [#13901](https://github.com/NVIDIA/TensorRT-LLM/pull/13901) | merged | [None][chore] Remove glm_moe_dsa tokenizer WAR after Transformers 5.x upgrade | `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`, `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py`, `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` |
| 2026-06-11 | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) | merged | [None][test] Update K2.5 andGLM-5 into CI Perf Test | `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` |
| 2026-07-23 | [#16524](https://github.com/NVIDIA/TensorRT-LLM/pull/16524) | merged | [None][feat] Default GLM-5 to the Python KV-cache transceiver | `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` |
| 2026-08-21 | [#18006](https://github.com/NVIDIA/TensorRT-LLM/pull/18006) | merged | [https://nvbugs/6329155][fix] Raise glm5 tep8 8k1k max_num_tokens to 8192 to fit isl=8192 prefill | `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` |
| 2026-08-31 | [#18388](https://github.com/NVIDIA/TensorRT-LLM/pull/18388) | merged | [None][doc] Update GLM-5 docs for GLM-5.3 and disaggregated serving | `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` |
| 2026-09-04 | [#18470](https://github.com/NVIDIA/TensorRT-LLM/pull/18470) | merged | [None][test] validate GLM-5.2 feature support matrix | `tests/integration/defs/accuracy/test_glm52.py` |
| 2026-09-28 | [#19136](https://github.com/NVIDIA/TensorRT-LLM/pull/19136) | merged | [None][feat] Add GLM-5.3-Flash (glm5_next) support | `tensorrt_llm/_torch/models/modeling_glm5_next.py`, `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py`, `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` |

## 逐 PR diff 审计卡

### PR #11990 - [None][feat] GLM 5 support and DSA MTP fixes

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11990
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`；关联提交 `07e2440108ec`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 22 个文件，+868/-63，可读 patch 1187 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` added +384/-0 (384 lines); hunks: -0,0 +1,384；`tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, from_pretrained，涉及 `_load_tokenizer_config, GlmMoeDsaTokenizer, __init__`；`tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +6/-0 (6 lines); hunks: -1784,6 +1784,7 @@ def forward(; -1820,6 +1821,11 @@ def __init__(self, model_config: ModelConfig[PretrainedCo...; symbols: forward, DeepseekV3ForCausalLM, __init__，涉及 `forward, DeepseekV3ForCausalLM, __init__`；`tensorrt_llm/_torch/models/modeling_speculative.py` modified +3/-3 (6 lines); hunks: -786,7 +786,7 @@ def __init__(; -830,7 +830,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` added +384/-0 (384 lines); hunks: -0,0 +1,384
  - `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, from_pretrained
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +6/-0 (6 lines); hunks: -1784,6 +1784,7 @@ def forward(; -1820,6 +1821,11 @@ def __init__(self, model_config: ModelConfig[PretrainedCo...; symbols: forward, DeepseekV3ForCausalLM, __init__
  - `tensorrt_llm/_torch/models/modeling_speculative.py` modified +3/-3 (6 lines); hunks: -786,7 +786,7 @@ def __init__(; -830,7 +830,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: __init__, load_weights
  - `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` added +5/-0 (5 lines); hunks: -0,0 +1,5
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` added +384/-0
  - runtime: `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` added +87/-0; `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +6/-0; `tensorrt_llm/_torch/models/modeling_speculative.py` modified +3/-3; `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` added +5/-0; `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +61/-37; `tensorrt_llm/_torch/speculative/interface.py` modified +83/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13901 - [None][chore] Remove glm_moe_dsa tokenizer WAR after Transformers 5.x upgrade

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13901
- 状态/时间: merged / 2026-05-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`；关联提交 `d68150c83771`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+6/-121，可读 patch 206 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +0/-6 (6 lines); hunks: -77,7 +77,6 @@ If you want to use the latest main branch, you can build from...; -99,7 +98,6 @@ GLM-5 natively supports Multi-Token Prediction (MTP), which en...；`tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` removed +0/-98 (98 lines); hunks: -1,98 +0,0; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, vocab_size，涉及 `_load_tokenizer_config, GlmMoeDsaTokenizer, __init__`；`tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` removed +0/-5 (5 lines); hunks: -1,5 +0,0；`tensorrt_llm/tokenizer/tokenizer.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6。
- 代码 diff 细节:
  - `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +0/-6 (6 lines); hunks: -77,7 +77,6 @@ If you want to use the latest main branch, you can build from...; -99,7 +98,6 @@ GLM-5 natively supports Multi-Token Prediction (MTP), which en...
  - `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` removed +0/-98 (98 lines); hunks: -1,98 +0,0; symbols: _load_tokenizer_config, GlmMoeDsaTokenizer, __init__, vocab_size
  - `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` removed +0/-5 (5 lines); hunks: -1,5 +0,0
  - `tensorrt_llm/tokenizer/tokenizer.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6
  - `tensorrt_llm/bench/utils/data.py` modified +3/-3 (6 lines); hunks: -37,9 +37,9 @@ def initialize_tokenizer(model_name: str,; symbols: initialize_tokenizer
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +0/-6
  - runtime: `tensorrt_llm/tokenizer/glm_moe_dsa/tokenizer.py` removed +0/-98; `tensorrt_llm/tokenizer/glm_moe_dsa/__init__.py` removed +0/-5; `tensorrt_llm/tokenizer/tokenizer.py` modified +0/-1; `tensorrt_llm/bench/utils/data.py` modified +3/-3; `tensorrt_llm/bench/benchmark/low_latency.py` modified +1/-1; `tensorrt_llm/bench/benchmark/throughput.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/unittest/llmapi/apps/_test_openai_chat_guided_decoding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14960 - [None][test] Update K2.5 andGLM-5 into CI Perf Test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14960
- 状态/时间: merged / 2026-06-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/scripts/perf-sanity/aggregated/glm5_fp4_2_nodes_grace_blackwell.yaml`, `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml`, `tests/scripts/perf-sanity/aggregated/glm5_fp4_grace_blackwell.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` 等 23 个文件；关联提交 `835fd6115bf7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 48 个文件，+2666/-41，可读 patch 2937 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108；`tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108；`tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108；`tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108。
- 代码 diff 细节:
  - `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` added +105/-0 (105 lines); hunks: -0,0 +1,105
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_1k1k_con4096_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb300_glm-5-fp4_8k1k_con1024_ctx1_dep2_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` added +108/-0; `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` added +105/-0; `tests/scripts/perf/disaggregated/gb200_glm-5-fp4_8k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` added +105/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/integration/test_lists/test-db/l0_b200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node8_gpu32.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16524 - [None][feat] Default GLM-5 to the Python KV-cache transceiver

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16524
- 状态/时间: merged / 2026-07-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` 等 13 个文件；关联提交 `e16dcc54fde0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+110/-3，可读 patch 299 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml` modified +4/-2 (6 lines); hunks: -20,7 +20,8 @@ context_servers:; -55,5 +56,6 @@ generation_servers:；`tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -61,6 +61,7 @@ worker_config:; -89,5 +90,6 @@ worker_config:；`tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ worker_config:; -93,5 +94,6 @@ worker_config:；`tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:。
- 代码 diff 细节:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml` modified +4/-2 (6 lines); hunks: -20,7 +20,8 @@ context_servers:; -55,5 +56,6 @@ generation_servers:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -61,6 +61,7 @@ worker_config:; -89,5 +90,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ worker_config:; -93,5 +94,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -66,6 +66,7 @@ worker_config:; -93,5 +94,6 @@ worker_config:
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml` modified +4/-2; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1024_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4ep4_gentp4ep4_glm5_nvfp4_dp_tllm.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb256_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_glm-5-fp4_1k1k_con512_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18006 - [https://nvbugs/6329155][fix] Raise glm5 tep8 8k1k max_num_tokens to 8192 to fit isl=8192 prefill

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18006
- 状态/时间: merged / 2026-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml`；关联提交 `f6d7404652f4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+1/-2，可读 patch 17 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` modified +1/-1 (2 lines); hunks: -13,7 +13,7 @@ server_configs:。
- 代码 diff 细节:
  - `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` modified +1/-1 (2 lines); hunks: -13,7 +13,7 @@ server_configs:
- 关键代码摘录:

```diff
diff -- tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml
@@ -13,7 +13,7 @@ server_configs:
-    max_num_tokens: 256
+    max_num_tokens: 8192
```

- 提取文件（未人工审阅）:
  - tests: `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/waives.txt`, `tests/scripts/perf-sanity/aggregated/glm5_fp4_blackwell.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18388 - [None][doc] Update GLM-5 docs for GLM-5.3 and disaggregated serving

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18388
- 状态/时间: merged / 2026-08-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`；关联提交 `6e6f506077cb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+63/-12，可读 patch 155 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +61/-11 (72 lines); hunks: -4,29 +4,44; -53,7 +68,7 @@ docker run --rm -it \。
- 代码 diff 细节:
  - `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +61/-11 (72 lines); hunks: -4,29 +4,44; -53,7 +68,7 @@ docker run --rm -it \
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md` modified +61/-11
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md`, `docs/source/models/supported-models.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #18470 - [None][test] validate GLM-5.2 feature support matrix

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18470
- 状态/时间: merged / 2026-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/integration/defs/accuracy/test_glm52.py`；关联提交 `eeaaf59f38b3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+546/-218，可读 patch 826 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/integration/defs/accuracy/test_glm52.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _ForceTokenLogitsProcessor, __init__, __call__, _force_token，涉及 `_ForceTokenLogitsProcessor, __init__, __call__`。
- 代码 diff 细节:
  - `tests/integration/defs/accuracy/test_glm52.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _ForceTokenLogitsProcessor, __init__, __call__, _force_token
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `tests/integration/defs/accuracy/test_glm52.py` added +529/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/.test_durations`, `tests/integration/defs/accuracy/test_glm52.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/perf/pytorch_model_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19136 - [None][feat] Add GLM-5.3-Flash (glm5_next) support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19136
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-glm-5.3-flash-on-trtllm.md`, `tensorrt_llm/_torch/configs/glm5_next.py`, `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_glm5_next.py`, `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` 等 7 个文件；关联提交 `d21a8cebbd16`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 59 个文件，+8867/-142，可读 patch 9860 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_glm5_next.py` added +2081/-0 (2081 lines); hunks: -0,0 +1,2081; symbols: Glm5NextSchedule, num_layers, attention_indices, mlp_indices，涉及 `Glm5NextSchedule, num_layers, attention_indices`；`tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` added +1210/-0 (1210 lines); hunks: -0,0 +1,1210; symbols: _image_encoder_cuda_graph_config, _text_dtype, _require_trtllm_vision_backend, _create_linear_weights，涉及 `_image_encoder_cuda_graph_config, _text_dtype, _require_trtllm_vision_backend`；`tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Glm5NextWeightAudit, remap_glm5_next_key, audit_glm5_next_checkpoint, glm5_next_is_quantized，涉及 `Glm5NextWeightAudit, remap_glm5_next_key, audit_glm5_next_checkpoint`；`tensorrt_llm/_torch/configs/glm5_next.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: Glm5NextTextConfig, __init__, Glm5NextVisionConfig, Glm5NextConfig，涉及 `Glm5NextTextConfig, __init__, Glm5NextVisionConfig`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_glm5_next.py` added +2081/-0 (2081 lines); hunks: -0,0 +1,2081; symbols: Glm5NextSchedule, num_layers, attention_indices, mlp_indices
  - `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` added +1210/-0 (1210 lines); hunks: -0,0 +1,1210; symbols: _image_encoder_cuda_graph_config, _text_dtype, _require_trtllm_vision_backend, _create_linear_weights
  - `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Glm5NextWeightAudit, remap_glm5_next_key, audit_glm5_next_checkpoint, glm5_next_is_quantized
  - `tensorrt_llm/_torch/configs/glm5_next.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: Glm5NextTextConfig, __init__, Glm5NextVisionConfig, Glm5NextConfig
  - `tests/unittest/_torch/modeling/test_glm5_next_contracts.py` added +872/-0 (872 lines); hunks: -0,0 +1,872; symbols: test_config_fallback_round_trip, test_config_fallback_matches_native, test_text_processing_without_native_processor, test_mixed_processor_kwargs
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_glm5_next.py` added +2081/-0; `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py` added +1210/-0; `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py` added +172/-0; `tensorrt_llm/_torch/configs/glm5_next.py` added +156/-0
  - tests: `tests/unittest/_torch/modeling/test_glm5_next_contracts.py` added +872/-0; `tests/integration/defs/accuracy/test_glm53_flash.py` added +466/-0
  - docs: `docs/source/deployment-guide/deployment-guide-for-glm-5.3-flash-on-trtllm.md` added +450/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/acceptance_length.yaml`, `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_glm53_flash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
