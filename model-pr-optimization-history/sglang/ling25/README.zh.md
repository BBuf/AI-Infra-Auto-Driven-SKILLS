# sglang Ling 2.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/InclusionAI/Ling-2.5-1T.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 44
- 当前文档总 PR 数: 43
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-06 | [#8680](https://github.com/sgl-project/sglang/pull/8680) | merged | Support bailing moe | `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py`, `docs/supported_models/generative_models.md` |
| 2025-09-12 | [#10359](https://github.com/sgl-project/sglang/pull/10359) | merged | Support LingV2 model | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` |
| 2025-09-12 | [#10362](https://github.com/sgl-project/sglang/pull/10362) | merged | Fix Bailing MoE model bugs | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py` |
| 2025-09-15 | [#9338](https://github.com/sgl-project/sglang/pull/9338) | merged | Refactor TopK to ensure readability and extensibility | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-09-24 | [#10860](https://github.com/sgl-project/sglang/pull/10860) | merged | fix bailing_moe with enable_dp_attention | `python/sglang/srt/models/bailing_moe.py` |
| 2025-09-26 | [#10749](https://github.com/sgl-project/sglang/pull/10749) | merged | Fuse write kv buffer into rope for qwen3 moe & bailing moe | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-10-12 | [#11465](https://github.com/sgl-project/sglang/pull/11465) | merged | bailingMoE: Fix Key error of deepep_mode | `python/sglang/srt/models/bailing_moe.py` |
| 2025-10-12 | [#11331](https://github.com/sgl-project/sglang/pull/11331) | merged | Deprecate `global_server_args_dict` | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-13 | [#11520](https://github.com/sgl-project/sglang/pull/11520) | merged | Revert "Deprecate `global_server_args_dict`" | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-13 | [#11528](https://github.com/sgl-project/sglang/pull/11528) | merged | Depreate `global_server_args_dict` | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-17 | [#11685](https://github.com/sgl-project/sglang/pull/11685) | merged | [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files | `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/models/qwen2_audio.py`, `python/sglang/srt/models/longcat_flash.py` |
| 2025-10-20 | [#11847](https://github.com/sgl-project/sglang/pull/11847) | merged | [9/N] MoE Refactor: cleanup dispatcher interfaces | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-10-31 | [#12369](https://github.com/sgl-project/sglang/pull/12369) | merged | Enable bailing_moe to support TP=16 | `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-07 | [#14337](https://github.com/sgl-project/sglang/pull/14337) | merged | remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.) | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py` |
| 2025-12-12 | [#13730](https://github.com/sgl-project/sglang/pull/13730) | merged | [bugfix] fix TBO crashes when attn_tp_size > 1 | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-21 | [#15526](https://github.com/sgl-project/sglang/pull/15526) | merged | Optimize Bailing-MoE with FlashInfer Fused All-Reduce | `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-28 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py` |
| 2026-01-10 | [#13715](https://github.com/sgl-project/sglang/pull/13715) | merged | Fix EPLB + FP4 Quantization Compatibility Issue | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-01-24 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-01-30 | [#17808](https://github.com/sgl-project/sglang/pull/17808) | merged | Fix the scenario where eh_proj is quantized in the bailing moe nextn weights | `python/sglang/srt/models/bailing_moe_nextn.py` |
| 2026-02-01 | [#15119](https://github.com/sgl-project/sglang/pull/15119) | merged | feat: Add Ling Flash v2.0 support for Eagle3 | `python/sglang/srt/models/bailing_moe.py` |
| 2026-02-13 | [#18598](https://github.com/sgl-project/sglang/pull/18598) | merged | Support LingV2_5 model | `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/layers/attention/linear/seg_la.py`, `python/sglang/srt/layers/attention/linear/lightning_attn.py` |
| 2026-02-13 | [#18793](https://github.com/sgl-project/sglang/pull/18793) | merged | Cleanup debug log for Ring model | `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/configs/mamba_utils.py` |
| 2026-02-15 | [#18860](https://github.com/sgl-project/sglang/pull/18860) | merged | update pre-commit config | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` |
| 2026-03-16 | [#19382](https://github.com/sgl-project/sglang/pull/19382) | merged | Add NPU basic function testcases | `test/registered/ascend/interface/test_npu_openai_function_calling.py`, `test/registered/ascend/llm_models/test_npu_phi_4_multimodal_llm.py`, `test/registered/ascend/vlm_models/test_npu_phi4_multimodal_instruct.py` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-03-19 | [#9744](https://github.com/sgl-project/sglang/pull/9744) | merged | [CPU] Add FP8 Bmm support | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-03-23 | [#20316](https://github.com/sgl-project/sglang/pull/20316) | merged | fix fused_set_kv_buffer for rope with Ling-v2 | `python/sglang/srt/models/bailing_moe.py` |
| 2026-04-01 | [#20751](https://github.com/sgl-project/sglang/pull/20751) | merged | [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture | `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` |
| 2026-04-26 | [#23732](https://github.com/sgl-project/sglang/pull/23732) | merged | Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) | `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-04-27 | [#23748](https://github.com/sgl-project/sglang/pull/23748) | merged | refactor(moe): centralize post-experts all-reduce skip predicate | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-30 | [#21126](https://github.com/sgl-project/sglang/pull/21126) | merged | [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split | `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` |
| 2026-05-04 | [#24333](https://github.com/sgl-project/sglang/pull/24333) | merged | nextn subclass owns post_load_weights is_nextn | `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_loader/loader.py` |
| 2026-05-11 | [#24977](https://github.com/sgl-project/sglang/pull/24977) | merged | fix gb envs in deployment guide | `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` |
| 2026-05-27 | [#23837](https://github.com/sgl-project/sglang/pull/23837) | merged | Add Ling_2_6 | `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` |
| 2026-05-29 | [#26474](https://github.com/sgl-project/sglang/pull/26474) | merged | [HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6 | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py` |
| 2026-06-02 | [#26623](https://github.com/sgl-project/sglang/pull/26623) | merged | Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T) | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py` |
| 2026-06-03 | [#27116](https://github.com/sgl-project/sglang/pull/27116) | merged | Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)" | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-06-03 | [#27120](https://github.com/sgl-project/sglang/pull/27120) | merged | Fix hybrid linear attention dispatch by layer id with draft-worker awareness | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |

## 逐 PR diff 审计卡

### PR #8680 - Support bailing moe

- 链接: https://github.com/sgl-project/sglang/pull/8680
- 状态/时间: merged / 2025-08-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+427/-0，可读 patch 441 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support bailing moe」；模型线: Ling 2.5；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py`, `docs/supported_models/generative_models.md`；技术摘要: 覆盖「Support bailing moe」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py`, `docs/supported_models/generative_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: BailingAttention, __init__, forward, BailingMLP，涉及 `BailingAttention, __init__, forward`；`test/srt/models/test_generation_models.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ class ModelCase:; symbols: ModelCase，涉及 `ModelCase`；`docs/supported_models/generative_models.md` modified +1/-0 (1 lines); hunks: -47,5 +47,6 @@ in the GitHub search bar.。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: BailingAttention, __init__, forward, BailingMLP
  - `test/srt/models/test_generation_models.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ class ModelCase:; symbols: ModelCase
  - `docs/supported_models/generative_models.md` modified +1/-0 (1 lines); hunks: -47,5 +47,6 @@ in the GitHub search bar.
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -0,0 +1,425 @@
+# Copyright 2023-2024 SGLang Team
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/bailing_moe.py
+from collections.abc import Iterable
+from typing import Optional, Tuple
+import torch
+import torch.nn.functional as F
diff -- test/srt/models/test_generation_models.py
@@ -67,6 +67,7 @@ class ModelCase:
+    ModelCase("inclusionAI/Ling-lite", trust_remote_code=True),
diff -- docs/supported_models/generative_models.md
@@ -47,5 +47,6 @@ in the GitHub search bar.
+| **Ling** (16.8B–290B) | `inclusionAI/Ling-lite`, `inclusionAI/Ling-plus` | InclusionAI’s open MoE models. Ling-Lite has 16.8B total / 2.75B active parameters, and Ling-Plus has
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` added +425/-0
  - tests: `test/srt/models/test_generation_models.py` modified +1/-0
  - docs: `docs/supported_models/generative_models.md` modified +1/-0
- 验证与风险: diff 自带测试面 `test/srt/models/test_generation_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10359 - Support LingV2 model

- 链接: https://github.com/sgl-project/sglang/pull/10359
- 状态/时间: merged / 2025-09-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+1165/-221，可读 patch 1642 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support LingV2 model」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json`；技术摘要: 覆盖「Support LingV2 model」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +795/-218 (1013 lines); hunks: -1,19 +1,51; -22,356 +54,828; symbols: BailingAttention, BailingMoEMLP, __init__, forward，涉及 `BailingAttention, BailingMoEMLP, __init__`；`python/sglang/srt/models/bailing_moe_nextn.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: BailingMoEModelNextN, __init__, forward, BailingMoeForCausalLMNextN，涉及 `BailingMoEModelNextN, __init__, forward`；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146；`python/sglang/srt/layers/linear.py` modified +32/-0 (32 lines); hunks: -893,6 +893,35 @@ def _load_fused_module_from_checkpoint(; -906,6 +935,9 @@ def weight_loader_v2(; symbols: _load_fused_module_from_checkpoint, _load_qkv_block_scale, weight_loader_v2，涉及 `_load_fused_module_from_checkpoint, _load_qkv_block_scale, weight_loader_v2`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +795/-218 (1013 lines); hunks: -1,19 +1,51; -22,356 +54,828; symbols: BailingAttention, BailingMoEMLP, __init__, forward
  - `python/sglang/srt/models/bailing_moe_nextn.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: BailingMoEModelNextN, __init__, forward, BailingMoeForCausalLMNextN
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/linear.py` modified +32/-0 (32 lines); hunks: -893,6 +893,35 @@ def _load_fused_module_from_checkpoint(; -906,6 +935,9 @@ def weight_loader_v2(; symbols: _load_fused_module_from_checkpoint, _load_qkv_block_scale, weight_loader_v2
  - `python/sglang/srt/configs/model_config.py` modified +5/-0 (5 lines); hunks: -141,6 +141,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -1,19 +1,51 @@
-# Copyright 2023-2024 SGLang Team
-# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/bailing_moe.py
-from collections.abc import Iterable
-from typing import Optional, Tuple
+# coding=utf-8
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -0,0 +1,168 @@
+# coding=utf-8
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
+# and OPT implementations in this library. It has been modified from its
+# original forms to accommodate minor architectural differences compared
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json
@@ -0,0 +1,146 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +795/-218; `python/sglang/srt/models/bailing_moe_nextn.py` added +168/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0; `python/sglang/srt/layers/linear.py` modified +32/-0; `python/sglang/srt/configs/model_config.py` modified +5/-0; `python/sglang/srt/server_args.py` modified +8/-1
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +11/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10362 - Fix Bailing MoE model bugs

- 链接: https://github.com/sgl-project/sglang/pull/10362
- 状态/时间: merged / 2025-09-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+8/-5，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Bailing MoE model bugs」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Fix Bailing MoE model bugs」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +7/-4 (11 lines); hunks: -128,7 +128,9 @@ def forward(; -328,7 +330,7 @@ def forward_normal_dual_stream(; symbols: forward, forward_normal_dual_stream, forward_normal，涉及 `forward, forward_normal_dual_stream, forward_normal`；`python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -757,7 +757,7 @@ def __post_init__(self):; symbols: __post_init__，涉及 `__post_init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +7/-4 (11 lines); hunks: -128,7 +128,9 @@ def forward(; -328,7 +330,7 @@ def forward_normal_dual_stream(; symbols: forward, forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -757,7 +757,7 @@ def __post_init__(self):; symbols: __post_init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -128,7 +128,9 @@ def forward(
-        hidden_states, _ = self.down_proj(hidden_states)
+        hidden_states, _ = self.down_proj(
+            hidden_states, skip_all_reduce=use_reduce_scatter
+        )
@@ -328,7 +330,7 @@ def forward_normal_dual_stream(
-        shared_output = self._forward_shared_experts(hidden_states)
diff -- python/sglang/srt/server_args.py
@@ -757,7 +757,7 @@ def __post_init__(self):
-                "BailingMoeV2ForCausalLM",
+                "BailingMoeForCausalLM",
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +7/-4; `python/sglang/srt/server_args.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9338 - Refactor TopK to ensure readability and extensibility

- 链接: https://github.com/sgl-project/sglang/pull/9338
- 状态/时间: merged / 2025-09-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+52/-47，可读 patch 296 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Refactor TopK to ensure readability and extensibility」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；技术摘要: 覆盖「Refactor TopK to ensure readability and extensibility」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native，涉及 `TopKConfig, __init__, forward_native`；`python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim，涉及 `_is_fp4_quantization_enabled, selection, _get_tile_tokens_dim`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class，涉及 `_forward_ll, get_moe_impl_class`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-2 (4 lines); hunks: -260,7 +260,7 @@ def __init__(; -853,7 +853,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -19,6 +19,7 @@
+    TYPE_CHECKING,
@@ -51,6 +52,9 @@
+if TYPE_CHECKING:
+    from sglang.srt.layers.quantization import QuantizationConfig
@@ -94,6 +98,7 @@ class TopKConfig:
+    output_format: Optional[TopKOutputFormat] = None
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -65,14 +65,10 @@
-    should_use_flashinfer_trtllm_moe,
-from sglang.srt.layers.moe.fused_moe_triton.layer import (
-    FusedMoE,
-    _is_fp4_quantization_enabled,
-)
-from sglang.srt.layers.moe.topk import TopK
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -74,16 +74,6 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +30/-9; `python/sglang/srt/models/deepseek_v2.py` modified +7/-12; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4; `python/sglang/srt/models/longcat_flash.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10860 - fix bailing_moe with enable_dp_attention

- 链接: https://github.com/sgl-project/sglang/pull/10860
- 状态/时间: merged / 2025-09-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix bailing_moe with enable_dp_attention」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「fix bailing_moe with enable_dp_attention」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -45,12 +45,12; -702,7 +702,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -45,12 +45,12; -702,7 +702,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -45,12 +45,12 @@
+    is_dp_attention_enabled,
-    ReplicatedLinear,
@@ -702,7 +702,7 @@ def __init__(
-                use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
+                enable_tp=not is_dp_attention_enabled(),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- 链接: https://github.com/sgl-project/sglang/pull/10749
- 状态/时间: merged / 2025-09-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+105/-34，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fuse write kv buffer into rope for qwen3 moe & bailing moe」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「Fuse write kv buffer into rope for qwen3 moe & bailing moe」；主要实现面是 `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg，涉及 `enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg`；`python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention，涉及 `forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg`；`python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core，涉及 `forward_prepare, forward_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/utils.py
@@ -0,0 +1,51 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/gpt_oss.py
@@ -66,6 +66,10 @@
+from sglang.srt.models.utils import (
+    create_fused_set_kv_buffer_arg,
+    enable_fused_set_kv_buffer,
+)
@@ -193,33 +197,6 @@ def forward_normal(
-def _enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
diff -- python/sglang/srt/models/bailing_moe.py
@@ -72,6 +72,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/utils.py` added +51/-0; `python/sglang/srt/models/gpt_oss.py` modified +7/-30; `python/sglang/srt/models/bailing_moe.py` modified +25/-2; `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11465 - bailingMoE: Fix Key error of deepep_mode

- 链接: https://github.com/sgl-project/sglang/pull/11465
- 状态/时间: merged / 2025-10-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「bailingMoE: Fix Key error of deepep_mode」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「bailingMoE: Fix Key error of deepep_mode」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -54,7 +54,7; -293,7 +293,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -54,7 +54,7; -293,7 +293,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -54,7 +54,7 @@
-from sglang.srt.layers.moe import get_moe_a2a_backend
+from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
@@ -293,7 +293,7 @@ def __init__(
-                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
+                deepep_mode=get_deepep_mode(),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11331 - Deprecate `global_server_args_dict`

- 链接: https://github.com/sgl-project/sglang/pull/11331
- 状态/时间: merged / 2025-10-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 54 个文件，+240/-321，可读 patch 1946 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Deprecate `global_server_args_dict`」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「Deprecate `global_server_args_dict`」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__，涉及 `__init__`；`python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str，涉及 `__init__, initialize, _get_attention_backend`；`python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts，涉及 `__init__, determine_num_fused_shared_experts`；`python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward，涉及 `__init__, compute_logprobs_for_multi_item_scoring, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__
  - `python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str
  - `python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11 (14 lines); hunks: -38,20 +38,12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -35,7 +35,6 @@
-from sglang.srt.debug_utils.dumper import dumper
@@ -108,10 +107,11 @@
-from sglang.srt.managers.schedule_batch import global_server_args_dict
+from sglang.srt.server_args import get_global_server_args
+from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
@@ -520,7 +520,7 @@ def __init__(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -83,10 +83,6 @@
-from sglang.srt.managers.schedule_batch import (
-    GLOBAL_SERVER_ARGS_KEYS,
-    global_server_args_dict,
-)
@@ -125,7 +121,11 @@
-from sglang.srt.server_args import ServerArgs
diff -- python/sglang/srt/models/glm4_moe.py
@@ -56,18 +56,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21; `python/sglang/srt/model_executor/model_runner.py` modified +16/-21; `python/sglang/srt/models/glm4_moe.py` modified +8/-12; `python/sglang/srt/layers/logits_processor.py` modified +6/-10; `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11; `python/sglang/srt/layers/communicator.py` modified +8/-5
- 验证与风险: diff 自带测试面 `test/srt/rl/test_fp32_lm_head.py`, `test/srt/test_gptqmodel_dynamic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11520 - Revert "Deprecate `global_server_args_dict`"

- 链接: https://github.com/sgl-project/sglang/pull/11520
- 状态/时间: merged / 2025-10-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 54 个文件，+321/-240，可读 patch 1946 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "Deprecate `global_server_args_dict`"」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「Revert "Deprecate `global_server_args_dict`"」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +21/-23 (44 lines); hunks: -35,6 +35,7; -107,11 +108,10; symbols: __init__，涉及 `__init__`；`python/sglang/srt/model_executor/model_runner.py` modified +21/-16 (37 lines); hunks: -83,6 +83,10; -121,11 +125,7; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str，涉及 `__init__, initialize, _get_attention_backend`；`python/sglang/srt/models/glm4_moe.py` modified +12/-8 (20 lines); hunks: -56,13 +56,18; -72,7 +77,6; symbols: __init__, determine_num_fused_shared_experts，涉及 `__init__, determine_num_fused_shared_experts`；`python/sglang/srt/layers/logits_processor.py` modified +10/-6 (16 lines); hunks: -38,15 +38,17; -228,8 +230,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward，涉及 `__init__, compute_logprobs_for_multi_item_scoring, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +21/-23 (44 lines); hunks: -35,6 +35,7; -107,11 +108,10; symbols: __init__
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-16 (37 lines); hunks: -83,6 +83,10; -121,11 +125,7; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str
  - `python/sglang/srt/models/glm4_moe.py` modified +12/-8 (20 lines); hunks: -56,13 +56,18; -72,7 +77,6; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/logits_processor.py` modified +10/-6 (16 lines); hunks: -38,15 +38,17; -228,8 +230,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +11/-3 (14 lines); hunks: -38,12 +38,20
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -35,6 +35,7 @@
+from sglang.srt.debug_utils.dumper import dumper
@@ -107,11 +108,10 @@
+from sglang.srt.managers.schedule_batch import global_server_args_dict
-from sglang.srt.server_args import get_global_server_args
-from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
@@ -520,7 +520,7 @@ def __init__(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -83,6 +83,10 @@
+from sglang.srt.managers.schedule_batch import (
+    GLOBAL_SERVER_ARGS_KEYS,
+    global_server_args_dict,
+)
@@ -121,11 +125,7 @@
-from sglang.srt.server_args import (
diff -- python/sglang/srt/models/glm4_moe.py
@@ -56,13 +56,18 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +21/-23; `python/sglang/srt/model_executor/model_runner.py` modified +21/-16; `python/sglang/srt/models/glm4_moe.py` modified +12/-8; `python/sglang/srt/layers/logits_processor.py` modified +10/-6; `python/sglang/srt/models/qwen3_vl_moe.py` modified +11/-3; `python/sglang/srt/layers/communicator.py` modified +5/-8
- 验证与风险: diff 自带测试面 `test/srt/rl/test_fp32_lm_head.py`, `test/srt/test_gptqmodel_dynamic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11528 - Depreate `global_server_args_dict`

- 链接: https://github.com/sgl-project/sglang/pull/11528
- 状态/时间: merged / 2025-10-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 54 个文件，+240/-321，可读 patch 1946 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Depreate `global_server_args_dict`」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「Depreate `global_server_args_dict`」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__，涉及 `__init__`；`python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str，涉及 `__init__, initialize, _get_attention_backend`；`python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts，涉及 `__init__, determine_num_fused_shared_experts`；`python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward，涉及 `__init__, compute_logprobs_for_multi_item_scoring, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__
  - `python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str
  - `python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11 (14 lines); hunks: -38,20 +38,12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -35,7 +35,6 @@
-from sglang.srt.debug_utils.dumper import dumper
@@ -108,10 +107,11 @@
-from sglang.srt.managers.schedule_batch import global_server_args_dict
+from sglang.srt.server_args import get_global_server_args
+from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
@@ -520,7 +520,7 @@ def __init__(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -83,10 +83,6 @@
-from sglang.srt.managers.schedule_batch import (
-    GLOBAL_SERVER_ARGS_KEYS,
-    global_server_args_dict,
-)
@@ -125,7 +121,11 @@
-from sglang.srt.server_args import ServerArgs
diff -- python/sglang/srt/models/glm4_moe.py
@@ -56,18 +56,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21; `python/sglang/srt/model_executor/model_runner.py` modified +16/-21; `python/sglang/srt/models/glm4_moe.py` modified +8/-12; `python/sglang/srt/layers/logits_processor.py` modified +6/-10; `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11; `python/sglang/srt/layers/communicator.py` modified +8/-5
- 验证与风险: diff 自带测试面 `test/srt/rl/test_fp32_lm_head.py`, `test/srt/test_gptqmodel_dynamic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11685 - [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files

- 链接: https://github.com/sgl-project/sglang/pull/11685
- 状态/时间: merged / 2025-10-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 151 个文件，+124/-406，可读 patch 1915 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/models/qwen2_audio.py`, `python/sglang/srt/models/longcat_flash.py`；技术摘要: 覆盖「[Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files」；主要实现面是 `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/models/qwen2_audio.py`, `python/sglang/srt/models/longcat_flash.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-18 (20 lines); hunks: -1,28 +1,12；`python/sglang/srt/models/qwen2_audio.py` modified +2/-15 (17 lines); hunks: -23,30 +23,18; -60,7 +48,6；`python/sglang/srt/models/longcat_flash.py` modified +1/-14 (15 lines); hunks: -44,9 +44,7; -87,20 +85,15；`python/sglang/srt/models/longcat_flash_nextn.py` modified +2/-13 (15 lines); hunks: -32,14 +32,10; -75,7 +71,6。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-18 (20 lines); hunks: -1,28 +1,12
  - `python/sglang/srt/models/qwen2_audio.py` modified +2/-15 (17 lines); hunks: -23,30 +23,18; -60,7 +48,6
  - `python/sglang/srt/models/longcat_flash.py` modified +1/-14 (15 lines); hunks: -44,9 +44,7; -87,20 +85,15
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +2/-13 (15 lines); hunks: -32,14 +32,10; -75,7 +71,6
  - `python/sglang/srt/models/mimo.py` modified +2/-13 (15 lines); hunks: -1,28 +1,17
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/w8a8_int8.py
@@ -1,28 +1,12 @@
-import importlib
-import sys
-from typing import (
-    TYPE_CHECKING,
-    Any,
-    Callable,
diff -- python/sglang/srt/models/qwen2_audio.py
@@ -23,30 +23,18 @@
-import math
-from functools import lru_cache, partial
-from typing import Any, Iterable, List, Optional, Tuple, Type, TypedDict
+from typing import Any, Iterable, List, Optional, Tuple
-import torch.nn.functional as F
-from einops import rearrange
diff -- python/sglang/srt/models/longcat_flash.py
@@ -44,9 +44,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-18; `python/sglang/srt/models/qwen2_audio.py` modified +2/-15; `python/sglang/srt/models/longcat_flash.py` modified +1/-14; `python/sglang/srt/models/longcat_flash_nextn.py` modified +2/-13; `python/sglang/srt/models/mimo.py` modified +2/-13; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-10
- 验证与风险: diff 自带测试面 `python/sglang/test/attention/test_flashattn_mla_backend.py`, `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/test/few_shot_gsm8k_engine.py`, `python/sglang/test/simple_eval_gpqa.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11847 - [9/N] MoE Refactor: cleanup dispatcher interfaces

- 链接: https://github.com/sgl-project/sglang/pull/11847
- 状态/时间: merged / 2025-10-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+394/-428，可读 patch 1948 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[9/N] MoE Refactor: cleanup dispatcher interfaces」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；技术摘要: 覆盖「[9/N] MoE Refactor: cleanup dispatcher interfaces」；主要实现面是 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunks: -7,6 +7,7; -15,6 +16,7; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, __init__，涉及 `DeepEPNormalOutput, format, DeepEPLLOutput`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunks: -20,18 +20,14; -109,23 +105,6 @@ def __init__(; symbols: __init__, forward, dispatch，涉及 `__init__, forward, dispatch`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunks: -11,14 +11,19; -32,6 +37,7; symbols: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__，涉及 `_get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported`；`python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunks: -5,13 +5,15; -27,16 +29,15; symbols: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_b，涉及 `MooncakeDispatchOutput, __init__, dispatch_a`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunks: -7,6 +7,7; -15,6 +16,7; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, __init__
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunks: -20,18 +20,14; -109,23 +105,6 @@ def __init__(; symbols: __init__, forward, dispatch
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunks: -11,14 +11,19; -32,6 +37,7; symbols: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunks: -5,13 +5,15; -27,16 +29,15; symbols: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_b
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-46 (60 lines); hunks: -74,7 +74,6; -113,10 +112,7; symbols: __init__, forward_deepep, _forward_shared_experts_and_put_results, op_select_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -7,6 +7,7 @@
+from sglang.srt.layers.dp_attention import get_is_extend_in_batch
@@ -15,6 +16,7 @@
+from sglang.srt.layers.moe.topk import TopKOutput
@@ -51,8 +53,6 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
@@ -61,9 +61,9 @@
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -20,18 +20,14 @@
+from sglang.srt.layers.moe.topk import TopKOutput
-from sglang.srt.layers.quantization.modelopt_quant import (
-    CUTEDSL_MOE_NVFP4_DISPATCH,
-    ModelOptNvFp4FusedMoEMethod,
-)
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -11,14 +11,19 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35; `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39; `python/sglang/srt/models/deepseek_v2.py` modified +14/-46; `python/sglang/srt/layers/moe/token_dispatcher/standard.py` modified +46/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12369 - Enable bailing_moe to support TP=16

- 链接: https://github.com/sgl-project/sglang/pull/12369
- 状态/时间: merged / 2025-10-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-2，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable bailing_moe to support TP=16」；模型线: Ling 2.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「Enable bailing_moe to support TP=16」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +9/-2 (11 lines); hunks: -420,14 +420,21 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-2 (11 lines); hunks: -420,14 +420,21 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -420,14 +420,21 @@ def __init__(
-        assert self.total_kv_heads % attn_tp_size == 0
+        if self.total_kv_heads >= attn_tp_size:
+            # Number of KV heads is greater than TP size, so we partition
+            # the KV heads across multiple tensor parallel GPUs.
+            assert self.total_kv_heads % attn_tp_size == 0
+        else:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +9/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14337 - remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)

- 链接: https://github.com/sgl-project/sglang/pull/14337
- 状态/时间: merged / 2025-12-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+0/-8，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`；技术摘要: 覆盖「remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal，涉及 `forward_normal`；`python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward，涉及 `forward`；`python/sglang/srt/models/llada2.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal，涉及 `forward_normal`；`python/sglang/srt/models/qwen2_moe.py` modified +0/-2 (2 lines); hunks: -275,11 +275,9 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
  - `python/sglang/srt/models/llada2.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/qwen2_moe.py` modified +0/-2 (2 lines); hunks: -275,11 +275,9 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -349,11 +349,9 @@ def forward_normal(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/kimi_linear.py
@@ -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/llada2.py
@@ -349,11 +349,9 @@ def forward_normal(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -275,11 +275,9 @@ def forward(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +0/-2; `python/sglang/srt/models/kimi_linear.py` modified +0/-2; `python/sglang/srt/models/llada2.py` modified +0/-2; `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13730 - [bugfix] fix TBO crashes when attn_tp_size > 1

- 链接: https://github.com/sgl-project/sglang/pull/13730
- 状态/时间: merged / 2025-12-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+285/-16，可读 patch 617 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] fix TBO crashes when attn_tp_size > 1」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「[bugfix] fix TBO crashes when attn_tp_size > 1」；主要实现面是 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/communicator.py` modified +14/-1 (15 lines); hunks: -217,14 +217,16 @@ class _LayerModeComputationContext:; -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationCo...; symbols: _LayerModeComputationContext, previous_layer, _compute_mlp_mode, _should_gather_for_tbo，涉及 `_LayerModeComputationContext, previous_layer, _compute_mlp_mode`；`python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0 (9 lines); hunks: -376,6 +376,7 @@ class ForwardBatch:; -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size，涉及 `ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size`；`python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -582,12 +582,16 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/falcon_h1.py` modified +3/-1 (4 lines); hunks: -198,15 +198,17 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/communicator.py` modified +14/-1 (15 lines); hunks: -217,14 +217,16 @@ class _LayerModeComputationContext:; -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationCo...; symbols: _LayerModeComputationContext, previous_layer, _compute_mlp_mode, _should_gather_for_tbo
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0 (9 lines); hunks: -376,6 +376,7 @@ class ForwardBatch:; -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -582,12 +582,16 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +3/-1 (4 lines); hunks: -198,15 +198,17 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/longcat_flash.py` modified +4/-0 (4 lines); hunks: -380,6 +380,8 @@ def __init__(; -398,6 +400,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/communicator.py
@@ -217,14 +217,16 @@ class _LayerModeComputationContext:
+    is_next_layer_sparse: Optional[bool]
+            num_layers=self.num_layers,
-            num_layers=self.num_layers,
+            is_next_layer_sparse=self.is_layer_sparse,
@@ -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
+    @classmethod
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -376,6 +376,7 @@ class ForwardBatch:
+    tbo_padded_len: Optional[int] = None
@@ -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
+        # TODO: The following is added to make sure sub-batch input_ids are padded
+        # to the multiple of attn_tp_size. It can likely be removed after this
+        # function is refactored and merged into the Scheduler.
+        if self.tbo_children:
diff -- python/sglang/srt/models/bailing_moe.py
@@ -582,12 +582,16 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +14/-1; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/falcon_h1.py` modified +3/-1; `python/sglang/srt/models/longcat_flash.py` modified +4/-0; `python/sglang/srt/models/qwen3_next.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/srt/ep/test_deepep_small.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15526 - Optimize Bailing-MoE with FlashInfer Fused All-Reduce

- 链接: https://github.com/sgl-project/sglang/pull/15526
- 状态/时间: merged / 2025-12-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+58/-20，可读 patch 182 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize Bailing-MoE with FlashInfer Fused All-Reduce」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「Optimize Bailing-MoE with FlashInfer Fused All-Reduce」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +58/-20 (78 lines); hunks: -19,7 +19,7; -54,7 +54,11; symbols: forward, forward_normal_dual_stream, forward_normal，涉及 `forward, forward_normal_dual_stream, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +58/-20 (78 lines); hunks: -19,7 +19,7; -54,7 +54,11; symbols: forward, forward_normal_dual_stream, forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -19,7 +19,7 @@
-from typing import Iterable, Optional, Tuple, Union
+from typing import Iterable, List, Optional, Tuple, Union
@@ -54,7 +54,11 @@
-from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
+from sglang.srt.layers.moe import (
+    get_deepep_mode,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +58/-20
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- 链接: https://github.com/sgl-project/sglang/pull/15835
- 状态/时间: merged / 2025-12-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+827/-127，可读 patch 1151 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] JIT Fused QK norm + qk norm clean up」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`；技术摘要: 覆盖「[Feature] JIT Fused QK norm + qk norm clean up」；主要实现面是 `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm，涉及 `create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm`；`python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope，涉及 `__init__, _apply_qk_norm, op_prepare`；`python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native，涉及 `__init__, _apply_qk_norm, forward_prepare_native`；`python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward，涉及 `__init__, _apply_qk_norm, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope
  - `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -250,28 +251,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, forward_prepare
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/utils.py
@@ -11,24 +11,27 @@
+from __future__ import annotations
-from typing import Any, Optional
+from typing import TYPE_CHECKING, Any, Optional, Tuple
+from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
+from sglang.jit_kernel.utils import register_jit_op
+from sglang.srt.environ import envs
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -57,12 +57,12 @@
-from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
+    apply_qk_norm,
@@ -498,31 +498,6 @@ def __init__(
-    def _apply_qk_norm(
-        self, q: torch.Tensor, k: torch.Tensor
-    ) -> Tuple[torch.Tensor, torch.Tensor]:
diff -- python/sglang/srt/models/qwen3.py
@@ -21,14 +21,14 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/utils.py` modified +80/-5; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27; `python/sglang/srt/models/qwen3.py` modified +9/-24; `python/sglang/srt/models/bailing_moe.py` modified +9/-23; `python/sglang/srt/models/glm4_moe.py` modified +9/-23; `python/sglang/srt/models/llada2.py` modified +9/-23
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_qknorm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13715 - Fix EPLB + FP4 Quantization Compatibility Issue

- 链接: https://github.com/sgl-project/sglang/pull/13715
- 状态/时间: merged / 2026-01-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+49/-3，可读 patch 157 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix EPLB + FP4 Quantization Compatibility Issue」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`；技术摘要: 覆盖「Fix EPLB + FP4 Quantization Compatibility Issue」；主要实现面是 `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather，涉及 `get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather`；`python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward，涉及 `get_moe_weights, forward`；`python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts，涉及 `get_moe_weights, _forward_shared_experts`；`python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal，涉及 `get_moe_weights, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -63,6 +63,7; -324,6 +325,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:
+def filter_moe_weight_param_global_expert(name, x, num_local_experts):
+    """
+    Filter out for MoE expert parameters that requires global expert.
+    """
+    return (
+        not getattr(x, "_sglang_require_global_experts", False)
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -103,7 +103,10 @@
-from sglang.srt.layers.moe.utils import RoutingMethodType
+from sglang.srt.layers.moe.utils import (
+    RoutingMethodType,
+    filter_moe_weight_param_global_expert,
+)
@@ -587,6 +590,9 @@ def get_moe_weights(self):
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -58,7 +58,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +12/-0; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1; `python/sglang/srt/models/qwen2_moe.py` modified +7/-1; `python/sglang/srt/models/qwen3_moe.py` modified +7/-1; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/glm4_moe.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17570 - Use attn tp group in embedding for more models

- 链接: https://github.com/sgl-project/sglang/pull/17570
- 状态/时间: merged / 2026-01-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+19/-19，可读 patch 171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use attn tp group in embedding for more models」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`；技术摘要: 覆盖「Use attn tp group in embedding for more models」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer，涉及 `__init__, get_layer`；`python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer
  - `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -895,7 +895,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -717,7 +717,7 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -62,7 +62,7 @@ def __init__(
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/falcon_h1.py
@@ -394,7 +394,7 @@ def __init__(
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/glm4.py
@@ -307,7 +307,7 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/glm4_moe.py
@@ -895,7 +895,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +1/-1; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/falcon_h1.py` modified +1/-1; `python/sglang/srt/models/glm4.py` modified +1/-1; `python/sglang/srt/models/glm4_moe.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17808 - Fix the scenario where eh_proj is quantized in the bailing moe nextn weights

- 链接: https://github.com/sgl-project/sglang/pull/17808
- 状态/时间: merged / 2026-01-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-2，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix the scenario where eh_proj is quantized in the bailing moe nextn weights」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/bailing_moe_nextn.py`；技术摘要: 覆盖「Fix the scenario where eh_proj is quantized in the bailing moe nextn weights」；主要实现面是 `python/sglang/srt/models/bailing_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2 (11 lines); hunks: -28,6 +28,7; -69,7 +70,13 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2 (11 lines); hunks: -28,6 +28,7; -69,7 +70,13 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -28,6 +28,7 @@
+from sglang.srt.layers.linear import ReplicatedLinear
@@ -69,7 +70,13 @@ def __init__(
-        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
+        self.eh_proj = ReplicatedLinear(
+            2 * config.hidden_size,
+            config.hidden_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15119 - feat: Add Ling Flash v2.0 support for Eagle3

- 链接: https://github.com/sgl-project/sglang/pull/15119
- 状态/时间: merged / 2026-02-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+30/-1，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: Add Ling Flash v2.0 support for Eagle3」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「feat: Add Ling Flash v2.0 support for Eagle3」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +30/-1 (31 lines); hunks: -738,6 +738,8 @@ def __init__(; -760,6 +762,10 @@ def forward(; symbols: __init__, forward, BailingMoEForCausalLM，涉及 `__init__, forward, BailingMoEForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +30/-1 (31 lines); hunks: -738,6 +738,8 @@ def __init__(; -760,6 +762,10 @@ def forward(; symbols: __init__, forward, BailingMoEForCausalLM
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -738,6 +738,8 @@ def __init__(
+        self.layers_to_capture = []
@@ -760,6 +762,10 @@ def forward(
+                if i in self.layers_to_capture:
+                    aux_hidden_states.append(
+                        hidden_states if residual is None else hidden_states + residual
+                    )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +30/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18598 - Support LingV2_5 model

- 链接: https://github.com/sgl-project/sglang/pull/18598
- 状态/时间: merged / 2026-02-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+4042/-23，可读 patch 4377 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support LingV2_5 model」；模型线: Ling 2.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/layers/attention/linear/seg_la.py`, `python/sglang/srt/layers/attention/linear/lightning_attn.py`；技术摘要: 覆盖「Support LingV2_5 model」；主要实现面是 `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/layers/attention/linear/seg_la.py`, `python/sglang/srt/layers/attention/linear/lightning_attn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe_linear.py` added +1571/-0 (1571 lines); hunks: -0,0 +1,1571; symbols: DsV3MLA, __init__, is_linear_layer, is_pp_missing_parameter，涉及 `DsV3MLA, __init__, is_linear_layer`；`python/sglang/srt/layers/attention/linear/seg_la.py` added +909/-0 (909 lines); hunks: -0,0 +1,909; symbols: SegLaMeta, seg_la_kernel, seg_la_p_kernel, seg_la_s_kernel，涉及 `SegLaMeta, seg_la_kernel, seg_la_p_kernel`；`python/sglang/srt/layers/attention/linear/lightning_attn.py` added +767/-0 (767 lines); hunks: -0,0 +1,767; symbols: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel，涉及 `_fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce`；`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +369/-0 (369 lines); hunks: -1,3 +1,5; -14,6 +16,12; symbols: forward_extend, LightningAttentionBackend, __init__, init_forward_metadata，涉及 `forward_extend, LightningAttentionBackend, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe_linear.py` added +1571/-0 (1571 lines); hunks: -0,0 +1,1571; symbols: DsV3MLA, __init__, is_linear_layer, is_pp_missing_parameter
  - `python/sglang/srt/layers/attention/linear/seg_la.py` added +909/-0 (909 lines); hunks: -0,0 +1,909; symbols: SegLaMeta, seg_la_kernel, seg_la_p_kernel, seg_la_s_kernel
  - `python/sglang/srt/layers/attention/linear/lightning_attn.py` added +767/-0 (767 lines); hunks: -0,0 +1,767; symbols: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +369/-0 (369 lines); hunks: -1,3 +1,5; -14,6 +16,12; symbols: forward_extend, LightningAttentionBackend, __init__, init_forward_metadata
  - `python/sglang/srt/configs/bailing_hybrid.py` added +188/-0 (188 lines); hunks: -0,0 +1,188; symbols: HybridLayerType, BailingHybridConfig, __init__, layers_block_type
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -0,0 +1,1571 @@
+# coding=utf-8
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
+import copy
+import logging
+from typing import Callable, Iterable, Optional, Set, Tuple, Union
+import torch
diff -- python/sglang/srt/layers/attention/linear/seg_la.py
@@ -0,0 +1,909 @@
+# -*- coding: utf-8 -*-
+"""
+Copyright (c) Ant Financial Service Group and its affiliates.
+"""
+# Copied from https://code.alipay.com/pia/PainlessInferenceAcceleration/blob/v0.0.6/flood/flood/ops/seg_la.py
+from dataclasses import dataclass
diff -- python/sglang/srt/layers/attention/linear/lightning_attn.py
@@ -0,0 +1,767 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe_linear.py` added +1571/-0; `python/sglang/srt/layers/attention/linear/seg_la.py` added +909/-0; `python/sglang/srt/layers/attention/linear/lightning_attn.py` added +767/-0; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +369/-0; `python/sglang/srt/configs/bailing_hybrid.py` added +188/-0; `python/sglang/srt/models/bailing_moe_nextn.py` modified +90/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/bailing_hybrid.py`, `python/sglang/srt/configs/model_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18793 - Cleanup debug log for Ring model

- 链接: https://github.com/sgl-project/sglang/pull/18793
- 状态/时间: merged / 2026-02-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-11，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Cleanup debug log for Ring model」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/configs/mamba_utils.py`；技术摘要: 覆盖「Cleanup debug log for Ring model」；主要实现面是 `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/configs/mamba_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10 (18 lines); hunks: -77,6 +77,7; -423,7 +424,7 @@ def __init__(; symbols: __init__, layer_fn, post_load_weights，涉及 `__init__, layer_fn, post_load_weights`；`python/sglang/srt/configs/mamba_utils.py` modified +1/-1 (2 lines); hunks: -102,7 +102,7 @@ def mamba2_state_dtype(config=None) -> Mamba2StateDType:; symbols: mamba2_state_dtype，涉及 `mamba2_state_dtype`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10 (18 lines); hunks: -77,6 +77,7; -423,7 +424,7 @@ def __init__(; symbols: __init__, layer_fn, post_load_weights
  - `python/sglang/srt/configs/mamba_utils.py` modified +1/-1 (2 lines); hunks: -102,7 +102,7 @@ def mamba2_state_dtype(config=None) -> Mamba2StateDType:; symbols: mamba2_state_dtype
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -77,6 +77,7 @@
+from sglang.srt.utils.common import rank0_log
@@ -423,7 +424,7 @@ def __init__(
-        logger.info(f"linear_backend in bailing_moe_linear: {self.linear_backend}")
+        logger.debug(f"linear_backend in bailing_moe_linear: {self.linear_backend}")
@@ -740,7 +741,7 @@ def __init__(
-                logger.info(f"==={layer_id=} use gqa")
diff -- python/sglang/srt/configs/mamba_utils.py
@@ -102,7 +102,7 @@ def mamba2_state_dtype(config=None) -> Mamba2StateDType:
-    logger.info(f"Mamba2 state dtype: conv_dtype={conv_dtype}, ssm_dtype={ssm_dtype}")
+    logger.debug(f"Mamba2 state dtype: conv_dtype={conv_dtype}, ssm_dtype={ssm_dtype}")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10; `python/sglang/srt/configs/mamba_utils.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/mamba_utils.py`, `python/sglang/srt/models/bailing_moe_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18860 - update pre-commit config

- 链接: https://github.com/sgl-project/sglang/pull/18860
- 状态/时间: merged / 2026-02-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 135 个文件，+239/-198，可读 patch 1632 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update pre-commit config」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`；技术摘要: 覆盖「update pre-commit config」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend，涉及 `forward_decode, forward_extend`；`python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15；`python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method，涉及 `get_moe_method`；`test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend
  - `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15
  - `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2 (4 lines); hunks: -1,6 +1,6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -670,9 +670,9 @@ def forward_decode(
-        (q_proj_states, k_proj_states, v_proj_states) = mixed_qkv
-        (q_conv_weights, k_conv_weights, v_conv_weights) = layer.conv_weights
-        (q_conv_bias, k_conv_bias, v_conv_bias) = layer.bias
+        q_proj_states, k_proj_states, v_proj_states = mixed_qkv
+        q_conv_weights, k_conv_weights, v_conv_weights = layer.conv_weights
+        q_conv_bias, k_conv_bias, v_conv_bias = layer.bias
diff -- python/sglang/srt/models/pixtral.py
@@ -23,11 +23,15 @@
-from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding
+from transformers.models.pixtral.modeling_pixtral import (
+    PixtralRotaryEmbedding,
+)
-from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid
+from transformers.models.pixtral.modeling_pixtral import (
diff -- python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py
@@ -63,11 +63,9 @@ def get_moe_method(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6; `python/sglang/srt/models/pixtral.py` modified +6/-2; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2; `python/sglang/srt/multimodal/processors/ernie45_vl.py` modified +3/-1
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1
- 验证与风险: diff 自带测试面 `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `test/manual/test_vlm_accuracy.py`, `test/registered/attention/test_triton_sliding_window.py`, `test/registered/layers/test_fla_layernorm_guard.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19382 - Add NPU basic function testcases

- 链接: https://github.com/sgl-project/sglang/pull/19382
- 状态/时间: merged / 2026-03-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 87 个文件，+4587/-333，可读 patch 5347 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add NPU basic function testcases」；模型线: Ling 2.5；类别: 文档/测试/CI；主要 diff: `test/registered/ascend/interface/test_npu_openai_function_calling.py`, `test/registered/ascend/llm_models/test_npu_phi_4_multimodal_llm.py`, `test/registered/ascend/vlm_models/test_npu_phi4_multimodal_instruct.py`；技术摘要: 覆盖「Add NPU basic function testcases」；主要实现面是 `test/registered/ascend/interface/test_npu_openai_function_calling.py`, `test/registered/ascend/llm_models/test_npu_phi_4_multimodal_llm.py`, `test/registered/ascend/vlm_models/test_npu_phi4_multimodal_instruct.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ascend/interface/test_npu_openai_function_calling.py` added +943/-0 (943 lines); hunks: -0,0 +1,943; symbols: TestOpenAIServerFunctionCalling, setUpClass, tearDownClass, test_function_calling_format，涉及 `TestOpenAIServerFunctionCalling, setUpClass, tearDownClass`；`test/registered/ascend/llm_models/test_npu_phi_4_multimodal_llm.py` renamed +0/-0 (0 lines)；`test/registered/ascend/vlm_models/test_npu_phi4_multimodal_instruct.py` renamed +0/-0 (0 lines)；`test/registered/ascend/interface/test_npu_api.py` added +732/-0 (732 lines); hunks: -0,0 +1,732; symbols: TestNpuApi, setUpClass, tearDownClass, does，涉及 `TestNpuApi, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/ascend/interface/test_npu_openai_function_calling.py` added +943/-0 (943 lines); hunks: -0,0 +1,943; symbols: TestOpenAIServerFunctionCalling, setUpClass, tearDownClass, test_function_calling_format
  - `test/registered/ascend/llm_models/test_npu_phi_4_multimodal_llm.py` renamed +0/-0 (0 lines)
  - `test/registered/ascend/vlm_models/test_npu_phi4_multimodal_instruct.py` renamed +0/-0 (0 lines)
  - `test/registered/ascend/interface/test_npu_api.py` added +732/-0 (732 lines); hunks: -0,0 +1,732; symbols: TestNpuApi, setUpClass, tearDownClass, does
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +495/-54 (549 lines); hunks: -11,99 +11,540; symbols: ModelTestConfig, run_command, get_benchmark_args, run_bench_serving
- 关键代码摘录:

```diff
diff -- test/registered/ascend/interface/test_npu_openai_function_calling.py
@@ -0,0 +1,943 @@
+import json
+import unittest
+import openai
+from sglang.srt.utils import kill_process_tree
+from sglang.srt.utils.hf_transformers_utils import get_tokenizer
+from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
diff -- test/registered/ascend/interface/test_npu_api.py
@@ -0,0 +1,732 @@
+import json
+import logging
+import os
+import shutil
+import unittest
+import requests
diff -- python/sglang/test/ascend/test_ascend_utils.py
@@ -11,99 +11,540 @@
```

- 已读文件:
  - tests: `test/registered/ascend/interface/test_npu_openai_function_calling.py` added +943/-0; `test/registered/ascend/llm_models/test_npu_phi_4_multimodal_llm.py` renamed +0/-0; `test/registered/ascend/vlm_models/test_npu_phi4_multimodal_instruct.py` renamed +0/-0; `test/registered/ascend/interface/test_npu_api.py` added +732/-0; `python/sglang/test/ascend/test_ascend_utils.py` modified +495/-54; `test/registered/ascend/interface/test_npu_matched_stop.py` added +163/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/ascend/disaggregation_utils.py`, `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache.py`, `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_mla.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17784 - Upgrade transformers==5.3.0

- 链接: https://github.com/sgl-project/sglang/pull/17784
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 95 个文件，+1136/-343，可读 patch 2752 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade transformers==5.3.0」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`；技术摘要: 覆盖「Upgrade transformers==5.3.0」；主要实现面是 `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update，涉及 `__init__, Gemma3RotaryEmbedding, _dynamic_frequency_update`；`python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope，涉及 `_get_rope_param, get_rope`；`python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes，涉及 `ModelImpl, is_deepseek_nsa, _derive_model_shapes`；`python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__，涉及 `compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__
  - `python/sglang/srt/models/midashenglm.py` modified +6/-14 (20 lines); hunks: -476,20 +476,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma3_causal.py
@@ -166,18 +166,36 @@ def __init__(
+        # In transformers v5, rope_parameters is nested per layer type:
+        #   {"sliding_attention": {"rope_theta": 10000}, "full_attention": {"rope_theta": 1000000}}
+        # In v4 it was flat: {"rope_type": "default", "rope_theta": ...}
+        rope_params = config.rope_parameters
+        is_nested = isinstance(rope_params, dict) and "full_attention" in rope_params
-            self.rope_theta = config.rope_local_base_freq
diff -- python/sglang/srt/layers/rotary_embedding/factory.py
@@ -2,6 +2,7 @@
+import logging
@@ -26,6 +27,29 @@
+logger = logging.getLogger(__name__)
+def _get_rope_param(rope_scaling, key, default, scaling_type):
+    """Get a parameter from rope_scaling dict, warn if missing.
+    In transformers v5, config.rope_scaling is an alias for rope_parameters
diff -- python/sglang/srt/configs/model_config.py
@@ -51,10 +51,20 @@ class ModelImpl(str, Enum):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13; `python/sglang/srt/configs/model_config.py` modified +38/-18; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7; `python/sglang/srt/models/midashenglm.py` modified +6/-14; `python/sglang/srt/models/glm4.py` modified +3/-14
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/registered/core/test_score_api.py`, `test/registered/quant/test_awq.py`, `test/registered/rl/test_multi_instance_release_memory_occupation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9744 - [CPU] Add FP8 Bmm support

- 链接: https://github.com/sgl-project/sglang/pull/9744
- 状态/时间: merged / 2026-03-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+585/-84，可读 patch 1014 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Add FP8 Bmm support」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`；技术摘要: 覆盖「[CPU] Add FP8 Bmm support」；主要实现面是 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28 (72 lines); hunks: -16,6 +16,7; -268,18 +269,24 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core，涉及 `forward_absorb_prepare, forward_absorb_core`；`python/sglang/srt/models/longcat_flash.py` modified +0/-12 (12 lines); hunks: -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights，涉及 `post_load_weights`；`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10 (10 lines); hunks: -46,8 +46,6; -583,14 +581,6 @@ def post_load_weights(; symbols: post_load_weights，涉及 `post_load_weights`；`python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8 (8 lines); hunks: -1208,14 +1208,6 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights，涉及 `post_load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28 (72 lines); hunks: -16,6 +16,7; -268,18 +269,24 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/longcat_flash.py` modified +0/-12 (12 lines); hunks: -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10 (10 lines); hunks: -46,8 +46,6; -583,14 +581,6 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8 (8 lines); hunks: -1208,14 +1208,6 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +0/-4 (4 lines); hunks: -426,10 +426,6 @@ def post_load_weights(self):; symbols: post_load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -16,6 +16,7 @@
+    _is_cpu,
@@ -268,18 +269,24 @@ def forward_absorb_prepare(
-            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
-            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
-                q_nope.transpose(0, 1),
-                (
diff -- python/sglang/srt/models/longcat_flash.py
@@ -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):
-                    # TODO: remove this after adding FP8 support in bmm cpu kernel
-                    if (
-                        _is_cpu
-                        and _is_cpu_amx_available
-                        and w.dtype == torch.float8_e4m3fn
-                    ):
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -46,8 +46,6 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28; `python/sglang/srt/models/longcat_flash.py` modified +0/-12; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8; `python/sglang/srt/models/longcat_flash_nextn.py` modified +0/-4; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +2/-1
  - other: `sgl-kernel/csrc/cpu/gemm_fp8.cpp` modified +310/-1; `sgl-kernel/csrc/cpu/bmm.cpp` modified +93/-11
- 验证与风险: diff 自带测试面 `test/srt/cpu/test_bmm.py`, `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20316 - fix fused_set_kv_buffer for rope with Ling-v2

- 链接: https://github.com/sgl-project/sglang/pull/20316
- 状态/时间: merged / 2026-03-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-2，可读 patch 29 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix fused_set_kv_buffer for rope with Ling-v2」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/bailing_moe.py`；技术摘要: 覆盖「fix fused_set_kv_buffer for rope with Ling-v2」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +6/-2 (8 lines); hunks: -532,6 +532,10 @@ def forward(; -542,7 +546,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +6/-2 (8 lines); hunks: -532,6 +532,10 @@ def forward(; -542,7 +546,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -532,6 +532,10 @@ def forward(
+        can_fuse_set_kv = (
+            self.head_dim == self.rotary_emb.rotary_dim
+            and enable_fused_set_kv_buffer(forward_batch)
+        )
@@ -542,7 +546,7 @@ def forward(
-                if enable_fused_set_kv_buffer(forward_batch)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +6/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20751 - [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture

- 链接: https://github.com/sgl-project/sglang/pull/20751
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 43 个文件，+673/-106，可读 patch 1465 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture」；模型线: Ling 2.5；类别: 文档/测试/CI；主要 diff: `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml`；技术摘要: 覆盖「[NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture」；主要实现面是 `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/full-test-npu.yml` added +355/-0 (355 lines); hunks: -0,0 +1,355；`.github/workflows/nightly-test-npu.yml` modified +124/-36 (160 lines); hunks: -2,7 +2,7 @@ name: Nightly Test (NPU); -21,40 +21,95 @@ on:；`.github/workflows/pr-test-npu.yml` modified +70/-40 (110 lines); hunks: -76,7 +76,7 @@ jobs:; -111,21 +111,8 @@ jobs:；`python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9 (18 lines); hunks: -117,9 +117,18; -133,15 +142,6。
- 代码 diff 细节:
  - `.github/workflows/full-test-npu.yml` added +355/-0 (355 lines); hunks: -0,0 +1,355
  - `.github/workflows/nightly-test-npu.yml` modified +124/-36 (160 lines); hunks: -2,7 +2,7 @@ name: Nightly Test (NPU); -21,40 +21,95 @@ on:
  - `.github/workflows/pr-test-npu.yml` modified +70/-40 (110 lines); hunks: -76,7 +76,7 @@ jobs:; -111,21 +111,8 @@ jobs:
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9 (18 lines); hunks: -117,9 +117,18; -133,15 +142,6
  - `test/registered/ascend/basic_function/quant/test_npu_autoround_moe.py` renamed +8/-1 (9 lines); hunks: -4,6 +4,10; -12,10 +16,13
- 关键代码摘录:

```diff
diff -- .github/workflows/full-test-npu.yml
@@ -0,0 +1,355 @@
+name: Full Test (NPU)
+on:
+#  pull_request:
+#    branches:
+#      - main
+#    paths:
diff -- .github/workflows/nightly-test-npu.yml
@@ -2,7 +2,7 @@ name: Nightly Test (NPU)
-    - cron: '0 17 * * *'  # Execute at 1:00 a.m. Beijing Time every day
+    - cron: '0 18 * * *'  # Execute at 2:00 a.m. Beijing Time every day
@@ -21,40 +21,95 @@ on:
+      image_a3:
+        description: 'The a3 running docker image of the test task.'
+        required: false
diff -- .github/workflows/pr-test-npu.yml
@@ -76,7 +76,7 @@ jobs:
```

- 已读文件:
  - ci: `.github/workflows/full-test-npu.yml` added +355/-0; `.github/workflows/nightly-test-npu.yml` modified +124/-36; `.github/workflows/pr-test-npu.yml` modified +70/-40
  - tests: `python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9; `test/registered/ascend/basic_function/quant/test_npu_autoround_moe.py` renamed +8/-1; `test/registered/ascend/basic_function/quant/test_npu_gptq_moe.py` renamed +8/-1; `test/registered/ascend/basic_function/quant/test_npu_autoround_dense.py` renamed +6/-1; `test/run_suite.py` modified +5/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mha.py`, `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mla.py`, `test/registered/ascend/basic_function/backends/test_npu_sampling_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- 链接: https://github.com/sgl-project/sglang/pull/23001
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 330 个文件，+80364/-0，可读 patch 68714 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add new Mintlify documentation site (docs_new/)」；模型线: Ling 2.5；类别: 文档/测试/CI；主要 diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`；技术摘要: 覆盖「Add new Mintlify documentation site (docs_new/)」；主要实现面是 `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in，涉及 `get_messages, get_current_weather, convert_dict_to_tool`；`docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages，涉及 `CapitalInfo, get_messages`；`docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317；`docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)。
- 代码 diff 细节:
  - `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0 (2911 lines)
- 关键代码摘录:

```diff
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -0,0 +1,740 @@
+---
+title: "Tool Parser"
+metatags:
+    description: "SGLang function calling: tool parsers for DeepSeek, Llama, Qwen, Mistral, GLM, Kimi K2. OpenAI-compatible tool use API."
+---
+This guide demonstrates how to use SGLang’s [Function calling](https://platform.openai.com/docs/guides/function-calling) functionality.
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -0,0 +1,663 @@
+---
+title: "Structured Outputs For Reasoning Models"
+metatags:
+    description: "SGLang structured outputs for reasoning models: free-form thinking with constrained final output for DeepSeek R1, QwQ models."
+---
+When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -0,0 +1,317 @@
```

- 已读文件:
  - docs: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0; `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0; `docs_new/docs/advanced_features/server_arguments.mdx` added +2871/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`, `docs_new/.mintignore`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- 链接: https://github.com/sgl-project/sglang/pull/23337
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 179 个文件，+16004/-8152，可读 patch 23604 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Sync docs_new with legacy docs and update migration redirects」；模型线: Ling 2.5；类别: 文档/测试/CI；主要 diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`；技术摘要: 覆盖「[Docs] Sync docs_new with legacy docs and update migration redirects」；主要实现面是 `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.；`docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines)；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines)。
- 代码 diff 细节:
  - `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines)
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486 (932 lines)
- 关键代码摘录:

```diff
diff -- docs_new/docs/supported-models/multimodal_language_models.mdx
@@ -1,15 +1,18 @@
+---
+title: "Multimodal Language Models"
+metatags:
+  description: "Documentation for Multimodal Language Models"
+---
-<CodeGroup>
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"
-When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
+When working with reasoning models that use special tokens like `<think>...</think>` to denote reasoning sections, you might want to allow free-form text within these sections whi
-To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `&lt;/think&gt;`, when launching the server. You can also specify the reasoning
+To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `</think>`, when launching the server. You can also specify the reasoning parse
-- [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d): The reasoning content is wrapped with `&lt;think&gt;` and `&lt;/think&
-- [QwQ](https://huggingface.co/Qwen/QwQ-32B): The reasoning content is wrapped with `&lt;think&gt;` and `&lt;/think&gt;` tags.
diff -- docs_new/docs/hardware-platforms/tpu.mdx
@@ -2,65 +2,67 @@
```

- 已读文件:
  - docs: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418; `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486; `docs_new/docs/hardware-platforms/tpu.mdx` modified +425/-468
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23732 - Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)

- 链接: https://github.com/sgl-project/sglang/pull/23732
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+59/-12，可读 patch 290 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`；技术摘要: 覆盖「Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)」；主要实现面是 `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal，涉及 `forward_normal`；`python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream，涉及 `_forward_single_stream, _forward_dual_stream`；`python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/llama4.py` modified +6/-1 (7 lines); hunks: -39,6 +39,7; -145,7 +146,11 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llada2.py
@@ -55,7 +55,11 @@
-from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
+from sglang.srt.layers.moe import (
+    get_deepep_mode,
+    get_moe_a2a_backend,
+    should_use_dp_reduce_scatterv,
+)
diff -- python/sglang/srt/models/hunyuan_v3.py
@@ -34,6 +34,7 @@
+from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
@@ -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        if self.ep_size > 1:
+        skip_post_reduce = should_use_dp_reduce_scatterv()
+        if self.ep_size > 1 and not skip_post_reduce:
-        if self.tp_size > 1:
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -34,6 +34,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llada2.py` modified +10/-2; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1; `python/sglang/srt/models/exaone_moe.py` modified +6/-2; `python/sglang/srt/models/llama4.py` modified +6/-1; `python/sglang/srt/models/sarvam_moe.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23748 - refactor(moe): centralize post-experts all-reduce skip predicate

- 链接: https://github.com/sgl-project/sglang/pull/23748
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+134/-132，可读 patch 532 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor(moe): centralize post-experts all-reduce skip predicate」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「refactor(moe): centralize post-experts all-reduce skip predicate」；主要实现面是 `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context，涉及 `should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context`；`python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal，涉及 `forward_normal_dual_stream, forward_normal`；`python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook，涉及 `forward_normal_dual_stream, _post_combine_hook`；`python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal，涉及 `forward_normal_dual_stream, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context
  - `python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-13 (22 lines); hunks: -50,8 +50,7; -332,20 +331,17 @@ def forward_normal(; symbols: forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():
+def should_skip_post_experts_all_reduce(
+    *,
+    is_tp_path: bool,
+    use_reduce_scatter: bool = False,
+    should_allreduce_fusion: bool = False,
+) -> bool:
diff -- python/sglang/srt/models/sarvam_moe.py
@@ -39,10 +39,7 @@
-from sglang.srt.layers.moe import (
-    should_use_dp_reduce_scatterv,
-    should_use_flashinfer_cutlass_moe_fp4_allgather,
-)
+from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
@@ -373,12 +370,10 @@ def forward_normal_dual_stream(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -85,7 +85,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +33/-0; `python/sglang/srt/models/sarvam_moe.py` modified +9/-16; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13; `python/sglang/srt/models/glm4_moe.py` modified +9/-13; `python/sglang/srt/models/qwen3_moe.py` modified +9/-13; `python/sglang/srt/models/hunyuan_v3.py` modified +13/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21126 - [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split

- 链接: https://github.com/sgl-project/sglang/pull/21126
- 状态/时间: merged / 2026-04-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+1419/-1031，可读 patch 2590 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`；技术摘要: 覆盖「[4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split」；主要实现面是 `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/awq.py` removed +0/-966 (966 lines); hunks: -1,966 +0,0; symbols: is_layer_skipped_awq, AWQConfig, for, __init__，涉及 `is_layer_skipped_awq, AWQConfig, for`；`python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: is_layer_skipped_awq, AWQConfig, for, __init__，涉及 `is_layer_skipped_awq, AWQConfig, for`；`python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: AWQMoEScheme, __init__, _init_kernel, create_weights，涉及 `AWQMoEScheme, __init__, _init_kernel`；`python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: AWQLinearScheme, __init__, _init_kernel, create_weights，涉及 `AWQLinearScheme, __init__, _init_kernel`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/awq.py` removed +0/-966 (966 lines); hunks: -1,966 +0,0; symbols: is_layer_skipped_awq, AWQConfig, for, __init__
  - `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: is_layer_skipped_awq, AWQConfig, for, __init__
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: AWQMoEScheme, __init__, _init_kernel, create_weights
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: AWQLinearScheme, __init__, _init_kernel, create_weights
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: AWQMarlinLinearScheme, __init__, create_weights, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/awq.py
@@ -1,966 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-from __future__ import annotations
-import logging
-import warnings
-from typing import TYPE_CHECKING, Any, Dict, List, Optional
-import torch
diff -- python/sglang/srt/layers/quantization/awq/awq.py
@@ -0,0 +1,484 @@
+# SPDX-License-Identifier: Apache-2.0
+from __future__ import annotations
+import logging
+import warnings
+from typing import TYPE_CHECKING, Any, Dict, List, Optional
+import torch
diff -- python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py
@@ -0,0 +1,156 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/awq.py` removed +0/-966; `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py` added +105/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py` renamed +35/-51
- 验证与风险: diff 自带测试面 `test/registered/quant/test_awq_dequant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24333 - nextn subclass owns post_load_weights is_nextn

- 链接: https://github.com/sgl-project/sglang/pull/24333
- 状态/时间: merged / 2026-05-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+30/-25，可读 patch 138 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「nextn subclass owns post_load_weights is_nextn」；模型线: Ling 2.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_loader/loader.py`；技术摘要: 覆盖「nextn subclass owns post_load_weights is_nextn」；主要实现面是 `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_loader/loader.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3 (12 lines); hunks: -217,7 +217,8 @@ def __init__(; -243,8 +244,13 @@ def set_embed_and_head(self, embed, head):; symbols: __init__, forward, set_embed_and_head, load_weights，涉及 `__init__, forward, set_embed_and_head`；`python/sglang/srt/models/deepseek_nextn.py` modified +6/-0 (6 lines); hunks: -313,5 +313,11 @@ def forward(; symbols: forward, load_weights, post_load_weights，涉及 `forward, load_weights, post_load_weights`；`python/sglang/srt/model_loader/loader.py` modified +15/-10 (25 lines); hunks: -78,7 +78,6; -286,6 +285,15 @@ def _initialize_model(; symbols: _initialize_model, _post_load_weights, BaseModelLoader, for，涉及 `_initialize_model, _post_load_weights, BaseModelLoader`；`python/sglang/srt/model_loader/utils.py` modified +0/-12 (12 lines); hunks: -247,18 +247,6 @@ def get_architecture_class_name(model_config: ModelConfig)...; symbols: get_architecture_class_name, post_load_weights, should_deepgemm_weight_requant_ue8m0，涉及 `get_architecture_class_name, post_load_weights, should_deepgemm_weight_requant_ue8m0`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3 (12 lines); hunks: -217,7 +217,8 @@ def __init__(; -243,8 +244,13 @@ def set_embed_and_head(self, embed, head):; symbols: __init__, forward, set_embed_and_head, load_weights
  - `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0 (6 lines); hunks: -313,5 +313,11 @@ def forward(; symbols: forward, load_weights, post_load_weights
  - `python/sglang/srt/model_loader/loader.py` modified +15/-10 (25 lines); hunks: -78,7 +78,6; -286,6 +285,15 @@ def _initialize_model(; symbols: _initialize_model, _post_load_weights, BaseModelLoader, for
  - `python/sglang/srt/model_loader/utils.py` modified +0/-12 (12 lines); hunks: -247,18 +247,6 @@ def get_architecture_class_name(model_config: ModelConfig)...; symbols: get_architecture_class_name, post_load_weights, should_deepgemm_weight_requant_ue8m0
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -217,7 +217,8 @@ def __init__(
-            self.post_load_weights_func = BailingMoEForCausalLM.post_load_weights
+            # V1 BailingMoeAttention is standard QKV (no kv_b_proj), no fixup needed.
+            self.post_load_weights_func = None
@@ -243,8 +244,13 @@ def set_embed_and_head(self, embed, head):
-    def post_load_weights(self, is_nextn=False, weight_names=None):
-        self.post_load_weights_func(self, is_nextn=is_nextn, weight_names=weight_names)
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -313,5 +313,11 @@ def forward(
+    def post_load_weights(self, is_nextn=True, weight_names=None):
+        # `is_nextn` is pinned to True for the NextN subclass; the parameter is kept
+        # only because the mixin's `do_load_weights` calls `self.post_load_weights`
+        # with `is_nextn=...` as a kwarg.
+        super().post_load_weights(is_nextn=True, weight_names=weight_names)
diff -- python/sglang/srt/model_loader/loader.py
@@ -78,7 +78,6 @@
-    post_load_weights,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3; `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0; `python/sglang/srt/model_loader/loader.py` modified +15/-10; `python/sglang/srt/model_loader/utils.py` modified +0/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/models/bailing_moe_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24977 - fix gb envs in deployment guide

- 链接: https://github.com/sgl-project/sglang/pull/24977
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix gb envs in deployment guide」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx`；技术摘要: 覆盖「fix gb envs in deployment guide」；主要实现面是 `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` modified +1/-1 (2 lines); hunks: -81,7 +81,7 @@ export const Ling251TDeployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` modified +1/-1 (2 lines); hunks: -81,7 +81,7 @@ export const Ling251TDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx
@@ -81,7 +81,7 @@ export const Ling251TDeployment = () => {
-    const envPrefix = isGB ? 'NCCL_IB_DISABLE=1 ' : '';
+    const envPrefix = isGB ? 'NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 ' : '';
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23837 - Add Ling_2_6

- 链接: https://github.com/sgl-project/sglang/pull/23837
- 状态/时间: merged / 2026-05-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+813/-68，可读 patch 1107 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Ling_2_6」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json`；技术摘要: 覆盖「Add Ling_2_6」；主要实现面是 `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164；`python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164；`python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` added +146/-0 (146 lines); hunks: -0,0 +1,146；`python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +78/-34 (112 lines); hunks: -57,6 +57,7; -243,9 +244,11 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json
@@ -0,0 +1,146 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json` added +164/-0; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json` added +164/-0; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` added +146/-0; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0; `python/sglang/srt/models/bailing_moe_linear.py` modified +78/-34; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +80/-24
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/linear/lightning_backend.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26474 - [HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6

- 链接: https://github.com/sgl-project/sglang/pull/26474
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+70/-1，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`；技术摘要: 覆盖「[HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +10/-1 (11 lines); hunks: -811,9 +811,18 @@ def __init__(; symbols: __init__, _is_full_attn，涉及 `__init__, _is_full_attn`；`python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/registered/8-gpu-models/test_ling_2_6_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestLing26Flash，涉及 `TestLing26Flash`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +10/-1 (11 lines); hunks: -811,9 +811,18 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__
  - `test/registered/8-gpu-models/test_ling_2_6_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestLing26Flash
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -811,9 +811,18 @@ def __init__(
-        # Dispatch by the layer's runtime type
+        # Explicit linear-attention subclass → strong linear signal (KDA, GDN,
+        # Qwen3-Next, Qwen3.5 main linear layers).
+        # Some hybrid models (Ling-2.5/2.6) wrap their linear layers in plain
+        # `RadixAttention` rather than `RadixLinearAttention`. Those wrappers
+        # set `_is_linear_attention=True` on the attn module so we can
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,6 +508,12 @@ def __init__(
+        # Marker for HybridLinearAttnBackend._is_full_attn: Bailing wraps
+        # linear-attention layers in a plain RadixAttention, so the
+        # dispatcher can't tell from the type alone that this is a linear
+        # layer (would otherwise default to the full-attn backend, e.g. the
+        # same way MTP/NEXTN draft layers are routed).
+        self.attn._is_linear_attention = True
diff -- test/registered/8-gpu-models/test_ling_2_6_flash.py
@@ -0,0 +1,54 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +10/-1; `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0
  - tests: `test/registered/8-gpu-models/test_ling_2_6_flash.py` added +54/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_ling_2_6_flash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26623 - Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)

- 链接: https://github.com/sgl-project/sglang/pull/26623
- 状态/时间: merged / 2026-06-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+10/-21，可读 patch 62 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`；技术摘要: 覆盖「Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +8/-13 (21 lines); hunks: -825,22 +825,17 @@ def __init__(; symbols: __init__, _is_full_attn，涉及 `__init__, _is_full_attn`；`python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/registered/8-gpu-models/test_ling_2_6_flash.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -12,7 +12,7; symbols: TestLing26Flash，涉及 `TestLing26Flash`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +8/-13 (21 lines); hunks: -825,22 +825,17 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__
  - `test/registered/8-gpu-models/test_ling_2_6_flash.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -12,7 +12,7; symbols: TestLing26Flash
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -825,22 +825,17 @@ def __init__(
-        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
+        self,
+        layer: Optional[Union[RadixAttention, RadixLinearAttention]],
+        layer_id: Optional[int] = None,
-        # Explicit linear-attention subclass → strong linear signal (KDA, GDN,
-        # Qwen3-Next, Qwen3.5 main linear layers).
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,12 +508,6 @@ def __init__(
-        # Marker for HybridLinearAttnBackend._is_full_attn: Bailing wraps
-        # linear-attention layers in a plain RadixAttention, so the
-        # dispatcher can't tell from the type alone that this is a linear
-        # layer (would otherwise default to the full-attn backend, e.g. the
-        # same way MTP/NEXTN draft layers are routed).
-        self.attn._is_linear_attention = True
diff -- test/registered/8-gpu-models/test_ling_2_6_flash.py
@@ -3,7 +3,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +8/-13; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6
  - tests: `test/registered/8-gpu-models/test_ling_2_6_flash.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_ling_2_6_flash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27116 - Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"

- 链接: https://github.com/sgl-project/sglang/pull/27116
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+19/-8，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`；技术摘要: 覆盖「Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +13/-8 (21 lines); hunks: -782,17 +782,22 @@ def __init__(; symbols: __init__, _is_full_attn，涉及 `__init__, _is_full_attn`；`python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +13/-8 (21 lines); hunks: -782,17 +782,22 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -782,17 +782,22 @@ def __init__(
-        self,
-        layer: Optional[Union[RadixAttention, RadixLinearAttention]],
-        layer_id: Optional[int] = None,
+        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
-        # RadixLinearAttention is unambiguously a linear-attention layer.
-        # Everything else (including plain RadixAttention) must be classified by
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,6 +508,12 @@ def __init__(
+        # Marker for HybridLinearAttnBackend._is_full_attn: Bailing wraps
+        # linear-attention layers in a plain RadixAttention, so the
+        # dispatcher can't tell from the type alone that this is a linear
+        # layer (would otherwise default to the full-attn backend, e.g. the
+        # same way MTP/NEXTN draft layers are routed).
+        self.attn._is_linear_attention = True
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +13/-8; `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27120 - Fix hybrid linear attention dispatch by layer id with draft-worker awareness

- 链接: https://github.com/sgl-project/sglang/pull/27120
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+5/-23，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix hybrid linear attention dispatch by layer id with draft-worker awareness」；模型线: Ling 2.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/bailing_moe_linear.py`；技术摘要: 覆盖「Fix hybrid linear attention dispatch by layer id with draft-worker awareness」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/bailing_moe_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +0/-16 (16 lines); hunks: -16,7 +16,6; -784,21 +783,6 @@ def __init__(; symbols: __init__, _is_full_attn，涉及 `__init__, _is_full_attn`；`python/sglang/srt/layers/attention/attention_registry.py` modified +5/-1 (6 lines); hunks: -314,7 +314,11 @@ def attn_backend_wrapper(runner: "ModelRunner", full_attn_b...; symbols: attn_backend_wrapper，涉及 `attn_backend_wrapper`；`python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +0/-16 (16 lines); hunks: -16,7 +16,6; -784,21 +783,6 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +5/-1 (6 lines); hunks: -314,7 +314,11 @@ def attn_backend_wrapper(runner: "ModelRunner", full_attn_b...; symbols: attn_backend_wrapper
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -16,7 +16,6 @@
-from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
@@ -784,21 +783,6 @@ def __init__(
-        # Explicit linear-attention subclass → strong linear signal (KDA, GDN,
-        # Qwen3-Next, Qwen3.5 main linear layers).
-        if isinstance(layer, RadixLinearAttention):
-            return False
diff -- python/sglang/srt/layers/attention/attention_registry.py
@@ -314,7 +314,11 @@ def attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBac
-        full_attn_layers = cfg.full_attention_layer_ids
+        if runner.is_draft_worker:
+            # FIXME: we assume that MTP/NEXTN always use full-attention.
+            full_attn_layers = [0]
+        else:
+            full_attn_layers = cfg.full_attention_layer_ids
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,12 +508,6 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +0/-16; `python/sglang/srt/layers/attention/attention_registry.py` modified +5/-1; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- 链接: https://github.com/sgl-project/sglang/pull/23906
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 160 个文件，+5197/-3068，可读 patch 12233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Cuda Graph Runner/Backend Refactor」；模型线: Ling 2.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`；技术摘要: 覆盖「[Refactor] Cuda Graph Runner/Backend Refactor」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool，涉及 `freeze_gc, _to_torch, patch_model`；`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype，涉及 `PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled`；`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode，涉及 `_make_graph_key, build_replay_fb_view, _allocate_decode_buffers`；`python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers，涉及 `BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode
  - `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: _grouped_foreach_copy_, foreach_copy, DecodeInputBuffers, create
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -1,860 +0,0 @@
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
@@ -0,0 +1,846 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,4 +1,4 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541; `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0; `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` added +225/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/doc_patch.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: Ling 2.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention，涉及 `ApertusMLP, __init__, forward`；`python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales，涉及 `__init__, forward, load_kv_cache_scales`；`python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__，涉及 `_resolve_moe_input_pad_multiple, __init__`；`python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
