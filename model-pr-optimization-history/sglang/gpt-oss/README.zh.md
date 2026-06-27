# sglang GPT-OSS 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `benchmark/gpt_oss/README.md` | [#9728](https://github.com/sgl-project/sglang/pull/9728) |
| `docs/basic_usage/gpt_oss.md` | [#9497](https://github.com/sgl-project/sglang/pull/9497), [#9613](https://github.com/sgl-project/sglang/pull/9613), [#9626](https://github.com/sgl-project/sglang/pull/9626) |
| `docs_new/cookbook/autoregressive/OpenAI/GPT-OSS.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/gpt-oss-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/gpt_oss_detector.py` | [#9043](https://github.com/sgl-project/sglang/pull/9043), [#9190](https://github.com/sgl-project/sglang/pull/9190), [#9657](https://github.com/sgl-project/sglang/pull/9657) |
| `python/sglang/srt/models/gpt_oss.py` | [#8824](https://github.com/sgl-project/sglang/pull/8824), [#8843](https://github.com/sgl-project/sglang/pull/8843), [#8944](https://github.com/sgl-project/sglang/pull/8944), [#9028](https://github.com/sgl-project/sglang/pull/9028), [#9146](https://github.com/sgl-project/sglang/pull/9146), [#9161](https://github.com/sgl-project/sglang/pull/9161), [#9359](https://github.com/sgl-project/sglang/pull/9359), [#9433](https://github.com/sgl-project/sglang/pull/9433), [#9469](https://github.com/sgl-project/sglang/pull/9469), [#9783](https://github.com/sgl-project/sglang/pull/9783), [#14197](https://github.com/sgl-project/sglang/pull/14197), [#16775](https://github.com/sgl-project/sglang/pull/16775), ... (19 total) |
| `python/sglang/test/gpt_oss_common.py` | [#16426](https://github.com/sgl-project/sglang/pull/16426) |
| `test/manual/core/test_gpt_oss_1gpu.py` | 无直接 PR 号提交 |
| `test/registered/8-gpu-models/test_gpt_oss_120b.py` | [#18134](https://github.com/sgl-project/sglang/pull/18134) |
| `test/registered/amd/accuracy/mi30x/test_gpt_oss_eval_amd.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` | [#26884](https://github.com/sgl-project/sglang/pull/26884) |
| `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` | [#21570](https://github.com/sgl-project/sglang/pull/21570) |
| `test/registered/models_e2e/test_gpt_oss_4gpu_bf16.py` | 无直接 PR 号提交 |
| `test/registered/models_e2e/test_gpt_oss_4gpu_mxfp4.py` | 无直接 PR 号提交 |
| `test/registered/models_e2e/test_gpt_oss_sm120.py` | 无直接 PR 号提交 |
| `test/registered/perf/test_gpt_oss_4gpu_perf.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 29
- 原文档显式引用补充 PR 数: 16
- 当前文档总 PR 数: 45
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-05 | [#8824](https://github.com/sgl-project/sglang/pull/8824) | merged | Add initial support for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-06 | [#8843](https://github.com/sgl-project/sglang/pull/8843) | merged | Support mxfp4 for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-08 | [#8944](https://github.com/sgl-project/sglang/pull/8944) | merged | Expert Parallelism for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-12 | [#9043](https://github.com/sgl-project/sglang/pull/9043) | merged | (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-13 | [#9146](https://github.com/sgl-project/sglang/pull/9146) | merged | Fix gpt-oss ~2x memory consumption issue | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9028](https://github.com/sgl-project/sglang/pull/9028) | merged | Support FA3 backend for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9161](https://github.com/sgl-project/sglang/pull/9161) | merged | Fix broken trtllm_mha attn backend with gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-20 | [#9359](https://github.com/sgl-project/sglang/pull/9359) | merged | Support DP attention with GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-21 | [#9433](https://github.com/sgl-project/sglang/pull/9433) | merged | [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9469](https://github.com/sgl-project/sglang/pull/9469) | merged | fix: tmp revert gpt oss tp sharding on hopper | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9497](https://github.com/sgl-project/sglang/pull/9497) | merged | [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools | `docs/basic_usage/gpt_oss.md` |
| 2025-08-25 | [#9190](https://github.com/sgl-project/sglang/pull/9190) | merged | Fix Harmony reasoning parser for and auto-separation for gpt-oss models | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-25 | [#9613](https://github.com/sgl-project/sglang/pull/9613) | merged | [docs] Refactor, remove compiled results and add gpt-oss | `docs/basic_usage/gpt_oss.md` |
| 2025-08-28 | [#9728](https://github.com/sgl-project/sglang/pull/9728) | merged | gpt-oss blog reproduction document | `benchmark/gpt_oss/README.md` |
| 2025-09-01 | [#9783](https://github.com/sgl-project/sglang/pull/9783) | merged | support fp8 kvcache for hybrid attn backend on GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-09-15 | [#9626](https://github.com/sgl-project/sglang/pull/9626) | merged | Add reasoning examples for GPT-OSS in Markdown examples | `docs/basic_usage/gpt_oss.md` |
| 2025-09-15 | [#9657](https://github.com/sgl-project/sglang/pull/9657) | merged | fix: gpt-oss streaming dropping normal content when tools are provided but not used | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-12-30 | [#14920](https://github.com/sgl-project/sglang/pull/14920) | merged | Eagle: GPT-OSS Eagle v2 support | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2026-01-07 | [#16426](https://github.com/sgl-project/sglang/pull/16426) | merged | Fix gpt_oss_common import path and migrate core tests | `python/sglang/test/gpt_oss_common.py` |
| 2026-01-18 | [#14197](https://github.com/sgl-project/sglang/pull/14197) | merged | [NPU]Support GPT-OSS for NPU | `python/sglang/srt/models/gpt_oss.py` |
| 2026-01-22 | [#17553](https://github.com/sgl-project/sglang/pull/17553) | merged | [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-03 | [#18134](https://github.com/sgl-project/sglang/pull/18134) | merged | feature: adding gpt-oss 120b nightly test | `test/registered/8-gpu-models/test_gpt_oss_120b.py` |
| 2026-02-12 | [#18405](https://github.com/sgl-project/sglang/pull/18405) | merged | [PCG] GPT OSS Triton Kernel Support | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-16 | [#18869](https://github.com/sgl-project/sglang/pull/18869) | merged | [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |
| 2026-02-20 | [#18988](https://github.com/sgl-project/sglang/pull/18988) | merged | [GPT-OSS] support fp8 online quantization for gpt-oss bf16 | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py` |
| 2026-03-06 | [#20056](https://github.com/sgl-project/sglang/pull/20056) | merged | [CI] Add GPT-OSS test for SM120 | `test/registered/core/test_gpt_oss_sm120.py` |
| 2026-03-24 | [#20755](https://github.com/sgl-project/sglang/pull/20755) | merged | Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+ | `python/sglang/srt/models/gpt_oss.py` |
| 2026-04-02 | [#21570](https://github.com/sgl-project/sglang/pull/21570) | merged | [4/n] Support gpt oss 20b lora | `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` |
| 2026-04-08 | [#22237](https://github.com/sgl-project/sglang/pull/22237) | merged | [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58 | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |
| 2026-05-15 | [#25335](https://github.com/sgl-project/sglang/pull/25335) | merged | [Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1 | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/flashinfer_comm_fusion.py` |
| 2026-05-20 | [#25831](https://github.com/sgl-project/sglang/pull/25831) | merged | [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests | `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py` |
| 2026-05-24 | [#26205](https://github.com/sgl-project/sglang/pull/26205) | merged | Clean up server startup log noise | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/utils/hf_transformers/tokenizer.py` |
| 2026-05-28 | [#26610](https://github.com/sgl-project/sglang/pull/26610) | merged | test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit) | `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` |
| 2026-05-29 | [#16775](https://github.com/sgl-project/sglang/pull/16775) | merged | [CPU] Add GPT-OSS model optimization for CPU | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-02 | [#26884](https://github.com/sgl-project/sglang/pull/26884) | merged | [AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path | `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` |
| 2026-06-03 | [#27001](https://github.com/sgl-project/sglang/pull/27001) | merged | [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests | `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` |
| 2026-06-06 | [#27201](https://github.com/sgl-project/sglang/pull/27201) | merged | [AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/server_args.py` |
| 2026-06-08 | [#27063](https://github.com/sgl-project/sglang/pull/27063) | merged | [AMD] Optimize gpt-oss-120B performance | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-08 | [#27528](https://github.com/sgl-project/sglang/pull/27528) | merged | Fix GPT-OSS MXFP4 hidden size reshape on SM10X | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | merged | [Spec] Retire Spec V1 | `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py` |
| 2026-06-12 | [#27941](https://github.com/sgl-project/sglang/pull/27941) | merged | Enable PDL for GPT-OSS tinygemm router | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |

## 逐 PR diff 审计卡

### PR #8824 - Add initial support for gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/8824
- 状态/时间: merged / 2025-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `c1d2061f97ae`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1595/-47，可读 patch 2185 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add initial support for gpt-oss」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add initial support for gpt-oss」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock，涉及 `GptOssConfig, __init__, get_attention_sliding_window_size`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -0,0 +1,923 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` added +923/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8843 - Support mxfp4 for GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/8843
- 状态/时间: merged / 2025-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `168033d5fb1e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+791/-325，可读 patch 1320 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support mxfp4 for GPT-OSS」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Support mxfp4 for GPT-OSS」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights，涉及 `__init__, _get_default_weight_mapping, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,8 @@
+    get_moe_expert_parallel_rank,
+    get_moe_expert_parallel_world_size,
@@ -108,11 +110,15 @@ def __init__(
+            quant_config_name = (
+                quant_config.get_name() if quant_config is not None else None
+            )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +209/-9
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8944 - Expert Parallelism for GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/8944
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `1d24db834803`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+269/-119，可读 patch 956 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Expert Parallelism for GPT-OSS」；模型线: GPT-OSS；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Expert Parallelism for GPT-OSS」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights，涉及 `__init__, _load_mxfp4_experts_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,6 +28,7 @@
+    get_moe_tensor_parallel_world_size,
@@ -96,11 +97,6 @@ def __init__(
-        if self.tp_size > config.num_local_experts:
-            raise ValueError(
-                f"Tensor parallel size {self.tp_size} is greater than "
-                f"the number of experts {config.num_local_experts}."
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +54/-47
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9043 - (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support

- 链接: https://github.com/sgl-project/sglang/pull/9043
- 状态/时间: merged / 2025-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gpt_oss_detector.py`；关联提交 `a21849013607`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+717/-409，可读 patch 1293 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「(gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/gpt_oss_detector.py`；技术摘要: 覆盖「(gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support」；主要实现面是 `python/sglang/srt/function_call/gpt_oss_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse，涉及 `GptOssDetector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -0,0 +1,331 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/harmony_utils.py`, `python/sglang/srt/entrypoints/http_server.py`, `python/sglang/srt/entrypoints/openai/protocol.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9146 - Fix gpt-oss ~2x memory consumption issue

- 链接: https://github.com/sgl-project/sglang/pull/9146
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `9394ed63867d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+19/-7，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix gpt-oss ~2x memory consumption issue」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Fix gpt-oss ~2x memory consumption issue」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights，涉及 `__init__, routed_experts_weights_of_layer, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -64,7 +64,13 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_flashinfer_available, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_flashinfer_available,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +19/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9028 - Support FA3 backend for gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/9028
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `0ff6d1fce122`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+24/-6，可读 patch 121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support FA3 backend for gpt-oss」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Support FA3 backend for gpt-oss」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -294,7 +294,7 @@ def __init__(
-            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
+            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/pyproject.toml`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9161 - Fix broken trtllm_mha attn backend with gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/9161
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `6b7c24712cda`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix broken trtllm_mha attn backend with gpt-oss」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Fix broken trtllm_mha attn backend with gpt-oss」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -293,8 +293,12 @@ def __init__(
+        # Choose dtype of sinks based on attention backend: trtllm_mha requires float32,
+        # others can use bfloat16
+        attn_backend = global_server_args_dict.get("attention_backend")
+        sinks_dtype = torch.float32 if attn_backend == "trtllm_mha" else torch.bfloat16
-            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
+            torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9359 - Support DP attention with GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/9359
- 状态/时间: merged / 2025-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `c10b8e6a0f2a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-5，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DP attention with GPT-OSS」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Support DP attention with GPT-OSS」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights，涉及 `_load_normal_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -1123,7 +1123,7 @@ def _load_normal_weights(
-                            start = tp_rank * param.numel()
+                            start = get_attention_tp_rank() * param.numel()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9433 - [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/9433
- 状态/时间: merged / 2025-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `dae9a80f43e8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-3，可读 patch 46 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「[fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights，涉及 `_load_mxfp4_experts_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -16,6 +16,7 @@
+import math
@@ -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):
+        assert (
+            intermediate_size % mxfp4_block == 0
+        ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
-        per_rank_intermediate_size_block = intermediate_size_block // moe_tp_size
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9469 - fix: tmp revert gpt oss tp sharding on hopper

- 链接: https://github.com/sgl-project/sglang/pull/9469
- 状态/时间: merged / 2025-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `849957bc76c3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-3，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: tmp revert gpt oss tp sharding on hopper」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「fix: tmp revert gpt oss tp sharding on hopper」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights，涉及 `_load_mxfp4_experts_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):
-        per_rank_intermediate_size_block = math.ceil(
-            intermediate_size_block / moe_tp_size
-        )
+        if _is_sm100_supported:
+            per_rank_intermediate_size_block = math.ceil(
+                intermediate_size_block / moe_tp_size
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +6/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9497 - [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools

- 链接: https://github.com/sgl-project/sglang/pull/9497
- 状态/时间: merged / 2025-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/gpt_oss.md`；关联提交 `fedfe91c1a6e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+106/-0，可读 patch 110 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add doc and quick demo for gpt-oss responses api & buildin tools」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `docs/basic_usage/gpt_oss.md`；技术摘要: 覆盖「[Docs] Add doc and quick demo for gpt-oss responses api & buildin tools」；主要实现面是 `docs/basic_usage/gpt_oss.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109。
- 代码 diff 细节:
  - `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109
- 关键代码摘录:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -1,3 +1,109 @@
+## Responses API & Built-in Tools
+### Responses API
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+### Built-in Tools
+GPT‑OSS can call built‑in tools for web search and Python execution. You can use the demo tool server or connect to external MCP tool servers.
+#### Python Tool
```

- 已读文件:
  - docs: `docs/basic_usage/gpt_oss.md` modified +106/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/gpt_oss.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #9190 - Fix Harmony reasoning parser for and auto-separation for gpt-oss models

- 链接: https://github.com/sgl-project/sglang/pull/9190
- 状态/时间: merged / 2025-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gpt_oss_detector.py`；关联提交 `a0a77d937b99`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+1681/-556，可读 patch 2406 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Harmony reasoning parser for and auto-separation for gpt-oss models」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/gpt_oss_detector.py`；技术摘要: 覆盖「Fix Harmony reasoning parser for and auto-separation for gpt-oss models」；主要实现面是 `python/sglang/srt/function_call/gpt_oss_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse，涉及 `GptOssDetector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -1,7 +1,7 @@
-from typing import List
+from typing import List, Optional
@@ -10,60 +10,31 @@
+from sglang.srt.harmony_parser import HarmonyParser
-    Detector for T4-style function calls with channel format.
+    Detector for T4-style function calls using HarmonyParser.
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/test_harmony_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9613 - [docs] Refactor, remove compiled results and add gpt-oss

- 链接: https://github.com/sgl-project/sglang/pull/9613
- 状态/时间: merged / 2025-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/gpt_oss.md`；关联提交 `9b08d975a0a5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+166/-611，可读 patch 638 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Refactor, remove compiled results and add gpt-oss」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `docs/basic_usage/gpt_oss.md`；技术摘要: 覆盖「[docs] Refactor, remove compiled results and add gpt-oss」；主要实现面是 `docs/basic_usage/gpt_oss.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe...。
- 代码 diff 细节:
  - `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe...
- 关键代码摘录:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python execution. You can
+### Tool & Reasoning Parser
+- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoni
```

- 已读文件:
  - docs: `docs/basic_usage/gpt_oss.md` modified +5/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/gpt_oss.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #9728 - gpt-oss blog reproduction document

- 链接: https://github.com/sgl-project/sglang/pull/9728
- 状态/时间: merged / 2025-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `benchmark/gpt_oss/README.md`；关联提交 `d0934a519257`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+163/-0，可读 patch 164 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「gpt-oss blog reproduction document」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `benchmark/gpt_oss/README.md`；技术摘要: 覆盖「gpt-oss blog reproduction document」；主要实现面是 `benchmark/gpt_oss/README.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163。
- 代码 diff 细节:
  - `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163
- 关键代码摘录:

```diff
diff -- benchmark/gpt_oss/README.md
@@ -0,0 +1,163 @@
+# How to reproduce the result of GPT-OSS with SGLang
+### Install the latest SGLang
+'''bash
+git clone https://github.com/sgl-project/sglang.git
+cd sglang
+git checkout v0.5.1.post3
```

- 已读文件:
  - other: `benchmark/gpt_oss/README.md` added +163/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #9783 - support fp8 kvcache for hybrid attn backend on GPT-OSS

- 链接: https://github.com/sgl-project/sglang/pull/9783
- 状态/时间: merged / 2025-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `9db8025376b2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-4，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support fp8 kvcache for hybrid attn backend on GPT-OSS」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「support fp8 kvcache for hybrid attn backend on GPT-OSS」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core，涉及 `forward_normal, _enable_fused_set_kv_buffer, forward_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -193,8 +193,9 @@ def forward_normal(
-def _enable_fused_set_kv_buffer():
-    return _is_cuda
+def _enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
+    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
+    return _is_cuda and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
@@ -341,7 +342,7 @@ def forward_prepare(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9626 - Add reasoning examples for GPT-OSS in Markdown examples

- 链接: https://github.com/sgl-project/sglang/pull/9626
- 状态/时间: merged / 2025-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/gpt_oss.md`；关联提交 `0b14159fc4e0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-2，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add reasoning examples for GPT-OSS in Markdown examples」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `docs/basic_usage/gpt_oss.md`；技术摘要: 覆盖「Add reasoning examples for GPT-OSS in Markdown examples」；主要实现面是 `docs/basic_usage/gpt_oss.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [。
- 代码 diff 细节:
  - `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [
- 关键代码摘录:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://gith
-GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use. Yo
@@ -69,6 +69,16 @@ tools = [
+# Reasoning level example
+response = client.responses.create(
+    model="openai/gpt-oss-120b",
```

- 已读文件:
  - docs: `docs/basic_usage/gpt_oss.md` modified +11/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/protocol.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9657 - fix: gpt-oss streaming dropping normal content when tools are provided but not used

- 链接: https://github.com/sgl-project/sglang/pull/9657
- 状态/时间: merged / 2025-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/gpt_oss_detector.py`；关联提交 `28c79dc84ab8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: gpt-oss streaming dropping normal content when tools are provided but not used」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/gpt_oss_detector.py`；技术摘要: 覆盖「fix: gpt-oss streaming dropping normal content when tools are provided but not used」；主要实现面是 `python/sglang/srt/function_call/gpt_oss_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment，涉及 `parse_streaming_increment`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -81,6 +81,29 @@ def parse_streaming_increment(
+        # If there are no parsed events and the chunk contains no Harmony structural
+        # markers, treat it as plain text and pass it through. This fixes a bug where
+        # normal content was held in the buffer when tools were provided but not used.
+        if not events:
+            has_harmony_markers = any(
+                marker in self._buffer
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/gpt_oss_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14920 - Eagle: GPT-OSS Eagle v2 support

- 链接: https://github.com/sgl-project/sglang/pull/14920
- 状态/时间: merged / 2025-12-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+48/-25，可读 patch 124 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Eagle: GPT-OSS Eagle v2 support」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`；技术摘要: 覆盖「Eagle: GPT-OSS Eagle v2 support」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run，涉及 `__init__, initialize, _dummy_run`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__，涉及 `__init__`；`python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode，涉及 `__init__, forward_draft_extend_after_decode`；`python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -345,6 +345,32 @@ def __init__(
+        # auxiliary hidden capture mode. TODO: expose this to server args?
+        self.eagle_use_aux_hidden_state = False
+        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
+            # load draft config
+            draft_model_config = ModelConfig.from_server_args(
+                server_args,
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):
-        if model_runner.spec_algorithm.is_eagle3():
+        if (
+            model_runner.spec_algorithm.is_eagle3()
+            and model_runner.eagle_use_aux_hidden_state
+        ):
diff -- python/sglang/srt/speculative/eagle_worker.py
@@ -186,6 +186,15 @@ def __init__(
+        self.eagle_use_aux_hidden_state = False
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0; `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16426 - Fix gpt_oss_common import path and migrate core tests

- 链接: https://github.com/sgl-project/sglang/pull/16426
- 状态/时间: merged / 2026-01-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/test/gpt_oss_common.py`；关联提交 `0c474273c514`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+48/-26，可读 patch 255 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix gpt_oss_common import path and migrate core tests」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/test/gpt_oss_common.py`；技术摘要: 覆盖「Fix gpt_oss_common import path and migrate core tests」；主要实现面是 `python/sglang/test/gpt_oss_common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
No textual patch was returned by GitHub for the selected changed files.
```

- 已读文件:
  - tests: `python/sglang/test/gpt_oss_common.py` renamed +0/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/gpt_oss_common.py`, `test/registered/core/test_deterministic.py`, `test/registered/core/test_gpt_oss_1gpu.py`, `test/registered/core/test_hidden_states.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14197 - [NPU]Support GPT-OSS for NPU

- 链接: https://github.com/sgl-project/sglang/pull/14197
- 状态/时间: merged / 2026-01-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `733de6be31e2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+96/-17，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Support GPT-OSS for NPU」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「[NPU]Support GPT-OSS for NPU」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare，涉及 `__init__, forward_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -71,9 +71,10 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_cuda, make_layers
+from sglang.srt.utils import LazyValue, add_prefix, is_cuda, is_npu, make_layers
+_is_npu = is_npu()
@@ -129,6 +130,7 @@ def __init__(
@@ -305,20 +307,20 @@ def forward_prepare(
-        q, k = self.rotary_emb(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +20/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17553 - [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py

- 链接: https://github.com/sgl-project/sglang/pull/17553
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `61abff66c150`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「[NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -492,7 +492,7 @@ def __init__(
-        if is_npu:
+        if _is_npu:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18134 - feature: adding gpt-oss 120b nightly test

- 链接: https://github.com/sgl-project/sglang/pull/18134
- 状态/时间: merged / 2026-02-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_gpt_oss_120b.py`；关联提交 `c8da307d7e63`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+88/-4，可读 patch 121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feature: adding gpt-oss 120b nightly test」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_gpt_oss_120b.py`；技术摘要: 覆盖「feature: adding gpt-oss 120b nightly test」；主要实现面是 `test/registered/8-gpu-models/test_gpt_oss_120b.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants，涉及 `TestGptOss120B, for, test_gpt_oss_120b_all_variants`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_gpt_oss_120b.py
@@ -0,0 +1,84 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
+# Runs on both H200 and B200 via nightly-8-gpu-common suite
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_gpt_oss_120b.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18405 - [PCG] GPT OSS Triton Kernel Support

- 链接: https://github.com/sgl-project/sglang/pull/18405
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `2bd8363486e4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+68/-32，可读 patch 228 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PCG] GPT OSS Triton Kernel Support」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「[PCG] GPT OSS Triton Kernel Support」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__，涉及 `forward_normal, moe_impl, GptOssAttention`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,10 @@
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -72,6 +76,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +21/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/compilation/piecewise_context_manager.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/model_executor/model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18869 - [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test

- 链接: https://github.com/sgl-project/sglang/pull/18869
- 状态/时间: merged / 2026-02-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-2，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Remove `--mem-fraction-static 0.93` from gpt-oss test」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；技术摘要: 覆盖「[CI] Remove `--mem-fraction-static 0.93` from gpt-oss test」；主要实现面是 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b，涉及 `test_mxfp4_120b`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -30,8 +30,6 @@ def test_mxfp4_120b(self):
-                "--mem-fraction-static",
-                "0.93",
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18988 - [GPT-OSS] support fp8 online quantization for gpt-oss bf16

- 链接: https://github.com/sgl-project/sglang/pull/18988
- 状态/时间: merged / 2026-02-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+31/-1，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GPT-OSS] support fp8 online quantization for gpt-oss bf16」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「[GPT-OSS] support fp8 online quantization for gpt-oss bf16」；主要实现面是 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply，涉及 `__init__, create_weights, apply`；`python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply
  - `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):
+        self.with_bias = False
@@ -706,8 +707,10 @@ def create_weights(
+        with_bias: bool = False,
+        self.with_bias = with_bias
@@ -782,6 +785,27 @@ def create_weights(
+        # BIAS (optional, e.g. GPT-OSS)
diff -- python/sglang/srt/server_args.py
@@ -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):
-                elif self.ep_size == 1 and is_triton_kernels_available():
+                elif (
+                    self.ep_size == 1
+                    and is_triton_kernels_available()
+                    and self.quantization is None
+                ):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0; `python/sglang/srt/server_args.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20056 - [CI] Add GPT-OSS test for SM120

- 链接: https://github.com/sgl-project/sglang/pull/20056
- 状态/时间: merged / 2026-03-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-0，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add GPT-OSS test for SM120」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `test/registered/core/test_gpt_oss_sm120.py`；技术摘要: 覆盖「[CI] Add GPT-OSS test for SM120」；主要实现面是 `test/registered/core/test_gpt_oss_sm120.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b，涉及 `TestGptOssSm120, setUpClass, test_mxfp4_20b`。
- 代码 diff 细节:
  - `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b
- 关键代码摘录:

```diff
diff -- test/registered/core/test_gpt_oss_sm120.py
@@ -0,0 +1,34 @@
+import unittest
+import torch
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.gpt_oss_common import BaseTestGptOss
+register_cuda_ci(est_time=500, suite="stage-b-test-small-1-gpu")
+@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
```

- 已读文件:
  - tests: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0
- 验证与风险: diff 自带测试面 `test/registered/core/test_gpt_oss_sm120.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20755 - Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+

- 链接: https://github.com/sgl-project/sglang/pull/20755
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `bbe25b24126d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+65/-2，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__，涉及 `GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -75,10 +75,34 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_npu, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_blackwell_supported,
+    is_cuda,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +65/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21570 - [4/n] Support gpt oss 20b lora

- 链接: https://github.com/sgl-project/sglang/pull/21570
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`；关联提交 `566b4a4f1ccc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+195/-24，可读 patch 328 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[4/n] Support gpt oss 20b lora」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`；技术摘要: 覆盖「[4/n] Support gpt oss 20b lora」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__，涉及 `forward, GptOssForCausalLM, should_apply_lora`；`test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__
  - `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -17,6 +17,7 @@
+import re
@@ -651,6 +652,13 @@ def forward(
+    _lora_pattern_moe = re.compile(
+        r"^(?:model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
+    )
+    def should_apply_lora(self, module_name: str) -> bool:
diff -- test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py
@@ -0,0 +1,151 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +8/-0
  - tests: `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0
- 验证与风险: diff 自带测试面 `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_8b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22237 - [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58

- 链接: https://github.com/sgl-project/sglang/pull/22237
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；技术摘要: 覆盖「[CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58」；主要实现面是 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b，涉及 `test_bf16_120b, test_mxfp4_120b`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -13,7 +13,7 @@ def test_bf16_120b(self):
-                "low": 0.60,
+                "low": 0.58,
@@ -23,7 +23,7 @@ def test_mxfp4_120b(self):
-                "low": 0.60,
+                "low": 0.58,
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25335 - [Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1

- 链接: https://github.com/sgl-project/sglang/pull/25335
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+147/-53，可读 patch 404 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/flashinfer_comm_fusion.py`；技术摘要: 覆盖「[Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1」；主要实现面是 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/flashinfer_comm_fusion.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/mxfp4.py` modified +46/-3 (49 lines); hunks: -141,6 +141,7 @@ def _get_flashinfer_mxfp4_device_permute_indices(; -156,6 +157,49 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, _patch_sm120_mxfp4_min_warps, _compute_num_warps_sm120_mxfp4, _swizzle_mxfp4，涉及 `_get_flashinfer_mxfp4_device_permute_indices, _patch_sm120_mxfp4_min_warps, _compute_num_warps_sm120_mxfp4`；`python/sglang/srt/layers/moe/topk.py` modified +44/-1 (45 lines); hunks: -32,7 +32,50; symbols: routing，涉及 `routing`；`python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +11/-13 (24 lines); hunks: -383,6 +383,11 @@ def initialize(; -515,8 +520,6 @@ def ensure_workspace_initialized(; symbols: initialize, ensure_workspace_initialized，涉及 `initialize, ensure_workspace_initialized`；`python/sglang/srt/layers/quantization/fp4_utils.py` modified +7/-7 (14 lines); hunks: -34,13 +34,13 @@ def _flashinfer_fp4_quantize_impl(; symbols: _flashinfer_fp4_quantize_impl，涉及 `_flashinfer_fp4_quantize_impl`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +46/-3 (49 lines); hunks: -141,6 +141,7 @@ def _get_flashinfer_mxfp4_device_permute_indices(; -156,6 +157,49 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, _patch_sm120_mxfp4_min_warps, _compute_num_warps_sm120_mxfp4, _swizzle_mxfp4
  - `python/sglang/srt/layers/moe/topk.py` modified +44/-1 (45 lines); hunks: -32,7 +32,50; symbols: routing
  - `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +11/-13 (24 lines); hunks: -383,6 +383,11 @@ def initialize(; -515,8 +520,6 @@ def ensure_workspace_initialized(; symbols: initialize, ensure_workspace_initialized
  - `python/sglang/srt/layers/quantization/fp4_utils.py` modified +7/-7 (14 lines); hunks: -34,13 +34,13 @@ def _flashinfer_fp4_quantize_impl(; symbols: _flashinfer_fp4_quantize_impl
  - `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py` modified +6/-2 (8 lines); hunks: -19,8 +19,12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -141,6 +141,7 @@ def _get_flashinfer_mxfp4_device_permute_indices(
+_sm120_mxfp4_min_warps_patched = False
@@ -156,6 +157,49 @@ def _get_flashinfer_mxfp4_device_permute_indices(
+def _patch_sm120_mxfp4_min_warps():
+    global _sm120_mxfp4_min_warps_patched
+    if _sm120_mxfp4_min_warps_patched:
+        return
diff -- python/sglang/srt/layers/moe/topk.py
@@ -32,7 +32,50 @@
-    from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx, routing
+    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
+    from triton_kernels.tensor import make_ragged_tensor_metadata
+    from triton_kernels.topk import topk as triton_kernels_topk
+    def routing(
+        logits,
diff -- python/sglang/srt/layers/flashinfer_comm_fusion.py
@@ -383,6 +383,11 @@ def initialize(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4.py` modified +46/-3; `python/sglang/srt/layers/moe/topk.py` modified +44/-1; `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +11/-13; `python/sglang/srt/layers/quantization/fp4_utils.py` modified +7/-7; `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py` modified +6/-2; `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +4/-3
  - tests: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` modified +1/-6
- 验证与风险: diff 自带测试面 `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/moe/test_cutedsl_moe.py`, `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25831 - [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests

- 链接: https://github.com/sgl-project/sglang/pull/25831
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 36 个文件，+572/-639，可读 patch 1504 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`；技术摘要: 覆盖「[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests」；主要实现面是 `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1 (2 lines); hunks: -1,4 +1,4；`python/sglang/test/kits/server_sanity_kit.py` removed +0/-228 (228 lines); hunks: -1,228 +0,0; symbols: ServerSanityMixin, _sanity_generate, test_health, test_health_generate，涉及 `ServerSanityMixin, _sanity_generate, test_health`；`python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: BasicSchedulerStressMixin, _stress_generate, test_streaming_response, test_concurrent_requests，涉及 `BasicSchedulerStressMixin, _stress_generate, test_streaming_response`；`python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france, test_basic_math，涉及 `BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france`。
- 代码 diff 细节:
  - `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1 (2 lines); hunks: -1,4 +1,4
  - `python/sglang/test/kits/server_sanity_kit.py` removed +0/-228 (228 lines); hunks: -1,228 +0,0; symbols: ServerSanityMixin, _sanity_generate, test_health, test_health_generate
  - `python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: BasicSchedulerStressMixin, _stress_generate, test_streaming_response, test_concurrent_requests
  - `python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france, test_basic_math
  - `test/registered/language/test_srt_backend.py` removed +0/-94 (94 lines); hunks: -1,94 +0,0; symbols: TestSRTBackend, setUpClass, tearDownClass, test_few_shot_qa
- 关键代码摘录:

```diff
diff -- test/manual/models/test_nvidia_nemotron_3_nano_archived.py
@@ -1,4 +1,4 @@
-"""Archived test classes split out of test/registered/models/test_nvidia_nemotron_3_nano.py.
+"""Archived test classes split out of test/registered/models_e2e/test_nvidia_nemotron_3_nano.py.
diff -- python/sglang/test/kits/server_sanity_kit.py
@@ -1,228 +0,0 @@
-"""Black-box server sanity prompts: cheap checks that catch silent
-correctness regressions (gibberish / repetition collapse / encoding),
-streaming/concurrent path bugs, and endpoint health.
-Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``
-and ``self.process``. Each test is independent and fast (≤ 5 s after
-warmup); the whole kit completes in < 1 min."""
diff -- python/sglang/test/kits/basic_scheduler_stress_kit.py
@@ -0,0 +1,135 @@
+"""Basic scheduler / cache / streaming stress sanity kit.
+Probes that catch bugs which only fire under multi-request or large-
+prompt conditions: scheduler hangs, radix prefix-cache cross-
+contamination, chunked-prefill multi-chunk kernel crashes, and SSE
```

- 已读文件:
  - tests: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1; `python/sglang/test/kits/server_sanity_kit.py` removed +0/-228; `python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0; `python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0; `test/registered/language/test_srt_backend.py` removed +0/-94; `test/registered/core/test_engine_child_pids.py` modified +40/-51
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/basic_api_contract_kit.py`, `python/sglang/test/kits/basic_decode_correctness_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`, `python/sglang/test/kits/server_sanity_kit.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26205 - Clean up server startup log noise

- 链接: https://github.com/sgl-project/sglang/pull/26205
- 状态/时间: merged / 2026-05-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+157/-68，可读 patch 380 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Clean up server startup log noise」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/utils/hf_transformers/tokenizer.py`；技术摘要: 覆盖「Clean up server startup log noise」；主要实现面是 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/utils/hf_transformers/tokenizer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/model_config.py` modified +3/-3 (6 lines); hunks: -1450,15 +1450,15 @@ def _get_and_verify_dtype(; symbols: _get_and_verify_dtype，涉及 `_get_and_verify_dtype`；`python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-4 (6 lines); hunks: -243,12 +243,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/utils/hf_transformers/tokenizer.py` modified +3/-3 (6 lines); hunks: -105,7 +105,7 @@ def _load_tokenizer_by_declared_class(tokenizer_name, *args,...; -208,7 +208,7 @@ def _resolve_tokenizers_backend(tokenizer_name, *args, **com...; symbols: _load_tokenizer_by_declared_class, _resolve_tokenizers_backend，涉及 `_load_tokenizer_by_declared_class, _resolve_tokenizers_backend`；`python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -219,7 +219,7 @@ def __init__(; -468,7 +468,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/model_config.py` modified +3/-3 (6 lines); hunks: -1450,15 +1450,15 @@ def _get_and_verify_dtype(; symbols: _get_and_verify_dtype
  - `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-4 (6 lines); hunks: -243,12 +243,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/utils/hf_transformers/tokenizer.py` modified +3/-3 (6 lines); hunks: -105,7 +105,7 @@ def _load_tokenizer_by_declared_class(tokenizer_name, *args,...; -208,7 +208,7 @@ def _resolve_tokenizers_backend(tokenizer_name, *args, **com...; symbols: _load_tokenizer_by_declared_class, _resolve_tokenizers_backend
  - `python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -219,7 +219,7 @@ def __init__(; -468,7 +468,7 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/model_executor/model_runner.py` modified +0/-2 (2 lines); hunks: -2214,8 +2214,6 @@ def configure_kv_cache_dtype(self):; symbols: configure_kv_cache_dtype, init_cublas
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/model_config.py
@@ -1450,15 +1450,15 @@ def _get_and_verify_dtype(
-            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
+            logger.debug("Upcasting %s to %s.", config_dtype, torch_dtype)
-            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
+            logger.debug("Downcasting %s to %s.", config_dtype, torch_dtype)
-            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)
+            logger.debug("Casting %s to %s.", config_dtype, torch_dtype)
diff -- python/sglang/srt/layers/attention/flashinfer_backend.py
@@ -243,12 +243,10 @@ def __init__(
-            # Disable CUTLASS backend when piecewise cuda graph is enabled
-            # due to TMA descriptor initialization issues on B200
-                logger.warning(
+                logger.info(
-                    "due to TMA descriptor initialization issues on B200. "
+                    "due to TMA descriptor initialization issues on SM100 GPUs. "
diff -- python/sglang/srt/utils/hf_transformers/tokenizer.py
@@ -105,7 +105,7 @@ def _load_tokenizer_by_declared_class(tokenizer_name, *args, **kwargs):
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +3/-3; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-4; `python/sglang/srt/utils/hf_transformers/tokenizer.py` modified +3/-3; `python/sglang/srt/models/gpt_oss.py` modified +2/-2; `python/sglang/srt/model_executor/model_runner.py` modified +0/-2; `python/sglang/srt/managers/template_manager.py` modified +7/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/managers/template_detection.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26610 - test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)

- 链接: https://github.com/sgl-project/sglang/pull/26610
- 状态/时间: merged / 2026-05-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+611/-816，可读 patch 1566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`；技术摘要: 覆盖「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；主要实现面是 `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass`；`python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache，涉及 `_random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching`；`test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass，涉及 `TestStep3p5FlashChainMTP, setUpClass, tearDownClass`；`test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV3MTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache
  - `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105 (105 lines); hunks: -1,105 +0,0; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -1,212 +0,0 @@
-import unittest
-from types import SimpleNamespace
-import requests
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-from sglang.test.run_eval import run_eval
diff -- python/sglang/test/kits/unified_radix_cache_kit.py
@@ -1,25 +1,12 @@
-import unittest
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-    get_input_ids,
-    make_mamba_decode_assert,
-    make_mamba_prefill_assert,
diff -- test/registered/models_e2e/test_step3p5_flash_chain_mtp.py
@@ -1,28 +1,20 @@
```

- 已读文件:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212; `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133; `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110; `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105; `test/registered/quant/test_deepseek_v3_fp4_4gpu.py` removed +0/-80
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/manual/core/test_dsv4_hicache_swa_translation_cache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16775 - [CPU] Add GPT-OSS model optimization for CPU

- 链接: https://github.com/sgl-project/sglang/pull/16775
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `3ecf2c76ad1b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+2023/-553，可读 patch 3996 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Add GPT-OSS model optimization for CPU」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「[CPU] Add GPT-OSS model optimization for CPU」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +20/-1 (21 lines); hunks: -81,6 +81,7; -89,6 +90,7; symbols: _load_mxfp4_experts_weights, _load_normal_weights，涉及 `_load_mxfp4_experts_weights, _load_normal_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +20/-1 (21 lines); hunks: -81,6 +81,7; -89,6 +90,7; symbols: _load_mxfp4_experts_weights, _load_normal_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -81,6 +81,7 @@
+    is_cpu,
@@ -89,6 +90,7 @@
+_is_cpu = is_cpu()
@@ -881,7 +883,8 @@ def _load_mxfp4_experts_weights(self, weights):
-            weight = weight.cuda()
+            if _is_cuda:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +20/-1
- 验证与风险: diff 自带测试面 `test/registered/cpu/test_decode.py`, `test/registered/cpu/test_extend.py`, `test/registered/cpu/test_gemm.py`, `test/registered/cpu/test_mla.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- 链接: https://github.com/sgl-project/sglang/pull/25813
- 状态/时间: merged / 2026-06-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+1262/-2154，可读 patch 4187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): port popular model usage guides into cookbook pages」；模型线: GPT-OSS；类别: 文档/测试/CI；主要 diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`；技术摘要: 覆盖「docs(cookbook): port popular model usage guides into cookbook pages」；主要实现面是 `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0；`docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...；`docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64，涉及 `image_to_base64`。
- 代码 diff 细节:
  - `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0
  - `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64
  - `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181 (181 lines); hunks: -1,181 +0,0
- 关键代码摘录:

```diff
diff -- docs_new/docs/basic_usage/deepseek_v32.mdx
@@ -1,601 +0,0 @@
-title: "DeepSeek V3.2/GLM-5 Usage"
-metatags:
-    description: "Deploy DeepSeek V3.2/GLM-5 with SGLang: DeepSeek Sparse Attention (DSA), long-context optimization, MTP speculative decoding, function calling. Supports H200, B2
-DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism power
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
-## Installation
diff -- docs_new/docs/basic_usage/deepseek_v3.mdx
@@ -1,375 +0,0 @@
-title: "DeepSeek V3/V3.1/R1 Usage"
-metatags:
-    description: "Deploy DeepSeek V3/R1 with SGLang: MLA optimization, FP8 quantization, multi-node TP, DP attention, MTP speculative decoding. Supports H200, B200, MI300X, A100."
-SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/dee
-This document outlines current optimizations for DeepSeek.
-For an overview of the implemented features see the completed [Roadmap](https://github.com/sgl-project/sglang/issues/2591).
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx
@@ -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose the most suitable in
```

- 已读文件:
  - docs: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26; `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181; `docs_new/docs/basic_usage/glmv.mdx` removed +0/-139
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26884 - [AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path

- 链接: https://github.com/sgl-project/sglang/pull/26884
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`；关联提交 `4226a6f13aa6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+87/-20，可读 patch 206 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`；技术摘要: 覆盖「[AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +12/-2 (14 lines); hunks: -75,7 +75,14 @@ def __post_init__(self):; -93,7 +100,10 @@ def __post_init__(self):; symbols: __post_init__，涉及 `__post_init__`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +12/-2 (14 lines); hunks: -75,7 +75,14 @@ def __post_init__(self):; -93,7 +100,10 @@ def __post_init__(self):; symbols: __post_init__
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py
@@ -75,7 +75,14 @@ def __post_init__(self):
-        env_vars={"SGLANG_USE_AITER": "1"},
+        # AITER MXFP4 fused-MoE for gpt-oss uses the SEPARATED gate/up tile
+        # layout (matches `gptoss_fp4_tuned_fmoe.csv` flydsl entries and the
+        # Mxfp4MoEMethod weight shuffle). Other AITER MXFP4 callers default
+        # to INTERLEAVE, so opt out explicitly here.
+        env_vars={
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +12/-2
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/27001
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+11/-471，可读 patch 936 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`；技术摘要: 覆盖「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；主要实现面是 `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x`；`test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x`；`test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x`；`test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models，涉及 `get_model_path, ModelConfig, get_display_name`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -39,21 +34,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
- 关键代码摘录:

```diff
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py
@@ -2,19 +2,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py
@@ -3,19 +3,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py
@@ -3,19 +3,10 @@
```

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` modified +1/-35
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27201 - [AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue

- 链接: https://github.com/sgl-project/sglang/pull/27201
- 状态/时间: merged / 2026-06-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+73/-55，可读 patch 233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「[AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue」；主要实现面是 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/mxfp4.py` modified +54/-43 (97 lines); hunks: -28,11 +28,11; -155,7 +155,9 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, swap_every_two_rows, apply，涉及 `_get_flashinfer_mxfp4_device_permute_indices, swap_every_two_rows, apply`；`python/sglang/srt/layers/attention/aiter_backend.py` modified +10/-3 (13 lines); hunks: -899,10 +899,12 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -1378,10 +1380,13 @@ def init_forward_metadata(self, forward_batch: ForwardBa...; symbols: init_forward_metadata, _apply_cuda_graph_metadata，涉及 `init_forward_metadata, _apply_cuda_graph_metadata`；`python/sglang/srt/server_args.py` modified +7/-7 (14 lines); hunks: -2179,13 +2179,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`；`test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +2/-2 (4 lines); hunks: -76,7 +76,7 @@ def __post_init__(self):; -97,7 +97,7 @@ def __post_init__(self):; symbols: __post_init__，涉及 `__post_init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +54/-43 (97 lines); hunks: -28,11 +28,11; -155,7 +155,9 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, swap_every_two_rows, apply
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +10/-3 (13 lines); hunks: -899,10 +899,12 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -1378,10 +1380,13 @@ def init_forward_metadata(self, forward_batch: ForwardBa...; symbols: init_forward_metadata, _apply_cuda_graph_metadata
  - `python/sglang/srt/server_args.py` modified +7/-7 (14 lines); hunks: -2179,13 +2179,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +2/-2 (4 lines); hunks: -76,7 +76,7 @@ def __post_init__(self):; -97,7 +97,7 @@ def __post_init__(self):; symbols: __post_init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -28,11 +28,11 @@
+from sglang.srt.environ import envs
@@ -155,7 +155,9 @@ def _get_flashinfer_mxfp4_device_permute_indices(
+            shuffle_scale_a16w4,
+            shuffle_weight_a16w4,
@@ -773,20 +775,13 @@ def swap_every_two_rows(x, axis=-1):
-            # Bias must be fp32 for the AITER kernels.
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -899,10 +899,12 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
+                        # AITER attention kernels require int32 page indices;
+                        # full_to_swa_index_mapping is stored as int64.
-                            )
+                            ).to(torch.int32)
@@ -1378,10 +1380,13 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
+                    # AITER attention kernels (e.g. mha_batch_prefill_func)
diff -- python/sglang/srt/server_args.py
@@ -2179,13 +2179,13 @@ def _handle_model_specific_adjustments(self):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4.py` modified +54/-43; `python/sglang/srt/layers/attention/aiter_backend.py` modified +10/-3; `python/sglang/srt/server_args.py` modified +7/-7
  - tests: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27063 - [AMD] Optimize gpt-oss-120B performance

- 链接: https://github.com/sgl-project/sglang/pull/27063
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `1c73ff8ad3fd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1874/-54，可读 patch 2160 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Optimize gpt-oss-120B performance」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「[AMD] Optimize gpt-oss-120B performance」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +74/-5 (79 lines); hunks: -84,6 +84,7; -92,6 +93,7; symbols: forward, _resolve_moe_input_pad_multiple, GptOssSparseMoeBlock, __init__，涉及 `forward, _resolve_moe_input_pad_multiple, GptOssSparseMoeBlock`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +74/-5 (79 lines); hunks: -84,6 +84,7; -92,6 +93,7; symbols: forward, _resolve_moe_input_pad_multiple, GptOssSparseMoeBlock, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -84,6 +84,7 @@
+    is_hip,
@@ -92,6 +93,7 @@
+_is_hip = is_hip()
@@ -165,6 +167,36 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]
+def _resolve_moe_input_pad_multiple(
+    quant_config: Optional[QuantizationConfig],
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +74/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/attention/aiter_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27528 - Fix GPT-OSS MXFP4 hidden size reshape on SM10X

- 链接: https://github.com/sgl-project/sglang/pull/27528
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `dc24a2682190`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GPT-OSS MXFP4 hidden size reshape on SM10X」；模型线: GPT-OSS；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Fix GPT-OSS MXFP4 hidden size reshape on SM10X」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +2/-1 (3 lines); hunks: -208,6 +208,7 @@ def __init__(; -291,7 +292,7 @@ def forward_normal(; symbols: __init__, forward_normal，涉及 `__init__, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +2/-1 (3 lines); hunks: -208,6 +208,7 @@ def __init__(; -291,7 +292,7 @@ def forward_normal(; symbols: __init__, forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -208,6 +208,7 @@ def __init__(
+        self.hidden_size = config.hidden_size
@@ -291,7 +292,7 @@ def forward_normal(
-        hidden_dim_unpadded = self.experts.hidden_size
+        hidden_dim_unpadded = self.hidden_size
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- 链接: https://github.com/sgl-project/sglang/pull/23906
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 160 个文件，+5197/-3068，可读 patch 12233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Cuda Graph Runner/Backend Refactor」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`；技术摘要: 覆盖「[Refactor] Cuda Graph Runner/Backend Refactor」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27964 - [Spec] Retire Spec V1

- 链接: https://github.com/sgl-project/sglang/pull/27964
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 46 个文件，+111/-252，可读 patch 1422 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec] Retire Spec V1」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`；技术摘要: 覆盖「[Spec] Retire Spec V1」；主要实现面是 `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass，涉及 `TestDeepseekMTP, setUpClass, tearDownClass`；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do；`python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family，涉及 `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`；`docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...。
- 代码 diff 细节:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- 关键代码摘录:

```diff
diff -- test/registered/ep/test_deepep_large.py
@@ -3,7 +3,6 @@
-from sglang.srt.environ import envs
@@ -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):
-        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-            cls.process = popen_launch_server(
-                cls.model,
-                cls.base_url,
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx
@@ -1108,7 +1108,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1227,7 +1226,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1351,7 +1349,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1476,7 +1473,6 @@ do
diff -- python/sglang/srt/arg_groups/speculative_hook.py
@@ -1,9 +1,8 @@
```

- 已读文件:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- 验证与风险: diff 自带测试面 `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27941 - Enable PDL for GPT-OSS tinygemm router

- 链接: https://github.com/sgl-project/sglang/pull/27941
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/gpt_oss.py`；关联提交 `cb9140ee6108`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable PDL for GPT-OSS tinygemm router」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Enable PDL for GPT-OSS tinygemm router」；主要实现面是 `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -26,6 +26,7; -161,7 +162,7 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Op...; symbols: forward, _load_normal_weights，涉及 `forward, _load_normal_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -26,6 +26,7; -161,7 +162,7 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Op...; symbols: forward, _load_normal_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -26,6 +26,7 @@
+from sglang.jit_kernel.utils import is_arch_support_pdl
@@ -161,7 +162,7 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]
-            tinygemm_bf16(x, self.weight, out, self.bias)
+            tinygemm_bf16(x, self.weight, out, self.bias, use_pdl=is_arch_support_pdl())
@@ -1094,7 +1095,6 @@ def _load_normal_weights(
-        tp_rank = get_tensor_model_parallel_rank()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/gpt_oss.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: GPT-OSS；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #28697 - [docs] Add B300 cookbook deployment options

- 链接: https://github.com/sgl-project/sglang/pull/28697
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+503/-69，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Add B300 cookbook deployment options」；模型线: GPT-OSS；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；技术摘要: 覆盖「[docs] Add B300 cookbook deployment options」；主要实现面是 `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167；`docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {；`docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {；`docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx
@@ -0,0 +1,167 @@
+export const InternS1Deployment = () => {
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
+      items: [
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx
@@ -9,6 +9,11 @@ const lookupData = {
+      {
+        "id": "b300",
+        "label": "B300",
+        "default": false
+      },
@@ -182,6 +187,66 @@ const lookupData = {
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -4,6 +4,7 @@ export const GLM5Deployment = () => {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
