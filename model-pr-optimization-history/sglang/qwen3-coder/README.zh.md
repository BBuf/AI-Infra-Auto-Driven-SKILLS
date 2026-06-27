# sglang Qwen3 Coder 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` | [#24435](https://github.com/sgl-project/sglang/pull/24435) |
| `docs_new/src/snippets/autoregressive/qwen3-coder-480b-a35b-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` | [#24435](https://github.com/sgl-project/sglang/pull/24435) |
| `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/qwen3_coder_detector.py` | [#8371](https://github.com/sgl-project/sglang/pull/8371), [#16744](https://github.com/sgl-project/sglang/pull/16744) |
| `python/sglang/srt/models/qwen3.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/amd/test_qwen3_coder_next_8gpu.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/ascend/llm_models/test_npu_qwen3_coder_480b_a35b.py` | 无直接 PR 号提交 |
| `test/registered/cpu/test_qwen3.py` | 无直接 PR 号提交 |
| `test/registered/lora/test_lora_qwen3.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 36
- 当前文档总 PR 数: 40
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-22 | [#8260](https://github.com/sgl-project/sglang/pull/8260) | merged | Preliminary Support for Qwen3XMLDetector | `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` |
| 2025-07-25 | [#8357](https://github.com/sgl-project/sglang/pull/8357) | merged | [Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py` |
| 2025-07-28 | [#8224](https://github.com/sgl-project/sglang/pull/8224) | merged | GLM-4.5 Model Support | `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-07-28 | [#8445](https://github.com/sgl-project/sglang/pull/8445) | merged | GLM-4.5 Model Support Follow-up | `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py` |
| 2025-08-08 | [#8371](https://github.com/sgl-project/sglang/pull/8371) | merged | Update qwen3_coder_detector.py for streaming | `python/sglang/srt/function_call/qwen3_coder_detector.py` |
| 2025-11-01 | [#12226](https://github.com/sgl-project/sglang/pull/12226) | merged | Forward unknown tool calls instead of dropping | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py` |
| 2025-11-13 | [#13163](https://github.com/sgl-project/sglang/pull/13163) | merged | Remove EBNF Composer | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py` |
| 2025-11-26 | [#13979](https://github.com/sgl-project/sglang/pull/13979) | open | Add Qwen3-Coder-480B to nightly tests | `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py` |
| 2026-01-19 | [#16744](https://github.com/sgl-project/sglang/pull/16744) | merged | support new qwen3_coder_detector | `python/sglang/srt/function_call/qwen3_coder_detector.py` |
| 2026-01-31 | [#17965](https://github.com/sgl-project/sglang/pull/17965) | merged | [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-02-04 | [#18195](https://github.com/sgl-project/sglang/pull/18195) | merged | Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2 | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2026-02-08 | [#18224](https://github.com/sgl-project/sglang/pull/18224) | merged | [ModelOPT] Support Qwen 3 Next Coder NVFP4 | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-25 | [#18700](https://github.com/sgl-project/sglang/pull/18700) | merged | [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu. | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` |
| 2026-02-25 | [#18355](https://github.com/sgl-project/sglang/pull/18355) | merged | [AMD] Support Qwen3-Coder-Next on AMD platform | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-02 | [#18608](https://github.com/sgl-project/sglang/pull/18608) | merged | [AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU | `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py` |
| 2026-03-03 | [#18882](https://github.com/sgl-project/sglang/pull/18882) | merged | feat: Add FP8 KV cache support for Triton attention backend | `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 2026-03-04 | [#19736](https://github.com/sgl-project/sglang/pull/19736) | merged | [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-04-01 | [#21458](https://github.com/sgl-project/sglang/pull/21458) | merged | [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write | `python/sglang/srt/models/qwen3.py` |
| 2026-04-01 | [#21818](https://github.com/sgl-project/sglang/pull/21818) | merged | [CI] Fix lint that was not applied in #21458 | `python/sglang/srt/models/qwen3.py` |
| 2026-04-01 | [#21829](https://github.com/sgl-project/sglang/pull/21829) | open | [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector | `python/sglang/srt/function_call/qwen3_coder_detector.py` |
| 2026-04-02 | [#21463](https://github.com/sgl-project/sglang/pull/21463) | merged | Migrate all callers from /get_server_info to /server_info | `test/registered/8-gpu-models/test_deepseek_v32_mtp.py`, `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`, `test/registered/amd/test_moriep_small.py` |
| 2026-04-05 | [#22140](https://github.com/sgl-project/sglang/pull/22140) | merged | [Fix] Fix nightly tests | `python/sglang/srt/models/deepseek_v2.py`, `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py`, `test/registered/8-gpu-models/test_qwen3_235b.py` |
| 2026-04-09 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-14 | [#22739](https://github.com/sgl-project/sglang/pull/22739) | merged | Restore Qwen3 rope config fallback | `python/sglang/srt/models/qwen3.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` |
| 2026-04-26 | [#19484](https://github.com/sgl-project/sglang/pull/19484) | merged | [CPU] Add Qwen3.5 model optimization for CPU | `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-05-02 | [#20520](https://github.com/sgl-project/sglang/pull/20520) | merged | [NPU]TP Communications compression For Qwen3 models for NPU | `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2.py` |
| 2026-05-04 | [#21722](https://github.com/sgl-project/sglang/pull/21722) | merged | feat: use structural tags to enable strict tool calling and reasoning for more models | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv4_detector.py`, `python/sglang/srt/function_call/function_call_parser.py` |
| 2026-05-20 | [#25825](https://github.com/sgl-project/sglang/pull/25825) | merged | [Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool | `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py` |
| 2026-05-21 | [#25983](https://github.com/sgl-project/sglang/pull/25983) | merged | feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-05-31 | [#26798](https://github.com/sgl-project/sglang/pull/26798) | merged | Make qwen3's set_embed_and_head idempotent | `python/sglang/srt/models/qwen3.py` |
| 2026-06-01 | [#24435](https://github.com/sgl-project/sglang/pull/24435) | merged | Update Qwen3-Coder docs_new NVIDIA guidance | `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` |
| 2026-06-03 | [#27001](https://github.com/sgl-project/sglang/pull/27001) | merged | [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests | `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` |
| 2026-06-06 | [#27248](https://github.com/sgl-project/sglang/pull/27248) | merged | [Doc][CPU]Update Cookbook with Xeon support info | `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-11 | [#13411](https://github.com/sgl-project/sglang/pull/13411) | closed | Improve Qwen3CoderDetector with schema-aware parameter type conversion | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-20 | [#28810](https://github.com/sgl-project/sglang/pull/28810) | merged | [CI] Remove deprecated test/srt legacy CI setup | `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py` |

## 逐 PR diff 审计卡

### PR #8260 - Preliminary Support for Qwen3XMLDetector

- 链接: https://github.com/sgl-project/sglang/pull/8260
- 状态/时间: merged / 2025-07-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+153/-0，可读 patch 175 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Preliminary Support for Qwen3XMLDetector」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Preliminary Support for Qwen3XMLDetector」；主要实现面是 `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: _safe_val, Qwen3XMLDetector, __init__, has_tool_call，涉及 `_safe_val, Qwen3XMLDetector, __init__`；`python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: -14,6 +14,7; -35,6 +36,7 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__，涉及 `FunctionCallParser, __init__`；`python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: -1099,6 +1099,7 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args，涉及 `add_cli_args`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: _safe_val, Qwen3XMLDetector, __init__, has_tool_call
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: -14,6 +14,7; -35,6 +36,7 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__
  - `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: -1099,6 +1099,7 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/qwen3_detector.py
@@ -0,0 +1,150 @@
+import ast
+import html
+import json
+import logging
+import re
+from typing import Any, Dict, List, Tuple
diff -- python/sglang/srt/function_call/function_call_parser.py
@@ -14,6 +14,7 @@
+from sglang.srt.function_call.qwen3_detector import Qwen3XMLDetector
@@ -35,6 +36,7 @@ class FunctionCallParser:
+        "qwen3": Qwen3XMLDetector,
diff -- python/sglang/srt/server_args.py
@@ -1099,6 +1099,7 @@ def add_cli_args(parser: argparse.ArgumentParser):
+                "qwen3",
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0; `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0; `python/sglang/srt/server_args.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8357 - [Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector

- 链接: https://github.com/sgl-project/sglang/pull/8357
- 状态/时间: merged / 2025-07-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+574/-83，可读 patch 868 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`；技术摘要: 覆盖「[Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector」；主要实现面是 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/test_function_call_parser.py` modified +455/-0 (455 lines); hunks: -10,6 +10,7; -507,6 +508,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_qwen3_coder_detector_ebnf，涉及 `setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf`；`python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63 (158 lines); hunks: -1,51 +1,73; -55,19 +77,20 @@ class EBNFComposer:; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val，涉及 `EBNFComposer, get_value_rule, _handle_enum`；`python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9 (19 lines); hunks: -9,7 +9,6; -29,7 +28,7 @@ def _safe_val(raw: str) -> Any:; symbols: _safe_val, Qwen3XMLDetector, Qwen3CoderDetector, _parse_block，涉及 `_safe_val, Qwen3XMLDetector, Qwen3CoderDetector`；`python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5 (9 lines); hunks: -8,7 +8,6; -216,11 +215,11 @@ def _get_parameter_value(self, val):; symbols: _get_parameter_value, structure_info, info, supports_structural_tag，涉及 `_get_parameter_value, structure_info, info`。
- 代码 diff 细节:
  - `test/srt/test_function_call_parser.py` modified +455/-0 (455 lines); hunks: -10,6 +10,7; -507,6 +508,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_qwen3_coder_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63 (158 lines); hunks: -1,51 +1,73; -55,19 +77,20 @@ class EBNFComposer:; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9 (19 lines); hunks: -9,7 +9,6; -29,7 +28,7 @@ def _safe_val(raw: str) -> Any:; symbols: _safe_val, Qwen3XMLDetector, Qwen3CoderDetector, _parse_block
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5 (9 lines); hunks: -8,7 +8,6; -216,11 +215,11 @@ def _get_parameter_value(self, val):; symbols: _get_parameter_value, structure_info, info, supports_structural_tag
  - `python/sglang/srt/function_call/function_call_parser.py` modified +4/-4 (8 lines); hunks: -14,7 +14,7; -36,7 +36,7 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__, get_structure_constraint
- 关键代码摘录:

```diff
diff -- test/srt/test_function_call_parser.py
@@ -10,6 +10,7 @@
+from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
@@ -507,6 +508,7 @@ def setUp(self):
+        self.qwen3_coder_detector = Qwen3CoderDetector()
@@ -620,6 +622,26 @@ def test_qwen25_detector_ebnf(self):
+    def test_qwen3_coder_detector_ebnf(self):
+        """Test that the Qwen3CoderDetector generates valid EBNF."""
diff -- python/sglang/srt/function_call/ebnf_composer.py
@@ -1,51 +1,73 @@
-from typing import Literal, Optional
+from typing import Any, Dict, Literal, Optional
-    json_grammar_ebnf_str = r"""
-        json ::= basic_array | basic_object
-        basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
-        basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
diff -- python/sglang/srt/function_call/qwen3_coder_detector.py
@@ -9,7 +9,6 @@
```

- 已读文件:
  - tests: `test/srt/test_function_call_parser.py` modified +455/-0
  - runtime: `python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63; `python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9; `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5; `python/sglang/srt/function_call/function_call_parser.py` modified +4/-4; `python/sglang/srt/function_call/base_format_detector.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8224 - GLM-4.5 Model Support

- 链接: https://github.com/sgl-project/sglang/pull/8224
- 状态/时间: merged / 2025-07-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1673/-7，可读 patch 1853 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.5 Model Support」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「GLM-4.5 Model Support」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunks: -0,0 +1,1034; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention，涉及 `Glm4MoeMLP, __init__, forward`；`test/srt/test_function_call_parser.py` modified +184/-0 (184 lines); hunks: -6,6 +6,7; -510,6 +511,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf，涉及 `setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf`；`python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN，涉及 `Glm4MoeModelNextN, __init__, forward`；`python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__，涉及 `get_argument_type, parse_arguments, Glm4MoeDetector`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunks: -0,0 +1,1034; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention
  - `test/srt/test_function_call_parser.py` modified +184/-0 (184 lines); hunks: -6,6 +6,7; -510,6 +511,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf
  - `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN
  - `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
  - `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +39/-1 (40 lines); hunks: -223,7 +223,10 @@ def test_function_calling_streaming_simple(self):; -910,5 +913,40 @@ def test_pythonic_tool_call_streaming(self):; symbols: test_function_calling_streaming_simple, test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -0,0 +1,1034 @@
+# Copyright 2025-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- test/srt/test_function_call_parser.py
@@ -6,6 +6,7 @@
+from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
@@ -510,6 +511,7 @@ def setUp(self):
+        self.glm45_detector = Glm4MoeDetector()
@@ -622,6 +624,29 @@ def test_qwen25_detector_ebnf(self):
+    def test_glm45_detector_ebnf(self):
+        """Test that the Glm4MoeDetector generates valid EBNF."""
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -0,0 +1,167 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` added +1034/-0; `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0; `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0; `python/sglang/srt/function_call/ebnf_composer.py` modified +10/-3; `python/sglang/srt/configs/model_config.py` modified +3/-0; `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0
  - tests: `test/srt/test_function_call_parser.py` modified +184/-0; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +39/-1
- 验证与风险: diff 自带测试面 `test/srt/openai_server/features/test_enable_thinking.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8445 - GLM-4.5 Model Support Follow-up

- 链接: https://github.com/sgl-project/sglang/pull/8445
- 状态/时间: merged / 2025-07-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+44/-15，可读 patch 168 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.5 Model Support Follow-up」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`；技术摘要: 覆盖「GLM-4.5 Model Support Follow-up」；主要实现面是 `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10 (49 lines); hunks: -135,7 +135,7 @@ def get_test_messages(self):; -203,7 +203,7 @@ def test_tool_choice_auto_non_streaming(self):; symbols: get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming, test_tool_choice_required_non_streaming，涉及 `get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunks: -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf，涉及 `build_ebnf`；`test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1 (2 lines); hunks: -913,7 +913,7 @@ def test_pythonic_tool_call_streaming(self):; symbols: test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass，涉及 `test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass`；`test/srt/test_function_call_parser.py` modified +1/-1 (2 lines); hunks: -2068,7 +2068,7 @@ def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id，涉及 `test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id`。
- 代码 diff 细节:
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10 (49 lines); hunks: -135,7 +135,7 @@ def get_test_messages(self):; -203,7 +203,7 @@ def test_tool_choice_auto_non_streaming(self):; symbols: get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming, test_tool_choice_required_non_streaming
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunks: -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf
  - `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1 (2 lines); hunks: -913,7 +913,7 @@ def test_pythonic_tool_call_streaming(self):; symbols: test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass
  - `test/srt/test_function_call_parser.py` modified +1/-1 (2 lines); hunks: -2068,7 +2068,7 @@ def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +1/-0 (1 lines); hunks: -148,4 +148,5 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf
- 关键代码摘录:

```diff
diff -- test/srt/openai_server/function_call/test_tool_choice.py
@@ -135,7 +135,7 @@ def get_test_messages(self):
-                "content": "Answer the following questions as best you can:\n\nYou will be given a trace of thinking process in the following format.\n\nQuestion: the input questi
+                "content": "Answer the following questions as best you can:\n\nYou will be given a trace of thinking process in the following format.\n\nQuestion: the input questi
@@ -203,7 +203,7 @@ def test_tool_choice_auto_non_streaming(self):
-            max_tokens=400,
+            max_tokens=2048,
@@ -220,7 +220,7 @@ def test_tool_choice_auto_streaming(self):
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):
-            # GLM4Moe is not compatible with multiple tool_calls under tool_choice condition: it will output unlimited tool_calls...
-            # tool_call_separator="\\n",
+            tool_call_separator="\\n",
diff -- test/srt/openai_server/function_call/test_openai_function_calling.py
@@ -913,7 +913,7 @@ def test_pythonic_tool_call_streaming(self):
-## Skip for ci test
+# Skip for ci test
diff -- test/srt/test_function_call_parser.py
```

- 已读文件:
  - tests: `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1; `test/srt/test_function_call_parser.py` modified +1/-1; `test/srt/openai_server/features/test_enable_thinking.py` modified +1/-1
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2; `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/srt/openai_server/features/test_enable_thinking.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8371 - Update qwen3_coder_detector.py for streaming

- 链接: https://github.com/sgl-project/sglang/pull/8371
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/qwen3_coder_detector.py`；关联提交 `b3359dc9bf5b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+348/-67，可读 patch 510 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update qwen3_coder_detector.py for streaming」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`；技术摘要: 覆盖「Update qwen3_coder_detector.py for streaming」；主要实现面是 `python/sglang/srt/function_call/qwen3_coder_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9 (228 lines); hunks: -57,6 +57,15 @@ def __init__(self):; -70,23 +79,224 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, _parse_and_stream_parameters，涉及 `__init__, has_tool_call, parse_streaming_increment`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9 (228 lines); hunks: -57,6 +57,15 @@ def __init__(self):; -70,23 +79,224 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, _parse_and_stream_parameters
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/qwen3_coder_detector.py
@@ -57,6 +57,15 @@ def __init__(self):
+        # Streaming state variables
+        self._current_function_name: str = ""
+        self._current_parameters: Dict[str, Any] = {}
+        self._streamed_parameters: Dict[str, str] = (
+            {}
+        )  # Track what parameter content we've streamed
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12226 - Forward unknown tool calls instead of dropping

- 链接: https://github.com/sgl-project/sglang/pull/12226
- 状态/时间: merged / 2025-11-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+145/-60，可读 patch 279 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Forward unknown tool calls instead of dropping」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`；技术摘要: 覆盖「Forward unknown tool calls instead of dropping」；主要实现面是 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37 (78 lines); hunks: -6,6 +6,7; -120,45 +121,48 @@ def parse_streaming_increment(; symbols: parse_streaming_increment，涉及 `parse_streaming_increment`；`test/srt/function_call/test_unknown_tool_name.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: DummyDetector, has_tool_call, detect_and_parse, test_unknown_tool_name_dropped_default，涉及 `DummyDetector, has_tool_call, detect_and_parse`；`python/sglang/srt/function_call/base_format_detector.py` modified +15/-12 (27 lines); hunks: -8,6 +8,7; -75,19 +76,21 @@ def parse_base_json(self, action: Any, tools: List[Tool]) ->...; symbols: parse_base_json，涉及 `parse_base_json`；`python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1 (5 lines); hunks: -5,6 +5,7; -91,7 +92,9 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: detect_and_parse，涉及 `detect_and_parse`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37 (78 lines); hunks: -6,6 +6,7; -120,45 +121,48 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
  - `test/srt/function_call/test_unknown_tool_name.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: DummyDetector, has_tool_call, detect_and_parse, test_unknown_tool_name_dropped_default
  - `python/sglang/srt/function_call/base_format_detector.py` modified +15/-12 (27 lines); hunks: -8,6 +8,7; -75,19 +76,21 @@ def parse_base_json(self, action: Any, tools: List[Tool]) ->...; symbols: parse_base_json
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1 (5 lines); hunks: -5,6 +5,7; -91,7 +92,9 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: detect_and_parse
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +3/-1 (4 lines); hunks: -4,6 +4,7; -220,7 +221,8 @@ def _extract_tool_call_from_event(; symbols: _extract_tool_call_from_event
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/qwen3_coder_detector.py
@@ -6,6 +6,7 @@
+from sglang.srt.environ import envs
@@ -120,45 +121,48 @@ def parse_streaming_increment(
-                    if function_name in self._tool_indices:
-                        self._current_function_name = function_name
-                        self._function_name_sent = True
-                        # Initialize tool call tracking
diff -- test/srt/function_call/test_unknown_tool_name.py
@@ -0,0 +1,69 @@
+import json
+import logging
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.environ import envs
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
+from sglang.srt.function_call.core_types import StreamingParseResult
diff -- python/sglang/srt/function_call/base_format_detector.py
@@ -8,6 +8,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37; `python/sglang/srt/function_call/base_format_detector.py` modified +15/-12; `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1; `python/sglang/srt/function_call/gpt_oss_detector.py` modified +3/-1; `python/sglang/srt/environ.py` modified +3/-0
  - tests: `test/srt/function_call/test_unknown_tool_name.py` added +69/-0
  - docs: `docs/references/environment_variables.md` modified +10/-9
- 验证与风险: diff 自带测试面 `test/srt/function_call/test_unknown_tool_name.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13163 - Remove EBNF Composer

- 链接: https://github.com/sgl-project/sglang/pull/13163
- 状态/时间: merged / 2025-11-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+6/-1081，可读 patch 1270 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove EBNF Composer」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`；技术摘要: 覆盖「Remove EBNF Composer」；主要实现面是 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/test_function_call_parser.py` modified +5/-459 (464 lines); hunks: -1,8 +1,6; -458,452 +456,6 @@ def test_detect_and_parse_with_text_before_tool_call(self):; symbols: test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp, test_pythonic_detector_ebnf，涉及 `test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp`；`python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344 (344 lines); hunks: -1,344 +0,0; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val，涉及 `EBNFComposer, get_value_rule, _handle_enum`；`test/srt/function_call/test_json_schema_constraint.py` modified +0/-52 (52 lines); hunks: -222,58 +222,6 @@ def test_tools_without_parameters(self):; symbols: test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror，涉及 `test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror`；`python/sglang/srt/function_call/function_call_parser.py` modified +0/-38 (38 lines); hunks: -195,41 +195,3 @@ def get_structure_constraint(; symbols: get_structure_constraint, get_ebnf，涉及 `get_structure_constraint, get_ebnf`。
- 代码 diff 细节:
  - `test/srt/test_function_call_parser.py` modified +5/-459 (464 lines); hunks: -1,8 +1,6; -458,452 +456,6 @@ def test_detect_and_parse_with_text_before_tool_call(self):; symbols: test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp, test_pythonic_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344 (344 lines); hunks: -1,344 +0,0; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val
  - `test/srt/function_call/test_json_schema_constraint.py` modified +0/-52 (52 lines); hunks: -222,58 +222,6 @@ def test_tools_without_parameters(self):; symbols: test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror
  - `python/sglang/srt/function_call/function_call_parser.py` modified +0/-38 (38 lines); hunks: -195,41 +195,3 @@ def get_structure_constraint(; symbols: get_structure_constraint, get_ebnf
  - `python/sglang/srt/function_call/step3_detector.py` modified +0/-29 (29 lines); hunks: -11,7 +11,6; -406,31 +405,3 @@ def supports_structural_tag(self) -> bool:; symbols: supports_structural_tag, structure_info, build_ebnf
- 关键代码摘录:

```diff
diff -- test/srt/test_function_call_parser.py
@@ -1,8 +1,6 @@
-from xgrammar import GrammarCompiler, TokenizerInfo
@@ -458,452 +456,6 @@ def test_detect_and_parse_with_text_before_tool_call(self):
-class TestEBNFGeneration(unittest.TestCase):
-    def setUp(self):
-        # Create sample tools for testing
-        self.tools = [
diff -- python/sglang/srt/function_call/ebnf_composer.py
@@ -1,344 +0,0 @@
-from typing import Any, Dict, Literal, Optional
-class EBNFComposer:
-    # Adapted from https://xgrammar.mlc.ai/docs/how_to/ebnf_guided_generation.html#try-out-via-hf-transformers
-    # Shared primitive grammar rules used across all formats
-    BASE_PRIMITIVE_GRAMMAR = r"""
-        basic_string ::= (([\"] basic_string_1 [\"]))
diff -- test/srt/function_call/test_json_schema_constraint.py
@@ -222,58 +222,6 @@ def test_tools_without_parameters(self):
```

- 已读文件:
  - tests: `test/srt/test_function_call_parser.py` modified +5/-459; `test/srt/function_call/test_json_schema_constraint.py` modified +0/-52
  - runtime: `python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344; `python/sglang/srt/function_call/function_call_parser.py` modified +0/-38; `python/sglang/srt/function_call/step3_detector.py` modified +0/-29; `python/sglang/srt/function_call/base_format_detector.py` modified +0/-27; `python/sglang/srt/function_call/kimik2_detector.py` modified +0/-19; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-13
- 验证与风险: diff 自带测试面 `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13979 - Add Qwen3-Coder-480B to nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/13979
- 状态/时间: open / 2025-11-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+288/-171，可读 patch 521 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Qwen3-Coder-480B to nightly tests」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`；技术摘要: 覆盖「Add Qwen3-Coder-480B to nightly tests」；主要实现面是 `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/nightly-test-nvidia.yml` modified +232/-170 (402 lines); hunks: -72,89 +72,118 @@ jobs:; -370,119 +399,152 @@ jobs:；`test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch，涉及 `TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch`；`test/nightly/nightly_utils.py` modified +3/-1 (4 lines); hunks: -211,6 +211,7 @@ def run_benchmark_for_model(; -228,6 +229,7 @@ def run_benchmark_for_model(; symbols: run_benchmark_for_model，涉及 `run_benchmark_for_model`。
- 代码 diff 细节:
  - `.github/workflows/nightly-test-nvidia.yml` modified +232/-170 (402 lines); hunks: -72,89 +72,118 @@ jobs:; -370,119 +399,152 @@ jobs:
  - `test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch
  - `test/nightly/nightly_utils.py` modified +3/-1 (4 lines); hunks: -211,6 +211,7 @@ def run_benchmark_for_model(; -228,6 +229,7 @@ def run_benchmark_for_model(; symbols: run_benchmark_for_model
- 关键代码摘录:

```diff
diff -- .github/workflows/nightly-test-nvidia.yml
@@ -72,89 +72,118 @@ jobs:
-      - name: Run test
-        timeout-minutes: 30
-        env:
-          GPU_CONFIG: "8-gpu-h200"
-        run: |
-          cd test
diff -- test/nightly/test_qwen3_coder_480b_perf.py
@@ -0,0 +1,53 @@
+import unittest
+from nightly_utils import NightlyBenchmarkRunner
+from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env
+QWEN3_CODER_480B_MODEL_PATH = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
+PROFILE_DIR = "performance_profiles_qwen3_coder_480b"
+class TestNightlyQwen3Coder480BPerformance(unittest.TestCase):
diff -- test/nightly/nightly_utils.py
@@ -211,6 +211,7 @@ def run_benchmark_for_model(
```

- 已读文件:
  - ci: `.github/workflows/nightly-test-nvidia.yml` modified +232/-170
  - tests: `test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0; `test/nightly/nightly_utils.py` modified +3/-1
- 验证与风险: diff 自带测试面 `test/nightly/nightly_utils.py`, `test/nightly/test_qwen3_coder_480b_perf.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16744 - support new qwen3_coder_detector

- 链接: https://github.com/sgl-project/sglang/pull/16744
- 状态/时间: merged / 2026-01-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/qwen3_coder_detector.py`；关联提交 `858a4d659b3e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+637/-667，可读 patch 1493 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support new qwen3_coder_detector」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`；技术摘要: 覆盖「support new qwen3_coder_detector」；主要实现面是 `python/sglang/srt/function_call/qwen3_coder_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271 (663 lines); hunks: -1,12 +1,10; -17,334 +15,457; symbols: _safe_val, Qwen3CoderDetector, __init__, already，涉及 `_safe_val, Qwen3CoderDetector, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271 (663 lines); hunks: -1,12 +1,10; -17,334 +15,457; symbols: _safe_val, Qwen3CoderDetector, __init__, already
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/qwen3_coder_detector.py
@@ -1,12 +1,10 @@
-import html
-from typing import Any, Dict, List, Tuple
+from typing import Any, List, Optional
-from sglang.srt.environ import envs
@@ -17,334 +15,457 @@
-def _safe_val(raw: str) -> Any:
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17965 - [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB

- 链接: https://github.com/sgl-project/sglang/pull/17965
- 状态/时间: merged / 2026-01-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+573/-16，可读 patch 705 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`；技术摘要: 覆盖「[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0 (128 lines); hunks: -0,0 +1,128；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0 (114 lines); hunks: -0,0 +1,114。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0 (128 lines); hunks: -0,0 +1,128
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0 (114 lines); hunks: -0,0 +1,114
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +4/-16 (20 lines); hunks: -8,6 +8,7; -21,7 +22,6; symbols: support_tensor_descriptor, should_enable_swap_ab, is_h20_device_and_sm90_supported
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
@@ -0,0 +1,128 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +4/-16
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +17/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18195 - Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2

- 链接: https://github.com/sgl-project/sglang/pull/18195
- 状态/时间: merged / 2026-02-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/18224
- 状态/时间: merged / 2026-02-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+35/-6，可读 patch 95 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ModelOPT] Support Qwen 3 Next Coder NVFP4」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[ModelOPT] Support Qwen 3 Next Coder NVFP4」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM，涉及 `__init__, HybridLayerType, Qwen3NextForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -665,6 +665,7 @@ def __init__(
+            quant_config=quant_config,
@@ -921,6 +922,15 @@ class HybridLayerType(enum.Enum):
+    # Map fused module names to their checkpoint (unfused) counterparts.
+    # This is needed so the quantization exclusion logic can match
+    # checkpoint-style names (e.g. "q_proj") against the fused sglang
+    # module names (e.g. "qkv_proj").
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +35/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18700 - [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu.

- 链接: https://github.com/sgl-project/sglang/pull/18700
- 状态/时间: merged / 2026-02-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+3/-3，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu.」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`；技术摘要: 覆盖「[NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu.」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7；`python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ def npu_fused_moe_without_routing_weights_bf16(; -129,7 +129,7 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: npu_fused_moe_without_routing_weights_bf16，涉及 `npu_fused_moe_without_routing_weights_bf16`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7
  - `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ def npu_fused_moe_without_routing_weights_bf16(; -129,7 +129,7 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: npu_fused_moe_without_routing_weights_bf16
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -43,7 +43,7 @@
-if not is_cpu() and not is_npu():
+if not is_cpu():
diff -- python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py
@@ -118,7 +118,7 @@ def npu_fused_moe_without_routing_weights_bf16(
-        weight=[layer.w13_weight.permute(0, 2, 1)],
+        weight=[layer.w13_weight],
@@ -129,7 +129,7 @@ def npu_fused_moe_without_routing_weights_bf16(
-        weight=[layer.w2_weight.permute(0, 2, 1)],
+        weight=[layer.w2_weight],
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1; `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- 链接: https://github.com/sgl-project/sglang/pull/18355
- 状态/时间: merged / 2026-02-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+213/-74，可读 patch 395 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Support Qwen3-Coder-Next on AMD platform」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[AMD] Support Qwen3-Coder-Next on AMD platform」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata，涉及 `ForwardMetadata, __init__, init_forward_metadata`；`python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj，涉及 `_forward_input_proj`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -89,6 +89,9 @@ class ForwardMetadata:
+    custom_mask: Optional[torch.Tensor] = None
+    mask_indptr: Optional[torch.Tensor] = None
+    max_extend_len: Optional[int] = None
@@ -123,7 +126,6 @@ def __init__(
-        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
@@ -133,6 +135,21 @@ def __init__(
diff -- python/sglang/srt/models/qwen3_next.py
@@ -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):
-            seq_len < DUAL_STREAM_TOKEN_THRESHOLD
-            and self.alt_stream is not None
+            self.alt_stream is not None
+            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18608 - [AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU

- 链接: https://github.com/sgl-project/sglang/pull/18608
- 状态/时间: merged / 2026-03-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`；关联提交 `98f47d817583`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+486/-0，可读 patch 488 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`；技术摘要: 覆盖「[AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: get_model_path, ModelConfig, __post_init__, get_display_name，涉及 `get_model_path, ModelConfig, __post_init__`；`test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: TestQwen3CoderNext, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestQwen3CoderNext, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: TestQwen3CoderNext, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py
@@ -0,0 +1,302 @@
+"""MI35x Qwen3-Coder-Next GSM8K Completion Evaluation Test (8-GPU)
+Tests Qwen3-Coder-Next model with basic and MTP configurations
+using few-shot completion benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-qwen3-coder-next suite
+"""
+import ast
diff -- test/registered/amd/test_qwen3_coder_next_8gpu.py
@@ -0,0 +1,184 @@
+"""MI35x Qwen3-Coder-Next Functionality Test (8-GPU)
+Tests Qwen3-Coder-Next model with basic configuration
+on MI35x. Covers GSM8K accuracy and BS=1 decode speed.
+Server args match run_qwen3-coder-next_spec.sh.
+Registry: stage-c-test-large-8-gpu-amd-mi35x-qwen3-coder-next suite
+"""
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0; `test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18882 - feat: Add FP8 KV cache support for Triton attention backend

- 链接: https://github.com/sgl-project/sglang/pull/18882
- 状态/时间: merged / 2026-03-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+180/-27，可读 patch 564 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: Add FP8 KV cache support for Triton attention backend」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`；技术摘要: 覆盖「feat: Add FP8 KV cache support for Triton attention backend」；主要实现面是 `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6 (69 lines); hunks: -7,6 +7,7; -86,6 +87,7 @@ def __init__(; symbols: __init__, forward_extend, _forward_extend_unified，涉及 `__init__, forward_extend, _forward_extend_unified`；`python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15 (41 lines); hunks: -46,7 +46,7 @@ def _fwd_kernel_stage1(; -124,7 +124,7 @@ def _fwd_kernel_stage1(; symbols: _fwd_kernel_stage1, _decode_att_m_fwd, _fwd_grouped_kernel_stage1，涉及 `_fwd_kernel_stage1, _decode_att_m_fwd, _fwd_grouped_kernel_stage1`；`python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6 (22 lines); hunks: -232,6 +232,8 @@ def _fwd_kernel(; -386,7 +388,7 @@ def _fwd_kernel(; symbols: _fwd_kernel, extend_attention_fwd，涉及 `_fwd_kernel, extend_attention_fwd`；`test/registered/quant/test_fp8kv_triton.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: TestFP8KVCacheTritonBackend, setUpClass, tearDownClass, test_gsm8k，涉及 `TestFP8KVCacheTritonBackend, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6 (69 lines); hunks: -7,6 +7,7; -86,6 +87,7 @@ def __init__(; symbols: __init__, forward_extend, _forward_extend_unified
  - `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15 (41 lines); hunks: -46,7 +46,7 @@ def _fwd_kernel_stage1(; -124,7 +124,7 @@ def _fwd_kernel_stage1(; symbols: _fwd_kernel_stage1, _decode_att_m_fwd, _fwd_grouped_kernel_stage1
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6 (22 lines); hunks: -232,6 +232,8 @@ def _fwd_kernel(; -386,7 +388,7 @@ def _fwd_kernel(; symbols: _fwd_kernel, extend_attention_fwd
  - `test/registered/quant/test_fp8kv_triton.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: TestFP8KVCacheTritonBackend, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/attention/test_triton_attention_kernels.py` modified +14/-0 (14 lines); hunks: -251,6 +251,8 @@ def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; -286,6 +288,8 @@ def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; symbols: _test_extend_attention_once, _test_extend_attention_sliding_window_once, _test_decode_attention_once, _test_grouped_decode_attention_once
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/triton_backend.py
@@ -7,6 +7,7 @@
+from sglang.srt.configs.model_config import AttentionArch
@@ -86,6 +87,7 @@ def __init__(
+        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
@@ -813,9 +815,24 @@ def forward_extend(
-            forward_batch.token_to_kv_pool.set_kv_buffer(
-                layer, forward_batch.out_cache_loc, k, v
diff -- python/sglang/srt/layers/attention/triton_ops/decode_attention.py
@@ -46,7 +46,7 @@ def _fwd_kernel_stage1(
-    sm_scale,
+    sm_scale_withk,
@@ -124,7 +124,7 @@ def _fwd_kernel_stage1(
-            qk *= sm_scale
+            qk *= sm_scale_withk
@@ -189,7 +189,7 @@ def _decode_att_m_fwd(
diff -- python/sglang/srt/layers/attention/triton_ops/extend_attention.py
@@ -232,6 +232,8 @@ def _fwd_kernel(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6; `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15; `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6
  - tests: `test/registered/quant/test_fp8kv_triton.py` added +58/-0; `test/registered/attention/test_triton_attention_kernels.py` modified +14/-0; `test/registered/attention/test_wave_attention_kernels.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/registered/attention/test_triton_attention_kernels.py`, `test/registered/attention/test_wave_attention_kernels.py`, `test/registered/quant/test_fp8kv_triton.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19736 - [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend

- 链接: https://github.com/sgl-project/sglang/pull/19736
- 状态/时间: merged / 2026-03-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`；技术摘要: 覆盖「[AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0 (2 lines); hunks: -1765,6 +1765,8 @@ def forward_extend(; symbols: forward_extend，涉及 `forward_extend`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0 (2 lines); hunks: -1765,6 +1765,8 @@ def forward_extend(; symbols: forward_extend
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -1765,6 +1765,8 @@ def forward_extend(
+                    1.0,  # k_scale
+                    1.0,  # v_scale
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- 链接: https://github.com/sgl-project/sglang/pull/21458
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+101/-3，可读 patch 152 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3.py`；技术摘要: 覆盖「[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write」；主要实现面是 `python/sglang/srt/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope，涉及 `__init__, forward_prepare_native, forward_prepare_npu`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -19,6 +19,7 @@
+from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
@@ -30,13 +31,25 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu
+from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu
+_is_hip = is_hip()
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +101/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21818 - [CI] Fix lint that was not applied in #21458

- 链接: https://github.com/sgl-project/sglang/pull/21818
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Fix lint that was not applied in #21458」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`；技术摘要: 覆盖「[CI] Fix lint that was not applied in #21458」；主要实现面是 `python/sglang/srt/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def forward_prepare_npu(self, positions, hidden_states, forw...; symbols: forward_prepare_npu, forward_prepare_aiter_fused_mrope，涉及 `forward_prepare_npu, forward_prepare_aiter_fused_mrope`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def forward_prepare_npu(self, positions, hidden_states, forw...; symbols: forward_prepare_npu, forward_prepare_aiter_fused_mrope
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -198,7 +198,9 @@ def forward_prepare_npu(self, positions, hidden_states, forward_batch):
-    def forward_prepare_aiter_fused_mrope(self, positions, hidden_states, forward_batch):
+    def forward_prepare_aiter_fused_mrope(
+        self, positions, hidden_states, forward_batch
+    ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21829 - [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector

- 链接: https://github.com/sgl-project/sglang/pull/21829
- 状态/时间: open / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+143/-0，可读 patch 171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`；技术摘要: 覆盖「[Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector」；主要实现面是 `python/sglang/srt/function_call/qwen3_coder_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +143/-0 (143 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,54 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param，涉及 `__init__, has_tool_call, _convert_param_value`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +143/-0 (143 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,54 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/qwen3_coder_detector.py
@@ -54,6 +54,13 @@ def __init__(self):
+        # Incremental parameter streaming state
+        # When a string parameter value is very long (e.g. code), we stream it
+        # incrementally instead of waiting for the complete </parameter> tag.
+        self._streaming_param_active: bool = False
+        self._streaming_param_emitted: int = 0  # chars processed in rest_of_slice
+        self._streaming_param_leading_checked: bool = False
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +143/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/qwen3_coder_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21463 - Migrate all callers from /get_server_info to /server_info

- 链接: https://github.com/sgl-project/sglang/pull/21463
- 状态/时间: merged / 2026-04-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 48 个文件，+74/-70，可读 patch 630 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Migrate all callers from /get_server_info to /server_info」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_deepseek_v32_mtp.py`, `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`, `test/registered/amd/test_moriep_small.py`；技术摘要: 覆盖「Migrate all callers from /get_server_info to /server_info」；主要实现面是 `test/registered/8-gpu-models/test_deepseek_v32_mtp.py`, `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`, `test/registered/amd/test_moriep_small.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_deepseek_v32_mtp.py` modified +4/-4 (8 lines); hunks: -76,7 +76,7 @@ def test_a_gsm8k(; -163,7 +163,7 @@ def test_a_gsm8k(; symbols: test_a_gsm8k，涉及 `test_a_gsm8k`；`sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` modified +4/-2 (6 lines); hunks: -269,6 +269,8 @@ async def flush_cache():; -277,10 +279,10 @@ async def get_server_info():; symbols: flush_cache, get_server_info，涉及 `flush_cache, get_server_info`；`test/registered/amd/test_moriep_small.py` modified +3/-3 (6 lines); hunks: -145,7 +145,7 @@ def test_gsm8k(; -397,7 +397,7 @@ def test_gsm8k(; symbols: test_gsm8k，涉及 `test_gsm8k`；`test/registered/distributed/test_data_parallelism.py` modified +3/-3 (6 lines); hunks: -57,13 +57,13 @@ def test_update_weight(self):; symbols: test_update_weight, test_get_memory_pool_size，涉及 `test_update_weight, test_get_memory_pool_size`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_deepseek_v32_mtp.py` modified +4/-4 (8 lines); hunks: -76,7 +76,7 @@ def test_a_gsm8k(; -163,7 +163,7 @@ def test_a_gsm8k(; symbols: test_a_gsm8k
  - `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` modified +4/-2 (6 lines); hunks: -269,6 +269,8 @@ async def flush_cache():; -277,10 +279,10 @@ async def get_server_info():; symbols: flush_cache, get_server_info
  - `test/registered/amd/test_moriep_small.py` modified +3/-3 (6 lines); hunks: -145,7 +145,7 @@ def test_gsm8k(; -397,7 +397,7 @@ def test_gsm8k(; symbols: test_gsm8k
  - `test/registered/distributed/test_data_parallelism.py` modified +3/-3 (6 lines); hunks: -57,13 +57,13 @@ def test_update_weight(self):; symbols: test_update_weight, test_get_memory_pool_size
  - `test/registered/ep/test_deepep_small.py` modified +3/-3 (6 lines); hunks: -412,7 +412,7 @@ def test_gsm8k(self):; -486,7 +486,7 @@ def test_gsm8k(self):; symbols: test_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_deepseek_v32_mtp.py
@@ -76,7 +76,7 @@ def test_a_gsm8k(
-        server_info = requests.get(self.base_url + "/get_server_info")
+        server_info = requests.get(self.base_url + "/server_info")
@@ -163,7 +163,7 @@ def test_a_gsm8k(
-        server_info = requests.get(self.base_url + "/get_server_info")
+        server_info = requests.get(self.base_url + "/server_info")
@@ -246,7 +246,7 @@ def test_a_gsm8k(
diff -- sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py
@@ -269,6 +269,8 @@ async def flush_cache():
+# TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
+@app.get("/server_info")
@@ -277,10 +279,10 @@ async def get_server_info():
-            server_info = await session.get(f"{server}/get_server_info")
+            server_info = await session.get(f"{server}/server_info")
-            server_info = await session.get(f"{server}/get_server_info")
diff -- test/registered/amd/test_moriep_small.py
@@ -145,7 +145,7 @@ def test_gsm8k(
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_deepseek_v32_mtp.py` modified +4/-4; `test/registered/amd/test_moriep_small.py` modified +3/-3; `test/registered/distributed/test_data_parallelism.py` modified +3/-3; `test/registered/ep/test_deepep_small.py` modified +3/-3
  - other: `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` modified +4/-2
  - docs: `docs/advanced_features/sgl_model_gateway.md` modified +2/-2
  - runtime: `python/sglang/bench_serving.py` modified +2/-2; `python/sglang/lang/backend/runtime_endpoint.py` modified +2/-2
- 验证与风险: diff 自带测试面 `python/sglang/test/bench_one_batch_server_internal.py`, `python/sglang/test/kits/cache_hit_kit.py`, `python/sglang/test/kl_test_utils.py`, `python/sglang/test/nightly_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22140 - [Fix] Fix nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/22140
- 状态/时间: merged / 2026-04-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+12/-12，可读 patch 108 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Fix nightly tests」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`；技术摘要: 覆盖「[Fix] Fix nightly tests」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -152,6 +152,7; -167,8 +168,6；`test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py` renamed +7/-6 (13 lines); hunks: -9,11 +9,11; -44,28 +44,29 @@ def test_deepseek_r1_fp4_all_variants(self):; symbols: TestDeepseekR1FP4Unified, for, test_deepseek_r1_fp4_all_variants，涉及 `TestDeepseekR1FP4Unified, for, test_deepseek_r1_fp4_all_variants`；`test/registered/8-gpu-models/test_qwen3_235b.py` modified +2/-1 (3 lines); hunks: -4,7 +4,7; -70,6 +70,7 @@ def test_qwen3_235b_fp8_all_variants(self):; symbols: test_qwen3_235b_fp8_all_variants, test_qwen3_235b_fp8_cp，涉及 `test_qwen3_235b_fp8_all_variants, test_qwen3_235b_fp8_cp`；`test/registered/lora/test_lora_qwen3.py` modified +1/-2 (3 lines); hunks: -15,7 +15,7; -27,7 +27,6; symbols: TestLoRAQwen3，涉及 `TestLoRAQwen3`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -152,6 +152,7; -167,8 +168,6
  - `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py` renamed +7/-6 (13 lines); hunks: -9,11 +9,11; -44,28 +44,29 @@ def test_deepseek_r1_fp4_all_variants(self):; symbols: TestDeepseekR1FP4Unified, for, test_deepseek_r1_fp4_all_variants
  - `test/registered/8-gpu-models/test_qwen3_235b.py` modified +2/-1 (3 lines); hunks: -4,7 +4,7; -70,6 +70,7 @@ def test_qwen3_235b_fp8_all_variants(self):; symbols: test_qwen3_235b_fp8_all_variants, test_qwen3_235b_fp8_cp
  - `test/registered/lora/test_lora_qwen3.py` modified +1/-2 (3 lines); hunks: -15,7 +15,7; -27,7 +27,6; symbols: TestLoRAQwen3
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` modified +1/-1 (2 lines); hunks: -42,7 +42,7; symbols: TestNvidiaNemotron3SuperNightly
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -152,6 +152,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
@@ -167,8 +168,6 @@
-    from sglang.srt.utils.custom_op import register_custom_op
diff -- test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py
@@ -9,11 +9,11 @@
-DEEPSEEK_R1_FP4_MODEL_PATH = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
+FULL_DEEPSEEK_V3_FP4_MODEL_PATH = "nvidia/DeepSeek-V3-0324-FP4"
-    """Unified test class for DeepSeek-R1-0528-NVFP4-v2 performance and accuracy.
+    """Unified test class for DeepSeek-V3-0324-FP4 performance and accuracy.
@@ -44,28 +44,29 @@ def test_deepseek_r1_fp4_all_variants(self):
-                DEEPSEEK_R1_FP4_MODEL_PATH,
diff -- test/registered/8-gpu-models/test_qwen3_235b.py
@@ -4,7 +4,7 @@
-from sglang.test.test_utils import ModelLaunchSettings
+from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system
@@ -70,6 +70,7 @@ def test_qwen3_235b_fp8_all_variants(self):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2
  - tests: `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py` renamed +7/-6; `test/registered/8-gpu-models/test_qwen3_235b.py` modified +2/-1; `test/registered/lora/test_lora_qwen3.py` modified +1/-2; `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`, `test/registered/lora/test_lora_qwen3.py`, `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22358 - Enable DFLASH support for additional model backends

- 链接: https://github.com/sgl-project/sglang/pull/22358
- 状态/时间: merged / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+152/-5，可读 patch 299 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable DFLASH support for additional model backends」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「Enable DFLASH support for additional model backends」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture，涉及 `forward, get_layer, get_input_embeddings`；`python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`；`python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture，涉及 `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -574,8 +574,15 @@ def forward(
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
+                residual,
diff -- python/sglang/srt/models/kimi_k25.py
@@ -849,6 +849,30 @@ def set_eagle3_layers_to_capture(
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
+        """Set the layers to capture for DFLASH draft model training."""
+        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
+            raise AttributeError(
+                "language_model does not support DFLASH layer capture."
+            )
diff -- python/sglang/srt/models/qwen3_next.py
@@ -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22739 - Restore Qwen3 rope config fallback

- 链接: https://github.com/sgl-project/sglang/pull/22739
- 状态/时间: merged / 2026-04-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-2，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Restore Qwen3 rope config fallback」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3.py`；技术摘要: 覆盖「Restore Qwen3 rope config fallback」；主要实现面是 `python/sglang/srt/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -316,8 +316,16 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
+        if (
+            hasattr(config, "rope_parameters")
+            and config.rope_parameters
+            and "rope_theta" in config.rope_parameters
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- 链接: https://github.com/sgl-project/sglang/pull/23001
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 330 个文件，+80364/-0，可读 patch 68714 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add new Mintlify documentation site (docs_new/)」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`；技术摘要: 覆盖「Add new Mintlify documentation site (docs_new/)」；主要实现面是 `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Docs] Sync docs_new with legacy docs and update migration redirects」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`；技术摘要: 覆盖「[Docs] Sync docs_new with legacy docs and update migration redirects」；主要实现面是 `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #19484 - [CPU] Add Qwen3.5 model optimization for CPU

- 链接: https://github.com/sgl-project/sglang/pull/19484
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+768/-209，可读 patch 1454 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Add Qwen3.5 model optimization for CPU」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py`；技术摘要: 覆盖「[CPU] Add Qwen3.5 model optimization for CPU」；主要实现面是 `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/update_config.py` modified +178/-75 (253 lines); hunks: -1,7 +1,13; -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):; symbols: get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim，涉及 `get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim`；`python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights，涉及 `__init__, weight_loader, forward`；`python/sglang/srt/models/qwen3_vl.py` modified +29/-6 (35 lines); hunks: -72,7 +72,13; -87,6 +93,9; symbols: Qwen3_VisionMLP, __init__，涉及 `Qwen3_VisionMLP, __init__`；`python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11 (30 lines); hunks: -375,14 +375,22 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/update_config.py` modified +178/-75 (253 lines); hunks: -1,7 +1,13; -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):; symbols: get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim
  - `python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights
  - `python/sglang/srt/models/qwen3_vl.py` modified +29/-6 (35 lines); hunks: -72,7 +72,13; -87,6 +93,9; symbols: Qwen3_VisionMLP, __init__
  - `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11 (30 lines); hunks: -375,14 +375,22 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +18/-2 (20 lines); hunks: -1,3 +1,4; -29,7 +30,12; symbols: mamba_v2_sharded_weight_loader, loader
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/update_config.py
@@ -1,7 +1,13 @@
+import logging
+from sglang.srt.utils import (
+    log_debug_on_rank0,
+)
+logger = logging.getLogger(__name__)
@@ -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):
diff -- python/sglang/srt/models/qwen3_5.py
@@ -124,8 +124,16 @@ def __init__(
-        self.num_v_heads = config.linear_num_value_heads
-        self.num_k_heads = config.linear_num_key_heads
+        self.num_v_heads = (
+            config.linear_num_value_heads
+            if not _is_cpu
+            else config.linear_num_value_heads_cpu
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -72,7 +72,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/update_config.py` modified +178/-75; `python/sglang/srt/models/qwen3_5.py` modified +37/-4; `python/sglang/srt/models/qwen3_vl.py` modified +29/-6; `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11; `python/sglang/srt/layers/attention/mamba/mamba.py` modified +18/-2; `python/sglang/srt/models/qwen3_next.py` modified +14/-5
- 验证与风险: diff 自带测试面 `test/srt/cpu/test_mamba.py`, `test/srt/cpu/test_qwen3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20520 - [NPU]TP Communications compression For Qwen3 models for NPU

- 链接: https://github.com/sgl-project/sglang/pull/20520
- 状态/时间: merged / 2026-05-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+191/-10，可读 patch 346 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]TP Communications compression For Qwen3 models for NPU」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2.py`；技术摘要: 覆盖「[NPU]TP Communications compression For Qwen3 models for NPU」；主要实现面是 `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/linear.py` modified +15/-2 (17 lines); hunks: -19,6 +19,7; -37,6 +38,7; symbols: weight_loader_v2, forward，涉及 `weight_loader_v2, forward`；`python/sglang/srt/layers/communicator.py` modified +12/-2 (14 lines); hunks: -22,6 +22,7; -1000,9 +1001,18 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual，涉及 `_gather_hidden_states_and_residual`；`python/sglang/srt/models/qwen2.py` modified +6/-2 (8 lines); hunks: -91,13 +91,17 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/linear.py` modified +15/-2 (17 lines); hunks: -19,6 +19,7; -37,6 +38,7; symbols: weight_loader_v2, forward
  - `python/sglang/srt/layers/communicator.py` modified +12/-2 (14 lines); hunks: -22,6 +22,7; -1000,9 +1001,18 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual
  - `python/sglang/srt/models/qwen2.py` modified +6/-2 (8 lines); hunks: -91,13 +91,17 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward
  - `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0 (37 lines); hunks: -0,0 +1,37; symbols: TestLlama
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/linear.py
@@ -19,6 +19,7 @@
+    tensor_model_parallel_quant_all_reduce,
@@ -37,6 +38,7 @@
+from sglang.srt.server_args import get_global_server_args
@@ -1512,7 +1514,7 @@ def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor
-    def forward(self, input_, skip_all_reduce=False):
+    def forward(self, input_, skip_all_reduce=False, forward_batch=None):
diff -- python/sglang/srt/layers/communicator.py
@@ -22,6 +22,7 @@
+    attention_tensor_model_parallel_quant_all_reduce,
@@ -1000,9 +1001,18 @@ def _gather_hidden_states_and_residual(
-                hidden_states = attention_tensor_model_parallel_all_reduce(
-                    hidden_states
+                quantize_communications = (
+                    not forward_batch.forward_mode.is_decode_or_idle()
diff -- python/sglang/srt/models/qwen2.py
@@ -91,13 +91,17 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/linear.py` modified +15/-2; `python/sglang/srt/layers/communicator.py` modified +12/-2; `python/sglang/srt/models/qwen2.py` modified +6/-2; `python/sglang/srt/models/qwen3.py` modified +1/-1; `python/sglang/srt/distributed/device_communicators/npu_communicator.py` modified +33/-1; `python/sglang/srt/server_args.py` modified +21/-0
  - tests: `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0; `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py` added +37/-0
- 验证与风险: diff 自带测试面 `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21722 - feat: use structural tags to enable strict tool calling and reasoning for more models

- 链接: https://github.com/sgl-project/sglang/pull/21722
- 状态/时间: merged / 2026-05-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+922/-49，可读 patch 1197 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: use structural tags to enable strict tool calling and reasoning for more models」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv4_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`；技术摘要: 覆盖「feat: use structural tags to enable strict tool calling and reasoning for more models」；主要实现面是 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv4_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_function_call_parser.py` modified +657/-9 (666 lines); hunks: -1,10 +1,16; -15,6 +21,7; symbols: test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, TestDeepSeekV4Detector, setUp，涉及 `test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, TestDeepSeekV4Detector`；`python/sglang/srt/function_call/deepseekv4_detector.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: DeepSeekV4Detector, __init__, get_structural_tag_name，涉及 `DeepSeekV4Detector, __init__, get_structural_tag_name`；`python/sglang/srt/function_call/function_call_parser.py` modified +35/-24 (59 lines); hunks: -152,7 +152,7 @@ def parse_stream_chunk(self, chunk_text: str) -> Tuple[str,...; -208,6 +208,7 @@ def get_structure_constraint(; symbols: parse_stream_chunk, get_structure_tag, get_legacy_structural_tag, get_structure_constraint，涉及 `parse_stream_chunk, get_structure_tag, get_legacy_structural_tag`；`python/sglang/srt/function_call/base_format_detector.py` modified +50/-2 (52 lines); hunks: -1,13 +1,19; -361,3 +367,45 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info, get_structural_tag_name, get_structural_tag，涉及 `structure_info, get_structural_tag_name, get_structural_tag`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +657/-9 (666 lines); hunks: -1,10 +1,16; -15,6 +21,7; symbols: test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, TestDeepSeekV4Detector, setUp
  - `python/sglang/srt/function_call/deepseekv4_detector.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: DeepSeekV4Detector, __init__, get_structural_tag_name
  - `python/sglang/srt/function_call/function_call_parser.py` modified +35/-24 (59 lines); hunks: -152,7 +152,7 @@ def parse_stream_chunk(self, chunk_text: str) -> Tuple[str,...; -208,6 +208,7 @@ def get_structure_constraint(; symbols: parse_stream_chunk, get_structure_tag, get_legacy_structural_tag, get_structure_constraint
  - `python/sglang/srt/function_call/base_format_detector.py` modified +50/-2 (52 lines); hunks: -1,13 +1,19; -361,3 +367,45 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info, get_structural_tag_name, get_structural_tag
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +10/-0 (10 lines); hunks: -253,3 +253,13 @@ def get_info(name: str) -> StructureInfo:; symbols: get_info
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -1,10 +1,16 @@
-from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.entrypoints.openai.protocol import (
+    Function,
+    Tool,
+    ToolChoice,
+    ToolChoiceFuncName,
diff -- python/sglang/srt/function_call/deepseekv4_detector.py
@@ -0,0 +1,67 @@
+import logging
+from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
+logger = logging.getLogger(__name__)
+class DeepSeekV4Detector(DeepSeekV32Detector):
+    """
+    Detector for DeepSeek V4 model function call format.
diff -- python/sglang/srt/function_call/function_call_parser.py
@@ -152,7 +152,7 @@ def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +657/-9
  - runtime: `python/sglang/srt/function_call/deepseekv4_detector.py` added +67/-0; `python/sglang/srt/function_call/function_call_parser.py` modified +35/-24; `python/sglang/srt/function_call/base_format_detector.py` modified +50/-2; `python/sglang/srt/function_call/kimik2_detector.py` modified +10/-0; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-0; `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +4/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/spec/test_spec_utils_traverse_tree.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25825 - [Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool

- 链接: https://github.com/sgl-project/sglang/pull/25825
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+59/-8，可读 patch 326 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`；技术摘要: 覆盖「[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool」；主要实现面是 `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu，涉及 `__init__, forward_prepare_native, forward_prepare_npu`；`python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare，涉及 `__init__, forward_prepare`；`python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare
  - `python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -64,6 +64,7 @@ def __init__(; -76,6 +77,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama.py
@@ -27,6 +27,7 @@
+    get_pp_indices,
@@ -131,6 +132,7 @@ def __init__(
+        start_layer: int = 0,
@@ -141,6 +143,7 @@ def __init__(
+        self.start_layer = start_layer
@@ -210,7 +213,7 @@ def forward_prepare_native(self, positions, hidden_states):
diff -- python/sglang/srt/models/glm4_moe.py
@@ -28,6 +28,7 @@
+    get_pp_indices,
@@ -187,6 +188,7 @@ def __init__(
+        start_layer: int = 0,
@@ -201,6 +203,7 @@ def __init__(
+        self.start_layer = start_layer
@@ -312,7 +315,7 @@ def forward_prepare(
diff -- python/sglang/srt/models/qwen2.py
@@ -24,6 +24,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama.py` modified +16/-2; `python/sglang/srt/models/glm4_moe.py` modified +12/-1; `python/sglang/srt/models/qwen2.py` modified +9/-0; `python/sglang/srt/models/qwen2_moe.py` modified +9/-0; `python/sglang/srt/models/qwen3.py` modified +5/-1; `python/sglang/srt/models/qwen3_moe.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama_eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25983 - feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext

- 链接: https://github.com/sgl-project/sglang/pull/25983
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 77 个文件，+1227/-905，可读 patch 5236 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；技术摘要: 覆盖「feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode，涉及 `get_spec_info, run_once, maybe_init_ngram_embedding`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once，涉及 `capture_one_batch_size, run_once`；`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size，涉及 `warmup_compile, _cache_loc_dtype, capture_one_batch_size`；`python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp，涉及 `_get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp
  - `python/sglang/srt/model_executor/forward_context.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: ForwardContext, set_forward_context, has_forward_context, get_forward_context
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -146,6 +146,11 @@
+from sglang.srt.model_executor.forward_context import (
+    ForwardContext,
+    forward_context,
+    has_forward_context,
+)
@@ -2638,9 +2643,6 @@ def get_spec_info():
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -65,6 +65,7 @@
+from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
@@ -1016,9 +1017,6 @@ def capture_one_batch_size(
-            req_to_token_pool=self.model_runner.req_to_token_pool,
-            token_to_kv_pool=self.model_runner.token_to_kv_pool,
-            attn_backend=attn_backend,
@@ -1040,85 +1038,90 @@ def capture_one_batch_size(
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -58,6 +58,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44; `python/sglang/srt/model_executor/forward_context.py` added +84/-0; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +39/-38
- 验证与风险: diff 自带测试面 `test/manual/attention/test_flashattn_backend.py`, `test/manual/attention/test_flashattn_mla_backend.py`, `test/manual/attention/test_prefix_chunk_info.py`, `test/manual/attention/test_trtllm_mla_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26798 - Make qwen3's set_embed_and_head idempotent

- 链接: https://github.com/sgl-project/sglang/pull/26798
- 状态/时间: merged / 2026-05-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-2，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Make qwen3's set_embed_and_head idempotent」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3.py`；技术摘要: 覆盖「Make qwen3's set_embed_and_head idempotent」；主要实现面是 `python/sglang/srt/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +4/-2 (6 lines); hunks: -674,8 +674,10 @@ def get_embed_and_head(self):; symbols: get_embed_and_head, set_embed_and_head，涉及 `get_embed_and_head, set_embed_and_head`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +4/-2 (6 lines); hunks: -674,8 +674,10 @@ def get_embed_and_head(self):; symbols: get_embed_and_head, set_embed_and_head
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -674,8 +674,10 @@ def get_embed_and_head(self):
-        del self.model.embed_tokens.weight
-        del self.lm_head.weight
+        if hasattr(self.model.embed_tokens, "weight"):
+            del self.model.embed_tokens.weight
+        if hasattr(self.lm_head, "weight"):
+            del self.lm_head.weight
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +4/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24435 - Update Qwen3-Coder docs_new NVIDIA guidance

- 链接: https://github.com/sgl-project/sglang/pull/24435
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`；关联提交 `106092123f01`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+285/-20，可读 patch 361 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update Qwen3-Coder docs_new NVIDIA guidance」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`；技术摘要: 覆盖「Update Qwen3-Coder docs_new NVIDIA guidance」；主要实现面是 `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` modified +273/-10 (283 lines); hunks: -47,7 +47,7 @@ This section provides deployment configurations verified on AM...; -292,7 +292,7 @@ Arguments: {"code": "def factorial(n):\n if n == 0 or n == 1...; symbols: factorial，涉及 `factorial`；`docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` modified +12/-10 (22 lines); hunks: -45,8 +45,8 @@ export const Qwen3CoderDeployment = () => {; -97,24 +97,26 @@ export const Qwen3CoderDeployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` modified +273/-10 (283 lines); hunks: -47,7 +47,7 @@ This section provides deployment configurations verified on AM...; -292,7 +292,7 @@ Arguments: {"code": "def factorial(n):\n if n == 0 or n == 1...; symbols: factorial
  - `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` modified +12/-10 (22 lines); hunks: -45,8 +45,8 @@ export const Qwen3CoderDeployment = () => {; -97,24 +97,26 @@ export const Qwen3CoderDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx
@@ -47,7 +47,7 @@ This section provides deployment configurations verified on AMD MI300X, MI325X,
-* **MOE Runner Backend**: FP8 uses `--moe-runner-backend triton`, NVFP4 uses `--moe-runner-backend flashinfer_cutlass`.
+* **GB200 Parallelism**: Use `--tp 4 --ep 4` on GB200. B200 uses the default NVIDIA settings generated above.
@@ -292,7 +292,7 @@ Arguments: {"code": "def factorial(n):\n    if n == 0 or n == 1:\n        return
-#### 5.1.1 Standard Scenario Benchmark
+#### 5.1.1 AMD Standard Scenario Benchmark
@@ -477,6 +477,269 @@ Max ITL (ms):                            36863.32
diff -- docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx
@@ -45,8 +45,8 @@ export const Qwen3CoderDeployment = () => {
-      b200: { tp: 8 },
-      gb200: { tp: 8 }
+      b200: { tp: 8, ep: 8 },
+      gb200: { tp: 4, ep: 4 }
@@ -97,24 +97,26 @@ export const Qwen3CoderDeployment = () => {
-    // EP and DP attention settings
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` modified +273/-10; `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` modified +12/-10
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/27001
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+11/-471，可读 patch 936 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`；技术摘要: 覆盖「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；主要实现面是 `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27248 - [Doc][CPU]Update Cookbook with Xeon support info

- 链接: https://github.com/sgl-project/sglang/pull/27248
- 状态/时间: merged / 2026-06-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+443/-121，可读 patch 1524 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc][CPU]Update Cookbook with Xeon support info」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`；技术摘要: 覆盖「[Doc][CPU]Update Cookbook with Xeon support info」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20 (85 lines); hunks: -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {; -57,8 +67,11 @@ export const DeepSeekV3Deployment = () => {；`docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15 (65 lines); hunks: -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {; -26,9 +28,9 @@ export const DeepSeekV31Deployment = () => {；`docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7 (56 lines); hunks: -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {; -35,9 +46,9 @@ export const DeepSeekR1BasicDeployment = () => {；`docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18 (51 lines); hunks: -64,7 +64,8 @@ export const Qwen35Deployment = () => {; -74,12 +75,13 @@ export const Qwen35Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20 (85 lines); hunks: -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {; -57,8 +67,11 @@ export const DeepSeekV3Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15 (65 lines); hunks: -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {; -26,9 +28,9 @@ export const DeepSeekV31Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7 (56 lines); hunks: -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {; -35,9 +46,9 @@ export const DeepSeekR1BasicDeployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18 (51 lines); hunks: -64,7 +64,8 @@ export const Qwen35Deployment = () => {; -74,12 +75,13 @@ export const Qwen35Deployment = () => {
  - `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` modified +31/-10 (41 lines); hunks: -13,7 +13,8 @@ export const Hunyuan3PreviewDeployment = () => {; -35,18 +36,22 @@ export const Hunyuan3PreviewDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx
@@ -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {
-        { id: 'mi355x', label: 'MI355X', default: false }
+        { id: 'mi355x', label: 'MI355X', default: false },
+        { id: 'xeon', label: 'XEON', default: false }
-      items: [
-        { id: 'fp8', label: 'FP8', default: true },
-        { id: 'fp4', label: 'FP4', default: false }
diff -- docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx
@@ -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {
-        { id: 'mi355x', label: 'MI355X', default: false }
+        { id: 'mi355x', label: 'MI355X', default: false },
+        { id: 'xeon', label: 'XEON', default: false }
-        { id: 'v31terminus', label: 'DeepSeek-V3.1-Terminus', default: false }
+        { id: 'v31terminus', label: 'DeepSeek-V3.1-Terminus', default: false },
+        { id: 'v31terminusint8', label: 'DeepSeek-V3.1-Terminus-Channel-int8', default: false, xeonOnly: true }
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx
@@ -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20; `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15; `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18; `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` modified +31/-10; `docs_new/src/snippets/autoregressive/deepseek-ocr-deployment.jsx` modified +29/-9
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- 链接: https://github.com/sgl-project/sglang/pull/23906
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 160 个文件，+5197/-3068，可读 patch 12233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Cuda Graph Runner/Backend Refactor」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`；技术摘要: 覆盖「[Refactor] Cuda Graph Runner/Backend Refactor」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #13411 - Improve Qwen3CoderDetector with schema-aware parameter type conversion

- 链接: https://github.com/sgl-project/sglang/pull/13411
- 状态/时间: closed / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+155/-10，可读 patch 222 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Improve Qwen3CoderDetector with schema-aware parameter type conversion」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`；技术摘要: 覆盖「Improve Qwen3CoderDetector with schema-aware parameter type conversion」；主要实现面是 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10 (145 lines); hunks: -17,15 +17,118; -84,6 +187,14 @@ def parse_streaming_increment(; symbols: _safe_val, _convert_param_value, Qwen3CoderDetector, parse_streaming_increment，涉及 `_safe_val, _convert_param_value, Qwen3CoderDetector`；`test/per_commit/function_call/test_function_call_parser.py` modified +20/-0 (20 lines); hunks: -1422,6 +1422,10 @@ def test_extract_tool_calls_type_conversion(self):; -1444,6 +1448,18 @@ def test_extract_tool_calls_type_conversion(self):; symbols: test_extract_tool_calls_type_conversion, test_parse_streaming_incremental，涉及 `test_extract_tool_calls_type_conversion, test_parse_streaming_incremental`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10 (145 lines); hunks: -17,15 +17,118; -84,6 +187,14 @@ def parse_streaming_increment(; symbols: _safe_val, _convert_param_value, Qwen3CoderDetector, parse_streaming_increment
  - `test/per_commit/function_call/test_function_call_parser.py` modified +20/-0 (20 lines); hunks: -1422,6 +1422,10 @@ def test_extract_tool_calls_type_conversion(self):; -1444,6 +1448,18 @@ def test_extract_tool_calls_type_conversion(self):; symbols: test_extract_tool_calls_type_conversion, test_parse_streaming_incremental
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/qwen3_coder_detector.py
@@ -17,15 +17,118 @@
-def _safe_val(raw: str) -> Any:
-    raw = html.unescape(raw.strip())
-    try:
-        return json.loads(raw)
-    except Exception:
+def _convert_param_value(
diff -- test/per_commit/function_call/test_function_call_parser.py
@@ -1422,6 +1422,10 @@ def test_extract_tool_calls_type_conversion(self):
+                        "str_param_int_content": {"type": "string"},
+                        "str_param_float_content": {"type": "string"},
+                        "str_param_bool_content": {"type": "string"},
+                        "str_param_obj_content": {"type": "string"},
@@ -1444,6 +1448,18 @@ def test_extract_tool_calls_type_conversion(self):
+<parameter=str_param_int_content>
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10
  - tests: `test/per_commit/function_call/test_function_call_parser.py` modified +20/-0
- 验证与风险: diff 自带测试面 `test/per_commit/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[docs] Add B300 cookbook deployment options」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；技术摘要: 覆盖「[docs] Add B300 cookbook deployment options」；主要实现面是 `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #28810 - [CI] Remove deprecated test/srt legacy CI setup

- 链接: https://github.com/sgl-project/sglang/pull/28810
- 状态/时间: merged / 2026-06-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 29 个文件，+2/-5773，可读 patch 5826 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Remove deprecated test/srt legacy CI setup」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py`；技术摘要: 覆盖「[CI] Remove deprecated test/srt legacy CI setup」；主要实现面是 `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: layernorm, rotary_emb, native_torch, native_torch_int8，涉及 `layernorm, rotary_emb, native_torch`；`test/srt/cpu/utils.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: parametrize, decorator, wrapper, SiluAndMul，涉及 `parametrize, decorator, wrapper`；`test/srt/cpu/test_norm.py` removed +0/-432 (432 lines); hunks: -1,432 +0,0; symbols: TestNorm, _forward_native, _norm, _gemma3_rmsnorm_native，涉及 `TestNorm, _forward_native, _norm`；`test/srt/cpu/test_extend.py` removed +0/-400 (400 lines); hunks: -1,400 +0,0; symbols: TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend, _run_sdpa_forward_extend_sink，涉及 `TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend`。
- 代码 diff 细节:
  - `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: layernorm, rotary_emb, native_torch, native_torch_int8
  - `test/srt/cpu/utils.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: parametrize, decorator, wrapper, SiluAndMul
  - `test/srt/cpu/test_norm.py` removed +0/-432 (432 lines); hunks: -1,432 +0,0; symbols: TestNorm, _forward_native, _norm, _gemma3_rmsnorm_native
  - `test/srt/cpu/test_extend.py` removed +0/-400 (400 lines); hunks: -1,400 +0,0; symbols: TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend, _run_sdpa_forward_extend_sink
  - `test/srt/cpu/test_mamba.py` removed +0/-394 (394 lines); hunks: -1,394 +0,0; symbols: l2norm, torch_chunk_gated_delta_rule, chunk_gated_delta_rule_update, torch_recurrent_gated_delta_rule
- 关键代码摘录:

```diff
diff -- test/srt/cpu/test_qkv_proj_with_rope.py
@@ -1,440 +0,0 @@
-import unittest
-import torch
-from utils import (
-    convert_weight,
-    native_w8a8_per_token_matmul,
-    per_token_quant_int8,
diff -- test/srt/cpu/utils.py
@@ -1,440 +0,0 @@
-import itertools
-import math
-import torch
-import torch.nn.functional as F
-precision = {
-    torch.bfloat16: 1e-2,
diff -- test/srt/cpu/test_norm.py
@@ -1,432 +0,0 @@
```

- 已读文件:
  - tests: `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440; `test/srt/cpu/utils.py` removed +0/-440; `test/srt/cpu/test_norm.py` removed +0/-432; `test/srt/cpu/test_extend.py` removed +0/-400; `test/srt/cpu/test_mamba.py` removed +0/-394; `test/srt/cpu/test_moe.py` removed +0/-352
- 验证与风险: diff 自带测试面 `test/README.md`, `test/srt/cpu/arm64/test_moe.py`, `test/srt/cpu/test_activation.py`, `test/srt/cpu/test_binding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
