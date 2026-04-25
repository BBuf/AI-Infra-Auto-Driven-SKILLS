# sglang Qwen3 Coder 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/qwen3-coder-480b-a35b-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/qwen3_coder_detector.py` | [#8371](https://github.com/sgl-project/sglang/pull/8371), [#16744](https://github.com/sgl-project/sglang/pull/16744) |
| `python/sglang/srt/models/qwen3.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/amd/test_qwen3_coder_next_8gpu.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/ascend/llm_models/test_npu_qwen3_coder_480b_a35b.py` | 无直接 PR 号提交 |
| `test/registered/lora/test_lora_qwen3.py` | 无直接 PR 号提交 |
| `test/srt/cpu/test_qwen3.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 3
- 原文档显式引用补充 PR 数: 16
- 当前文档总 PR 数: 19
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
| 2025-11-17 | [#13411](https://github.com/sgl-project/sglang/pull/13411) | open | Improve Qwen3CoderDetector with schema-aware parameter type conversion | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py` |
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
| 2026-04-01 | [#21829](https://github.com/sgl-project/sglang/pull/21829) | open | [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector | `python/sglang/srt/function_call/qwen3_coder_detector.py` |

## 逐 PR diff 审计卡

### PR #8260 - Preliminary Support for Qwen3XMLDetector

- 链接: https://github.com/sgl-project/sglang/pull/8260
- 状态/时间: merged / 2025-07-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+153/-0，可读 patch 175 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Preliminary Support for Qwen3XMLDetector」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`；PR 正文摘要: This is a very basic, preliminary implementation of the Qwen3XMLDetector, intended to provide initial support. Please note that the related functionality still requires extensiv...。
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
- 动机: 标题「[Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`；PR 正文摘要: Follow up to #8260 Here are some issues of `Qwen3XMLDetector`: 1. **EBNFComposer XML Support:** `tool_choice="required"` and specific tool selection fail because EBNFComposer on...。
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
- 动机: 标题「GLM-4.5 Model Support」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；PR 正文摘要: The SGLang version of the complete implementation of the GLM-4.5 model, which includes: 1. Model implementation (with MTP and without MTP) 2. Tool call and Reasoning parser 3. t...。
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
- 动机: 标题「GLM-4.5 Model Support Follow-up」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`；PR 正文摘要: - Address PR comments in https://github.com/sgl-project/sglang/pull/8224 - support more than one tool_choice logic for GLM4-moe - add key_value_separator for qwen3 coder。
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
- 动机: 标题「Update qwen3_coder_detector.py for streaming」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`；PR 正文摘要: This PR is to improve the streaming tool call parser from reading all values to incremental flow. Update the tool call parser for qwen3_coder_detector using XML format. - **Stre...。
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
- 动机: 标题「Forward unknown tool calls instead of dropping」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`；PR 正文摘要: Closes #12223 - Avoid silent data loss when models emit unknown tool names. - Make behavior consistent with paths that already forward malformed arguments (no schema validation)...。
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
- 动机: 标题「Remove EBNF Composer」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`；PR 正文摘要: Resolve issue: https://github.com/sgl-project/sglang/issues/11078 We are now using json schema to constrain model output for required and named tool_choice cases, so we no longe...。
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

### PR #13411 - Improve Qwen3CoderDetector with schema-aware parameter type conversion

- 链接: https://github.com/sgl-project/sglang/pull/13411
- 状态/时间: open / 2025-11-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+155/-10，可读 patch 222 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Improve Qwen3CoderDetector with schema-aware parameter type conversion」；模型线: Qwen3 Coder；类别: 模型实现调整；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`；PR 正文摘要: In the original implementation of Qwen3CoderDetector, there is a function called `_safe_val` to help convert the parameter type of function calls: However, due to the unique str...。
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

### PR #13979 - Add Qwen3-Coder-480B to nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/13979
- 状态/时间: open / 2025-11-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+288/-171，可读 patch 521 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Qwen3-Coder-480B to nightly tests」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`；PR 正文未提供可用摘要。
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
- 动机: 标题「support new qwen3_coder_detector」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`；PR 正文摘要: UT !39a93fe0305fa84ec404a08211d84853 E2E Test !76e2386d0e007fb9c295f7f8b61f2904 The test results, provided by Zeyu Cui from the Qwen Team @cyente, have been confirmed to meet ex...。
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
- 动机: 标题「[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`；PR 正文摘要: Enable SwapAB on H200, which has been verified in this PR. And also retune the configuration of TP=8 EP=1 DeepseekV3 and TP=8 EP=2 Qwen3 Coder, in the latency scenario. Also ena...。
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
- 动机: 标题「Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`；PR 正文摘要: Add optimized Triton MoE kernel configuration for Qwen3-Coder-Next-FP8 on H100 with TP=2. `python -m sglang.bench_serving --backend sglang-oai-chat --model Qwen/Qwen3-Coder-Next...。
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
- 动机: 标题「[ModelOPT] Support Qwen 3 Next Coder NVFP4」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；PR 正文摘要: This branch include important bugfix for qwen 3 coder next nvfp4 B300 `sglang serve --model vincentzed-hf/Qwen3-Coder-Next-NVFP4 --quantization modelopt_fp4` **We provide cmd to...。
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
- 动机: 标题「[NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu.」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`；PR 正文摘要: 1. The original code contained an import error that affected the proper loading of weight-related modules. 2. Duplicate dimension transformations were applied to weight.data: Th...。
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
- 动机: 标题「[AMD] Support Qwen3-Coder-Next on AMD platform」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`；PR 正文摘要: Enable Qwen3-Coder-Next model on AMD GPU platform. With this PR, we are able to support non-MTP (fp8 kv cache) and MTP on Qwen3-Coder-Next. - aiter_backend.py: - Handle v_head_d...。
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
- 动机: 标题「[AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU」；模型线: Qwen3 Coder；类别: 文档/测试/CI；主要 diff: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`；PR 正文摘要: Add CI test coverage for the Qwen3-Coder-Next model on AMD MI35x (gfx950) 8-GPU systems. This model features a hybrid architecture combining full attention (GQA) with linear att...。
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
- 动机: 标题「feat: Add FP8 KV cache support for Triton attention backend」；模型线: Qwen3 Coder；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`；PR 正文摘要: The Triton attention backend does not correctly handle FP8 KV cache. It does not pass k_scale/v_scale to set_kv_buffer, causing scales to default to 1.0. In addition, the Triton...。
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
- 动机: 标题「[AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend」；模型线: Qwen3 Coder；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`；PR 正文摘要: - Fix `TypeError: extend_attention_fwd() missing 1 required positional argument: 'v_scale'` crash in aiter_backend when running non-MLA speculative decoding (target_verify / dra...。
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

### PR #21829 - [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector

- 链接: https://github.com/sgl-project/sglang/pull/21829
- 状态/时间: open / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+140/-0，可读 patch 168 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector」；模型线: Qwen3 Coder；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`；PR 正文摘要: When using --tool-call-parser qwen3_coder with streaming enabled, the arguments field of toolcall deltas is not streamed incrementally. Instead, the entire parameter value is bu...。
- 实现要点: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0 (140 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,51 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param，涉及 `__init__, has_tool_call, _convert_param_value`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0 (140 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,51 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param
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
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/qwen3_coder_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
