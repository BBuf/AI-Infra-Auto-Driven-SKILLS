# sglang Qwen3 Coder Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/qwen3-coder-480b-a35b-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/qwen3_coder_detector.py` | [#8371](https://github.com/sgl-project/sglang/pull/8371), [#16744](https://github.com/sgl-project/sglang/pull/16744) |
| `python/sglang/srt/models/qwen3.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/amd/test_qwen3_coder_next_8gpu.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/ascend/llm_models/test_npu_qwen3_coder_480b_a35b.py` | no direct PR-number commit |
| `test/registered/lora/test_lora_qwen3.py` | no direct PR-number commit |
| `test/srt/cpu/test_qwen3.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 16
- Total PRs in this document: 19
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #8260 - Preliminary Support for Qwen3XMLDetector

- Link: https://github.com/sgl-project/sglang/pull/8260
- Status/date: merged / 2025-07-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +153/-0, 175 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Preliminary Support for Qwen3XMLDetector"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; PR body summary: This is a very basic, preliminary implementation of the Qwen3XMLDetector, intended to provide initial support. Please note that the related functionality still requires extensiv....
- Key implementation: `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: _safe_val, Qwen3XMLDetector, __init__, has_tool_call, touching `_safe_val, Qwen3XMLDetector, __init__`; `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: -14,6 +14,7; -35,6 +36,7 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__, touching `FunctionCallParser, __init__`; `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: -1099,6 +1099,7 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args, touching `add_cli_args`.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: _safe_val, Qwen3XMLDetector, __init__, has_tool_call
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: -14,6 +14,7; -35,6 +36,7 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__
  - `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: -1099,6 +1099,7 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0; `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0; `python/sglang/srt/server_args.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8357 - [Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector

- Link: https://github.com/sgl-project/sglang/pull/8357
- Status/date: merged / 2025-07-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +574/-83, 868 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector"; model line: Qwen3 Coder; category: bug fix; main diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`; PR body summary: Follow up to #8260 Here are some issues of `Qwen3XMLDetector`: 1. **EBNFComposer XML Support:** `tool_choice="required"` and specific tool selection fail because EBNFComposer on....
- Key implementation: `test/srt/test_function_call_parser.py` modified +455/-0 (455 lines); hunks: -10,6 +10,7; -507,6 +508,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_qwen3_coder_detector_ebnf, touching `setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf`; `python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63 (158 lines); hunks: -1,51 +1,73; -55,19 +77,20 @@ class EBNFComposer:; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val, touching `EBNFComposer, get_value_rule, _handle_enum`; `python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9 (19 lines); hunks: -9,7 +9,6; -29,7 +28,7 @@ def _safe_val(raw: str) -> Any:; symbols: _safe_val, Qwen3XMLDetector, Qwen3CoderDetector, _parse_block, touching `_safe_val, Qwen3XMLDetector, Qwen3CoderDetector`; `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5 (9 lines); hunks: -8,7 +8,6; -216,11 +215,11 @@ def _get_parameter_value(self, val):; symbols: _get_parameter_value, structure_info, info, supports_structural_tag, touching `_get_parameter_value, structure_info, info`.
- Code diff details:
  - `test/srt/test_function_call_parser.py` modified +455/-0 (455 lines); hunks: -10,6 +10,7; -507,6 +508,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_qwen3_coder_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63 (158 lines); hunks: -1,51 +1,73; -55,19 +77,20 @@ class EBNFComposer:; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9 (19 lines); hunks: -9,7 +9,6; -29,7 +28,7 @@ def _safe_val(raw: str) -> Any:; symbols: _safe_val, Qwen3XMLDetector, Qwen3CoderDetector, _parse_block
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5 (9 lines); hunks: -8,7 +8,6; -216,11 +215,11 @@ def _get_parameter_value(self, val):; symbols: _get_parameter_value, structure_info, info, supports_structural_tag
  - `python/sglang/srt/function_call/function_call_parser.py` modified +4/-4 (8 lines); hunks: -14,7 +14,7; -36,7 +36,7 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__, get_structure_constraint
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/srt/test_function_call_parser.py` modified +455/-0
  - runtime: `python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63; `python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9; `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5; `python/sglang/srt/function_call/function_call_parser.py` modified +4/-4; `python/sglang/srt/function_call/base_format_detector.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8224 - GLM-4.5 Model Support

- Link: https://github.com/sgl-project/sglang/pull/8224
- Status/date: merged / 2025-07-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1673/-7, 1853 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "GLM-4.5 Model Support"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; PR body summary: The SGLang version of the complete implementation of the GLM-4.5 model, which includes: 1. Model implementation (with MTP and without MTP) 2. Tool call and Reasoning parser 3. t....
- Key implementation: `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunks: -0,0 +1,1034; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention, touching `Glm4MoeMLP, __init__, forward`; `test/srt/test_function_call_parser.py` modified +184/-0 (184 lines); hunks: -6,6 +6,7; -510,6 +511,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf, touching `setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf`; `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN, touching `Glm4MoeModelNextN, __init__, forward`; `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__, touching `get_argument_type, parse_arguments, Glm4MoeDetector`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunks: -0,0 +1,1034; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention
  - `test/srt/test_function_call_parser.py` modified +184/-0 (184 lines); hunks: -6,6 +6,7; -510,6 +511,7 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf
  - `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN
  - `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
  - `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +39/-1 (40 lines); hunks: -223,7 +223,10 @@ def test_function_calling_streaming_simple(self):; -910,5 +913,40 @@ def test_pythonic_tool_call_streaming(self):; symbols: test_function_calling_streaming_simple, test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` added +1034/-0; `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0; `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0; `python/sglang/srt/function_call/ebnf_composer.py` modified +10/-3; `python/sglang/srt/configs/model_config.py` modified +3/-0; `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0
  - tests: `test/srt/test_function_call_parser.py` modified +184/-0; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +39/-1
- Risk and verification: The diff ships test coverage in `test/srt/openai_server/features/test_enable_thinking.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8445 - GLM-4.5 Model Support Follow-up

- Link: https://github.com/sgl-project/sglang/pull/8445
- Status/date: merged / 2025-07-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +44/-15, 168 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "GLM-4.5 Model Support Follow-up"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`; PR body summary: - Address PR comments in https://github.com/sgl-project/sglang/pull/8224 - support more than one tool_choice logic for GLM4-moe - add key_value_separator for qwen3 coder.
- Key implementation: `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10 (49 lines); hunks: -135,7 +135,7 @@ def get_test_messages(self):; -203,7 +203,7 @@ def test_tool_choice_auto_non_streaming(self):; symbols: get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming, test_tool_choice_required_non_streaming, touching `get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming`; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunks: -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf, touching `build_ebnf`; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1 (2 lines); hunks: -913,7 +913,7 @@ def test_pythonic_tool_call_streaming(self):; symbols: test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass, touching `test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass`; `test/srt/test_function_call_parser.py` modified +1/-1 (2 lines); hunks: -2068,7 +2068,7 @@ def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id, touching `test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id`.
- Code diff details:
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10 (49 lines); hunks: -135,7 +135,7 @@ def get_test_messages(self):; -203,7 +203,7 @@ def test_tool_choice_auto_non_streaming(self):; symbols: get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming, test_tool_choice_required_non_streaming
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunks: -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf
  - `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1 (2 lines); hunks: -913,7 +913,7 @@ def test_pythonic_tool_call_streaming(self):; symbols: test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass
  - `test/srt/test_function_call_parser.py` modified +1/-1 (2 lines); hunks: -2068,7 +2068,7 @@ def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +1/-0 (1 lines); hunks: -148,4 +148,5 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1; `test/srt/test_function_call_parser.py` modified +1/-1; `test/srt/openai_server/features/test_enable_thinking.py` modified +1/-1
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2; `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/openai_server/features/test_enable_thinking.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8371 - Update qwen3_coder_detector.py for streaming

- Link: https://github.com/sgl-project/sglang/pull/8371
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/qwen3_coder_detector.py`; associated commits `b3359dc9bf5b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +348/-67, 510 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update qwen3_coder_detector.py for streaming"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`; PR body summary: This PR is to improve the streaming tool call parser from reading all values to incremental flow. Update the tool call parser for qwen3_coder_detector using XML format. - **Stre....
- Key implementation: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9 (228 lines); hunks: -57,6 +57,15 @@ def __init__(self):; -70,23 +79,224 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, _parse_and_stream_parameters, touching `__init__, has_tool_call, parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9 (228 lines); hunks: -57,6 +57,15 @@ def __init__(self):; -70,23 +79,224 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, _parse_and_stream_parameters
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12226 - Forward unknown tool calls instead of dropping

- Link: https://github.com/sgl-project/sglang/pull/12226
- Status/date: merged / 2025-11-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +145/-60, 279 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Forward unknown tool calls instead of dropping"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`; PR body summary: Closes #12223 - Avoid silent data loss when models emit unknown tool names. - Make behavior consistent with paths that already forward malformed arguments (no schema validation)....
- Key implementation: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37 (78 lines); hunks: -6,6 +6,7; -120,45 +121,48 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, touching `parse_streaming_increment`; `test/srt/function_call/test_unknown_tool_name.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: DummyDetector, has_tool_call, detect_and_parse, test_unknown_tool_name_dropped_default, touching `DummyDetector, has_tool_call, detect_and_parse`; `python/sglang/srt/function_call/base_format_detector.py` modified +15/-12 (27 lines); hunks: -8,6 +8,7; -75,19 +76,21 @@ def parse_base_json(self, action: Any, tools: List[Tool]) ->...; symbols: parse_base_json, touching `parse_base_json`; `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1 (5 lines); hunks: -5,6 +5,7; -91,7 +92,9 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: detect_and_parse, touching `detect_and_parse`.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37 (78 lines); hunks: -6,6 +6,7; -120,45 +121,48 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
  - `test/srt/function_call/test_unknown_tool_name.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: DummyDetector, has_tool_call, detect_and_parse, test_unknown_tool_name_dropped_default
  - `python/sglang/srt/function_call/base_format_detector.py` modified +15/-12 (27 lines); hunks: -8,6 +8,7; -75,19 +76,21 @@ def parse_base_json(self, action: Any, tools: List[Tool]) ->...; symbols: parse_base_json
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1 (5 lines); hunks: -5,6 +5,7; -91,7 +92,9 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: detect_and_parse
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +3/-1 (4 lines); hunks: -4,6 +4,7; -220,7 +221,8 @@ def _extract_tool_call_from_event(; symbols: _extract_tool_call_from_event
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37; `python/sglang/srt/function_call/base_format_detector.py` modified +15/-12; `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1; `python/sglang/srt/function_call/gpt_oss_detector.py` modified +3/-1; `python/sglang/srt/environ.py` modified +3/-0
  - tests: `test/srt/function_call/test_unknown_tool_name.py` added +69/-0
  - docs: `docs/references/environment_variables.md` modified +10/-9
- Risk and verification: The diff ships test coverage in `test/srt/function_call/test_unknown_tool_name.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13163 - Remove EBNF Composer

- Link: https://github.com/sgl-project/sglang/pull/13163
- Status/date: merged / 2025-11-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +6/-1081, 1270 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove EBNF Composer"; model line: Qwen3 Coder; category: model implementation change; main diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`; PR body summary: Resolve issue: https://github.com/sgl-project/sglang/issues/11078 We are now using json schema to constrain model output for required and named tool_choice cases, so we no longe....
- Key implementation: `test/srt/test_function_call_parser.py` modified +5/-459 (464 lines); hunks: -1,8 +1,6; -458,452 +456,6 @@ def test_detect_and_parse_with_text_before_tool_call(self):; symbols: test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp, test_pythonic_detector_ebnf, touching `test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp`; `python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344 (344 lines); hunks: -1,344 +0,0; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val, touching `EBNFComposer, get_value_rule, _handle_enum`; `test/srt/function_call/test_json_schema_constraint.py` modified +0/-52 (52 lines); hunks: -222,58 +222,6 @@ def test_tools_without_parameters(self):; symbols: test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror, touching `test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror`; `python/sglang/srt/function_call/function_call_parser.py` modified +0/-38 (38 lines); hunks: -195,41 +195,3 @@ def get_structure_constraint(; symbols: get_structure_constraint, get_ebnf, touching `get_structure_constraint, get_ebnf`.
- Code diff details:
  - `test/srt/test_function_call_parser.py` modified +5/-459 (464 lines); hunks: -1,8 +1,6; -458,452 +456,6 @@ def test_detect_and_parse_with_text_before_tool_call(self):; symbols: test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp, test_pythonic_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344 (344 lines); hunks: -1,344 +0,0; symbols: EBNFComposer, get_value_rule, _handle_enum, format_enum_val
  - `test/srt/function_call/test_json_schema_constraint.py` modified +0/-52 (52 lines); hunks: -222,58 +222,6 @@ def test_tools_without_parameters(self):; symbols: test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror
  - `python/sglang/srt/function_call/function_call_parser.py` modified +0/-38 (38 lines); hunks: -195,41 +195,3 @@ def get_structure_constraint(; symbols: get_structure_constraint, get_ebnf
  - `python/sglang/srt/function_call/step3_detector.py` modified +0/-29 (29 lines); hunks: -11,7 +11,6; -406,31 +405,3 @@ def supports_structural_tag(self) -> bool:; symbols: supports_structural_tag, structure_info, build_ebnf
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/srt/test_function_call_parser.py` modified +5/-459; `test/srt/function_call/test_json_schema_constraint.py` modified +0/-52
  - runtime: `python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344; `python/sglang/srt/function_call/function_call_parser.py` modified +0/-38; `python/sglang/srt/function_call/step3_detector.py` modified +0/-29; `python/sglang/srt/function_call/base_format_detector.py` modified +0/-27; `python/sglang/srt/function_call/kimik2_detector.py` modified +0/-19; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-13
- Risk and verification: The diff ships test coverage in `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13411 - Improve Qwen3CoderDetector with schema-aware parameter type conversion

- Link: https://github.com/sgl-project/sglang/pull/13411
- Status/date: open / 2025-11-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +155/-10, 222 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Improve Qwen3CoderDetector with schema-aware parameter type conversion"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`; PR body summary: In the original implementation of Qwen3CoderDetector, there is a function called `_safe_val` to help convert the parameter type of function calls: However, due to the unique str....
- Key implementation: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10 (145 lines); hunks: -17,15 +17,118; -84,6 +187,14 @@ def parse_streaming_increment(; symbols: _safe_val, _convert_param_value, Qwen3CoderDetector, parse_streaming_increment, touching `_safe_val, _convert_param_value, Qwen3CoderDetector`; `test/per_commit/function_call/test_function_call_parser.py` modified +20/-0 (20 lines); hunks: -1422,6 +1422,10 @@ def test_extract_tool_calls_type_conversion(self):; -1444,6 +1448,18 @@ def test_extract_tool_calls_type_conversion(self):; symbols: test_extract_tool_calls_type_conversion, test_parse_streaming_incremental, touching `test_extract_tool_calls_type_conversion, test_parse_streaming_incremental`.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10 (145 lines); hunks: -17,15 +17,118; -84,6 +187,14 @@ def parse_streaming_increment(; symbols: _safe_val, _convert_param_value, Qwen3CoderDetector, parse_streaming_increment
  - `test/per_commit/function_call/test_function_call_parser.py` modified +20/-0 (20 lines); hunks: -1422,6 +1422,10 @@ def test_extract_tool_calls_type_conversion(self):; -1444,6 +1448,18 @@ def test_extract_tool_calls_type_conversion(self):; symbols: test_extract_tool_calls_type_conversion, test_parse_streaming_incremental
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10
  - tests: `test/per_commit/function_call/test_function_call_parser.py` modified +20/-0
- Risk and verification: The diff ships test coverage in `test/per_commit/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13979 - Add Qwen3-Coder-480B to nightly tests

- Link: https://github.com/sgl-project/sglang/pull/13979
- Status/date: open / 2025-11-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +288/-171, 521 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Qwen3-Coder-480B to nightly tests"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`; no usable PR-body summary.
- Key implementation: `.github/workflows/nightly-test-nvidia.yml` modified +232/-170 (402 lines); hunks: -72,89 +72,118 @@ jobs:; -370,119 +399,152 @@ jobs:; `test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch, touching `TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch`; `test/nightly/nightly_utils.py` modified +3/-1 (4 lines); hunks: -211,6 +211,7 @@ def run_benchmark_for_model(; -228,6 +229,7 @@ def run_benchmark_for_model(; symbols: run_benchmark_for_model, touching `run_benchmark_for_model`.
- Code diff details:
  - `.github/workflows/nightly-test-nvidia.yml` modified +232/-170 (402 lines); hunks: -72,89 +72,118 @@ jobs:; -370,119 +399,152 @@ jobs:
  - `test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch
  - `test/nightly/nightly_utils.py` modified +3/-1 (4 lines); hunks: -211,6 +211,7 @@ def run_benchmark_for_model(; -228,6 +229,7 @@ def run_benchmark_for_model(; symbols: run_benchmark_for_model
- Key code excerpts:

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

- Reviewed files:
  - ci: `.github/workflows/nightly-test-nvidia.yml` modified +232/-170
  - tests: `test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0; `test/nightly/nightly_utils.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `test/nightly/nightly_utils.py`, `test/nightly/test_qwen3_coder_480b_perf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16744 - support new qwen3_coder_detector

- Link: https://github.com/sgl-project/sglang/pull/16744
- Status/date: merged / 2026-01-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/qwen3_coder_detector.py`; associated commits `858a4d659b3e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +637/-667, 1493 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support new qwen3_coder_detector"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`; PR body summary: UT !39a93fe0305fa84ec404a08211d84853 E2E Test !76e2386d0e007fb9c295f7f8b61f2904 The test results, provided by Zeyu Cui from the Qwen Team @cyente, have been confirmed to meet ex....
- Key implementation: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271 (663 lines); hunks: -1,12 +1,10; -17,334 +15,457; symbols: _safe_val, Qwen3CoderDetector, __init__, already, touching `_safe_val, Qwen3CoderDetector, __init__`.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271 (663 lines); hunks: -1,12 +1,10; -17,334 +15,457; symbols: _safe_val, Qwen3CoderDetector, __init__, already
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17965 - [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB

- Link: https://github.com/sgl-project/sglang/pull/17965
- Status/date: merged / 2026-01-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +573/-16, 705 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB"; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; PR body summary: Enable SwapAB on H200, which has been verified in this PR. And also retune the configuration of TP=8 EP=1 DeepseekV3 and TP=8 EP=2 Qwen3 Coder, in the latency scenario. Also ena....
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0 (128 lines); hunks: -0,0 +1,128; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0 (114 lines); hunks: -0,0 +1,114.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0 (128 lines); hunks: -0,0 +1,128
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0 (114 lines); hunks: -0,0 +1,114
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +4/-16 (20 lines); hunks: -8,6 +8,7; -21,7 +22,6; symbols: support_tensor_descriptor, should_enable_swap_ab, is_h20_device_and_sm90_supported
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +4/-16
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +17/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18195 - Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2

- Link: https://github.com/sgl-project/sglang/pull/18195
- Status/date: merged / 2026-02-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`; PR body summary: Add optimized Triton MoE kernel configuration for Qwen3-Coder-Next-FP8 on H100 with TP=2. `python -m sglang.bench_serving --backend sglang-oai-chat --model Qwen/Qwen3-Coder-Next....
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- Link: https://github.com/sgl-project/sglang/pull/18224
- Status/date: merged / 2026-02-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +35/-6, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ModelOPT] Support Qwen 3 Next Coder NVFP4"; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/models/qwen3_next.py`; PR body summary: This branch include important bugfix for qwen 3 coder next nvfp4 B300 `sglang serve --model vincentzed-hf/Qwen3-Coder-Next-NVFP4 --quantization modelopt_fp4` **We provide cmd to....
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM, touching `__init__, HybridLayerType, Qwen3NextForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +35/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18700 - [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu.

- Link: https://github.com/sgl-project/sglang/pull/18700
- Status/date: merged / 2026-02-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu."; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`; PR body summary: 1. The original code contained an import error that affected the proper loading of weight-related modules. 2. Duplicate dimension transformations were applied to weight.data: Th....
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7; `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ def npu_fused_moe_without_routing_weights_bf16(; -129,7 +129,7 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: npu_fused_moe_without_routing_weights_bf16, touching `npu_fused_moe_without_routing_weights_bf16`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7
  - `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ def npu_fused_moe_without_routing_weights_bf16(; -129,7 +129,7 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: npu_fused_moe_without_routing_weights_bf16
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1; `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- Link: https://github.com/sgl-project/sglang/pull/18355
- Status/date: merged / 2026-02-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +213/-74, 395 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Support Qwen3-Coder-Next on AMD platform"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; PR body summary: Enable Qwen3-Coder-Next model on AMD GPU platform. With this PR, we are able to support non-MTP (fp8 kv cache) and MTP on Qwen3-Coder-Next. - aiter_backend.py: - Handle v_head_d....
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata, touching `ForwardMetadata, __init__, init_forward_metadata`; `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj, touching `_forward_input_proj`.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18608 - [AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU

- Link: https://github.com/sgl-project/sglang/pull/18608
- Status/date: merged / 2026-03-02
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`; associated commits `98f47d817583`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +486/-0, 488 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`; PR body summary: Add CI test coverage for the Qwen3-Coder-Next model on AMD MI35x (gfx950) 8-GPU systems. This model features a hybrid architecture combining full attention (GQA) with linear att....
- Key implementation: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: get_model_path, ModelConfig, __post_init__, get_display_name, touching `get_model_path, ModelConfig, __post_init__`; `test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: TestQwen3CoderNext, setUpClass, tearDownClass, test_a_gsm8k, touching `TestQwen3CoderNext, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0 (302 lines); hunks: -0,0 +1,302; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: TestQwen3CoderNext, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0; `test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18882 - feat: Add FP8 KV cache support for Triton attention backend

- Link: https://github.com/sgl-project/sglang/pull/18882
- Status/date: merged / 2026-03-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +180/-27, 564 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Add FP8 KV cache support for Triton attention backend"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; PR body summary: The Triton attention backend does not correctly handle FP8 KV cache. It does not pass k_scale/v_scale to set_kv_buffer, causing scales to default to 1.0. In addition, the Triton....
- Key implementation: `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6 (69 lines); hunks: -7,6 +7,7; -86,6 +87,7 @@ def __init__(; symbols: __init__, forward_extend, _forward_extend_unified, touching `__init__, forward_extend, _forward_extend_unified`; `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15 (41 lines); hunks: -46,7 +46,7 @@ def _fwd_kernel_stage1(; -124,7 +124,7 @@ def _fwd_kernel_stage1(; symbols: _fwd_kernel_stage1, _decode_att_m_fwd, _fwd_grouped_kernel_stage1, touching `_fwd_kernel_stage1, _decode_att_m_fwd, _fwd_grouped_kernel_stage1`; `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6 (22 lines); hunks: -232,6 +232,8 @@ def _fwd_kernel(; -386,7 +388,7 @@ def _fwd_kernel(; symbols: _fwd_kernel, extend_attention_fwd, touching `_fwd_kernel, extend_attention_fwd`; `test/registered/quant/test_fp8kv_triton.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: TestFP8KVCacheTritonBackend, setUpClass, tearDownClass, test_gsm8k, touching `TestFP8KVCacheTritonBackend, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6 (69 lines); hunks: -7,6 +7,7; -86,6 +87,7 @@ def __init__(; symbols: __init__, forward_extend, _forward_extend_unified
  - `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15 (41 lines); hunks: -46,7 +46,7 @@ def _fwd_kernel_stage1(; -124,7 +124,7 @@ def _fwd_kernel_stage1(; symbols: _fwd_kernel_stage1, _decode_att_m_fwd, _fwd_grouped_kernel_stage1
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6 (22 lines); hunks: -232,6 +232,8 @@ def _fwd_kernel(; -386,7 +388,7 @@ def _fwd_kernel(; symbols: _fwd_kernel, extend_attention_fwd
  - `test/registered/quant/test_fp8kv_triton.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: TestFP8KVCacheTritonBackend, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/attention/test_triton_attention_kernels.py` modified +14/-0 (14 lines); hunks: -251,6 +251,8 @@ def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; -286,6 +288,8 @@ def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; symbols: _test_extend_attention_once, _test_extend_attention_sliding_window_once, _test_decode_attention_once, _test_grouped_decode_attention_once
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6; `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15; `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6
  - tests: `test/registered/quant/test_fp8kv_triton.py` added +58/-0; `test/registered/attention/test_triton_attention_kernels.py` modified +14/-0; `test/registered/attention/test_wave_attention_kernels.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/registered/attention/test_triton_attention_kernels.py`, `test/registered/attention/test_wave_attention_kernels.py`, `test/registered/quant/test_fp8kv_triton.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19736 - [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend

- Link: https://github.com/sgl-project/sglang/pull/19736
- Status/date: merged / 2026-03-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend"; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`; PR body summary: - Fix `TypeError: extend_attention_fwd() missing 1 required positional argument: 'v_scale'` crash in aiter_backend when running non-MLA speculative decoding (target_verify / dra....
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0 (2 lines); hunks: -1765,6 +1765,8 @@ def forward_extend(; symbols: forward_extend, touching `forward_extend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0 (2 lines); hunks: -1765,6 +1765,8 @@ def forward_extend(; symbols: forward_extend
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -1765,6 +1765,8 @@ def forward_extend(
+                    1.0,  # k_scale
+                    1.0,  # v_scale
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21829 - [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector

- Link: https://github.com/sgl-project/sglang/pull/21829
- Status/date: open / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +140/-0, 168 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`; PR body summary: When using --tool-call-parser qwen3_coder with streaming enabled, the arguments field of toolcall deltas is not streamed incrementally. Instead, the entire parameter value is bu....
- Key implementation: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0 (140 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,51 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param, touching `__init__, has_tool_call, _convert_param_value`.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0 (140 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,51 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/qwen3_coder_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
