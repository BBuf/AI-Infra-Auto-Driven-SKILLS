# sglang Qwen3 Coder Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` | [#24435](https://github.com/sgl-project/sglang/pull/24435) |
| `docs_new/src/snippets/autoregressive/qwen3-coder-480b-a35b-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` | [#24435](https://github.com/sgl-project/sglang/pull/24435) |
| `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/qwen3_coder_detector.py` | [#8371](https://github.com/sgl-project/sglang/pull/8371), [#16744](https://github.com/sgl-project/sglang/pull/16744) |
| `python/sglang/srt/models/qwen3.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/amd/test_qwen3_coder_next_8gpu.py` | [#18608](https://github.com/sgl-project/sglang/pull/18608) |
| `test/registered/ascend/llm_models/test_npu_qwen3_coder_480b_a35b.py` | no direct PR-number commit |
| `test/registered/cpu/test_qwen3.py` | no direct PR-number commit |
| `test/registered/lora/test_lora_qwen3.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 36
- Total PRs in this document: 40
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

## Per-PR Diff Audit Cards

### PR #8260 - Preliminary Support for Qwen3XMLDetector

- Link: https://github.com/sgl-project/sglang/pull/8260
- Status/date: merged / 2025-07-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +153/-0, 175 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Preliminary Support for Qwen3XMLDetector"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "Preliminary Support for Qwen3XMLDetector"; the main implementation surface is `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector"; model line: Qwen3 Coder; category: bug fix; main diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`; technical summary: Covers "[Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector"; the main implementation surface is `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "GLM-4.5 Model Support"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; technical summary: Covers "GLM-4.5 Model Support"; the main implementation surface is `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "GLM-4.5 Model Support Follow-up"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`; technical summary: Covers "GLM-4.5 Model Support Follow-up"; the main implementation surface is `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Update qwen3_coder_detector.py for streaming"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`; technical summary: Covers "Update qwen3_coder_detector.py for streaming"; the main implementation surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Forward unknown tool calls instead of dropping"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`; technical summary: Covers "Forward unknown tool calls instead of dropping"; the main implementation surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Remove EBNF Composer"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`; technical summary: Covers "Remove EBNF Composer"; the main implementation surface is `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #13979 - Add Qwen3-Coder-480B to nightly tests

- Link: https://github.com/sgl-project/sglang/pull/13979
- Status/date: open / 2025-11-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +288/-171, 521 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Qwen3-Coder-480B to nightly tests"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`; technical summary: Covers "Add Qwen3-Coder-480B to nightly tests"; the main implementation surface is `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "support new qwen3_coder_detector"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`; technical summary: Covers "support new qwen3_coder_detector"; the main implementation surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB"; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; technical summary: Covers "[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[ModelOPT] Support Qwen 3 Next Coder NVFP4"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_next.py`; technical summary: Covers "[ModelOPT] Support Qwen 3 Next Coder NVFP4"; the main implementation surface is `python/sglang/srt/models/qwen3_next.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu."; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`; technical summary: Covers "[NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu."; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[AMD] Support Qwen3-Coder-Next on AMD platform"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; technical summary: Covers "[AMD] Support Qwen3-Coder-Next on AMD platform"; the main implementation surface is `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`; technical summary: Covers "[AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "feat: Add FP8 KV cache support for Triton attention backend"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; technical summary: Covers "feat: Add FP8 KV cache support for Triton attention backend"; the main implementation surface is `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend"; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`; technical summary: Covers "[AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend"; the main implementation surface is `python/sglang/srt/layers/attention/aiter_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- Link: https://github.com/sgl-project/sglang/pull/21458
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +101/-3, 152 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope, touching `__init__, forward_prepare_native, forward_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +101/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21818 - [CI] Fix lint that was not applied in #21458

- Link: https://github.com/sgl-project/sglang/pull/21818
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Fix lint that was not applied in #21458"; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[CI] Fix lint that was not applied in #21458"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def forward_prepare_npu(self, positions, hidden_states, forw...; symbols: forward_prepare_npu, forward_prepare_aiter_fused_mrope, touching `forward_prepare_npu, forward_prepare_aiter_fused_mrope`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def forward_prepare_npu(self, positions, hidden_states, forw...; symbols: forward_prepare_npu, forward_prepare_aiter_fused_mrope
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -198,7 +198,9 @@ def forward_prepare_npu(self, positions, hidden_states, forward_batch):
-    def forward_prepare_aiter_fused_mrope(self, positions, hidden_states, forward_batch):
+    def forward_prepare_aiter_fused_mrope(
+        self, positions, hidden_states, forward_batch
+    ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21829 - [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector

- Link: https://github.com/sgl-project/sglang/pull/21829
- Status/date: open / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +143/-0, 171 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`; technical summary: Covers "[Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector"; the main implementation surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +143/-0 (143 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,54 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param, touching `__init__, has_tool_call, _convert_param_value`.
- Code diff details:
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +143/-0 (143 lines); hunks: -54,6 +54,13 @@ def __init__(self):; -169,6 +176,54 @@ def _convert_param_value(; symbols: __init__, has_tool_call, _convert_param_value, _should_stream_param
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
  - runtime: `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +143/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/qwen3_coder_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21463 - Migrate all callers from /get_server_info to /server_info

- Link: https://github.com/sgl-project/sglang/pull/21463
- Status/date: merged / 2026-04-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 48 files, +74/-70, 630 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate all callers from /get_server_info to /server_info"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_deepseek_v32_mtp.py`, `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`, `test/registered/amd/test_moriep_small.py`; technical summary: Covers "Migrate all callers from /get_server_info to /server_info"; the main implementation surface is `test/registered/8-gpu-models/test_deepseek_v32_mtp.py`, `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`, `test/registered/amd/test_moriep_small.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_deepseek_v32_mtp.py` modified +4/-4 (8 lines); hunks: -76,7 +76,7 @@ def test_a_gsm8k(; -163,7 +163,7 @@ def test_a_gsm8k(; symbols: test_a_gsm8k, touching `test_a_gsm8k`; `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` modified +4/-2 (6 lines); hunks: -269,6 +269,8 @@ async def flush_cache():; -277,10 +279,10 @@ async def get_server_info():; symbols: flush_cache, get_server_info, touching `flush_cache, get_server_info`; `test/registered/amd/test_moriep_small.py` modified +3/-3 (6 lines); hunks: -145,7 +145,7 @@ def test_gsm8k(; -397,7 +397,7 @@ def test_gsm8k(; symbols: test_gsm8k, touching `test_gsm8k`; `test/registered/distributed/test_data_parallelism.py` modified +3/-3 (6 lines); hunks: -57,13 +57,13 @@ def test_update_weight(self):; symbols: test_update_weight, test_get_memory_pool_size, touching `test_update_weight, test_get_memory_pool_size`.
- Code diff details:
  - `test/registered/8-gpu-models/test_deepseek_v32_mtp.py` modified +4/-4 (8 lines); hunks: -76,7 +76,7 @@ def test_a_gsm8k(; -163,7 +163,7 @@ def test_a_gsm8k(; symbols: test_a_gsm8k
  - `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` modified +4/-2 (6 lines); hunks: -269,6 +269,8 @@ async def flush_cache():; -277,10 +279,10 @@ async def get_server_info():; symbols: flush_cache, get_server_info
  - `test/registered/amd/test_moriep_small.py` modified +3/-3 (6 lines); hunks: -145,7 +145,7 @@ def test_gsm8k(; -397,7 +397,7 @@ def test_gsm8k(; symbols: test_gsm8k
  - `test/registered/distributed/test_data_parallelism.py` modified +3/-3 (6 lines); hunks: -57,13 +57,13 @@ def test_update_weight(self):; symbols: test_update_weight, test_get_memory_pool_size
  - `test/registered/ep/test_deepep_small.py` modified +3/-3 (6 lines); hunks: -412,7 +412,7 @@ def test_gsm8k(self):; -486,7 +486,7 @@ def test_gsm8k(self):; symbols: test_gsm8k
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_deepseek_v32_mtp.py` modified +4/-4; `test/registered/amd/test_moriep_small.py` modified +3/-3; `test/registered/distributed/test_data_parallelism.py` modified +3/-3; `test/registered/ep/test_deepep_small.py` modified +3/-3
  - other: `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` modified +4/-2
  - docs: `docs/advanced_features/sgl_model_gateway.md` modified +2/-2
  - runtime: `python/sglang/bench_serving.py` modified +2/-2; `python/sglang/lang/backend/runtime_endpoint.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `python/sglang/test/bench_one_batch_server_internal.py`, `python/sglang/test/kits/cache_hit_kit.py`, `python/sglang/test/kl_test_utils.py`, `python/sglang/test/nightly_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22140 - [Fix] Fix nightly tests

- Link: https://github.com/sgl-project/sglang/pull/22140
- Status/date: merged / 2026-04-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +12/-12, 108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Fix nightly tests"; model line: Qwen3 Coder; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`; technical summary: Covers "[Fix] Fix nightly tests"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -152,6 +152,7; -167,8 +168,6; `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py` renamed +7/-6 (13 lines); hunks: -9,11 +9,11; -44,28 +44,29 @@ def test_deepseek_r1_fp4_all_variants(self):; symbols: TestDeepseekR1FP4Unified, for, test_deepseek_r1_fp4_all_variants, touching `TestDeepseekR1FP4Unified, for, test_deepseek_r1_fp4_all_variants`; `test/registered/8-gpu-models/test_qwen3_235b.py` modified +2/-1 (3 lines); hunks: -4,7 +4,7; -70,6 +70,7 @@ def test_qwen3_235b_fp8_all_variants(self):; symbols: test_qwen3_235b_fp8_all_variants, test_qwen3_235b_fp8_cp, touching `test_qwen3_235b_fp8_all_variants, test_qwen3_235b_fp8_cp`; `test/registered/lora/test_lora_qwen3.py` modified +1/-2 (3 lines); hunks: -15,7 +15,7; -27,7 +27,6; symbols: TestLoRAQwen3, touching `TestLoRAQwen3`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -152,6 +152,7; -167,8 +168,6
  - `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py` renamed +7/-6 (13 lines); hunks: -9,11 +9,11; -44,28 +44,29 @@ def test_deepseek_r1_fp4_all_variants(self):; symbols: TestDeepseekR1FP4Unified, for, test_deepseek_r1_fp4_all_variants
  - `test/registered/8-gpu-models/test_qwen3_235b.py` modified +2/-1 (3 lines); hunks: -4,7 +4,7; -70,6 +70,7 @@ def test_qwen3_235b_fp8_all_variants(self):; symbols: test_qwen3_235b_fp8_all_variants, test_qwen3_235b_fp8_cp
  - `test/registered/lora/test_lora_qwen3.py` modified +1/-2 (3 lines); hunks: -15,7 +15,7; -27,7 +27,6; symbols: TestLoRAQwen3
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` modified +1/-1 (2 lines); hunks: -42,7 +42,7; symbols: TestNvidiaNemotron3SuperNightly
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2
  - tests: `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py` renamed +7/-6; `test/registered/8-gpu-models/test_qwen3_235b.py` modified +2/-1; `test/registered/lora/test_lora_qwen3.py` modified +1/-2; `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`, `test/registered/lora/test_lora_qwen3.py`, `test/registered/perf/test_dpsk_v3_fp4_4gpu_perf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22358 - Enable DFLASH support for additional model backends

- Link: https://github.com/sgl-project/sglang/pull/22358
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +152/-5, 299 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable DFLASH support for additional model backends"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`; technical summary: Covers "Enable DFLASH support for additional model backends"; the main implementation surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture, touching `forward, get_layer, get_input_embeddings`; `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`; `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture, touching `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22739 - Restore Qwen3 rope config fallback

- Link: https://github.com/sgl-project/sglang/pull/22739
- Status/date: merged / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-2, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Restore Qwen3 rope config fallback"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "Restore Qwen3 rope config fallback"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- Link: https://github.com/sgl-project/sglang/pull/23001
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 330 files, +80364/-0, 68714 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`; technical summary: Covers "Add new Mintlify documentation site (docs_new/)"; the main implementation surface is `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in, touching `get_messages, get_current_weather, convert_dict_to_tool`; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages, touching `CapitalInfo, get_messages`; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines).
- Code diff details:
  - `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0 (2911 lines)
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0; `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0; `docs_new/docs/advanced_features/server_arguments.mdx` added +2871/-0
- Risk and verification: This is mostly docs/examples in `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`, `docs_new/.mintignore`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- Link: https://github.com/sgl-project/sglang/pull/23337
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 179 files, +16004/-8152, 23604 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Sync docs_new with legacy docs and update migration redirects"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`; technical summary: Covers "[Docs] Sync docs_new with legacy docs and update migration redirects"; the main implementation surface is `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines); `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines).
- Code diff details:
  - `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines)
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486 (932 lines)
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418; `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486; `docs_new/docs/hardware-platforms/tpu.mdx` modified +425/-468
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #19484 - [CPU] Add Qwen3.5 model optimization for CPU

- Link: https://github.com/sgl-project/sglang/pull/19484
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +768/-209, 1454 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Add Qwen3.5 model optimization for CPU"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[CPU] Add Qwen3.5 model optimization for CPU"; the main implementation surface is `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/update_config.py` modified +178/-75 (253 lines); hunks: -1,7 +1,13; -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):; symbols: get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim, touching `get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim`; `python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights, touching `__init__, weight_loader, forward`; `python/sglang/srt/models/qwen3_vl.py` modified +29/-6 (35 lines); hunks: -72,7 +72,13; -87,6 +93,9; symbols: Qwen3_VisionMLP, __init__, touching `Qwen3_VisionMLP, __init__`; `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11 (30 lines); hunks: -375,14 +375,22 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/configs/update_config.py` modified +178/-75 (253 lines); hunks: -1,7 +1,13; -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):; symbols: get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim
  - `python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights
  - `python/sglang/srt/models/qwen3_vl.py` modified +29/-6 (35 lines); hunks: -72,7 +72,13; -87,6 +93,9; symbols: Qwen3_VisionMLP, __init__
  - `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11 (30 lines); hunks: -375,14 +375,22 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +18/-2 (20 lines); hunks: -1,3 +1,4; -29,7 +30,12; symbols: mamba_v2_sharded_weight_loader, loader
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/configs/update_config.py` modified +178/-75; `python/sglang/srt/models/qwen3_5.py` modified +37/-4; `python/sglang/srt/models/qwen3_vl.py` modified +29/-6; `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11; `python/sglang/srt/layers/attention/mamba/mamba.py` modified +18/-2; `python/sglang/srt/models/qwen3_next.py` modified +14/-5
- Risk and verification: The diff ships test coverage in `test/srt/cpu/test_mamba.py`, `test/srt/cpu/test_qwen3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20520 - [NPU]TP Communications compression For Qwen3 models for NPU

- Link: https://github.com/sgl-project/sglang/pull/20520
- Status/date: merged / 2026-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +191/-10, 346 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]TP Communications compression For Qwen3 models for NPU"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2.py`; technical summary: Covers "[NPU]TP Communications compression For Qwen3 models for NPU"; the main implementation surface is `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/linear.py` modified +15/-2 (17 lines); hunks: -19,6 +19,7; -37,6 +38,7; symbols: weight_loader_v2, forward, touching `weight_loader_v2, forward`; `python/sglang/srt/layers/communicator.py` modified +12/-2 (14 lines); hunks: -22,6 +22,7; -1000,9 +1001,18 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual, touching `_gather_hidden_states_and_residual`; `python/sglang/srt/models/qwen2.py` modified +6/-2 (8 lines); hunks: -91,13 +91,17 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/linear.py` modified +15/-2 (17 lines); hunks: -19,6 +19,7; -37,6 +38,7; symbols: weight_loader_v2, forward
  - `python/sglang/srt/layers/communicator.py` modified +12/-2 (14 lines); hunks: -22,6 +22,7; -1000,9 +1001,18 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual
  - `python/sglang/srt/models/qwen2.py` modified +6/-2 (8 lines); hunks: -91,13 +91,17 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward
  - `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0 (37 lines); hunks: -0,0 +1,37; symbols: TestLlama
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/linear.py` modified +15/-2; `python/sglang/srt/layers/communicator.py` modified +12/-2; `python/sglang/srt/models/qwen2.py` modified +6/-2; `python/sglang/srt/models/qwen3.py` modified +1/-1; `python/sglang/srt/distributed/device_communicators/npu_communicator.py` modified +33/-1; `python/sglang/srt/server_args.py` modified +21/-0
  - tests: `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0; `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py` added +37/-0
- Risk and verification: The diff ships test coverage in `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21722 - feat: use structural tags to enable strict tool calling and reasoning for more models

- Link: https://github.com/sgl-project/sglang/pull/21722
- Status/date: merged / 2026-05-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +922/-49, 1197 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: use structural tags to enable strict tool calling and reasoning for more models"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv4_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`; technical summary: Covers "feat: use structural tags to enable strict tool calling and reasoning for more models"; the main implementation surface is `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv4_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_function_call_parser.py` modified +657/-9 (666 lines); hunks: -1,10 +1,16; -15,6 +21,7; symbols: test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, TestDeepSeekV4Detector, setUp, touching `test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, TestDeepSeekV4Detector`; `python/sglang/srt/function_call/deepseekv4_detector.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: DeepSeekV4Detector, __init__, get_structural_tag_name, touching `DeepSeekV4Detector, __init__, get_structural_tag_name`; `python/sglang/srt/function_call/function_call_parser.py` modified +35/-24 (59 lines); hunks: -152,7 +152,7 @@ def parse_stream_chunk(self, chunk_text: str) -> Tuple[str,...; -208,6 +208,7 @@ def get_structure_constraint(; symbols: parse_stream_chunk, get_structure_tag, get_legacy_structural_tag, get_structure_constraint, touching `parse_stream_chunk, get_structure_tag, get_legacy_structural_tag`; `python/sglang/srt/function_call/base_format_detector.py` modified +50/-2 (52 lines); hunks: -1,13 +1,19; -361,3 +367,45 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info, get_structural_tag_name, get_structural_tag, touching `structure_info, get_structural_tag_name, get_structural_tag`.
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +657/-9 (666 lines); hunks: -1,10 +1,16; -15,6 +21,7; symbols: test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, TestDeepSeekV4Detector, setUp
  - `python/sglang/srt/function_call/deepseekv4_detector.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: DeepSeekV4Detector, __init__, get_structural_tag_name
  - `python/sglang/srt/function_call/function_call_parser.py` modified +35/-24 (59 lines); hunks: -152,7 +152,7 @@ def parse_stream_chunk(self, chunk_text: str) -> Tuple[str,...; -208,6 +208,7 @@ def get_structure_constraint(; symbols: parse_stream_chunk, get_structure_tag, get_legacy_structural_tag, get_structure_constraint
  - `python/sglang/srt/function_call/base_format_detector.py` modified +50/-2 (52 lines); hunks: -1,13 +1,19; -361,3 +367,45 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info, get_structural_tag_name, get_structural_tag
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +10/-0 (10 lines); hunks: -253,3 +253,13 @@ def get_info(name: str) -> StructureInfo:; symbols: get_info
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +657/-9
  - runtime: `python/sglang/srt/function_call/deepseekv4_detector.py` added +67/-0; `python/sglang/srt/function_call/function_call_parser.py` modified +35/-24; `python/sglang/srt/function_call/base_format_detector.py` modified +50/-2; `python/sglang/srt/function_call/kimik2_detector.py` modified +10/-0; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-0; `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/spec/test_spec_utils_traverse_tree.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25825 - [Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool

- Link: https://github.com/sgl-project/sglang/pull/25825
- Status/date: merged / 2026-05-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +59/-8, 326 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`; technical summary: Covers "[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool"; the main implementation surface is `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu, touching `__init__, forward_prepare_native, forward_prepare_npu`; `python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare, touching `__init__, forward_prepare`; `python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare
  - `python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -64,6 +64,7 @@ def __init__(; -76,6 +77,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama.py` modified +16/-2; `python/sglang/srt/models/glm4_moe.py` modified +12/-1; `python/sglang/srt/models/qwen2.py` modified +9/-0; `python/sglang/srt/models/qwen2_moe.py` modified +9/-0; `python/sglang/srt/models/qwen3.py` modified +5/-1; `python/sglang/srt/models/qwen3_moe.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama_eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25983 - feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext

- Link: https://github.com/sgl-project/sglang/pull/25983
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 77 files, +1227/-905, 5236 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; technical summary: Covers "feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode, touching `get_spec_info, run_once, maybe_init_ngram_embedding`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once, touching `capture_one_batch_size, run_once`; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size, touching `warmup_compile, _cache_loc_dtype, capture_one_batch_size`; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp, touching `_get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp
  - `python/sglang/srt/model_executor/forward_context.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: ForwardContext, set_forward_context, has_forward_context, get_forward_context
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44; `python/sglang/srt/model_executor/forward_context.py` added +84/-0; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +39/-38
- Risk and verification: The diff ships test coverage in `test/manual/attention/test_flashattn_backend.py`, `test/manual/attention/test_flashattn_mla_backend.py`, `test/manual/attention/test_prefix_chunk_info.py`, `test/manual/attention/test_trtllm_mla_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26798 - Make qwen3's set_embed_and_head idempotent

- Link: https://github.com/sgl-project/sglang/pull/26798
- Status/date: merged / 2026-05-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Make qwen3's set_embed_and_head idempotent"; model line: Qwen3 Coder; category: model implementation change; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "Make qwen3's set_embed_and_head idempotent"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +4/-2 (6 lines); hunks: -674,8 +674,10 @@ def get_embed_and_head(self):; symbols: get_embed_and_head, set_embed_and_head, touching `get_embed_and_head, set_embed_and_head`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +4/-2 (6 lines); hunks: -674,8 +674,10 @@ def get_embed_and_head(self):; symbols: get_embed_and_head, set_embed_and_head
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24435 - Update Qwen3-Coder docs_new NVIDIA guidance

- Link: https://github.com/sgl-project/sglang/pull/24435
- Status/date: merged / 2026-06-01
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`; associated commits `106092123f01`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +285/-20, 361 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update Qwen3-Coder docs_new NVIDIA guidance"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`; technical summary: Covers "Update Qwen3-Coder docs_new NVIDIA guidance"; the main implementation surface is `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` modified +273/-10 (283 lines); hunks: -47,7 +47,7 @@ This section provides deployment configurations verified on AM...; -292,7 +292,7 @@ Arguments: {"code": "def factorial(n):\n if n == 0 or n == 1...; symbols: factorial, touching `factorial`; `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` modified +12/-10 (22 lines); hunks: -45,8 +45,8 @@ export const Qwen3CoderDeployment = () => {; -97,24 +97,26 @@ export const Qwen3CoderDeployment = () => {.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` modified +273/-10 (283 lines); hunks: -47,7 +47,7 @@ This section provides deployment configurations verified on AM...; -292,7 +292,7 @@ Arguments: {"code": "def factorial(n):\n if n == 0 or n == 1...; symbols: factorial
  - `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` modified +12/-10 (22 lines); hunks: -45,8 +45,8 @@ export const Qwen3CoderDeployment = () => {; -97,24 +97,26 @@ export const Qwen3CoderDeployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx` modified +273/-10; `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx` modified +12/-10
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- Link: https://github.com/sgl-project/sglang/pull/27001
- Status/date: merged / 2026-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +11/-471, 936 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`; technical summary: Covers "[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests"; the main implementation surface is `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x`; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x`; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x`; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models, touching `get_model_path, ModelConfig, get_display_name`.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -39,21 +34,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` modified +1/-35
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27248 - [Doc][CPU]Update Cookbook with Xeon support info

- Link: https://github.com/sgl-project/sglang/pull/27248
- Status/date: merged / 2026-06-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +443/-121, 1524 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc][CPU]Update Cookbook with Xeon support info"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`; technical summary: Covers "[Doc][CPU]Update Cookbook with Xeon support info"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20 (85 lines); hunks: -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {; -57,8 +67,11 @@ export const DeepSeekV3Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15 (65 lines); hunks: -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {; -26,9 +28,9 @@ export const DeepSeekV31Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7 (56 lines); hunks: -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {; -35,9 +46,9 @@ export const DeepSeekR1BasicDeployment = () => {; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18 (51 lines); hunks: -64,7 +64,8 @@ export const Qwen35Deployment = () => {; -74,12 +75,13 @@ export const Qwen35Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20 (85 lines); hunks: -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {; -57,8 +67,11 @@ export const DeepSeekV3Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15 (65 lines); hunks: -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {; -26,9 +28,9 @@ export const DeepSeekV31Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7 (56 lines); hunks: -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {; -35,9 +46,9 @@ export const DeepSeekR1BasicDeployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18 (51 lines); hunks: -64,7 +64,8 @@ export const Qwen35Deployment = () => {; -74,12 +75,13 @@ export const Qwen35Deployment = () => {
  - `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` modified +31/-10 (41 lines); hunks: -13,7 +13,8 @@ export const Hunyuan3PreviewDeployment = () => {; -35,18 +36,22 @@ export const Hunyuan3PreviewDeployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20; `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15; `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18; `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` modified +31/-10; `docs_new/src/snippets/autoregressive/deepseek-ocr-deployment.jsx` modified +29/-9
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- Link: https://github.com/sgl-project/sglang/pull/23906
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 160 files, +5197/-3068, 12233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Cuda Graph Runner/Backend Refactor"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`; technical summary: Covers "[Refactor] Cuda Graph Runner/Backend Refactor"; the main implementation surface is `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool, touching `freeze_gc, _to_torch, patch_model`; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype, touching `PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled`; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode, touching `_make_graph_key, build_replay_fb_view, _allocate_decode_buffers`; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers, touching `BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank`.
- Code diff details:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode
  - `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: _grouped_foreach_copy_, foreach_copy, DecodeInputBuffers, create
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541; `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0; `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` added +225/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/doc_patch.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13411 - Improve Qwen3CoderDetector with schema-aware parameter type conversion

- Link: https://github.com/sgl-project/sglang/pull/13411
- Status/date: closed / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +155/-10, 222 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Improve Qwen3CoderDetector with schema-aware parameter type conversion"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`; technical summary: Covers "Improve Qwen3CoderDetector with schema-aware parameter type conversion"; the main implementation surface is `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: Qwen3 Coder; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention, touching `ApertusMLP, __init__, forward`; `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales, touching `__init__, forward, load_kv_cache_scales`; `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__, touching `_resolve_moe_input_pad_multiple, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28697 - [docs] Add B300 cookbook deployment options

- Link: https://github.com/sgl-project/sglang/pull/28697
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +503/-69, 1291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Add B300 cookbook deployment options"; model line: Qwen3 Coder; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; technical summary: Covers "[docs] Add B300 cookbook deployment options"; the main implementation surface is `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28810 - [CI] Remove deprecated test/srt legacy CI setup

- Link: https://github.com/sgl-project/sglang/pull/28810
- Status/date: merged / 2026-06-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +2/-5773, 5826 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Remove deprecated test/srt legacy CI setup"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py`; technical summary: Covers "[CI] Remove deprecated test/srt legacy CI setup"; the main implementation surface is `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: layernorm, rotary_emb, native_torch, native_torch_int8, touching `layernorm, rotary_emb, native_torch`; `test/srt/cpu/utils.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: parametrize, decorator, wrapper, SiluAndMul, touching `parametrize, decorator, wrapper`; `test/srt/cpu/test_norm.py` removed +0/-432 (432 lines); hunks: -1,432 +0,0; symbols: TestNorm, _forward_native, _norm, _gemma3_rmsnorm_native, touching `TestNorm, _forward_native, _norm`; `test/srt/cpu/test_extend.py` removed +0/-400 (400 lines); hunks: -1,400 +0,0; symbols: TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend, _run_sdpa_forward_extend_sink, touching `TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend`.
- Code diff details:
  - `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: layernorm, rotary_emb, native_torch, native_torch_int8
  - `test/srt/cpu/utils.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: parametrize, decorator, wrapper, SiluAndMul
  - `test/srt/cpu/test_norm.py` removed +0/-432 (432 lines); hunks: -1,432 +0,0; symbols: TestNorm, _forward_native, _norm, _gemma3_rmsnorm_native
  - `test/srt/cpu/test_extend.py` removed +0/-400 (400 lines); hunks: -1,400 +0,0; symbols: TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend, _run_sdpa_forward_extend_sink
  - `test/srt/cpu/test_mamba.py` removed +0/-394 (394 lines); hunks: -1,394 +0,0; symbols: l2norm, torch_chunk_gated_delta_rule, chunk_gated_delta_rule_update, torch_recurrent_gated_delta_rule
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440; `test/srt/cpu/utils.py` removed +0/-440; `test/srt/cpu/test_norm.py` removed +0/-432; `test/srt/cpu/test_extend.py` removed +0/-400; `test/srt/cpu/test_mamba.py` removed +0/-394; `test/srt/cpu/test_moe.py` removed +0/-352
- Risk and verification: The diff ships test coverage in `test/README.md`, `test/srt/cpu/arm64/test_moe.py`, `test/srt/cpu/test_activation.py`, `test/srt/cpu/test_binding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
