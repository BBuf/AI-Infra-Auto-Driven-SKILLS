# sglang GLM-4.5 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 28
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/glm45.md` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.5V.mdx` | no direct PR-number commit |
| `docs_new/docs/basic_usage/glm45.mdx` | no direct PR-number commit |
| `python/sglang/srt/function_call/glm4_moe_detector.py` | [#8224](https://github.com/sgl-project/sglang/pull/8224), [#8445](https://github.com/sgl-project/sglang/pull/8445), [#11017](https://github.com/sgl-project/sglang/pull/11017) |
| `python/sglang/srt/models/glm4_moe.py` | [#8224](https://github.com/sgl-project/sglang/pull/8224), [#8456](https://github.com/sgl-project/sglang/pull/8456), [#8729](https://github.com/sgl-project/sglang/pull/8729), [#8804](https://github.com/sgl-project/sglang/pull/8804), [#11017](https://github.com/sgl-project/sglang/pull/11017), [#11665](https://github.com/sgl-project/sglang/pull/11665), [#11800](https://github.com/sgl-project/sglang/pull/11800) |
| `python/sglang/srt/models/glm4_moe_lite.py` | no direct PR-number commit |
| `python/sglang/srt/models/glm4_moe_nextn.py` | [#8224](https://github.com/sgl-project/sglang/pull/8224), [#11017](https://github.com/sgl-project/sglang/pull/11017), [#11800](https://github.com/sgl-project/sglang/pull/11800) |
| `test/registered/moe/test_glm4_moe_models.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-28 | [#8224](https://github.com/sgl-project/sglang/pull/8224) | merged | GLM-4.5 Model Support | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-07-28 | [#8445](https://github.com/sgl-project/sglang/pull/8445) | merged | GLM-4.5 Model Support Follow-up | `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-07-28 | [#8456](https://github.com/sgl-project/sglang/pull/8456) | merged | fix GLM4_MOE launch with compressed_tensor quant model | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-01 | [#8647](https://github.com/sgl-project/sglang/pull/8647) | merged | Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647) | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-03 | [#8729](https://github.com/sgl-project/sglang/pull/8729) | merged | use fp32 for e_score_correction_bias in GLM-4.5 | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-05 | [#8804](https://github.com/sgl-project/sglang/pull/8804) | merged | GLM-4.5 and GLM-4.5-Air both support | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-07 | [#8883](https://github.com/sgl-project/sglang/pull/8883) | merged | fix glm4 moe | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-14 | [#9136](https://github.com/sgl-project/sglang/pull/9136) | merged | [DP Attention] Refactor: adding some utility functions | `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2025-08-15 | [#9223](https://github.com/sgl-project/sglang/pull/9223) | merged | Cleanup MoE Refactor | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-17 | [#9264](https://github.com/sgl-project/sglang/pull/9264) | merged | Quick Fix GLM | `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py` |
| 2025-09-04 | [#10008](https://github.com/sgl-project/sglang/pull/10008) | merged | Optimized deepseek-v3/r1 model performance on mxfp4 run | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` |
| 2025-09-28 | [#11017](https://github.com/sgl-project/sglang/pull/11017) | merged | Update GLM-4.5 Model Doc | `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-10-16 | [#11692](https://github.com/sgl-project/sglang/pull/11692) | merged | Fix missing a2a backend init of GLM4.5 MoE Block | `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-18 | [#11665](https://github.com/sgl-project/sglang/pull/11665) | merged | fix(glm45): disable reduce scatter | `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-20 | [#11847](https://github.com/sgl-project/sglang/pull/11847) | merged | [9/N] MoE Refactor: cleanup dispatcher interfaces | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-10-24 | [#11800](https://github.com/sgl-project/sglang/pull/11800) | merged | Refactoring GLM-4.5 and GLM-4.5V related implementations | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-11-05 | [#12456](https://github.com/sgl-project/sglang/pull/12456) | merged | [fix] Handle escaped characters in GLM tool call parser to prevent double serialization | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-11-05 | [#12572](https://github.com/sgl-project/sglang/pull/12572) | merged | Register allgather/reducescatter buffers with symm memory | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-10 | [#12834](https://github.com/sgl-project/sglang/pull/12834) | merged | Refactor KTransformers heterogeneous compute with unified GPU-quantization backend | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-11-10 | [#12957](https://github.com/sgl-project/sglang/pull/12957) | merged | clean redundant code in previous PR | `python/sglang/srt/models/glm4_moe.py` |
| 2025-11-21 | [#13711](https://github.com/sgl-project/sglang/pull/13711) | open | [fused-moe] Add TP2 RTX Pro 6000 for GLM-4.5-Air and GLM-4.5V | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` |
| 2025-11-25 | [#13786](https://github.com/sgl-project/sglang/pull/13786) | merged | Overlap glm moe gemms in two cuda streams | `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-01 | [#13873](https://github.com/sgl-project/sglang/pull/13873) | merged | Feat: GLM-4.6 supports shared experts fusion | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2025-12-13 | [#13989](https://github.com/sgl-project/sglang/pull/13989) | merged | Fix GLM-4.6 tool calls don't support streaming output for arguments i… | `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-20 | [#15333](https://github.com/sgl-project/sglang/pull/15333) | merged | [GLM-4.7] GLM-4.7 Tool Parser and Doc Update | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-12-21 | [#12162](https://github.com/sgl-project/sglang/pull/12162) | merged | [Feature] Enable return routed experts | `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/managers/detokenizer_manager.py` |
| 2025-12-30 | [#15754](https://github.com/sgl-project/sglang/pull/15754) | merged | Fix: Handle empty func_name and None values in GLM MoE detectors | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-09 | [#15753](https://github.com/sgl-project/sglang/pull/15753) | merged | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py` |
| 2026-01-15 | [#12497](https://github.com/sgl-project/sglang/pull/12497) | merged | [Fix] Remove assertion for padding for NVFP4 weight scales to fix GLM 4.5 NVFP4 | `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-01-24 | [#14668](https://github.com/sgl-project/sglang/pull/14668) | merged | [NVIDIA] Add flashinfer all-to-all MOE dispatcher | `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-02-21 | [#19106](https://github.com/sgl-project/sglang/pull/19106) | open | Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-03-02 | [#17714](https://github.com/sgl-project/sglang/pull/17714) | merged | Add GLM45 tool interruption support | `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-03-03 | [#19728](https://github.com/sgl-project/sglang/pull/19728) | open | Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `test/registered/moe/test_fused_moe.py` |
| 2026-03-19 | [#20917](https://github.com/sgl-project/sglang/pull/20917) | open | fix(serving_responses): check enable_thinking for qwen3/glm45 models | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/entrypoints/openai/serving_responses.py`, `PR_DESCRIPTION.md` |
| 2026-04-09 | [#20543](https://github.com/sgl-project/sglang/pull/20543) | merged | fix: do not strip whitespace from GLM tool call values | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-04-17 | [#23067](https://github.com/sgl-project/sglang/pull/23067) | open | Fix: forward continue_final_message kwargs in Glm45Detector | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |

## Per-PR Diff Audit Cards

### PR #8224 - GLM-4.5 Model Support

- Link: https://github.com/sgl-project/sglang/pull/8224
- Status/date: merged / 2025-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; associated commits `6d6a8bc278ea`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1673/-7, 1853 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "GLM-4.5 Model Support". The diff centers on `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`. PR body context: The SGLang version of the complete implementation of the GLM-4.5 model, which includes: 1. Model implementation (with MTP and without MTP) 2. Tool call and Reasoning parser 3. t...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunks: -0,0 +1,1034; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention, touching `Glm4MoeMLP, __init__, forward`; `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN, touching `Glm4MoeModelNextN, __init__, forward`; `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__, touching `get_argument_type, parse_arguments, Glm4MoeDetector`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunks: -0,0 +1,1034; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention
  - `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN
  - `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
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
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -0,0 +1,167 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -0,0 +1,165 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` added +1034/-0; `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0; `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0
- Risk and verification: The diff ships test coverage in `test/srt/openai_server/features/test_enable_thinking.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8445 - GLM-4.5 Model Support Follow-up

- Link: https://github.com/sgl-project/sglang/pull/8445
- Status/date: merged / 2025-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/glm4_moe_detector.py`; associated commits `581e7dcb92a7`
- Diff scope read: GitHub Pull Request files API returned 6 files, +44/-15, 168 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "GLM-4.5 Model Support Follow-up". The diff centers on `python/sglang/srt/function_call/glm4_moe_detector.py`. PR body context: ## Motivation - Address PR comments in https://github.com/sgl-project/sglang/pull/8224 - support more than one tool_choice logic for GLM4-moe - add key_value_separator for qwen3...
- Key implementation: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunks: -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf, touching `build_ebnf`.
- Code diff details:
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunks: -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -156,8 +156,7 @@ def build_ebnf(self, tools: List[Tool]):
-            # GLM4Moe is not compatible with multiple tool_calls under tool_choice condition: it will output unlimited tool_calls...
-            # tool_call_separator="\\n",
+            tool_call_separator="\\n",
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2
- Risk and verification: The diff ships test coverage in `test/srt/openai_server/features/test_enable_thinking.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8456 - fix GLM4_MOE launch with compressed_tensor quant model

- Link: https://github.com/sgl-project/sglang/pull/8456
- Status/date: merged / 2025-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4_moe.py`; associated commits `25f73c6cf3c2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix GLM4_MOE launch with compressed_tensor quant model". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation fix-compressed_tensor_quant Without the change: With the fix, it could launch successfully, and gsm8k benchmark is good: ## Modifications ## Checklist - [ ] Format...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -795,6 +795,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -795,6 +795,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -795,6 +795,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+                    or self.quant_config.get_name() == "compressed_tensors"
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8647 - Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647)

- Link: https://github.com/sgl-project/sglang/pull/8647
- Status/date: merged / 2025-08-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +73/-5, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647)". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation Currently when we enable **EPMoe** of the **glm4_moe** model, it still TP splits the shared experts, which blocks us to launch **FP8 per block quant GLM4.5** with...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +73/-5 (78 lines); hunks: -387,6 +387,7 @@ def __init__(; -480,11 +481,7 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, forward_normal, touching `__init__, forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +73/-5 (78 lines); hunks: -387,6 +387,7 @@ def __init__(; -480,11 +481,7 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -387,6 +387,7 @@ def __init__(
+        self.ep_size = get_moe_expert_parallel_world_size()
@@ -480,11 +481,7 @@ def __init__(
-                **(
-                    dict(tp_rank=0, tp_size=1)
-                    if global_server_args_dict["moe_a2a_backend"].is_deepep()
-                    else {}
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +73/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8729 - use fp32 for e_score_correction_bias in GLM-4.5

- Link: https://github.com/sgl-project/sglang/pull/8729
- Status/date: merged / 2025-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4_moe.py`; associated commits `760286e3d378`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR changes model-related implementation. Title: "use fp32 for e_score_correction_bias in GLM-4.5". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: Force e_score_correction_bias to FP32 instead of using the model's own dtype. This is to maintain consistency with what was passed during training.
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -343,7 +343,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -343,7 +343,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -343,7 +343,7 @@ def __init__(
-            torch.empty((config.n_routed_experts))
+            torch.empty((config.n_routed_experts), dtype=torch.float32)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8804 - GLM-4.5 and GLM-4.5-Air both support

- Link: https://github.com/sgl-project/sglang/pull/8804
- Status/date: merged / 2025-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4_moe.py`; associated commits `a4b0d5c9e5cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "GLM-4.5 and GLM-4.5-Air both support". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: determine_num_fused_shared_experts changed
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +1/-2 (3 lines); hunks: -785,7 +785,7 @@ def __init__(; -797,7 +797,6 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-2 (3 lines); hunks: -785,7 +785,7 @@ def __init__(; -797,7 +797,6 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -785,7 +785,7 @@ def __init__(
-        self, architecture: str = "DeepseekV3ForCausalLM"
+        self, architecture: str = "Glm4MoeForCausalLM"
@@ -797,7 +797,6 @@ def determine_num_fused_shared_experts(
-            or self.config.n_routed_experts != 128
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8883 - fix glm4 moe

- Link: https://github.com/sgl-project/sglang/pull/8883
- Status/date: merged / 2025-08-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-4, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix glm4 moe". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation GLM4 inherits deepseek but #8539's update is not applied to GLM4. Some logits are broken. ## Modifications ## Accuracy Test ## Benchmark & Profiling ## Checklist -...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +19/-4 (23 lines); hunks: -527,7 +527,10 @@ def __init__(; -548,21 +551,32 @@ def forward_normal_dual_stream(; symbols: __init__, forward_normal_dual_stream, forward_normal, touching `__init__, forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +19/-4 (23 lines); hunks: -527,7 +527,10 @@ def __init__(; -548,21 +551,32 @@ def forward_normal_dual_stream(; symbols: __init__, forward_normal_dual_stream, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -527,7 +527,10 @@ def __init__(
-        self, hidden_states: torch.Tensor, can_fuse_mlp_allreduce: bool = False
+        self,
+        hidden_states: torch.Tensor,
+        can_fuse_mlp_allreduce: bool = False,
+        use_reduce_scatter: bool = False,
@@ -548,21 +551,32 @@ def forward_normal_dual_stream(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +19/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9136 - [DP Attention] Refactor: adding some utility functions

- Link: https://github.com/sgl-project/sglang/pull/9136
- Status/date: merged / 2025-08-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 21 files, +216/-159, 987 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR changes model-related implementation. Title: "[DP Attention] Refactor: adding some utility functions". The diff centers on `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`. PR body context: ## Motivation - This PR provides `get_global_gathered_buffer` and `get_local_gathered_buffer` that allow gathered_buffer creation anywhere in the repo. Here is one use case. - T...
- Key implementation: `python/sglang/srt/layers/dp_attention.py` modified +114/-27 (141 lines); hunks: -4,7 +4,7; -18,35 +18,40; symbols: DPPaddingMode, DpPaddingMode, is_max_len, is_sum_len, touching `DPPaddingMode, DpPaddingMode, is_max_len`; `python/sglang/srt/layers/logits_processor.py` modified +12/-18 (30 lines); hunks: -27,15 +27,17; -108,14 +110,12 @@ class LogitsMetadata:; symbols: LogitsMetadata, from_forward_batch, compute_dp_attention_metadata, LogitsProcessor, touching `LogitsMetadata, from_forward_batch, compute_dp_attention_metadata`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +8/-21 (29 lines); hunks: -34,9 +34,10; -349,30 +350,15 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, capture_one_batch_size, touching `__init__, capture_one_batch_size`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +8/-10 (18 lines); hunks: -40,9 +40,10; -274,13 +275,13 @@ class ForwardBatch:; symbols: ForwardBatch, prepare_mlp_sync_batch, touching `ForwardBatch, prepare_mlp_sync_batch`.
- Code diff details:
  - `python/sglang/srt/layers/dp_attention.py` modified +114/-27 (141 lines); hunks: -4,7 +4,7; -18,35 +18,40; symbols: DPPaddingMode, DpPaddingMode, is_max_len, is_sum_len
  - `python/sglang/srt/layers/logits_processor.py` modified +12/-18 (30 lines); hunks: -27,15 +27,17; -108,14 +110,12 @@ class LogitsMetadata:; symbols: LogitsMetadata, from_forward_batch, compute_dp_attention_metadata, LogitsProcessor
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +8/-21 (29 lines); hunks: -34,9 +34,10; -349,30 +350,15 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, capture_one_batch_size
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +8/-10 (18 lines); hunks: -40,9 +40,10; -274,13 +275,13 @@ class ForwardBatch:; symbols: ForwardBatch, prepare_mlp_sync_batch
  - `python/sglang/srt/layers/communicator.py` modified +7/-7 (14 lines); hunks: -32,6 +32,8; -319,7 +321,7 @@ def _scattered_to_tp_attn_full(; symbols: _scattered_to_tp_attn_full, _gather_hidden_states_and_residual, _scatter_hidden_states, _gather
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/dp_attention.py
@@ -4,7 +4,7 @@
-from typing import TYPE_CHECKING, List, Tuple
+from typing import TYPE_CHECKING, List, Optional, Tuple
@@ -18,35 +18,40 @@
+if TYPE_CHECKING:
+    from sglang.srt.configs.model_config import ModelConfig
+    from sglang.srt.server_args import ServerArgs
diff -- python/sglang/srt/layers/logits_processor.py
@@ -27,15 +27,17 @@
-    DPPaddingMode,
+    DpPaddingMode,
+    get_global_dp_buffer,
+    set_dp_buffer_len,
@@ -108,14 +110,12 @@ class LogitsMetadata:
-    gathered_buffer: Optional[torch.Tensor] = None
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -34,9 +34,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/dp_attention.py` modified +114/-27; `python/sglang/srt/layers/logits_processor.py` modified +12/-18; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +8/-21; `python/sglang/srt/model_executor/forward_batch_info.py` modified +8/-10; `python/sglang/srt/layers/communicator.py` modified +7/-7; `python/sglang/srt/model_executor/model_runner.py` modified +2/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9223 - Cleanup MoE Refactor

- Link: https://github.com/sgl-project/sglang/pull/9223
- Status/date: merged / 2025-08-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +18/-16, 90 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "Cleanup MoE Refactor". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +6/-7 (13 lines); hunks: -459,15 +459,15 @@ def forward_normal_dual_stream(; -489,10 +489,9 @@ def forward_normal(; symbols: forward_normal_dual_stream, forward_normal, touching `forward_normal_dual_stream, forward_normal`; `python/sglang/srt/layers/quantization/mxfp4.py` modified +8/-3 (11 lines); hunks: -573,15 +573,20 @@ def apply(; -602,8 +607,8 @@ def apply(; symbols: apply, touching `apply`; `python/sglang/srt/models/glm4_moe.py` modified +4/-6 (10 lines); hunks: -509,9 +509,8 @@ def forward_normal_dual_stream(; -552,9 +551,8 @@ def forward_normal(; symbols: forward_normal_dual_stream, forward_normal, touching `forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-7 (13 lines); hunks: -459,15 +459,15 @@ def forward_normal_dual_stream(; -489,10 +489,9 @@ def forward_normal(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +8/-3 (11 lines); hunks: -573,15 +573,20 @@ def apply(; -602,8 +607,8 @@ def apply(; symbols: apply
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-6 (10 lines); hunks: -509,9 +509,8 @@ def forward_normal_dual_stream(; -552,9 +551,8 @@ def forward_normal(; symbols: forward_normal_dual_stream, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -459,15 +459,15 @@ def forward_normal_dual_stream(
-            kwargs = {"hidden_states": hidden_states}
-            kwargs["topk_output"] = self.topk(hidden_states, router_logits)
-            final_hidden_states = self.experts(**kwargs)
+            topk_output = self.topk(hidden_states, router_logits)
+            final_hidden_states = self.experts(hidden_states, topk_output)
@@ -489,10 +489,9 @@ def forward_normal(
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -573,15 +573,20 @@ def apply(
+        from sglang.srt.layers.moe.topk import TopKOutputChecker
+            assert TopKOutputChecker.format_is_bypassed(topk_output)
-            top_k, router_logits = topk_output
+            top_k = topk_output.topk_config.top_k
+            router_logits = topk_output.router_logits
@@ -602,8 +607,8 @@ def apply(
diff -- python/sglang/srt/models/glm4_moe.py
@@ -509,9 +509,8 @@ def forward_normal_dual_stream(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +6/-7; `python/sglang/srt/layers/quantization/mxfp4.py` modified +8/-3; `python/sglang/srt/models/glm4_moe.py` modified +4/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9264 - Quick Fix GLM

- Link: https://github.com/sgl-project/sglang/pull/9264
- Status/date: merged / 2025-08-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-1, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Quick Fix GLM". The diff centers on `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py`. PR body context: ## Motivation GLM inference is failing after https://github.com/sgl-project/sglang/pull/8846/files#diff-5b9e34dd492bd8a14702a18b594721091092276fad1cf8736fba6ef1f33c1b04 Long ter...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +5/-0 (5 lines); hunks: -24,6 +24,7; -719,6 +720,9 @@ def __init__(; symbols: __init__, touching `__init__`; `test/srt/test_nightly_gsm8k_eval.py` modified +1/-1 (2 lines); hunks: -30,7 +30,7.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-0 (5 lines); hunks: -24,6 +24,7; -719,6 +720,9 @@ def __init__(; symbols: __init__
  - `test/srt/test_nightly_gsm8k_eval.py` modified +1/-1 (2 lines); hunks: -30,7 +30,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -24,6 +24,7 @@
+    get_pp_group,
@@ -719,6 +720,9 @@ def __init__(
+        self.pp_group = get_pp_group()
+        self.start_layer = 0
+        self.end_layer = config.num_hidden_layers
@@ -735,6 +739,7 @@ def __init__(
diff -- test/srt/test_nightly_gsm8k_eval.py
@@ -30,7 +30,7 @@
-    "zai-org/GLM-4.5-Air-FP8": 0.94,
+    "zai-org/GLM-4.5-Air-FP8": 0.78,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +5/-0
  - tests: `test/srt/test_nightly_gsm8k_eval.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_nightly_gsm8k_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10008 - Optimized deepseek-v3/r1 model performance on mxfp4 run

- Link: https://github.com/sgl-project/sglang/pull/10008
- Status/date: merged / 2025-09-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +489/-67, 850 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR optimizes an inference path or backend selection. Title: "Optimized deepseek-v3/r1 model performance on mxfp4 run". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`. PR body context: ## Motivation In order to decrease the activation tensor quantized overhead, fused the quantized behavior into the different ops (activation, layernorm, gemm, flatten) ## Modifi...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +228/-32 (260 lines); hunks: -112,6 +112,7; -129,6 +130,22; symbols: forward, __init__, touching `forward, __init__`; `python/sglang/srt/layers/quantization/quark/utils.py` modified +97/-0 (97 lines); hunks: -5,6 +5,10; -105,3 +109,96 @@ def _is_equal_or_regex_match(; symbols: deep_compare, _is_equal_or_regex_match, b_dynamic_mxfp4_quant, mxfp4_to_f32, touching `deep_compare, _is_equal_or_regex_match, b_dynamic_mxfp4_quant`; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +49/-30 (79 lines); hunks: -8,6 +8,7; -38,15 +39,6 @@ def get_min_capability(cls) -> int:; symbols: get_min_capability, process_weights_after_loading, create_weights, apply_weights, touching `get_min_capability, process_weights_after_loading, create_weights`; `python/sglang/srt/layers/rocm_linear_utils.py` added +44/-0 (44 lines); hunks: -0,0 +1,44; symbols: aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size, touching `aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +228/-32 (260 lines); hunks: -112,6 +112,7; -129,6 +130,22; symbols: forward, __init__
  - `python/sglang/srt/layers/quantization/quark/utils.py` modified +97/-0 (97 lines); hunks: -5,6 +5,10; -105,3 +109,96 @@ def _is_equal_or_regex_match(; symbols: deep_compare, _is_equal_or_regex_match, b_dynamic_mxfp4_quant, mxfp4_to_f32
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +49/-30 (79 lines); hunks: -8,6 +8,7; -38,15 +39,6 @@ def get_min_capability(cls) -> int:; symbols: get_min_capability, process_weights_after_loading, create_weights, apply_weights
  - `python/sglang/srt/layers/rocm_linear_utils.py` added +44/-0 (44 lines); hunks: -0,0 +1,44; symbols: aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size
  - `python/sglang/srt/layers/communicator.py` modified +36/-4 (40 lines); hunks: -43,15 +43,23; -207,6 +215,7 @@ def prepare_attn(; symbols: prepare_attn
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -112,6 +112,7 @@
+    is_gfx95_supported,
@@ -129,6 +130,22 @@
+_is_gfx95_supported = is_gfx95_supported()
+_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
+if _use_aiter_gfx95:
+    from sglang.srt.layers.quantization.quark.utils import quark_post_load_weights
diff -- python/sglang/srt/layers/quantization/quark/utils.py
@@ -5,6 +5,10 @@
+import torch
+from aiter.ops.triton.quant import dynamic_mxfp4_quant
+from torch import nn
@@ -105,3 +109,96 @@ def _is_equal_or_regex_match(
+# utility for tensor dims > 2 cases
+def b_dynamic_mxfp4_quant(x):
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
@@ -8,6 +8,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +228/-32; `python/sglang/srt/layers/quantization/quark/utils.py` modified +97/-0; `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +49/-30; `python/sglang/srt/layers/rocm_linear_utils.py` added +44/-0; `python/sglang/srt/layers/communicator.py` modified +36/-4; `python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py` added +13/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11017 - Update GLM-4.5 Model Doc

- Link: https://github.com/sgl-project/sglang/pull/11017
- Status/date: merged / 2025-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; associated commits `abb6781573a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +11/-12, 88 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR extends deployment docs, tests, or CI coverage. Title: "Update GLM-4.5 Model Doc". The diff centers on `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`. PR body context: This is a update of GLM-4.5 Doc, including format and model change
- Key implementation: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +3/-3 (6 lines); hunks: -39,7 +39,7 @@ def parse_arguments(json_value):; -53,7 +53,7 @@ def __init__(self):; symbols: parse_arguments, Glm4MoeDetector, __init__, has_tool_call, touching `parse_arguments, Glm4MoeDetector, __init__`; `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunks: -12,7 +12,7; -785,9 +785,9 @@ def determine_num_fused_shared_experts(; symbols: determine_num_fused_shared_experts, touching `determine_num_fused_shared_experts`; `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -48,7 +48,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +3/-3 (6 lines); hunks: -39,7 +39,7 @@ def parse_arguments(json_value):; -53,7 +53,7 @@ def __init__(self):; symbols: parse_arguments, Glm4MoeDetector, __init__, has_tool_call
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunks: -12,7 +12,7; -785,9 +785,9 @@ def determine_num_fused_shared_experts(; symbols: determine_num_fused_shared_experts
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -48,7 +48,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -39,7 +39,7 @@ def parse_arguments(json_value):
-    Detector for GLM-4.5 models.
+    Detector for GLM-4.5 and GLM-4.6 models.
@@ -53,7 +53,7 @@ def __init__(self):
-        """Check if the text contains a glm-4.5 format tool call."""
+        """Check if the text contains a glm-4.5 / glm-4.6 format tool call."""
@@ -102,7 +102,7 @@ def parse_streaming_increment(
diff -- python/sglang/srt/models/glm4_moe.py
@@ -12,7 +12,7 @@
-"""Inference-only GLM-4.5 model compatible with HuggingFace weights"""
+"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights"""
@@ -785,9 +785,9 @@ def determine_num_fused_shared_experts(
-            disable_reason = "Only GLM-4.5 on NV-platform with capability >= 80 can use shared experts fusion optimization."
+            disable_reason = "Only GLM-4.5 or GLM-4.6 on NV-platform with capability >= 80 can use shared experts fusion optimization."
-            disable_reason = "Deepseek and GLM-4.5 can not use shared experts fusion optimization under expert parallelism."
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -12,7 +12,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +3/-3; `python/sglang/srt/models/glm4_moe.py` modified +3/-3; `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11692 - Fix missing a2a backend init of GLM4.5 MoE Block

- Link: https://github.com/sgl-project/sglang/pull/11692
- Status/date: merged / 2025-10-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix missing a2a backend init of GLM4.5 MoE Block". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation Fix https://github.com/sgl-project/sglang/actions/runs/18546407737/job/52865228007#step:4:8006 ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ##...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +4/-2 (6 lines); hunks: -467,7 +467,7 @@ def __init__(; -496,7 +496,9 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, touching `__init__, forward_normal_dual_stream`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-2 (6 lines); hunks: -467,7 +467,7 @@ def __init__(; -496,7 +496,9 @@ def __init__(; symbols: __init__, forward_normal_dual_stream
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -467,7 +467,7 @@ def __init__(
-        if get_moe_a2a_backend().is_deepep():
+        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
@@ -496,7 +496,9 @@ def __init__(
-        self._enable_deepep_moe = get_moe_a2a_backend().is_deepep()
+        self._enable_a2a_moe = (
+            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11665 - fix(glm45): disable reduce scatter

- Link: https://github.com/sgl-project/sglang/pull/11665
- Status/date: merged / 2025-10-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4_moe.py`; associated commits `f7ab9554554f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "fix(glm45): disable reduce scatter". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation glm45_moe not support reduce-scatter but enable it ## Modifications disable reduce scatter
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -662,7 +662,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -662,7 +662,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -662,7 +662,7 @@ def __init__(
-            allow_reduce_scatter=True,
+            allow_reduce_scatter=False,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11847 - [9/N] MoE Refactor: cleanup dispatcher interfaces

- Link: https://github.com/sgl-project/sglang/pull/11847
- Status/date: merged / 2025-10-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +394/-428, 1948 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "[9/N] MoE Refactor: cleanup dispatcher interfaces". The diff centers on `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. PR body context: - Unified initialization for `StandardDispatcher` and `DeepEPDispatcher`. - Added `get_is_extend_in_batch` so that there is no need to pass `forward_batch` to dispatcher. - Adde...
- Key implementation: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunks: -7,6 +7,7; -15,6 +16,7; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, __init__, touching `DeepEPNormalOutput, format, DeepEPLLOutput`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunks: -20,18 +20,14; -109,23 +105,6 @@ def __init__(; symbols: __init__, forward, dispatch, touching `__init__, forward, dispatch`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunks: -11,14 +11,19; -32,6 +37,7; symbols: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__, touching `_get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported`; `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunks: -5,13 +5,15; -27,16 +29,15; symbols: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_b, touching `MooncakeDispatchOutput, __init__, dispatch_a`.
- Code diff details:
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunks: -7,6 +7,7; -15,6 +16,7; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, __init__
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunks: -20,18 +20,14; -109,23 +105,6 @@ def __init__(; symbols: __init__, forward, dispatch
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunks: -11,14 +11,19; -32,6 +37,7; symbols: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunks: -5,13 +5,15; -27,16 +29,15; symbols: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_b
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-46 (60 lines); hunks: -74,7 +74,6; -113,10 +112,7; symbols: __init__, forward_deepep, _forward_shared_experts_and_put_results, op_select_experts
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35; `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39; `python/sglang/srt/models/deepseek_v2.py` modified +14/-46; `python/sglang/srt/layers/moe/token_dispatcher/standard.py` modified +46/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11800 - Refactoring GLM-4.5 and GLM-4.5V related implementations

- Link: https://github.com/sgl-project/sglang/pull/11800
- Status/date: merged / 2025-10-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`; associated commits `4060ed37cb67`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +356/-565, 1370 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR changes model-related implementation. Title: "Refactoring GLM-4.5 and GLM-4.5V related implementations". The diff centers on `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`. PR body context: To resolve the inheritance conflict with the DeepSeek-V2 model, the GLM-4.5 model implementation has been refactored, streamlining the code and inference logic, with relevant be...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +322/-354 (676 lines); hunks: -15,7 +15,7; -27,10 +27,16; symbols: __init__, forward, Glm4MoeSparseMoeBlock, touching `__init__, forward, Glm4MoeSparseMoeBlock`; `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-14 (18 lines); hunks: -12,7 +12,8; -33,7 +34,7; symbols: forward, Glm4MoeForCausalLMNextN, __init__, touching `forward, Glm4MoeForCausalLMNextN, __init__`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +322/-354 (676 lines); hunks: -15,7 +15,7; -27,10 +27,16; symbols: __init__, forward, Glm4MoeSparseMoeBlock
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-14 (18 lines); hunks: -12,7 +12,8; -33,7 +34,7; symbols: forward, Glm4MoeForCausalLMNextN, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -15,7 +15,7 @@
-from typing import Any, Dict, Iterable, Optional, Tuple
+from typing import Any, Dict, Iterable, Optional, Tuple, Union
@@ -27,10 +27,16 @@
+    parallel_state,
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
+    use_symmetric_memory,
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -12,7 +12,8 @@
-"""Inference-only GLM-4.5, GLM-4.6 NextN Speculative Decoding."""
+"""Inference-only GLM-4.5, GLM-4.6 Speculative Decoding."""
@@ -33,7 +34,7 @@
-from sglang.srt.utils import BumpAllocator, add_prefix
+from sglang.srt.utils import add_prefix
@@ -84,14 +85,6 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +322/-354; `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/models/glm4v_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12456 - [fix] Handle escaped characters in GLM tool call parser to prevent double serialization

- Link: https://github.com/sgl-project/sglang/pull/12456
- Status/date: merged / 2025-11-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +127/-13, 172 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[fix] Handle escaped characters in GLM tool call parser to prevent double serialization". The diff centers on `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`. PR body context: ## Motivation This PR fixes a bug where the tool call parser fails when the model's output contains literal escaped characters, such as `\n`, `\"`. **The Problem:** When GLM-4 o...
- Key implementation: `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: -2191,6 +2191,109 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos, touching `test_partial_tool_call, test_array_argument_with_escaped_json, check_params`; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; -45,8 +55,13 @@ def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__, touching `get_argument_type, parse_arguments, Glm4MoeDetector`.
- Code diff details:
  - `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: -2191,6 +2191,109 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; -45,8 +55,13 @@ def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
- Key code excerpts:

```diff
diff -- test/srt/test_function_call_parser.py
@@ -2191,6 +2191,109 @@ def test_partial_tool_call(self):
+    def test_array_argument_with_escaped_json(self):
+        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
+        # Add a tool with array parameter
+        tools_with_array = [
+            Tool(
+                type="function",
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
-        try:
-            parsed_value = json.loads(json_value)
-        except:
-            parsed_value = ast.literal_eval(json_value)
+        parsed_value = json.loads(json_value)
-        return json_value, False
```

- Reviewed files:
  - tests: `test/srt/test_function_call_parser.py` modified +103/-0
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12572 - Register allgather/reducescatter buffers with symm memory

- Link: https://github.com/sgl-project/sglang/pull/12572
- Status/date: merged / 2025-11-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +250/-114, 840 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR changes model-related implementation. Title: "Register allgather/reducescatter buffers with symm memory". The diff centers on `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/moe/topk.py`. PR body context: ## Motivation Rebase version of https://github.com/sgl-project/sglang/pull/9358 on top main (including https://github.com/sgl-project/sglang/pull/12524) ## Modifications ## Accu...
- Key implementation: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +30/-22 (52 lines); hunks: -11,7 +11,11; -663,7 +667,9 @@ def apply(; symbols: apply, touching `apply`; `python/sglang/srt/layers/dp_attention.py` modified +34/-11 (45 lines); hunks: -17,6 +17,9; -86,6 +89,7 @@ class _DpGatheredBufferWrapper:; symbols: _DpGatheredBufferWrapper, set_dp_buffer_len, get_global_dp_buffer, get_local_dp_buffer, touching `_DpGatheredBufferWrapper, set_dp_buffer_len, get_global_dp_buffer`; `python/sglang/srt/layers/moe/topk.py` modified +21/-9 (30 lines); hunks: -32,12 +32,17; -279,13 +284,17 @@ def forward_cuda(; symbols: forward_cuda, forward_cpu, forward_npu, empty_topk_output, touching `forward_cuda, forward_cpu, forward_npu`; `python/sglang/srt/layers/quantization/fp8.py` modified +13/-6 (19 lines); hunks: -10,6 +10,12; -1033,9 +1039,10 @@ def apply(; symbols: apply, apply_with_router_logits, maybe_apply_hip_fused_experts, touching `apply, apply_with_router_logits, maybe_apply_hip_fused_experts`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +30/-22 (52 lines); hunks: -11,7 +11,11; -663,7 +667,9 @@ def apply(; symbols: apply
  - `python/sglang/srt/layers/dp_attention.py` modified +34/-11 (45 lines); hunks: -17,6 +17,9; -86,6 +89,7 @@ class _DpGatheredBufferWrapper:; symbols: _DpGatheredBufferWrapper, set_dp_buffer_len, get_global_dp_buffer, get_local_dp_buffer
  - `python/sglang/srt/layers/moe/topk.py` modified +21/-9 (30 lines); hunks: -32,12 +32,17; -279,13 +284,17 @@ def forward_cuda(; symbols: forward_cuda, forward_cpu, forward_npu, empty_topk_output
  - `python/sglang/srt/layers/quantization/fp8.py` modified +13/-6 (19 lines); hunks: -10,6 +10,12; -1033,9 +1039,10 @@ def apply(; symbols: apply, apply_with_router_logits, maybe_apply_hip_fused_experts
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +7/-6 (13 lines); hunks: -18,6 +18,7; -841,16 +842,16 @@ def forward(self, hidden_states: torch.Tensor, topk_output...; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -11,7 +11,11 @@
-from sglang.srt.layers.dp_attention import get_dp_global_num_tokens, get_local_dp_buffer
+from sglang.srt.layers.dp_attention import (
+    get_dp_global_num_tokens,
+    get_local_dp_buffer,
+    is_allocation_symmetric,
+)
diff -- python/sglang/srt/layers/dp_attention.py
@@ -17,6 +17,9 @@
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
+    use_symmetric_memory,
+)
@@ -86,6 +89,7 @@ class _DpGatheredBufferWrapper:
+    _dp_max_padding: bool
@@ -100,27 +104,33 @@ def set_dp_buffer_len(
diff -- python/sglang/srt/layers/moe/topk.py
@@ -32,12 +32,17 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +30/-22; `python/sglang/srt/layers/dp_attention.py` modified +34/-11; `python/sglang/srt/layers/moe/topk.py` modified +21/-9; `python/sglang/srt/layers/quantization/fp8.py` modified +13/-6; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +7/-6; `python/sglang/srt/layers/communicator.py` modified +11/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/communicator.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12834 - Refactor KTransformers heterogeneous compute with unified GPU-quantization backend

- Link: https://github.com/sgl-project/sglang/pull/12834
- Status/date: merged / 2025-11-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +494/-507, 1298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "Refactor KTransformers heterogeneous compute with unified GPU-quantization backend". The diff centers on `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. PR body context: ## Motivation Replace scattered, hard-coded paths and env-var checks with one clean architecture that can load any GPU quant method. ## Modifications Introduced KTEPWrapperMetho...
- Key implementation: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-411 (412 lines); hunks: -4,7 +4,6; -15,19 +14,10; symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, get_moe_method, touching `_mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState`; `python/sglang/srt/layers/moe/kt_ep_wrapper.py` added +393/-0 (393 lines); hunks: -0,0 +1,393; symbols: KTConfig, create_kt_config_from_server_args, mask_cpu_expert_ids, KTEPWrapperMethod, touching `KTConfig, create_kt_config_from_server_args, mask_cpu_expert_ids`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +24/-19 (43 lines); hunks: -25,6 +25,10; -36,15 +40,11; symbols: __init__, _weight_loader_physical, _weight_loader_impl, touching `__init__, _weight_loader_physical, _weight_loader_impl`; `python/sglang/srt/models/glm4_moe.py` modified +37/-0 (37 lines); hunks: -61,6 +61,7; -454,6 +455,42 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal, touching `forward, forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-411 (412 lines); hunks: -4,7 +4,6; -15,19 +14,10; symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, get_moe_method
  - `python/sglang/srt/layers/moe/kt_ep_wrapper.py` added +393/-0 (393 lines); hunks: -0,0 +1,393; symbols: KTConfig, create_kt_config_from_server_args, mask_cpu_expert_ids, KTEPWrapperMethod
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +24/-19 (43 lines); hunks: -25,6 +25,10; -36,15 +40,11; symbols: __init__, _weight_loader_physical, _weight_loader_impl
  - `python/sglang/srt/models/glm4_moe.py` modified +37/-0 (37 lines); hunks: -61,6 +61,7; -454,6 +455,42 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +25/-8 (33 lines); hunks: -19,7 +19,6; -71,8 +70,6 @@ def to_int(self) -> int:; symbols: to_int, CompressedTensorsConfig, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -4,7 +4,6 @@
-import re
@@ -15,19 +14,10 @@
-try:
-    from kt_kernel import AMXMoEWrapper
-    KTRANSFORMERS_AVAILABLE = True
-except ImportError:
diff -- python/sglang/srt/layers/moe/kt_ep_wrapper.py
@@ -0,0 +1,393 @@
+# SPDX-License-Identifier: Apache-2.0
+"""
+KT Expert Parallelism Wrapper for MoE layers.
+This module provides a generic wrapper that enables CPU-GPU expert parallelism
+for any MoE quantization method. It coordinates parallel execution of GPU experts
+(using any quantization method) and CPU experts (using AMX/AVX instructions).
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -25,6 +25,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-411; `python/sglang/srt/layers/moe/kt_ep_wrapper.py` added +393/-0; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +24/-19; `python/sglang/srt/models/glm4_moe.py` modified +37/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +25/-8; `python/sglang/srt/models/deepseek_v2.py` modified +5/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12957 - clean redundant code in previous PR

- Link: https://github.com/sgl-project/sglang/pull/12957
- Status/date: merged / 2025-11-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-37, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "clean redundant code in previous PR". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Motivation #12834 add some redundant code in python/sglang/srt/models/glm4_moe.py. This PR remove it. ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Chec...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +0/-37 (37 lines); hunks: -61,7 +61,6; -455,42 +454,6 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal, touching `forward, forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +0/-37 (37 lines); hunks: -61,7 +61,6; -455,42 +454,6 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -61,7 +61,6 @@
-from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
@@ -455,42 +454,6 @@ def forward(
-    def forward_normal_dual_stream(
-        self,
-        hidden_states: torch.Tensor,
-        should_allreduce_fusion: bool = False,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +0/-37
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13711 - [fused-moe] Add TP2 RTX Pro 6000 for GLM-4.5-Air and GLM-4.5V

- Link: https://github.com/sgl-project/sglang/pull/13711
- Status/date: open / 2025-11-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +585/-0, 596 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "[fused-moe] Add TP2 RTX Pro 6000 for GLM-4.5-Air and GLM-4.5V". The diff centers on `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`. PR body context: ## Motivation Support GLM-4.5-Air and GLM-4.5V with 2x RTX Pro 6000 ## Modifications Added a new fused_moe_triton config ## Checklist - [x] Format your code according to the For...
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -82,6 +82,7 @@ def get_model_config(; symbols: get_model_config
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json
@@ -0,0 +1,146 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0
  - other: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13786 - Overlap glm moe gemms in two cuda streams

- Link: https://github.com/sgl-project/sglang/pull/13786
- Status/date: merged / 2025-11-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +47/-3, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR optimizes an inference path or backend selection. Title: "Overlap glm moe gemms in two cuda streams". The diff centers on `python/sglang/srt/models/glm4_moe.py`. PR body context: ## Modifications Before this pr After this pr ## Accuracy Tests ## Benchmarking and Profiling Launch Bench serving before this pr: 60.40 token/s, after this pr: 66.31 token/s ##...
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunks: -448,12 +448,56 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal, touching `forward, forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunks: -448,12 +448,56 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -448,12 +448,56 @@ def forward(
-            return self.forward_normal(
-                hidden_states, should_allreduce_fusion, use_reduce_scatter
-            )
+            if (
+                self.alt_stream is not None
+                and hidden_states.shape[0] > 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +47/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13873 - Feat: GLM-4.6 supports shared experts fusion

- Link: https://github.com/sgl-project/sglang/pull/13873
- Status/date: merged / 2025-12-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +252/-24, 431 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "Feat: GLM-4.6 supports shared experts fusion". The diff centers on `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`. PR body context: Hi from novita.ai team 👋 ## Motivation Fuse shared experts with routed experts for better performance. ## Modifications The changes involve modifying the model initialization to...
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunks: -85,6 +85,7; -352,8 +353,14 @@ def __init__(; symbols: __init__, forward, forward_normal_dual_stream, touching `__init__, forward, forward_normal_dual_stream`; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +19/-2 (21 lines); hunks: -208,6 +208,7 @@ def try_get_optimal_moe_config(; -222,7 +223,15 @@ def try_get_optimal_moe_config(; symbols: try_get_optimal_moe_config, touching `try_get_optimal_moe_config`; `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunks: -139,6 +139,10 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunks: -85,6 +85,7; -352,8 +353,14 @@ def __init__(; symbols: __init__, forward, forward_normal_dual_stream
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +19/-2 (21 lines); hunks: -208,6 +208,7 @@ def try_get_optimal_moe_config(; -222,7 +223,15 @@ def try_get_optimal_moe_config(; symbols: try_get_optimal_moe_config
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunks: -139,6 +139,10 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-0 (1 lines); hunks: -434,6 +434,7 @@ def fused_experts_impl(; symbols: fused_experts_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/models/glm4_moe.py
@@ -85,6 +85,7 @@
+    log_info_on_rank0,
@@ -352,8 +353,14 @@ def __init__(
+        self.moe_ep_size = get_moe_expert_parallel_world_size()
+        self.num_fused_shared_experts = (
+            0
+            if get_global_server_args().disable_shared_experts_fusion
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py
@@ -208,6 +208,7 @@ def try_get_optimal_moe_config(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0; `python/sglang/srt/models/glm4_moe.py` modified +74/-19; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +19/-2; `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-0; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +1/-0
  - other: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13989 - Fix GLM-4.6 tool calls don't support streaming output for arguments i…

- Link: https://github.com/sgl-project/sglang/pull/13989
- Status/date: merged / 2025-12-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +527/-81, 700 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "Fix GLM-4.6 tool calls don't support streaming output for arguments i…". The diff centers on `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py`. PR body context: ## Motivation The original `glm_moe_detector_old_normal.py` implementation, unable to achieve character-level streaming output, resulting in suboptimal user experience: - #11888...
- Key implementation: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunks: -2,16 +2,43; -21,32 +48,90 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; symbols: get_argument_type, StreamState, parse_arguments, touching `get_argument_type, StreamState, parse_arguments`; `test/registered/function_call/test_function_call_parser.py` modified +29/-15 (44 lines); hunks: -2034,12 +2034,12 @@ def test_streaming_tool_call(self):; -2066,12 +2066,12 @@ def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_tool_call, test_streaming_multiple_tool_calls, test_invalid_tool_call, test_partial_tool_call, touching `test_streaming_tool_call, test_streaming_multiple_tool_calls, test_invalid_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunks: -2,16 +2,43; -21,32 +48,90 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; symbols: get_argument_type, StreamState, parse_arguments
  - `test/registered/function_call/test_function_call_parser.py` modified +29/-15 (44 lines); hunks: -2034,12 +2034,12 @@ def test_streaming_tool_call(self):; -2066,12 +2066,12 @@ def test_streaming_multiple_tool_calls(self):; symbols: test_streaming_tool_call, test_streaming_multiple_tool_calls, test_invalid_tool_call, test_partial_tool_call
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -2,16 +2,43 @@
-from typing import List
+from enum import Enum
+from typing import Any, Dict, List, Optional, Tuple
-from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc
+from sglang.srt.function_call.core_types import (
+    StreamingParseResult,
diff -- test/registered/function_call/test_function_call_parser.py
@@ -2034,12 +2034,12 @@ def test_streaming_tool_call(self):
-                        tool_calls.append({"name": "", "parameters": {}})
+                        tool_calls.append({"name": "", "parameters": ""})
-                        tc["parameters"] = tool_call_chunk.parameters
+                        tc["parameters"] += tool_call_chunk.parameters
@@ -2066,12 +2066,12 @@ def test_streaming_multiple_tool_calls(self):
-                        tool_calls.append({"name": "", "parameters": {}})
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66
  - tests: `test/registered/function_call/test_function_call_parser.py` modified +29/-15
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15333 - [GLM-4.7] GLM-4.7 Tool Parser and Doc Update

- Link: https://github.com/sgl-project/sglang/pull/15333
- Status/date: merged / 2025-12-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +809/-394, 1356 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "[GLM-4.7] GLM-4.7 Tool Parser and Doc Update". The diff centers on `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`. PR body context: Support for GLM-4.7 model's Tool Parser and partial documentation
- Key implementation: `test/registered/function_call/test_function_call_parser.py` modified +212/-388 (600 lines); hunks: -7,10 +7,10; -1159,9 +1159,6 @@ def setUp(self):; symbols: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_json_format, touching `setUp, test_detect_and_parse_xml_format, test_streaming_xml_format`; `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: StreamState, get_argument_type, _convert_to_number, parse_arguments, touching `StreamState, get_argument_type, _convert_to_number`; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunks: -43,9 +43,12 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number, touching `get_argument_type, _convert_to_number`; `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: -15,6 +15,7; -46,6 +47,7 @@ class FunctionCallParser:; symbols: FunctionCallParser, touching `FunctionCallParser`.
- Code diff details:
  - `test/registered/function_call/test_function_call_parser.py` modified +212/-388 (600 lines); hunks: -7,10 +7,10; -1159,9 +1159,6 @@ def setUp(self):; symbols: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_json_format
  - `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: StreamState, get_argument_type, _convert_to_number, parse_arguments
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunks: -43,9 +43,12 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: -15,6 +15,7; -46,6 +47,7 @@ class FunctionCallParser:; symbols: FunctionCallParser
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -12,7 +12,7
- Key code excerpts:

```diff
diff -- test/registered/function_call/test_function_call_parser.py
@@ -7,10 +7,10 @@
+from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
-from sglang.srt.function_call.mimo_detector import MiMoDetector
@@ -1159,9 +1159,6 @@ def setUp(self):
-        from transformers import AutoTokenizer
-        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")
@@ -1239,16 +1236,12 @@ def test_streaming_xml_format(self):
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -0,0 +1,584 @@
+import ast
+import json
+import logging
+import re
+from enum import Enum
+from typing import Any, Dict, List, Optional, Tuple
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -43,9 +43,12 @@ def get_argument_type(
```

- Reviewed files:
  - tests: `test/registered/function_call/test_function_call_parser.py` modified +212/-388
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2; `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0; `python/sglang/srt/models/glm4_moe.py` modified +1/-1
  - docs: `docs/basic_usage/glm45.md` modified +4/-2; `docs/advanced_features/server_arguments.md` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12162 - [Feature] Enable return routed experts

- Link: https://github.com/sgl-project/sglang/pull/12162
- Status/date: merged / 2025-12-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +646/-10, 1059 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "[Feature] Enable return routed experts". The diff centers on `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/managers/detokenizer_manager.py`. PR body context: ## Motivation As per the request from the RL community, this PR enables sglang to return routed experts (topk) during fwd for later usage in training phase. Thanks the MiMo team...
- Key implementation: `python/sglang/srt/layers/moe/routed_experts_capturer.py` added +289/-0 (289 lines); hunks: -0,0 +1,289; symbols: get_tensor_size_bytes, _RoutedExpertsDeviceCache, __init__, get_buffer_size_bytes, touching `get_tensor_size_bytes, _RoutedExpertsDeviceCache, __init__`; `python/sglang/srt/model_executor/model_runner.py` modified +61/-0 (61 lines); hunks: -97,6 +97,11; -557,6 +562,21 @@ def initialize(self, min_per_gpu_memory: float):; symbols: initialize, init_routed_experts_capturer, remote_instance_init_transfer_engine, forward, touching `initialize, init_routed_experts_capturer, remote_instance_init_transfer_engine`; `python/sglang/srt/managers/detokenizer_manager.py` modified +18/-0 (18 lines); hunks: -21,6 +21,7; -266,8 +267,24 @@ def _decode_batch_token_id_output(self, recv_obj: BatchToke...; symbols: _decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out, touching `_decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out`; `python/sglang/srt/layers/moe/topk.py` modified +11/-1 (12 lines); hunks: -48,6 +48,7; -203,6 +204,7 @@ def __init__(; symbols: __init__, forward_native, forward_cuda, forward_cpu, touching `__init__, forward_native, forward_cuda`.
- Code diff details:
  - `python/sglang/srt/layers/moe/routed_experts_capturer.py` added +289/-0 (289 lines); hunks: -0,0 +1,289; symbols: get_tensor_size_bytes, _RoutedExpertsDeviceCache, __init__, get_buffer_size_bytes
  - `python/sglang/srt/model_executor/model_runner.py` modified +61/-0 (61 lines); hunks: -97,6 +97,11; -557,6 +562,21 @@ def initialize(self, min_per_gpu_memory: float):; symbols: initialize, init_routed_experts_capturer, remote_instance_init_transfer_engine, forward
  - `python/sglang/srt/managers/detokenizer_manager.py` modified +18/-0 (18 lines); hunks: -21,6 +21,7; -266,8 +267,24 @@ def _decode_batch_token_id_output(self, recv_obj: BatchToke...; symbols: _decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out
  - `python/sglang/srt/layers/moe/topk.py` modified +11/-1 (12 lines); hunks: -48,6 +48,7; -203,6 +204,7 @@ def __init__(; symbols: __init__, forward_native, forward_cuda, forward_cpu
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +4/-0 (4 lines); hunks: -835,6 +835,7 @@ def _create_tokenized_object(; -1574,6 +1575,9 @@ def _handle_batch_output(; symbols: _create_tokenized_object, _handle_batch_output
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/routed_experts_capturer.py
@@ -0,0 +1,289 @@
+import logging
+from abc import ABC
+from typing import Optional
+import numpy as np
+import pybase64
+import torch
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -97,6 +97,11 @@
+from sglang.srt.layers.moe.routed_experts_capturer import (
+    RoutedExpertsCapturer,
+    get_global_experts_capturer,
+    set_global_experts_capturer,
+)
@@ -557,6 +562,21 @@ def initialize(self, min_per_gpu_memory: float):
diff -- python/sglang/srt/managers/detokenizer_manager.py
@@ -21,6 +21,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/routed_experts_capturer.py` added +289/-0; `python/sglang/srt/model_executor/model_runner.py` modified +61/-0; `python/sglang/srt/managers/detokenizer_manager.py` modified +18/-0; `python/sglang/srt/layers/moe/topk.py` modified +11/-1; `python/sglang/srt/managers/tokenizer_manager.py` modified +4/-0; `python/sglang/srt/managers/multi_tokenizer_mixin.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/srt/rl/test_return_routed_experts.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15754 - Fix: Handle empty func_name and None values in GLM MoE detectors

- Link: https://github.com/sgl-project/sglang/pull/15754
- Status/date: merged / 2025-12-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1513/-140, 1786 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "Fix: Handle empty func_name and None values in GLM MoE detectors". The diff centers on `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`. PR body context: - Fix AssertionError when func_name is empty by adding proper checks - Handle None values before calling strip() method to prevent AttributeError - Add boundary case handling fo...
- Key implementation: `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunks: -0,0 +1,1176; symbols: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls, touching `TestGlm47MoeDetector, setUp, test_single_tool_call`; `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunks: -40,15 +40,27 @@ def get_argument_type(; -143,6 +155,10 @@ def __init__(self):; symbols: get_argument_type, _convert_to_number, __init__, _reset_streaming_state, touching `get_argument_type, _convert_to_number, __init__`; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunks: -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; -426,10 +428,19 @@ def parse_streaming_increment(; symbols: detect_and_parse, parse_streaming_increment, touching `detect_and_parse, parse_streaming_increment`; `test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunks: -2257,6 +2257,21 @@ def check_single_todos(tool_result, expected):; symbols: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp, touching `check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector`.
- Code diff details:
  - `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunks: -0,0 +1,1176; symbols: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunks: -40,15 +40,27 @@ def get_argument_type(; -143,6 +155,10 @@ def __init__(self):; symbols: get_argument_type, _convert_to_number, __init__, _reset_streaming_state
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunks: -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; -426,10 +428,19 @@ def parse_streaming_increment(; symbols: detect_and_parse, parse_streaming_increment
  - `test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunks: -2257,6 +2257,21 @@ def check_single_todos(tool_result, expected):; symbols: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp
- Key code excerpts:

```diff
diff -- test/registered/function_call/test_glm47_moe_detector.py
@@ -0,0 +1,1176 @@
+import json
+import unittest
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.function_call.core_types import StreamingParseResult
+from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -40,15 +40,27 @@ def get_argument_type(
-    if func_name not in name2tool:
+    # Check if function exists
+    tool = name2tool.get(func_name)
+    if not tool:
+        return None
+    # Get parameters safely using getattr
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
```

- Reviewed files:
  - tests: `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0; `test/registered/function_call/test_function_call_parser.py` modified +15/-0
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`, `test/registered/function_call/test_glm47_moe_detector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15753 - Fix GLM-4.7 MoE Detector complex JSON Schema type parsing

- Link: https://github.com/sgl-project/sglang/pull/15753
- Status/date: merged / 2026-01-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +869/-20, 989 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix GLM-4.7 MoE Detector complex JSON Schema type parsing". The diff centers on `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`. PR body context: - Modify get_argument_type function to correctly parse complex JSON Schema structures including anyOf/oneOf/allOf and OpenAI-style array types like ["string", "null"] - Improve...
- Key implementation: `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunks: -3,7 +3,11; -1172,5 +1176,676 @@ def test_streamed_raw_length_multiple_empty_returns(self):; symbols: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type, touching `test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp`; `python/sglang/srt/function_call/utils.py` modified +104/-1 (105 lines); hunks: -1,6 +1,6; -101,6 +101,109 @@ def _get_tool_schema(tool: Tool) -> dict:; symbols: _get_tool_schema, infer_type_from_json_schema, get_json_schema_constraint, touching `_get_tool_schema, infer_type_from_json_schema, get_json_schema_constraint`; `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _get_value_type, _format_value_complete, touching `get_argument_type, _get_value_type, _format_value_complete`; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number, _get_value_type, _format_value_complete, touching `get_argument_type, _convert_to_number, _get_value_type`.
- Code diff details:
  - `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunks: -3,7 +3,11; -1172,5 +1176,676 @@ def test_streamed_raw_length_multiple_empty_returns(self):; symbols: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type
  - `python/sglang/srt/function_call/utils.py` modified +104/-1 (105 lines); hunks: -1,6 +1,6; -101,6 +101,109 @@ def _get_tool_schema(tool: Tool) -> dict:; symbols: _get_tool_schema, infer_type_from_json_schema, get_json_schema_constraint
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _get_value_type, _format_value_complete
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number, _get_value_type, _format_value_complete
- Key code excerpts:

```diff
diff -- test/registered/function_call/test_glm47_moe_detector.py
@@ -3,7 +3,11 @@
-from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
+from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
+from sglang.srt.function_call.glm47_moe_detector import (
+    Glm47MoeDetector,
+    get_argument_type,
+)
diff -- python/sglang/srt/function_call/utils.py
@@ -1,6 +1,6 @@
-from typing import Any, List, Literal, Optional, Tuple, Union
+from typing import Any, Dict, List, Literal, Optional, Tuple, Union
@@ -101,6 +101,109 @@ def _get_tool_schema(tool: Tool) -> dict:
+def infer_type_from_json_schema(schema: Dict[str, Any]) -> Optional[str]:
+    """
+    Infer the primary type of a parameter from JSON Schema.
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -12,6 +12,7 @@
```

- Reviewed files:
  - tests: `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3
  - runtime: `python/sglang/srt/function_call/utils.py` modified +104/-1; `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_glm47_moe_detector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12497 - [Fix] Remove assertion for padding for NVFP4 weight scales to fix GLM 4.5 NVFP4

- Link: https://github.com/sgl-project/sglang/pull/12497
- Status/date: merged / 2026-01-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-4, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix] Remove assertion for padding for NVFP4 weight scales to fix GLM 4.5 NVFP4". The diff centers on `python/sglang/srt/layers/quantization/modelopt_quant.py`. PR body context: Fix https://github.com/sgl-project/sglang/issues/12208 Fix https://github.com/sgl-project/sglang/issues/14388 I didn't run into the original error, but actually a padding error...
- Key implementation: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +7/-4 (11 lines); hunks: -1510,10 +1510,13 @@ def _slice_scale(w):; symbols: _slice_scale, touching `_slice_scale`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +7/-4 (11 lines); hunks: -1510,10 +1510,13 @@ def _slice_scale(w):; symbols: _slice_scale
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -1510,10 +1510,13 @@ def _slice_scale(w):
-                # For other backends, ensure the per-input block dimension is aligned to 16.
-                assert (
-                    weight_scale.shape[assert_dim] % block_size == 0
-                ), f"Expected {name}_weight_scale.dim({assert_dim}) to be divisible by {block_size}"
+                if weight_scale.shape[assert_dim] % 4 != 0:
+                    logger.warning(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/modelopt_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14668 - [NVIDIA] Add flashinfer all-to-all MOE dispatcher

- Link: https://github.com/sgl-project/sglang/pull/14668
- Status/date: merged / 2026-01-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +723/-16, 935 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "[NVIDIA] Add flashinfer all-to-all MOE dispatcher". The diff centers on `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`. PR body context: ~Draft PR since https://github.com/flashinfer-ai/flashinfer/pull/2102 is not yet merged into flashinfer.~ https://github.com/flashinfer-ai/flashinfer/pull/2102 is now merged. ##...
- Key implementation: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: FlashinferDispatchOutput, format, FlashinferCombineInput, FlashinferDispatcher, touching `FlashinferDispatchOutput, format, FlashinferCombineInput`; `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: CommBackend, when, TorchDistributedCommBackend, __init__, touching `CommBackend, when, TorchDistributedCommBackend`; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunks: -18,6 +18,7; -1479,6 +1480,7 @@ def _slice_scale(w):; symbols: _slice_scale, apply, touching `_slice_scale, apply`; `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0 (19 lines); hunks: -25,6 +25,8; -149,12 +151,19 @@ def format_is_deepep(; symbols: format_is_deepep, format_is_flashinfer, DispatchOutputFormat, is_standard, touching `format_is_deepep, format_is_flashinfer, DispatchOutputFormat`.
- Code diff details:
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: FlashinferDispatchOutput, format, FlashinferCombineInput, FlashinferDispatcher
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: CommBackend, when, TorchDistributedCommBackend, __init__
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunks: -18,6 +18,7; -1479,6 +1480,7 @@ def _slice_scale(w):; symbols: _slice_scale, apply
  - `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0 (19 lines); hunks: -25,6 +25,8; -149,12 +151,19 @@ def format_is_deepep(; symbols: format_is_deepep, format_is_flashinfer, DispatchOutputFormat, is_standard
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0 (9 lines); hunks: -37,6 +37,7; -117,6 +118,14 @@ def create_moe_dispatcher(moe_runner_config: MoeRunnerConfi...; symbols: create_moe_dispatcher
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py
@@ -0,0 +1,263 @@
+from __future__ import annotations
+import logging
+from typing import NamedTuple, Optional
+import torch
+from sglang.srt.environ import envs
+from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
diff -- python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py
@@ -0,0 +1,47 @@
+import torch.distributed as dist
+from sglang.srt.utils import is_flashinfer_available
+if is_flashinfer_available():
+    from flashinfer.comm.mnnvl import CommBackend
+else:
+    class CommBackend:
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -18,6 +18,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0; `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14; `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0; `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +6/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_flashinfer_dispatcher.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19106 - Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks

- Link: https://github.com/sgl-project/sglang/pull/19106
- Status/date: open / 2026-02-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +505/-37, 677 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`. PR body context: ## Summary This PR fixes `Glm4MoeLiteForCausalLM` serving failures for CompressedTensors checkpoints (e.g. `GLM-4.7-Flash-REAP-23B-A3B-AWQ-4bit`) in SGLang, while keeping regula...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunks: -1275,40 +1275,66 @@ def __init__(; -2791,8 +2817,18 @@ def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM, touching `__init__, forward, DeepseekV2ForCausalLM`; `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunks: -132,16 +132,13 @@ def forward(; -467,6 +464,17 @@ def __init__(; symbols: forward, __init__, Glm4MoeLiteForCausalLM, determine_num_fused_shared_experts, touching `forward, __init__, Glm4MoeLiteForCausalLM`; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunks: -35,6 +35,7; -93,6 +94,55 @@ class DeepseekV2WeightLoaderMixin:; symbols: DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights, touching `DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights`; `python/sglang/srt/models/glm4_moe.py` modified +16/-0 (16 lines); hunks: -1001,6 +1001,13 @@ def forward(; -1047,6 +1054,15 @@ def determine_num_fused_shared_experts(self):; symbols: forward, Glm4MoeForCausalLM, __init__, determine_num_fused_shared_experts, touching `forward, Glm4MoeForCausalLM, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunks: -1275,40 +1275,66 @@ def __init__(; -2791,8 +2817,18 @@ def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunks: -132,16 +132,13 @@ def forward(; -467,6 +464,17 @@ def __init__(; symbols: forward, __init__, Glm4MoeLiteForCausalLM, determine_num_fused_shared_experts
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunks: -35,6 +35,7; -93,6 +94,55 @@ class DeepseekV2WeightLoaderMixin:; symbols: DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights
  - `python/sglang/srt/models/glm4_moe.py` modified +16/-0 (16 lines); hunks: -1001,6 +1001,13 @@ def forward(; -1047,6 +1054,15 @@ def determine_num_fused_shared_experts(self):; symbols: forward, Glm4MoeForCausalLM, __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/configs/model_config.py` modified +14/-1 (15 lines); hunks: -1009,7 +1009,20 @@ def _verify_transformers_version(self):; symbols: _verify_transformers_version
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1275,40 +1275,66 @@ def __init__(
-        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU, we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight kernel
-        # which requires self.w_kc and self.w_vc to be packed.
-        # If not, we will use torch.bmm and weight shouldn't be packed in this case
-        has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
+        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU,
+        # we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -132,16 +132,13 @@ def forward(
-        # Some quantization wrappers store the underlying parameter as `weight_packed`.
-        if not hasattr(self.gate_up_proj, "weight"):
-            self.gate_up_proj.weight = getattr(self.gate_up_proj, "weight_packed")
-        if not hasattr(self.down_proj, "weight"):
-            self.down_proj.weight = getattr(self.down_proj, "weight_packed")
+        gate_up_proj_weight = getattr(self.gate_up_proj, "weight", None)
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -35,6 +35,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +52/-27; `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0; `python/sglang/srt/models/glm4_moe.py` modified +16/-0; `python/sglang/srt/configs/model_config.py` modified +14/-1; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +6/-1
  - tests: `test/registered/core/test_deepseek_weight_loader.py` added +86/-0; `test/registered/core/test_model_config_transformers_version.py` added +84/-0
- Risk and verification: The diff ships test coverage in `test/registered/core/test_deepseek_attention_backend_handler.py`, `test/registered/core/test_deepseek_packed_modules_mapping.py`, `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_glm4_moe_lite_shared_experts_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17714 - Add GLM45 tool interruption support

- Link: https://github.com/sgl-project/sglang/pull/17714
- Status/date: merged / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +238/-3, 318 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "Add GLM45 tool interruption support". The diff centers on `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`. PR body context: # Motivation This PR adds support for GLM-4.5 models where tool calls can interrupt reasoning blocks without an explicit closing tag. Specifically, GLM-4.5 uses the pattern ` .....
- Key implementation: `test/registered/parser/test_reasoning_parser.py` modified +182/-0 (182 lines); hunks: -3,6 +3,7; -313,6 +314,158 @@ def test_streaming_kimi_format(self):; symbols: test_streaming_kimi_format, TestGlm45Detector, setUp, test_init, touching `test_streaming_kimi_format, TestGlm45Detector, setUp`; `python/sglang/srt/parser/reasoning_parser.py` modified +56/-3 (59 lines); hunks: -25,11 +25,13 @@ def __init__(; -66,7 +68,21 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:; symbols: __init__, detect_and_parse, parse_streaming_increment, touching `__init__, detect_and_parse, parse_streaming_increment`.
- Code diff details:
  - `test/registered/parser/test_reasoning_parser.py` modified +182/-0 (182 lines); hunks: -3,6 +3,7; -313,6 +314,158 @@ def test_streaming_kimi_format(self):; symbols: test_streaming_kimi_format, TestGlm45Detector, setUp, test_init
  - `python/sglang/srt/parser/reasoning_parser.py` modified +56/-3 (59 lines); hunks: -25,11 +25,13 @@ def __init__(; -66,7 +68,21 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:; symbols: __init__, detect_and_parse, parse_streaming_increment
- Key code excerpts:

```diff
diff -- test/registered/parser/test_reasoning_parser.py
@@ -3,6 +3,7 @@
+    Glm45Detector,
@@ -313,6 +314,158 @@ def test_streaming_kimi_format(self):
+class TestGlm45Detector(CustomTestCase):
+    """Test cases for GLM45 detector with tool interruption support."""
+    def setUp(self):
+        self.detector = Glm45Detector()
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -25,11 +25,13 @@ def __init__(
+        tool_start_token: Optional[str] = None,
+        self.tool_start_token = tool_start_token
@@ -66,7 +68,21 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:
-            # Assume reasoning was truncated before `</think>` token
+            # Check for tool_start_token interruption
+            if (
```

- Reviewed files:
  - tests: `test/registered/parser/test_reasoning_parser.py` modified +182/-0
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +56/-3
- Risk and verification: The diff ships test coverage in `test/registered/parser/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19728 - Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback

- Link: https://github.com/sgl-project/sglang/pull/19728
- Status/date: open / 2026-03-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +104/-4, 179 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback". The diff centers on `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `test/registered/moe/test_fused_moe.py`. PR body context: ## Summary This fixes two ROCm regressions that blocked `GLM-4.5V-FP8` startup on MI300X when launching with: ## Root Cause There were two independent failures on ROCm: 1. `fuse...
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(; -1549,10 +1566,10 @@ def _native_dynamic_per_token_quant_fp8(output, input, s...; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8, touching `per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8`; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: -351,6 +351,12 @@ def fused_experts_impl(; symbols: fused_experts_impl, touching `fused_experts_impl`; `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: -1,9 +1,11; -239,6 +241,70 @@ def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding, touching `test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding`; `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: -3,11 +3,14; -141,6 +144,14 @@ def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding, touching `test_scaled_fp8_quant_with_padding`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(; -1549,10 +1566,10 @@ def _native_dynamic_per_token_quant_fp8(output, input, s...; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: -351,6 +351,12 @@ def fused_experts_impl(; symbols: fused_experts_impl
  - `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: -1,9 +1,11; -239,6 +241,70 @@ def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding
  - `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: -3,11 +3,14; -141,6 +144,14 @@ def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -1541,6 +1541,23 @@ def per_token_group_quant_mla_deep_gemm_masked_fp8(
+    def _copy_with_optional_row_padding(
+        dst: torch.Tensor,
+        src: torch.Tensor,
+        pad_value: float = 0.0,
+    ) -> None:
+        if dst.shape == src.shape:
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -351,6 +351,12 @@ def fused_experts_impl(
+    elif hidden_states.shape[1] == w1.shape[2]:
+        # Some ROCm FP8 MoE checkpoints load unpadded expert weights even when
+        # SGLANG_MOE_PADDING is enabled globally. In that case the runtime
+        # shape already matches the true hidden size and subtracting
+        # `padding_size` would reject a valid layout.
+        padded_size = 0
diff -- test/registered/moe/test_fused_moe.py
@@ -1,9 +1,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0
  - tests: `test/registered/moe/test_fused_moe.py` modified +66/-0; `python/sglang/test/test_custom_ops.py` modified +11/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_custom_ops.py`, `test/registered/moe/test_fused_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20917 - fix(serving_responses): check enable_thinking for qwen3/glm45 models

- Link: https://github.com/sgl-project/sglang/pull/20917
- Status/date: open / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +130/-19, 233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR adds or enables a model support/runtime surface. Title: "fix(serving_responses): check enable_thinking for qwen3/glm45 models". The diff centers on `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/entrypoints/openai/serving_responses.py`, `PR_DESCRIPTION.md`. PR body context: ## Motivation For reasoning-capable models like Qwen3, GLM45, Nemotron-3, and InternS1, the /v1/responses endpoint was not properly checking the enable_thinking parameter when d...
- Key implementation: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +18/-9 (27 lines); hunks: -796,15 +796,18 @@ def forward_extend(; -1201,11 +1204,17 @@ def forward_decode(; symbols: forward_extend, forward_decode, touching `forward_extend, forward_decode`; `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +10/-1 (11 lines); hunks: -531,7 +531,16 @@ def _make_response_output_items(; symbols: _make_response_output_items, touching `_make_response_output_items`; `PR_DESCRIPTION.md` added +69/-0 (69 lines); hunks: -0,0 +1,69; `python/pyproject.toml` modified +5/-0 (5 lines); hunks: -128,6 +128,10 @@ tracing = [; -151,6 +155,7 @@ dev = ["sglang[test]"].
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +18/-9 (27 lines); hunks: -796,15 +796,18 @@ def forward_extend(; -1201,11 +1204,17 @@ def forward_decode(; symbols: forward_extend, forward_decode
  - `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +10/-1 (11 lines); hunks: -531,7 +531,16 @@ def _make_response_output_items(; symbols: _make_response_output_items
  - `PR_DESCRIPTION.md` added +69/-0 (69 lines); hunks: -0,0 +1,69
  - `python/pyproject.toml` modified +5/-0 (5 lines); hunks: -128,6 +128,10 @@ tracing = [; -151,6 +155,7 @@ dev = ["sglang[test]"]
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -796,15 +796,18 @@ def forward_extend(
+        # 5) k_scale and v_scale are actually available (not None).
+        # If k_scale is None, converting to fp8 would produce garbage output due to
+        # missing descaling factors, so we fall back to non-fp8 path.
+            and layer.k_scale is not None
-            if layer.k_scale is not None:
-                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
diff -- python/sglang/srt/entrypoints/openai/serving_responses.py
@@ -531,7 +531,16 @@ def _make_response_output_items(
-        if self.reasoning_parser:
+        # For models like qwen3/glm45/nemotron_3/interns1, check enable_thinking
+        # to determine if reasoning should be parsed, mirroring serving_chat.py logic
+        enable_reasoning = True
+        if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
+            enable_reasoning = (
diff -- PR_DESCRIPTION.md
@@ -0,0 +1,69 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +18/-9; `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +10/-1; `python/pyproject.toml` modified +5/-0
  - other: `PR_DESCRIPTION.md` added +69/-0
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20543 - fix: do not strip whitespace from GLM tool call values

- Link: https://github.com/sgl-project/sglang/pull/20543
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +66/-2, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix: do not strip whitespace from GLM tool call values". The diff centers on `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`. PR body context: ## Motivation Fix #20542 ## Modifications Remove `arg_value = arg_value.strip()`
- Key implementation: `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunks: -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):; -2546,6 +2579,39 @@ def check_single_todos(tool_result, expected):; symbols: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp, touching `test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector`; `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunks: -759,7 +759,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs, touching `_parse_argument_pairs`; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunks: -613,7 +613,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs, touching `_parse_argument_pairs`.
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunks: -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):; -2546,6 +2579,39 @@ def check_single_todos(tool_result, expected):; symbols: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunks: -759,7 +759,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunks: -613,7 +613,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs
- Key code excerpts:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):
+    def test_whitespace_preserved_in_arg_values(self):
+        """Test that leading/trailing whitespace in arg values is not stripped."""
+        tools_with_string = [
+            Tool(
+                type="function",
+                function=Function(
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -759,7 +759,6 @@ def _parse_argument_pairs(
-            arg_value = arg_value.strip()
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -613,7 +613,6 @@ def _parse_argument_pairs(
-            arg_value = arg_value.strip()
```

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23067 - Fix: forward continue_final_message kwargs in Glm45Detector

- Link: https://github.com/sgl-project/sglang/pull/23067
- Status/date: open / 2026-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +66/-1, 94 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM-4.5, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix: forward continue_final_message kwargs in Glm45Detector". The diff centers on `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`. PR body context: ## Motivation `Glm45Detector.__init__` does not accept `continue_final_message` or `previous_content`, while `ReasoningParser.__init__` passes these kwargs whenever the request...
- Key implementation: `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunks: -518,6 +518,39 @@ def test_forced_reasoning_mode(self):; -1248,6 +1281,30 @@ def test_continue_final_message_with_request(self):; symbols: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous, touching `test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous`; `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunks: -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):; symbols: Glm45Detector, __init__, touching `Glm45Detector, __init__`.
- Code diff details:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunks: -518,6 +518,39 @@ def test_forced_reasoning_mode(self):; -1248,6 +1281,30 @@ def test_continue_final_message_with_request(self):; symbols: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous
  - `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunks: -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):; symbols: Glm45Detector, __init__
- Key code excerpts:

```diff
diff -- test/registered/unit/parser/test_reasoning_parser.py
@@ -518,6 +518,39 @@ def test_forced_reasoning_mode(self):
+    def test_continue_final_message_accepts_kwargs(self):
+        """Regression: Glm45Detector must accept continue_final_message and
+        previous_content kwargs (forwarded by ReasoningParser when the request
+        sets continue_final_message=True with a trailing assistant message)."""
+        detector = Glm45Detector(
+            continue_final_message=True,
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):
-    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
+    def __init__(
+        self,
+        stream_reasoning: bool = True,
+        force_reasoning: bool = False,
+        continue_final_message: bool = False,
```

- Reviewed files:
  - tests: `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/parser/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
