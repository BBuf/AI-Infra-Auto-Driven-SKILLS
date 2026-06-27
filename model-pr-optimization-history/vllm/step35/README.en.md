# vllm Step 3.5 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/models/multimodal/processing/test_step3_vl_image_embeds.py` | no direct PR-number commit |
| `tests/reasoning/test_step3p5_reasoning_parser.py` | [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `tests/tool_parsers/test_step3p5_tool_parser.py` | [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/model_executor/models/step3_text.py` | no direct PR-number commit |
| `vllm/model_executor/models/step3_vl.py` | no direct PR-number commit |
| `vllm/model_executor/models/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33755](https://github.com/vllm-project/vllm/pull/33755), [#34478](https://github.com/vllm-project/vllm/pull/34478), [#41892](https://github.com/vllm-project/vllm/pull/41892) |
| `vllm/model_executor/models/step3p5_mtp.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523) |
| `vllm/reasoning/step3p5_reasoning_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `vllm/tool_parsers/step3p5_tool_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/transformers_utils/configs/step3_vl.py` | no direct PR-number commit |
| `vllm/transformers_utils/configs/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523) |
| `vllm/transformers_utils/processors/step3_vl.py` | no direct PR-number commit |
| `vllm/v1/spec_decode/step3p5.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 11
- Total PRs in this document: 5
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-13 | [#41892](https://github.com/vllm-project/vllm/pull/41892) | merged | [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports) | `vllm/model_executor/models/step3p5.py` |
| 2026-05-18 | [#42224](https://github.com/vllm-project/vllm/pull/42224) | merged | [MM][CG] Enable encoder Cudagraph for Step3VL | `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py` |
| 2026-06-03 | [#44346](https://github.com/vllm-project/vllm/pull/44346) | merged | [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers | `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | merged | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |

## Per-PR Diff Audit Cards

### PR #41892 - [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports)

- Link: https://github.com/vllm-project/vllm/pull/41892
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/step3p5.py`; associated commits `3b1ef03be4a3`
- Diff scope read: GitHub Pull Request files API returned 3 files, +46/-4, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports)"; model line: Step 3.5; category: bug fix; main diff: `vllm/model_executor/models/step3p5.py`; technical summary: Covers "[Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports)"; the main implementation surface is `vllm/model_executor/models/step3p5.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/step3p5.py` modified +6/-0 (6 lines); hunks: -817,6 +817,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, Step3p5ForCausalLM, touching `load_weights, Step3p5ForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/step3p5.py` modified +6/-0 (6 lines); hunks: -817,6 +817,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, Step3p5ForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/step3p5.py
@@ -817,6 +817,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+    # Required so quantization exclude lists match fused module prefixes.
+    packed_modules_mapping = {
+        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
+        "gate_up_proj": ["gate_proj", "up_proj"],
+    }
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/step3p5.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`, `vllm/model_executor/models/step3p5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42224 - [MM][CG] Enable encoder Cudagraph for Step3VL

- Link: https://github.com/vllm-project/vllm/pull/42224
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +384/-22, 534 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Enable encoder Cudagraph for Step3VL"; model line: Step 3.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py`; technical summary: Covers "[MM][CG] Enable encoder Cudagraph for Step3VL"; the main implementation surface is `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/step3_vl.py` modified +323/-2 (325 lines); hunks: -46,7 +46,12; -487,7 +492,9 @@ def forward(; symbols: forward, Step3VLForConditionalGeneration, __init__, device, touching `forward, Step3VLForConditionalGeneration, __init__`; `vllm/model_executor/models/interfaces.py` modified +21/-0 (21 lines); hunks: -1594,6 +1594,27 @@ def select_encoder_cudagraph_items(; symbols: select_encoder_cudagraph_items, postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs, touching `select_encoder_cudagraph_items, postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs`; `vllm/model_executor/models/utils.py` modified +16/-0 (16 lines); hunks: -884,3 +884,19 @@ def get_layer_index(feature_layer_index: int, num_hidden_la...; symbols: get_layer_index, scatter_output_slices, touching `get_layer_index, scatter_output_slices`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0 (12 lines); hunks: -41,6 +41,13 @@ def qwen_vl_chat_template(content: str) -> str:; -90,6 +97,11 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, step3_vl_chat_template, touching `qwen_vl_chat_template, step3_vl_chat_template`.
- Code diff details:
  - `vllm/model_executor/models/step3_vl.py` modified +323/-2 (325 lines); hunks: -46,7 +46,12; -487,7 +492,9 @@ def forward(; symbols: forward, Step3VLForConditionalGeneration, __init__, device
  - `vllm/model_executor/models/interfaces.py` modified +21/-0 (21 lines); hunks: -1594,6 +1594,27 @@ def select_encoder_cudagraph_items(; symbols: select_encoder_cudagraph_items, postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs
  - `vllm/model_executor/models/utils.py` modified +16/-0 (16 lines); hunks: -884,3 +884,19 @@ def get_layer_index(feature_layer_index: int, num_hidden_la...; symbols: get_layer_index, scatter_output_slices
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0 (12 lines); hunks: -41,6 +41,13 @@ def qwen_vl_chat_template(content: str) -> str:; -90,6 +97,11 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, step3_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0 (2 lines); hunks: -77,6 +77,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...; -89,6 +90,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/step3_vl.py
@@ -46,7 +46,12 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsEncoderCudaGraph,
+    SupportsMultiModal,
+    SupportsPP,
diff -- vllm/model_executor/models/interfaces.py
@@ -1594,6 +1594,27 @@ def select_encoder_cudagraph_items(
+    def postprocess_encoder_output(
+        self,
+        output: torch.Tensor,
+        indices: list[int],
+        per_item_out_tokens: list[int],
+        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
diff -- vllm/model_executor/models/utils.py
@@ -884,3 +884,19 @@ def get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/step3_vl.py` modified +323/-2; `vllm/model_executor/models/interfaces.py` modified +21/-0; `vllm/model_executor/models/utils.py` modified +16/-0; `vllm/model_executor/models/step_vl.py` modified +1/-0; `vllm/v1/worker/encoder_cudagraph.py` modified +8/-20
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +2/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44346 - [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers

- Link: https://github.com/vllm-project/vllm/pull/44346
- Status/date: merged / 2026-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +20/-15, 178 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers"; model line: Step 3.5; category: model implementation change; main diff: `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`; technical summary: Covers "[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers"; the main implementation surface is `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/utils.py` modified +7/-0 (7 lines); hunks: -3,6 +3,7; -31,6 +32,12; symbols: safe_literal_eval, partial_tag_overlap, touching `safe_literal_eval, partial_tag_overlap`; `vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3 (6 lines); hunks: -1,7 +1,6; -27,6 +26,7; symbols: _try_parse_wildcard_number, _deserialize, touching `_try_parse_wildcard_number, _deserialize`; `vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,7 +1,6; -28,7 +27,7; symbols: _parse_arguments, touching `_parse_arguments`; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,6 +1,5; -26,7 +25,7; symbols: _end_element, touching `_end_element`.
- Code diff details:
  - `vllm/tool_parsers/utils.py` modified +7/-0 (7 lines); hunks: -3,6 +3,7; -31,6 +32,12; symbols: safe_literal_eval, partial_tag_overlap
  - `vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3 (6 lines); hunks: -1,7 +1,6; -27,6 +26,7; symbols: _try_parse_wildcard_number, _deserialize
  - `vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,7 +1,6; -28,7 +27,7; symbols: _parse_arguments
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,6 +1,5; -26,7 +25,7; symbols: _end_element
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-2 (4 lines); hunks: -11,7 +11,6; -42,6 +41,7; symbols: _deserialize
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/utils.py
@@ -3,6 +3,7 @@
+import warnings
@@ -31,6 +32,12 @@
+def safe_literal_eval(text: str):
+    with warnings.catch_warnings():
+        warnings.simplefilter("ignore", SyntaxWarning)
+        return ast.literal_eval(text)
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -1,7 +1,6 @@
-import ast
@@ -27,6 +26,7 @@
+from vllm.tool_parsers.utils import safe_literal_eval
@@ -183,13 +183,13 @@ def _try_parse_wildcard_number(value: str) -> int | float | None:
-        """Deserialize a string value using json.loads then ast.literal_eval."""
+        """Deserialize a string value using json.loads then safe_literal_eval."""
diff -- vllm/tool_parsers/minicpm5xml_tool_parser.py
@@ -1,7 +1,6 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/utils.py` modified +7/-0; `vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3; `vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-2; `vllm/tool_parsers/poolside_v1_tool_parser.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/glm4_moe_tool_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- Link: https://github.com/vllm-project/vllm/pull/41184
- Status/date: merged / 2026-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 90 files, +2734/-2027, 7329 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; model line: Step 3.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`; technical summary: Covers "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts, touching `FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE`; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method, touching `FusedMoeWeightScaleSupported, RoutedExperts, __init__`; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward, touching `register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward`; `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__, touching `FusedMoEWithLoRA, __init__`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts
  - `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward
  - `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__
  - `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1 (107 lines); hunks: -13,7 +13,7; -1633,3 +1633,108 @@ def maybe_remap_kv_scale_name(name: str, params_dict: di...; symbols: maybe_remap_kv_scale_name, maybe_remap_moe_expert_param_name, remap_moe_expert_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1,1424 +1,404 @@
-from collections.abc import Callable, Iterable
-from enum import Enum
-from typing import Literal, cast, overload
+from collections.abc import Callable
+from typing import Any
-from torch.nn.parameter import UninitializedParameter
diff -- vllm/model_executor/layers/fused_moe/routed_experts.py
@@ -0,0 +1,1144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable, Iterable
+from enum import Enum
+from typing import TYPE_CHECKING, Any, Literal, cast, overload
+import torch
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner.py
@@ -1,28 +1,39 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82; `vllm/lora/layers/fused_moe.py` modified +76/-43; `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1; `vllm/model_executor/layers/fused_moe/runner/moe_runner_interface.py` modified +102/-2
- Risk and verification: The diff ships test coverage in `tests/distributed/test_eplb_fused_moe_layer.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43586 - [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR

- Link: https://github.com/vllm-project/vllm/pull/43586
- Status/date: merged / 2026-06-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +809/-69, 1559 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; model line: Step 3.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; technical summary: Covers "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features, touching `get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__`; `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata, touching `BudgetGraphMetadata`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template, touching `VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template`; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2, touching `run_tarsier2`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features
  - `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template
  - `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2
  - `vllm/model_executor/models/interfaces.py` modified +5/-0 (5 lines); hunks: -1623,6 +1623,7 @@ def postprocess_encoder_output(; -1643,6 +1644,7 @@ def prepare_encoder_cudagraph_capture_inputs(; symbols: postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs, prepare_encoder_cudagraph_replay_buffers, encoder_cudagraph_forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -4,7 +4,7 @@
-from typing import Annotated, Literal
+from typing import Annotated, Any, Literal
@@ -15,6 +15,7 @@
+    SupportsEncoderCudaGraph,
@@ -52,6 +53,7 @@
+    IMAGE_SIZE,
diff -- docs/design/cuda_graphs_multimodal.md
@@ -2,6 +2,8 @@
+For two-tower vision encoders (e.g., DeepSeek-OCR's SAM + CLIP with dynamic tiling), a **dual-path graph** mode captures two independent sets of CUDA graphs — one for the global i
@@ -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on the host side. Th
+For two-tower vision encoders such as DeepSeek-OCR (SAM + CLIP with dynamic tiling), the global image path and local patch path have independent token profiles (272 tokens per glo
@@ -37,17 +41,57 @@ class BudgetGraphMetadata:
+When `EncoderCudaGraphConfig.enable_dual_path_graph` is `True`, the manager generates two independent budget lists — `global_token_budgets` (multiples of `global_token_per_image`)
+For dual-path models, the manager routes to `_execute_local_dual_path()`, which constrains both global and local token budgets simultaneously during packing (see [Dual-Path graph
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -29,6 +29,7 @@ class VitCudagraphTestConfig:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5; `vllm/model_executor/models/interfaces.py` modified +5/-0; `vllm/model_executor/models/step3_vl.py` modified +5/-0; `vllm/model_executor/models/glm4_1v.py` modified +4/-0; `vllm/model_executor/models/internvl.py` modified +4/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +63/-16; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
