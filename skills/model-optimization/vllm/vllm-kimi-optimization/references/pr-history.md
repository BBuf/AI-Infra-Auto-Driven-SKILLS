# vllm Kimi K2/K2.5/Linear/VL PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 1
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/reasoning/test_kimi_k2_reasoning_parser.py` | [#37438](https://github.com/vllm-project/vllm/pull/37438) |
| `tests/tool_parsers/test_kimi_k2_tool_parser.py` | [#31207](https://github.com/vllm-project/vllm/pull/31207), [#38579](https://github.com/vllm-project/vllm/pull/38579) |
| `vllm/model_executor/models/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127), [#36903](https://github.com/vllm-project/vllm/pull/36903) |
| `vllm/model_executor/models/kimi_k25.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33320](https://github.com/vllm-project/vllm/pull/33320), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#33562](https://github.com/vllm-project/vllm/pull/33562), [#33876](https://github.com/vllm-project/vllm/pull/33876), [#34427](https://github.com/vllm-project/vllm/pull/34427), [#34501](https://github.com/vllm-project/vllm/pull/34501), [#36192](https://github.com/vllm-project/vllm/pull/36192), [#36361](https://github.com/vllm-project/vllm/pull/36361), [#37693](https://github.com/vllm-project/vllm/pull/37693), [#39344](https://github.com/vllm-project/vllm/pull/39344) |
| `vllm/model_executor/models/kimi_k25_vit.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#34501](https://github.com/vllm-project/vllm/pull/34501) |
| `vllm/model_executor/models/kimi_linear.py` | [#27809](https://github.com/vllm-project/vllm/pull/27809), [#27834](https://github.com/vllm-project/vllm/pull/27834), [#27885](https://github.com/vllm-project/vllm/pull/27885), [#37371](https://github.com/vllm-project/vllm/pull/37371) |
| `vllm/model_executor/models/kimi_vl.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#16833](https://github.com/vllm-project/vllm/pull/16833), [#17156](https://github.com/vllm-project/vllm/pull/17156), [#21769](https://github.com/vllm-project/vllm/pull/21769), [#23114](https://github.com/vllm-project/vllm/pull/23114), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#31738](https://github.com/vllm-project/vllm/pull/31738) |
| `vllm/model_executor/models/moonvit.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#29309](https://github.com/vllm-project/vllm/pull/29309), [#31738](https://github.com/vllm-project/vllm/pull/31738) |
| `vllm/reasoning/kimi_k2_reasoning_parser.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33646](https://github.com/vllm-project/vllm/pull/33646) |
| `vllm/tokenizers/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/tool_parsers/kimi_k2_tool_parser.py` | [#31207](https://github.com/vllm-project/vllm/pull/31207), [#38579](https://github.com/vllm-project/vllm/pull/38579) |
| `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/transformers_utils/configs/kimi_k25.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131) |
| `vllm/transformers_utils/configs/kimi_linear.py` | [#27809](https://github.com/vllm-project/vllm/pull/27809) |
| `vllm/transformers_utils/configs/kimi_vl.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387) |
| `vllm/transformers_utils/configs/moonvit.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387) |
| `vllm/transformers_utils/processors/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/transformers_utils/processors/kimi_k25.py` | [#37693](https://github.com/vllm-project/vllm/pull/37693) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-14 | [#16387](https://github.com/vllm-project/vllm/pull/16387) | merged | [Model][VLM] Add Kimi-VL model support | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py` |
| 2025-04-18 | [#16833](https://github.com/vllm-project/vllm/pull/16833) | merged | [Misc] Clean up Kimi-VL | `vllm/model_executor/models/kimi_vl.py` |
| 2025-04-25 | [#17156](https://github.com/vllm-project/vllm/pull/17156) | merged | fix float16 support for kimi-vl | `vllm/model_executor/models/kimi_vl.py` |
| 2025-08-05 | [#21769](https://github.com/vllm-project/vllm/pull/21769) | merged | Migrate KimiVLImagePixelInputs to TensorSchema | `vllm/model_executor/models/kimi_vl.py` |
| 2025-08-19 | [#23114](https://github.com/vllm-project/vllm/pull/23114) | merged | [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506 | `vllm/model_executor/models/kimi_vl.py` |
| 2025-09-01 | [#23817](https://github.com/vllm-project/vllm/pull/23817) | merged | [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506 | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2025-10-30 | [#27809](https://github.com/vllm-project/vllm/pull/27809) | merged | [Model] Introduce Kimi Linear to vLLM | `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py` |
| 2025-10-31 | [#27834](https://github.com/vllm-project/vllm/pull/27834) | merged | [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants | `vllm/model_executor/models/kimi_linear.py` |
| 2025-10-31 | [#27885](https://github.com/vllm-project/vllm/pull/27885) | merged | fix incorrect type annotation in KimiMLP | `vllm/model_executor/models/kimi_linear.py` |
| 2025-11-24 | [#29309](https://github.com/vllm-project/vllm/pull/29309) | merged | [XPU]fix Kimi-VL-A3B-thinking on xpu | `vllm/model_executor/models/moonvit.py` |
| 2025-12-15 | [#30125](https://github.com/vllm-project/vllm/pull/30125) | merged | [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it. | `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-12-30 | [#31207](https://github.com/vllm-project/vllm/pull/31207) | merged | fix: update kimi k2 tool parser logic | `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py` |
| 2026-01-06 | [#31738](https://github.com/vllm-project/vllm/pull/31738) | merged | [Models]: Use `MMEncoderAttention` for MoonViT | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2026-01-27 | [#33131](https://github.com/vllm-project/vllm/pull/33131) | merged | [Models] Kimi-K2.5 | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py` |
| 2026-01-29 | [#33320](https://github.com/vllm-project/vllm/pull/33320) | merged | [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d… | `vllm/model_executor/models/kimi_k25.py` |
| 2026-01-30 | [#33346](https://github.com/vllm-project/vllm/pull/33346) | merged | [Models] Refactor Kimi-K2.5 weight loading | `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-02-02 | [#33562](https://github.com/vllm-project/vllm/pull/33562) | merged | [Bugfix] Enable Kimi k25 processor test | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-05 | [#33876](https://github.com/vllm-project/vllm/pull/33876) | merged | [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-13 | [#34427](https://github.com/vllm-project/vllm/pull/34427) | merged | [Bugfix] Delete unused redundant code in Kimi-K2.5 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-13 | [#34501](https://github.com/vllm-project/vllm/pull/34501) | merged | [Bugfix] Add quant_config in ViT of Kimi-K2.5 | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-27 | [#33646](https://github.com/vllm-project/vllm/pull/33646) | merged | [Bugfix] Handle case when kimi ends reasoning with a tool call | `vllm/reasoning/kimi_k2_reasoning_parser.py` |
| 2026-03-06 | [#36192](https://github.com/vllm-project/vllm/pull/36192) | merged | [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-03-11 | [#36127](https://github.com/vllm-project/vllm/pull/36127) | merged | [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct | `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py` |
| 2026-03-11 | [#36361](https://github.com/vllm-project/vllm/pull/36361) | merged | Kimi k2.5 MLA based eagle3 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-03-14 | [#36903](https://github.com/vllm-project/vllm/pull/36903) | merged | [Misc] Clean up Kimi-audio whisper encoder loading | `vllm/model_executor/models/kimi_audio.py` |
| 2026-03-18 | [#37371](https://github.com/vllm-project/vllm/pull/37371) | merged | standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01 | `vllm/model_executor/models/kimi_linear.py` |
| 2026-03-19 | [#37438](https://github.com/vllm-project/vllm/pull/37438) | merged | [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support | `tests/reasoning/test_kimi_k2_reasoning_parser.py` |
| 2026-03-20 | [#37693](https://github.com/vllm-project/vllm/pull/37693) | merged | [Model] Update Kimi-K25 and Isaac processors to fit HF-style | `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-04-12 | [#39344](https://github.com/vllm-project/vllm/pull/39344) | merged | fix(kimi_k25): resolve media_placeholder_token_id from tokenizer | `vllm/model_executor/models/kimi_k25.py` |
| 2026-04-19 | [#38579](https://github.com/vllm-project/vllm/pull/38579) | merged | [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping | `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py` |

## Per-PR Diff Audit Cards

### PR #16387 - [Model][VLM] Add Kimi-VL model support

- Link: https://github.com/vllm-project/vllm/pull/16387
- Status/date: merged / 2025-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`, `vllm/transformers_utils/configs/kimi_vl.py`, `vllm/transformers_utils/configs/moonvit.py`; associated commits `b1308b84a3a6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +1436/-14, 1618 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][VLM] Add Kimi-VL model support"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py`; technical summary: Covers "[Model][VLM] Add Kimi-VL model support"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope, touching `multihead_attention, sdpa_attention, _apply_rope_input_validation`; `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward, touching `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`; `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__, touching `KimiVLConfig, __init__`; `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__, touching `MoonViTConfig, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
  - `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__
  - `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -0,0 +1,628 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# This file is meant to be used in kimi_vl.py only
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/kimi_vl.py
@@ -0,0 +1,608 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
+# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
diff -- vllm/transformers_utils/configs/kimi_vl.py
@@ -0,0 +1,36 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` added +628/-0; `vllm/model_executor/models/kimi_vl.py` added +608/-0; `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0; `vllm/transformers_utils/configs/moonvit.py` added +32/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16833 - [Misc] Clean up Kimi-VL

- Link: https://github.com/vllm-project/vllm/pull/16833
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `aadb6565628c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-44, 139 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Kimi-VL"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Misc] Clean up Kimi-VL"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens, touching `KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,7 +56,6 @@
-from vllm.logger import init_logger
@@ -70,22 +69,20 @@
-from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
-                                    NestedTensors)
+from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
+                                    MultiModalKwargs, NestedTensors)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-40
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17156 - fix float16 support for kimi-vl

- Link: https://github.com/vllm-project/vllm/pull/17156
- Status/date: merged / 2025-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `69bff9bc8934`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix float16 support for kimi-vl"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "fix float16 support for kimi-vl"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, touching `_parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -340,8 +340,7 @@ def _parse_and_validate_image_input(
-        # fp32 -> bf16
-        pixel_values = pixel_values.to(torch.bfloat16)
+        pixel_values = pixel_values.to(self.vision_tower.dtype)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21769 - Migrate KimiVLImagePixelInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/21769
- Status/date: merged / 2025-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `05fae021750b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-9, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate KimiVLImagePixelInputs to TensorSchema"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "Migrate KimiVLImagePixelInputs to TensorSchema"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input, touching `forward, KimiVLImagePixelInputs, _parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -46,7 +46,7 @@
-from typing import Any, Literal, Optional, TypedDict, Union
+from typing import Annotated, Any, Literal, Optional, Union
@@ -79,6 +79,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -118,15 +119,22 @@ def forward(self, image_features: torch.Tensor) -> torch.Tensor:
-class KimiVLImagePixelInputs(TypedDict):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +15/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23114 - [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506

- Link: https://github.com/vllm-project/vllm/pull/23114
- Status/date: merged / 2025-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`; associated commits `fda9537c5e61`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-13, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506"; the main implementation surface is `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__, touching `get_replacement, KimiVLForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -54,16 +54,16 @@
-from vllm.distributed import (get_tensor_model_parallel_rank,
-                              get_tensor_model_parallel_world_size)
+from vllm.distributed import get_pp_group
-from vllm.model_executor.models.interfaces import SupportsMultiModal
+from vllm.model_executor.models.interfaces import (SupportsMultiModal,
+                                                   SupportsPP)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23817 - [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506

- Link: https://github.com/vllm-project/vllm/pull/23817
- Status/date: merged / 2025-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `a0e0efd6bdcf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +157/-62, 478 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer, touching `MLP2, __init__, forward`; `vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward, touching `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer
  - `vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -42,7 +42,6 @@
-import math
@@ -55,6 +54,8 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.models.utils import maybe_prefix
@@ -383,21 +384,30 @@ class MLP2(nn.Module):
-    def __init__(self, dims: list[int], activation, bias=True):
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,6 +56,7 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
@@ -76,6 +77,7 @@
+from vllm.multimodal.utils import run_dp_sharded_mrope_vision_model
@@ -93,29 +95,35 @@ class MaxImageTokenMeta:
-    def __init__(self, config: KimiVLConfig):
+    def __init__(self, config: KimiVLConfig, \
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +55/-22; `vllm/model_executor/models/kimi_vl.py` modified +39/-15
- Risk and verification: The diff ships test coverage in `tests/multimodal/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27809 - [Model] Introduce Kimi Linear to vLLM

- Link: https://github.com/vllm-project/vllm/pull/27809
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`; associated commits `4e68cc9b6aa2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +1326/-49, 1510 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Introduce Kimi Linear to vLLM"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`; technical summary: Covers "[Model] Introduce Kimi Linear to vLLM"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE, touching `KimiMLP, __init__, forward`; `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe, touching `KimiLinearConfig, __init__, is_mla`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE
  - `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -0,0 +1,663 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from typing import Any
+import torch
+from torch import nn
diff -- vllm/transformers_utils/configs/kimi_linear.py
@@ -0,0 +1,144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from transformers.configuration_utils import PretrainedConfig
+from vllm.logger import init_logger
+logger = init_logger(__name__)
+class KimiLinearConfig(PretrainedConfig):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` added +663/-0; `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27834 - [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants

- Link: https://github.com/vllm-project/vllm/pull/27834
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `e5ef4dfc11ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-Linear] Correct prefixes and add compatibility to AWQ quants"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "[Kimi-Linear] Correct prefixes and add compatibility to AWQ quants"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -155,6 +155,7 @@ def __init__(
+                prefix=f"{prefix}.shared_experts",
@@ -340,7 +341,7 @@ def __init__(
-                prefix=f"{prefix}.mlp",
+                prefix=f"{prefix}.block_sparse_moe",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27885 - fix incorrect type annotation in KimiMLP

- Link: https://github.com/vllm-project/vllm/pull/27885
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `bc306fe5e978`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix incorrect type annotation in KimiMLP"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "fix incorrect type annotation in KimiMLP"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -22,7 +22,6 @@
-    QKVParallelLinear,
@@ -61,7 +60,7 @@ def __init__(
-        quant_config: QKVParallelLinear | None = None,
+        quant_config: QuantizationConfig | None = None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29309 - [XPU]fix Kimi-VL-A3B-thinking on xpu

- Link: https://github.com/vllm-project/vllm/pull/29309
- Status/date: merged / 2025-11-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/moonvit.py`; associated commits `3cfa63ad9916`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-6, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU]fix Kimi-VL-A3B-thinking on xpu"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/moonvit.py`; technical summary: Covers "[XPU]fix Kimi-VL-A3B-thinking on xpu"; the main implementation surface is `vllm/model_executor/models/moonvit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__, touching `multihead_attention, Rope2DPosEmb, __init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -56,10 +56,13 @@
+from vllm.platforms import current_platform
+elif current_platform.is_xpu():
+    from vllm.attention.utils.fa_utils import flash_attn_varlen_func
@@ -106,10 +109,10 @@ def multihead_attention(
-        q_cu_seqlens,
-        k_cu_seqlens,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +14/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/moonvit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30125 - [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it.

- Link: https://github.com/vllm-project/vllm/pull/30125
- Status/date: merged / 2025-12-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +1264/-853, 3625 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it."; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it."; the main implementation surface is `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `vllm/attention/layers/mm_encoder_attention.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt, touching `build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt`; `vllm/attention/layers/mm_encoder_attention.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__, enabled, touching `maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__`; `vllm/model_executor/models/qwen2_vl.py` modified +47/-96 (143 lines); hunks: -33,7 +33,6; -45,10 +44,8; symbols: __init__, split_qkv, forward, touching `__init__, split_qkv, forward`; `vllm/model_executor/models/glm4_1v.py` modified +46/-91 (137 lines); hunks: -47,8 +47,10; -191,10 +193,15 @@ def __init__(; symbols: __init__, split_qkv, forward, touching `__init__, split_qkv, forward`.
- Code diff details:
  - `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt
  - `vllm/attention/layers/mm_encoder_attention.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: maybe_get_vit_flash_attn_backend, MMEncoderAttention, __init__, enabled
  - `vllm/model_executor/models/qwen2_vl.py` modified +47/-96 (143 lines); hunks: -33,7 +33,6; -45,10 +44,8; symbols: __init__, split_qkv, forward
  - `vllm/model_executor/models/glm4_1v.py` modified +46/-91 (137 lines); hunks: -47,8 +47,10; -191,10 +193,15 @@ def __init__(; symbols: __init__, split_qkv, forward
  - `vllm/model_executor/models/dots_ocr.py` modified +46/-83 (129 lines); hunks: -5,15 +5,14; -254,11 +253,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tests/models/multimodal/generation/test_vit_backend_functionality.py
@@ -0,0 +1,434 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Consolidated test for ViT attention backend functionality across multiple models.
+This test validates that each multimodal model can successfully generate outputs
+using different ViT attention backends. Tests are parametrized by model and backend.
diff -- vllm/attention/layers/mm_encoder_attention.py
@@ -0,0 +1,284 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable
+import torch
+from vllm.attention.backends.registry import AttentionBackendEnum
+from vllm.attention.ops.vit_attn_wrappers import (
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -33,7 +33,6 @@
```

- Reviewed files:
  - tests: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0
  - runtime: `vllm/attention/layers/mm_encoder_attention.py` added +284/-0; `vllm/model_executor/models/qwen2_vl.py` modified +47/-96; `vllm/model_executor/models/glm4_1v.py` modified +46/-91; `vllm/model_executor/models/dots_ocr.py` modified +46/-83; `vllm/model_executor/models/siglip2navit.py` modified +45/-84; `vllm/model_executor/models/qwen2_5_vl.py` modified +48/-76
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_backend_functionality.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31207 - fix: update kimi k2 tool parser logic

- Link: https://github.com/vllm-project/vllm/pull/31207
- Status/date: merged / 2025-12-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; associated commits `358bfd315cad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +211/-202, 511 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: update kimi k2 tool parser logic"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; technical summary: Covers "fix: update kimi k2 tool parser logic"; the main implementation surface is `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin, touching `assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools`; `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming, touching `_check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming`.
- Code diff details:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_kimi_k2_tool_parser.py
@@ -44,6 +44,33 @@ def assert_tool_calls(
+def run_streaming_sequence(parser, deltas):
+    """Helper to simulate a streaming sequence and return results."""
+    previous_text = ""
+    previous_token_ids: list[int] = []
+    results = []
+    for delta_text, delta_token_ids in deltas:
diff -- vllm/tool_parsers/kimi_k2_tool_parser.py
@@ -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
@@ -238,6 +237,7 @@ def extract_tool_calls_streaming(
@@ -252,13 +252,18 @@ def extract_tool_calls_streaming(
-                remaining = buffered_text
-                # Return remaining text as reasoning content if non-empty
-                if remaining.strip():
-                    return DeltaMessage(content=remaining)
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_kimi_k2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31738 - [Models]: Use `MMEncoderAttention` for MoonViT

- Link: https://github.com/vllm-project/vllm/pull/31738
- Status/date: merged / 2026-01-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; associated commits `7101e0851f73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +72/-158, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models]: Use `MMEncoderAttention` for MoonViT"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`; technical summary: Covers "[Models]: Use `MMEncoderAttention` for MoonViT"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__, touching `multihead_attention, sdpa_attention, _apply_rope_input_validation`; `vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -51,118 +51,20 @@
-from transformers.utils import is_flash_attn_2_available
+from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
+from vllm.config import MultiModalConfig
+from vllm.distributed import divide, get_tensor_model_parallel_world_size
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import (
diff -- vllm/model_executor/models/kimi_vl.py
@@ -325,7 +325,7 @@ def __init__(
-            self.use_data_parallel,
+            multimodal_config=model_config.multimodal_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +71/-157; `vllm/model_executor/models/kimi_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33131 - [Models] Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/33131
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `vllm/transformers_utils/configs/kimi_k25.py`; associated commits `b539f988e1ee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1799/-8, 2011 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py`; technical summary: Covers "[Models] Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape, touching `_apply_rope_input_validation, get_rope_shape_decorate, wrapper`; `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__, touching `MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor`; `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size, touching `KimiK25VisionConfig, __init__, KimiK25Config`; `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming, touching `KimiK2ReasoningParser, __init__, is_reasoning_end`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape
  - `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__
  - `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -0,0 +1,678 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Vision tower implementation for Kimi-K2.5 model.
+This module provides the vision encoder components for Kimi-K2.5,
+including 3D patch embedding, RoPE position embedding, and
diff -- vllm/model_executor/models/kimi_k25.py
@@ -0,0 +1,581 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+"""
+Kimi-K2.5 Model Implementation for vLLM.
+Kimi-K2.5 extends Kimi-K2 with vision support
diff -- vllm/transformers_utils/configs/kimi_k25.py
@@ -0,0 +1,129 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0; `vllm/model_executor/models/kimi_k25.py` added +581/-0; `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0; `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33320 - [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…

- Link: https://github.com/vllm-project/vllm/pull/33320
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `17b17c068453`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -58,6 +58,7 @@
+from vllm.platforms import current_platform
@@ -320,7 +321,7 @@ def __init__(
-        self.device = torch.cuda.current_device()
+        self.device = current_platform.current_device()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33346 - [Models] Refactor Kimi-K2.5 weight loading

- Link: https://github.com/vllm-project/vllm/pull/33346
- Status/date: merged / 2026-01-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `8bfc8d5600ed`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +40/-176, 282 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Refactor Kimi-K2.5 weight loading"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; technical summary: Covers "[Models] Refactor Kimi-K2.5 weight loading"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input, touching `KimiK25ForConditionalGeneration, get_placeholder_str, __init__`; `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -23,16 +23,7 @@
-from vllm.distributed import get_pp_group
-from vllm.model_executor.layers.fused_moe import SharedFusedMoE
-from vllm.model_executor.layers.logits_processor import LogitsProcessor
-from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
-from vllm.model_executor.model_loader.weight_utils import (
-    default_weight_loader,
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -660,13 +660,13 @@ def __init__(
-            prefix=maybe_prefix(prefix, "linear_1"),
+            prefix=f"{prefix}.linear_1",
-            prefix=maybe_prefix(prefix, "linear_2"),
+            prefix=f"{prefix}.linear_2",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +38/-174; `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33562 - [Bugfix] Enable Kimi k25 processor test

- Link: https://github.com/vllm-project/vllm/pull/33562
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `4061dcf4c51a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +96/-12, 221 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Enable Kimi k25 processor test"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Enable Kimi k25 processor test"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__, touching `MoonshotKimiVAutoProcessor, __init__, __call__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):
-    def __init__(self, media_processor=None, tokenizer=None):
+    def __init__(
+        self, media_processor=None, tokenizer=None, media_token_id: int | None = None
+    ):
+        self.media_token_id = media_token_id
+        assert self.media_token_id is not None
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +27/-5
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- Link: https://github.com/vllm-project/vllm/pull/33876
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `a2522839d87d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-5, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -24,7 +24,11 @@
-from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
+from vllm.model_executor.models.interfaces import (
+    SupportsMultiModal,
+    SupportsPP,
+    SupportsQuant,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34427 - [Bugfix] Delete unused redundant code in Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/34427
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `62788f99a4d0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-5, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Delete unused redundant code in Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Delete unused redundant code in Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -11,7 +11,6 @@
-import copy
@@ -378,10 +377,6 @@ def __init__(
-        sub_vllm_config = copy.deepcopy(vllm_config)
-        sub_vllm_config.model_config.hf_config = (
-            sub_vllm_config.model_config.hf_config.text_config
-        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +0/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34501 - [Bugfix] Add quant_config in ViT of Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/34501
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; associated commits `4a9952ec1b15`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-0, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add quant_config in ViT of Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Bugfix] Add quant_config in ViT of Kimi-K2.5"; the main implementation surface is `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input, touching `__init__, _maybe_ignore_quant_config, _parse_and_validate_media_input`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -28,6 +28,7 @@
+from vllm.model_executor.layers.quantization import QuantizationConfig
@@ -304,6 +305,7 @@ def __init__(
+        quant_config: QuantizationConfig | None = None,
@@ -314,13 +316,15 @@ def __init__(
+            quant_config=quant_config,
+            quant_config=quant_config,
diff -- vllm/model_executor/models/kimi_k25.py
@@ -23,6 +23,10 @@
+from vllm.model_executor.layers.quantization import QuantizationConfig
+from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
+    CompressedTensorsConfig,
+)
@@ -361,6 +365,7 @@ def __init__(
+                quant_config=self._maybe_ignore_quant_config(quant_config),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0; `vllm/model_executor/models/kimi_k25.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33646 - [Bugfix] Handle case when kimi ends reasoning with a tool call

- Link: https://github.com/vllm-project/vllm/pull/33646
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/kimi_k2_reasoning_parser.py`; associated commits `9251ed5c4fc6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +230/-2, 240 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Handle case when kimi ends reasoning with a tool call"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/reasoning/kimi_k2_reasoning_parser.py`; technical summary: Covers "[Bugfix] Handle case when kimi ends reasoning with a tool call"; the main implementation surface is `vllm/reasoning/kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end, touching `KimiK2ReasoningParser, __init__, _is_identity_mode`.
- Code diff details:
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end
- Key code excerpts:

```diff
diff -- vllm/reasoning/kimi_k2_reasoning_parser.py
@@ -0,0 +1,228 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from transformers import PreTrainedTokenizerBase
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
```

- Reviewed files:
  - runtime: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/__init__.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36192 - [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25

- Link: https://github.com/vllm-project/vllm/pull/36192
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `00bd08edeee5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-2, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:
-            self.ctx.model_config.model, trust_remote_code=True
+            self.ctx.model_config.model,
+            trust_remote_code=self.ctx.model_config.trust_remote_code,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36127 - [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct

- Link: https://github.com/vllm-project/vllm/pull/36127
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja`, `vllm/transformers_utils/processors/kimi_audio.py`; associated commits `42fadebecb79`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1446/-29, 1583 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add support for moonshotai/Kimi-Audio-7B-Instruct"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`; technical summary: Covers "[Model] Add support for moonshotai/Kimi-Audio-7B-Instruct"; the main implementation surface is `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo, touching `_get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__`; `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__, touching `_load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained`; `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class, touching `_get_feat_extract_output_lengths, KimiAudioProcessor, __init__`; `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config, touching `KimiAudioRenderer, from_config`.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo
  - `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__
  - `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class
  - `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config
  - `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0 (13 lines); hunks: -0,0 +1,13
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_audio.py
@@ -0,0 +1,725 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""
+import os
+from collections.abc import Iterable, Mapping, Sequence
+from typing import Any, ClassVar, Literal
diff -- vllm/tokenizers/kimi_audio.py
@@ -0,0 +1,410 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tokenizer for Kimi-Audio using TikToken."""
+import contextlib
+import json
+from pathlib import Path
diff -- vllm/transformers_utils/processors/kimi_audio.py
@@ -0,0 +1,163 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_audio.py` added +725/-0; `vllm/tokenizers/kimi_audio.py` added +410/-0; `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0; `vllm/renderers/kimi_audio.py` added +49/-0; `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36361 - Kimi k2.5 MLA based eagle3

- Link: https://github.com/vllm-project/vllm/pull/36361
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `557389473755`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +499/-8, 649 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Kimi k2.5 MLA based eagle3"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "Kimi k2.5 MLA based eagle3"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -28,6 +28,8 @@
+    SupportsEagle,
+    SupportsEagle3,
@@ -311,7 +313,12 @@ def split_video_chunks(self, video):
-    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant
+    nn.Module,
+    SupportsMultiModal,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36903 - [Misc] Clean up Kimi-audio whisper encoder loading

- Link: https://github.com/vllm-project/vllm/pull/36903
- Status/date: merged / 2026-03-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_audio.py`; associated commits `a8e8d62dd80f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +89/-116, 382 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Kimi-audio whisper encoder loading"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `vllm/model_executor/models/kimi_audio.py`; technical summary: Covers "[Misc] Clean up Kimi-audio whisper encoder loading"; the main implementation surface is `vllm/model_executor/models/kimi_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, touching `_get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder`.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_audio.py
@@ -3,25 +3,21 @@
-import os
-from huggingface_hub import snapshot_download
-from safetensors import safe_open
-from vllm.model_executor.model_loader.weight_utils import (
-    default_weight_loader,
-)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_audio.py` modified +61/-111
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/default_loader.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/kimi_audio.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37371 - standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01

- Link: https://github.com/vllm-project/vllm/pull/37371
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_linear.py`; associated commits `17808394bc48`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +235/-219, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; the main implementation surface is `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids, touching `forward, KimiLinearForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -46,6 +46,7 @@
+    AutoWeightsLoader,
@@ -472,94 +473,7 @@ def forward(
-class KimiLinearForCausalLM(
-    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
-):
-    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +97/-88
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37438 - [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support

- Link: https://github.com/vllm-project/vllm/pull/37438
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_kimi_k2_reasoning_parser.py`; associated commits `c63ca2b2e696`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +173/-18, 227 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/reasoning/test_kimi_k2_reasoning_parser.py`; technical summary: Covers "[Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support"; the main implementation surface is `tests/reasoning/test_kimi_k2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags, touching `kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled`.
- Code diff details:
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags
- Key code excerpts:

```diff
diff -- tests/reasoning/test_kimi_k2_reasoning_parser.py
@@ -0,0 +1,155 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.engine.protocol import DeltaMessage
+from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
```

- Reviewed files:
  - tests: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_kimi_k2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37693 - [Model] Update Kimi-K25 and Isaac processors to fit HF-style

- Link: https://github.com/vllm-project/vllm/pull/37693
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/kimi_k25.py`; associated commits `37aadf623786`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +128/-95, 366 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Update Kimi-K25 and Isaac processors to fit HF-style"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "[Model] Update Kimi-K25 and Isaac processors to fit HF-style"; the main implementation surface is `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__, touching `KimiK25Processor, __init__, __call__`; `vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits, touching `KimiK25ProcessingInfo, __init__, get_hf_processor`.
- Code diff details:
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__
  - `vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/kimi_k25.py
@@ -1,38 +1,41 @@
-import torch
-from transformers import BatchFeature
+from transformers import BaseImageProcessor, BatchFeature, TensorType
+from vllm.tokenizers.hf import HfTokenizer
-    attributes = ["tokenizer"]
-    tokenizer_class = "AutoTokenizer"
diff -- vllm/model_executor/models/kimi_k25.py
@@ -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):
-        self.hf_config = self.get_hf_config()
-        self.media_token_id = self.hf_config.media_placeholder_token_id
-        media_processor = cached_get_image_processor(
+        self.hf_config = hf_config = self.get_hf_config()
+        tokenizer = self.get_tokenizer()
+        image_processor = cached_get_image_processor(
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38; `vllm/model_executor/models/kimi_k25.py` modified +16/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/isaac.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/isaac.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39344 - fix(kimi_k25): resolve media_placeholder_token_id from tokenizer

- Link: https://github.com/vllm-project/vllm/pull/39344
- Status/date: merged / 2026-04-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/kimi_k25.py`; associated commits `17e787a7792b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +24/-3, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(kimi_k25): resolve media_placeholder_token_id from tokenizer"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "fix(kimi_k25): resolve media_placeholder_token_id from tokenizer"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement, touching `__init__, _get_prompt_updates, get_replacement`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:
-        self.media_token_id = media_token_id = hf_config.media_placeholder_token_id
+        # Resolve token ID from the tokenizer because transformers v5
+        # may remap token IDs vs config.json.
+        config_token_id = hf_config.media_placeholder_token_id
+        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
+        is_valid_resolved = isinstance(resolved_token_id, int) and (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +24/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38579 - [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping

- Link: https://github.com/vllm-project/vllm/pull/38579
- Status/date: merged / 2026-04-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; associated commits `03ce1c6ed908`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +684/-1405, 2206 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`; technical summary: Covers "[Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping"; the main implementation surface is `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls, touching `kimi_k2_tokenizer, kimi_k2_tool_parser, parser`; `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state, touching `KimiK2ToolParser, __init__, _check_and_strip_markers`.
- Code diff details:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_kimi_k2_tool_parser.py
@@ -3,14 +3,20 @@
+from unittest.mock import MagicMock
-from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
+from tests.tool_parsers.utils import (
+    run_tool_extraction,
+    run_tool_extraction_streaming,
+)
diff -- vllm/tool_parsers/kimi_k2_tool_parser.py
@@ -1,6 +1,5 @@
-# code modified from deepseekv3_tool_parser.py
@@ -17,137 +16,59 @@
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
+from vllm.tool_parsers.utils import partial_tag_overlap
-        self.current_tool_name_sent: bool = False
+        # Streaming state
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_kimi_k2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
