# vllm DeepSeek OCR 2 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `vllm/model_executor/models/deepseek_ocr2.py` | [#33165](https://github.com/vllm-project/vllm/pull/33165), [#33909](https://github.com/vllm-project/vllm/pull/33909) |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 37
- Total PRs in this document: 39
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-10-22 | [#27247](https://github.com/vllm-project/vllm/pull/27247) | merged | [Model] Upstream Deepseek-OCR model | `vllm/model_executor/models/deepencoder.py`, `vllm/model_executor/models/deepseek_ocr.py`, `vllm/transformers_utils/processors/deepseek_ocr.py` |
| 2025-10-23 | [#27361](https://github.com/vllm-project/vllm/pull/27361) | merged | [Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support | `vllm/model_executor/models/deepseek_ocr.py`, `vllm/transformers_utils/processors/deepseek_ocr.py`, `tests/models/multimodal/processing/test_common.py` |
| 2025-11-05 | [#27560](https://github.com/vllm-project/vllm/pull/27560) | merged | [Bugfix] Validate custom logits processor xargs for online serving | `vllm/model_executor/models/deepseek_ocr.py`, `vllm/entrypoints/openai/serving_completion.py`, `vllm/entrypoints/openai/serving_chat.py` |
| 2025-11-08 | [#28101](https://github.com/vllm-project/vllm/pull/28101) | merged | [Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2 | `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_ocr.py` |
| 2025-11-13 | [#27583](https://github.com/vllm-project/vllm/pull/27583) | merged | Rename clashing method names for vLLM model protocol | `vllm/model_executor/models/interfaces_base.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py` |
| 2025-11-13 | [#28617](https://github.com/vllm-project/vllm/pull/28617) | merged | [BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path | `vllm/model_executor/models/deepseek_ocr.py` |
| 2025-12-02 | [#29793](https://github.com/vllm-project/vllm/pull/29793) | merged | [Chore] Move tokenizer initialization methods | `vllm/transformers_utils/tokenizer.py`, `vllm/tokenizers/registry.py`, `vllm/tokenizers/__init__.py` |
| 2025-12-04 | [#30035](https://github.com/vllm-project/vllm/pull/30035) | merged | [Chore] Deprecate `merge_by_field_config` arg | `vllm/multimodal/inputs.py`, `tests/multimodal/test_inputs.py`, `vllm/multimodal/utils.py` |
| 2025-12-06 | [#30170](https://github.com/vllm-project/vllm/pull/30170) | merged | [Chore] Deprecate `SupportsMultiModal.merge_by_field_config` | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/aya_vision.py` |
| 2025-12-07 | [#30145](https://github.com/vllm-project/vllm/pull/30145) | merged | [Renderer] Separate out `RendererConfig` from `ModelConfig` | `tests/entrypoints/test_chat_utils.py`, `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/registry.py` |
| 2025-12-07 | [#30199](https://github.com/vllm-project/vllm/pull/30199) | merged | Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)" | `tests/entrypoints/test_chat_utils.py`, `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/registry.py` |
| 2026-01-02 | [#31569](https://github.com/vllm-project/vllm/pull/31569) | merged | feat: support LoRA for DeepSeek-OCR(Language Model part) | `vllm/model_executor/models/deepseek_ocr.py`, `docs/models/supported_models.md` |
| 2026-01-08 | [#31947](https://github.com/vllm-project/vllm/pull/31947) | merged | [Model] Standardize common vision encoders | `vllm/model_executor/models/siglip.py`, `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/phi3v.py` |
| 2026-01-09 | [#32016](https://github.com/vllm-project/vllm/pull/32016) | merged | [Model] Remove redundant None check in DeepSeekOCR image input processing | `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-01-14 | [#32327](https://github.com/vllm-project/vllm/pull/32327) | merged | [1/N] Reorganize multimodal processing code | `vllm/multimodal/processing/processor.py`, `vllm/multimodal/processing/context.py`, `vllm/multimodal/processing/__init__.py` |
| 2026-01-20 | [#32632](https://github.com/vllm-project/vllm/pull/32632) | merged | [1/N] Initialize MM components in context managers (A-D) | `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/deepseek_vl2.py` |
| 2026-01-24 | [#31972](https://github.com/vllm-project/vllm/pull/31972) | merged | [Models]: Make Multimodal config implicit in ViT implementation | `vllm/model_executor/models/vision.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/dots_ocr.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py` |
| 2026-02-02 | [#33165](https://github.com/vllm-project/vllm/pull/33165) | merged | [Model] Support DeepSeek-OCR-2 | `vllm/model_executor/models/deepseek_ocr2.py`, `vllm/transformers_utils/processors/deepseek_ocr2.py` |
| 2026-02-05 | [#33909](https://github.com/vllm-project/vllm/pull/33909) | merged | [Models] Consolidate Deepseek-OCR2 processor | `vllm/transformers_utils/processors/deepseek_ocr2.py`, `vllm/model_executor/models/deepseek_ocr2.py` |
| 2026-02-11 | [#34330](https://github.com/vllm-project/vllm/pull/34330) | merged | [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder` | `vllm/model_executor/models/idefics3.py`, `vllm/multimodal/processing/dummy_inputs.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2026-02-12 | [#34085](https://github.com/vllm-project/vllm/pull/34085) | merged | Fix DeepSeek-OCR tensor validation for all size variants | `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-02-23 | [#35025](https://github.com/vllm-project/vllm/pull/35025) | merged | [Refactor] Simplify dummy data generation | `vllm/config/multimodal.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/multimodal/registry.py` |
| 2026-03-06 | [#36024](https://github.com/vllm-project/vllm/pull/36024) | merged | [Misc] Lazy import registered processors | `vllm/transformers_utils/processors/__init__.py`, `tests/models/registry.py`, `vllm/transformers_utils/processors/deepseek_ocr.py` |
| 2026-03-12 | [#36670](https://github.com/vllm-project/vllm/pull/36670) | merged | [Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop | `tests/models/multimodal/processing/test_deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr.py` |
| 2026-03-17 | [#37289](https://github.com/vllm-project/vllm/pull/37289) | merged | [Bugfix] Standardize custom HF Processor init | `vllm/transformers_utils/processors/qwen_vl.py`, `vllm/model_executor/models/glm4v.py`, `vllm/model_executor/models/qwen_vl.py` |
| 2026-03-25 | [#35182](https://github.com/vllm-project/vllm/pull/35182) | merged | [Misc] Reorganize inputs | `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py` |
| 2026-04-27 | [#36464](https://github.com/vllm-project/vllm/pull/36464) | merged | [Examples] Resettle generate examples. | `docs/features/multimodal_inputs.md`, `examples/generate/multimodal/qwen2_5_omni/README.md`, `docs/features/reasoning_outputs.md` |
| 2026-05-02 | [#40830](https://github.com/vllm-project/vllm/pull/40830) | merged | [MM][CG] Support ViT CG for Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |
| 2026-05-13 | [#42151](https://github.com/vllm-project/vllm/pull/42151) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5 | `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md` |
| 2026-05-13 | [#41736](https://github.com/vllm-project/vllm/pull/41736) | merged | [MM][CG] Support ViT CG for Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md` |
| 2026-05-18 | [#42224](https://github.com/vllm-project/vllm/pull/42224) | merged | [MM][CG] Enable encoder Cudagraph for Step3VL | `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py` |
| 2026-06-04 | [#41759](https://github.com/vllm-project/vllm/pull/41759) | merged | [MM][Perf][CG] Support ViT full CUDA graph for InternVL | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md` |
| 2026-06-09 | [#40576](https://github.com/vllm-project/vllm/pull/40576) | merged | [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference | `vllm/model_executor/models/glm4_1v.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md` |
| 2026-06-10 | [#45131](https://github.com/vllm-project/vllm/pull/45131) | merged | Deprecated 1st generation Qwen and QwenVL models | `vllm/model_executor/models/qwen_vl.py`, `vllm/model_executor/models/qwen.py`, `vllm/tokenizers/qwen_vl.py` |
| 2026-06-12 | [#40660](https://github.com/vllm-project/vllm/pull/40660) | merged | [MM][Perf][CG] Support ViT full cudagraphs for mllama4 | `vllm/model_executor/models/mllama4.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | merged | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |
| 2026-06-17 | [#41992](https://github.com/vllm-project/vllm/pull/41992) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |
| 2026-06-22 | [#45993](https://github.com/vllm-project/vllm/pull/45993) | merged | [Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM | `tests/tool_parsers/test_minimax_tool_parser.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/tool_parsers/minimax_tool_parser.py` |

## Per-PR Diff Audit Cards

### PR #27247 - [Model] Upstream Deepseek-OCR model

- Link: https://github.com/vllm-project/vllm/pull/27247
- Status/date: merged / 2025-10-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +1821/-40, 1953 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Upstream Deepseek-OCR model"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/deepencoder.py`, `vllm/model_executor/models/deepseek_ocr.py`, `vllm/transformers_utils/processors/deepseek_ocr.py`; technical summary: Covers "[Model] Upstream Deepseek-OCR model"; the main implementation surface is `vllm/model_executor/models/deepencoder.py`, `vllm/model_executor/models/deepseek_ocr.py`, `vllm/transformers_utils/processors/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepencoder.py` added +673/-0 (673 lines); hunks: -0,0 +1,673; symbols: MLPBlock, __init__, forward, LayerNorm2d, touching `MLPBlock, __init__, forward`; `vllm/model_executor/models/deepseek_ocr.py` added +594/-0 (594 lines); hunks: -0,0 +1,594; symbols: NoRepeatNGramLogitsProcessor, __init__, __call__, NGramPerReqLogitsProcessor, touching `NoRepeatNGramLogitsProcessor, __init__, __call__`; `vllm/transformers_utils/processors/deepseek_ocr.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: find_closest_aspect_ratio, calculate_aspect_ratios, count_tiles, dynamic_preprocess, touching `find_closest_aspect_ratio, calculate_aspect_ratios, count_tiles`; `vllm/model_executor/models/deepseek_vl2.py` modified +23/-20 (43 lines); hunks: -101,9 +101,10 @@ def __init__(self, cfg: MlpProjectorConfig):; -120,7 +121,8 @@ def __init__(self, cfg: MlpProjectorConfig):; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepencoder.py` added +673/-0 (673 lines); hunks: -0,0 +1,673; symbols: MLPBlock, __init__, forward, LayerNorm2d
  - `vllm/model_executor/models/deepseek_ocr.py` added +594/-0 (594 lines); hunks: -0,0 +1,594; symbols: NoRepeatNGramLogitsProcessor, __init__, __call__, NGramPerReqLogitsProcessor
  - `vllm/transformers_utils/processors/deepseek_ocr.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: find_closest_aspect_ratio, calculate_aspect_ratios, count_tiles, dynamic_preprocess
  - `vllm/model_executor/models/deepseek_vl2.py` modified +23/-20 (43 lines); hunks: -101,9 +101,10 @@ def __init__(self, cfg: MlpProjectorConfig):; -120,7 +121,8 @@ def __init__(self, cfg: MlpProjectorConfig):; symbols: __init__, forward
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -585,6 +585,9 @@ def check_available_online(; symbols: check_available_online
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepencoder.py
@@ -0,0 +1,673 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from
+# https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/sam_vary_sdpa.py
+# Copyright (c) Meta Platforms, Inc. and affiliates.
+# All rights reserved.
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -0,0 +1,594 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""
+import math
+from collections.abc import Iterable, Mapping, Sequence
+import torch
diff -- vllm/transformers_utils/processors/deepseek_ocr.py
@@ -0,0 +1,442 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepencoder.py` added +673/-0; `vllm/model_executor/models/deepseek_ocr.py` added +594/-0; `vllm/transformers_utils/processors/deepseek_ocr.py` added +442/-0; `vllm/model_executor/models/deepseek_vl2.py` modified +23/-20; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs: `docs/models/supported_models.md` modified +1/-0; `examples/offline_inference/vision_language.py` modified +69/-20
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27361 - [Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support

- Link: https://github.com/vllm-project/vllm/pull/27361
- Status/date: merged / 2025-10-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +112/-66, 306 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `vllm/transformers_utils/processors/deepseek_ocr.py`, `tests/models/multimodal/processing/test_common.py`; technical summary: Covers "[Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `vllm/transformers_utils/processors/deepseek_ocr.py`, `tests/models/multimodal/processing/test_common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55 (113 lines); hunks: -4,6 +4,7; -53,6 +54,7; symbols: DeepseekOCRImagePixelInputs, NoRepeatNGramLogitsProcessor, __init__, _get_mm_fields_config, touching `DeepseekOCRImagePixelInputs, NoRepeatNGramLogitsProcessor, __init__`; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9 (14 lines); hunks: -411,20 +411,16 @@ def tokenize_with_images(; symbols: tokenize_with_images, touching `tokenize_with_images`; `tests/models/multimodal/processing/test_common.py` modified +1/-0 (1 lines); hunks: -332,6 +332,7 @@ def _test_processing_correctness_one(; symbols: _test_processing_correctness_one, touching `_test_processing_correctness_one`; `examples/offline_inference/vision_language_multi_image.py` modified +48/-2 (50 lines); hunks: -44,6 +44,7 @@ class ModelRequestData(NamedTuple):; -201,6 +202,46 @@ def load_deepseek_vl2(question: str, image_urls: list[str])...; symbols: ModelRequestData, load_deepseek_vl2, load_deepseek_ocr, load_gemma3, touching `ModelRequestData, load_deepseek_vl2, load_deepseek_ocr`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55 (113 lines); hunks: -4,6 +4,7; -53,6 +54,7; symbols: DeepseekOCRImagePixelInputs, NoRepeatNGramLogitsProcessor, __init__, _get_mm_fields_config
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9 (14 lines); hunks: -411,20 +411,16 @@ def tokenize_with_images(; symbols: tokenize_with_images
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0 (1 lines); hunks: -332,6 +332,7 @@ def _test_processing_correctness_one(; symbols: _test_processing_correctness_one
  - `examples/offline_inference/vision_language_multi_image.py` modified +48/-2 (50 lines); hunks: -44,6 +44,7 @@ class ModelRequestData(NamedTuple):; -201,6 +202,46 @@ def load_deepseek_vl2(question: str, image_urls: list[str])...; symbols: ModelRequestData, load_deepseek_vl2, load_deepseek_ocr, load_gemma3
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -4,6 +4,7 @@
+from typing import Annotated, Literal
@@ -53,6 +54,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -65,6 +67,28 @@
+class DeepseekOCRImagePixelInputs(TensorSchema):
+    """
diff -- vllm/transformers_utils/processors/deepseek_ocr.py
@@ -411,20 +411,16 @@ def tokenize_with_images(
-            pixel_values = torch.zeros((1, 3, self.base_size, self.base_size))
-            images_spatial_crop = torch.zeros((1, 1), dtype=torch.long)
-            images_crop = torch.zeros(
-                (1, 3, self.image_size, self.image_size)
-            ).unsqueeze(0)
+            pixel_values = torch.zeros((0, 3, self.base_size, self.base_size))
diff -- tests/models/multimodal/processing/test_common.py
@@ -332,6 +332,7 @@ def _test_processing_correctness_one(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +58/-55; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +5/-9
  - tests: `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - docs: `examples/offline_inference/vision_language_multi_image.py` modified +48/-2
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27560 - [Bugfix] Validate custom logits processor xargs for online serving

- Link: https://github.com/vllm-project/vllm/pull/27560
- Status/date: merged / 2025-11-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +232/-49, 574 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Validate custom logits processor xargs for online serving"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `vllm/entrypoints/openai/serving_completion.py`, `vllm/entrypoints/openai/serving_chat.py`; technical summary: Covers "[Bugfix] Validate custom logits processor xargs for online serving"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `vllm/entrypoints/openai/serving_completion.py`, `vllm/entrypoints/openai/serving_chat.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +23/-17 (40 lines); hunks: -131,25 +131,18 @@ class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):; -163,13 +156,26 @@ def new_req_logits_processor(; symbols: NGramPerReqLogitsProcessor, __init__, is_argmax_invariant, new_req_logits_processor, touching `NGramPerReqLogitsProcessor, __init__, is_argmax_invariant`; `vllm/entrypoints/openai/serving_completion.py` modified +9/-0 (9 lines); hunks: -36,6 +36,7; -59,6 +60,10 @@ def __init__(; symbols: __init__, create_completion, touching `__init__, create_completion`; `vllm/entrypoints/openai/serving_chat.py` modified +8/-0 (8 lines); hunks: -71,6 +71,7; -110,6 +111,9 @@ def __init__(; symbols: __init__, create_chat_completion, touching `__init__, create_chat_completion`; `vllm/transformers_utils/configs/deepseek_vl2.py` modified +6/-0 (6 lines); hunks: -218,3 +218,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +23/-17 (40 lines); hunks: -131,25 +131,18 @@ class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):; -163,13 +156,26 @@ def new_req_logits_processor(; symbols: NGramPerReqLogitsProcessor, __init__, is_argmax_invariant, new_req_logits_processor
  - `vllm/entrypoints/openai/serving_completion.py` modified +9/-0 (9 lines); hunks: -36,6 +36,7; -59,6 +60,10 @@ def __init__(; symbols: __init__, create_completion
  - `vllm/entrypoints/openai/serving_chat.py` modified +8/-0 (8 lines); hunks: -71,6 +71,7; -110,6 +111,9 @@ def __init__(; symbols: __init__, create_chat_completion
  - `vllm/transformers_utils/configs/deepseek_vl2.py` modified +6/-0 (6 lines); hunks: -218,3 +218,9 @@ def __init__(; symbols: __init__
  - `vllm/entrypoints/openai/protocol.py` modified +2/-2 (4 lines); hunks: -772,10 +772,10 @@ class ChatCompletionRequest(OpenAIBaseModel):; symbols: ChatCompletionRequest
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -131,25 +131,18 @@ class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):
-    def __init__(
-        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
-    ):
-        super().__init__(vllm_config, device, is_pin_memory)
-    def is_argmax_invariant(self) -> bool:
-        return True
diff -- vllm/entrypoints/openai/serving_completion.py
@@ -36,6 +36,7 @@
+from vllm.v1.sample.logits_processor import validate_logits_processors_parameters
@@ -59,6 +60,10 @@ def __init__(
+        # set up logits processors
+        self.logits_processors = self.model_config.logits_processors
@@ -181,6 +186,10 @@ async def create_completion(
+                    validate_logits_processors_parameters(
diff -- vllm/entrypoints/openai/serving_chat.py
@@ -71,6 +71,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +23/-17; `vllm/entrypoints/openai/serving_completion.py` modified +9/-0; `vllm/entrypoints/openai/serving_chat.py` modified +8/-0; `vllm/transformers_utils/configs/deepseek_vl2.py` modified +6/-0; `vllm/entrypoints/openai/protocol.py` modified +2/-2
  - tests: `tests/entrypoints/openai/test_lora_resolvers.py` modified +1/-0; `tests/entrypoints/openai/test_serving_chat.py` modified +1/-0
  - docs: `docs/features/custom_logitsprocs.md` modified +33/-9
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/test_lora_resolvers.py`, `tests/entrypoints/openai/test_serving_chat.py`, `tests/v1/logits_processors/test_custom_online.py`, `tests/v1/logits_processors/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28101 - [Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2

- Link: https://github.com/vllm-project/vllm/pull/28101
- Status/date: merged / 2025-11-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +144/-548, 825 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_ocr.py`; technical summary: Covers "[Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2"; the main implementation surface is `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek.py` removed +0/-517 (517 lines); hunks: -1,517 +0,0; symbols: DeepseekMLP, __init__, forward, DeepseekMoE, touching `DeepseekMLP, __init__, forward`; `vllm/model_executor/models/deepseek_v2.py` modified +139/-13 (152 lines); hunks: -58,6 +58,7; -104,6 +105,92; symbols: DeepseekAttention, __init__, forward, DeepseekV2MLP, touching `DeepseekAttention, __init__, forward`; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8 (8 lines); hunks: -417,18 +417,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/deepseek_vl2.py` modified +0/-8 (8 lines); hunks: -403,18 +403,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek.py` removed +0/-517 (517 lines); hunks: -1,517 +0,0; symbols: DeepseekMLP, __init__, forward, DeepseekMoE
  - `vllm/model_executor/models/deepseek_v2.py` modified +139/-13 (152 lines); hunks: -58,6 +58,7; -104,6 +105,92; symbols: DeepseekAttention, __init__, forward, DeepseekV2MLP
  - `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8 (8 lines); hunks: -417,18 +417,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_vl2.py` modified +0/-8 (8 lines); hunks: -403,18 +403,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `tests/models/registry.py` modified +4/-1 (5 lines); hunks: -219,7 +219,10 @@ def check_available_online(; symbols: check_available_online
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek.py
@@ -1,517 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -58,6 +58,7 @@
+    QKVParallelLinear,
@@ -104,6 +105,92 @@
+class DeepseekAttention(nn.Module):
+    """Normal MHA implementation used by Deepseek v1."""
+    def __init__(
+        self,
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -417,18 +417,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek.py` removed +0/-517; `vllm/model_executor/models/deepseek_v2.py` modified +139/-13; `vllm/model_executor/models/deepseek_ocr.py` modified +0/-8; `vllm/model_executor/models/deepseek_vl2.py` modified +0/-8; `vllm/model_executor/models/registry.py` modified +1/-1
  - tests: `tests/models/registry.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27583 - Rename clashing method names for vLLM model protocol

- Link: https://github.com/vllm-project/vllm/pull/27583
- Status/date: merged / 2025-11-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 164 files, +574/-583, 4116 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Rename clashing method names for vLLM model protocol"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/interfaces_base.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py`; technical summary: Covers "Rename clashing method names for vLLM model protocol"; the main implementation surface is `vllm/model_executor/models/interfaces_base.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/bert.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interfaces_base.py` modified +23/-20 (43 lines); hunks: -41,36 +41,39; -110,7 +113,7 @@ def is_vllm_model(; symbols: VllmModel, __init__, get_input_embeddings, embed_input_ids, touching `VllmModel, __init__, get_input_embeddings`; `vllm/model_executor/models/interfaces.py` modified +19/-13 (32 lines); hunks: -94,7 +94,7 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | N...; -104,7 +104,13 @@ def get_multimodal_embeddings(self, **kwargs: object) -> Mu...; symbols: get_placeholder_str, get_multimodal_embeddings, embed_multimodal, get_language_model, touching `get_placeholder_str, get_multimodal_embeddings, embed_multimodal`; `vllm/model_executor/models/bert.py` modified +7/-7 (14 lines); hunks: -375,7 +375,7 @@ def __init__(; -486,8 +486,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, embed_input_ids, forward, touching `__init__, get_input_embeddings, embed_input_ids`; `vllm/model_executor/models/modernbert.py` modified +7/-7 (14 lines); hunks: -46,7 +46,7 @@ def __init__(self, config: ModernBertConfig):; -225,8 +225,8 @@ def __init__(; symbols: __init__, get_input_embeddings, embed_input_ids, forward, touching `__init__, get_input_embeddings, embed_input_ids`.
- Code diff details:
  - `vllm/model_executor/models/interfaces_base.py` modified +23/-20 (43 lines); hunks: -41,36 +41,39; -110,7 +113,7 @@ def is_vllm_model(; symbols: VllmModel, __init__, get_input_embeddings, embed_input_ids
  - `vllm/model_executor/models/interfaces.py` modified +19/-13 (32 lines); hunks: -94,7 +94,7 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | N...; -104,7 +104,13 @@ def get_multimodal_embeddings(self, **kwargs: object) -> Mu...; symbols: get_placeholder_str, get_multimodal_embeddings, embed_multimodal, get_language_model
  - `vllm/model_executor/models/bert.py` modified +7/-7 (14 lines); hunks: -375,7 +375,7 @@ def __init__(; -486,8 +486,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, embed_input_ids, forward
  - `vllm/model_executor/models/modernbert.py` modified +7/-7 (14 lines); hunks: -46,7 +46,7 @@ def __init__(self, config: ModernBertConfig):; -225,8 +225,8 @@ def __init__(; symbols: __init__, get_input_embeddings, embed_input_ids, forward
  - `vllm/model_executor/models/qwen3_vl.py` modified +6/-8 (14 lines); hunks: -1100,7 +1100,7 @@ def forward(; -1493,9 +1493,7 @@ def get_mrope_input_positions(; symbols: forward, get_mrope_input_positions, get_language_model, get_multimodal_embeddings
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interfaces_base.py
@@ -41,36 +41,39 @@
-    def __init__(
-        self,
-        vllm_config: VllmConfig,
-        prefix: str = "",
-    ) -> None: ...
+    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
diff -- vllm/model_executor/models/interfaces.py
@@ -94,7 +94,7 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | None:
-    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
+    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
@@ -104,7 +104,13 @@ def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
-        ...
+        if hasattr(self, "get_multimodal_embeddings"):
+            logger.warning_once(
diff -- vllm/model_executor/models/bert.py
@@ -375,7 +375,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces_base.py` modified +23/-20; `vllm/model_executor/models/interfaces.py` modified +19/-13; `vllm/model_executor/models/bert.py` modified +7/-7; `vllm/model_executor/models/modernbert.py` modified +7/-7; `vllm/model_executor/models/qwen3_vl.py` modified +6/-8; `vllm/model_executor/models/clip.py` modified +6/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/apertus.py`, `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/arctic.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28617 - [BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path

- Link: https://github.com/vllm-project/vllm/pull/28617
- Status/date: merged / 2025-11-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_ocr.py`; technical summary: Covers "[BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def validate_params(cls, params: SamplingParams):; symbols: validate_params, is_argmax_invariant, new_req_logits_processor, touching `validate_params, is_argmax_invariant, new_req_logits_processor`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def validate_params(cls, params: SamplingParams):; symbols: validate_params, is_argmax_invariant, new_req_logits_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -161,7 +161,7 @@ def validate_params(cls, params: SamplingParams):
-        return True
+        return False
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29793 - [Chore] Move tokenizer initialization methods

- Link: https://github.com/vllm-project/vllm/pull/29793
- Status/date: merged / 2025-12-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 51 files, +150/-129, 761 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Move tokenizer initialization methods"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `vllm/transformers_utils/tokenizer.py`, `vllm/tokenizers/registry.py`, `vllm/tokenizers/__init__.py`; technical summary: Covers "[Chore] Move tokenizer initialization methods"; the main implementation surface is `vllm/transformers_utils/tokenizer.py`, `vllm/tokenizers/registry.py`, `vllm/tokenizers/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/tokenizer.py` modified +43/-48 (91 lines); hunks: -2,17 +2,10; -28,18 +21,54 @@ def __getattr__(name: str):; symbols: __getattr__, encode_tokens, cached_tokenizer_from_config, init_tokenizer_from_configs, touching `__getattr__, encode_tokens, cached_tokenizer_from_config`; `vllm/tokenizers/registry.py` modified +37/-1 (38 lines); hunks: -2,10 +2,12; -21,6 +23,9; symbols: get_tokenizer, cached_tokenizer_from_config, init_tokenizer_from_config, touching `get_tokenizer, cached_tokenizer_from_config, init_tokenizer_from_config`; `vllm/tokenizers/__init__.py` modified +10/-1 (11 lines); hunks: -4,12 +4,21; `vllm/model_executor/models/granite_speech.py` modified +4/-4 (8 lines); hunks: -59,8 +59,8; -862,7 +862,7 @@ def get_generation_prompt(; symbols: get_generation_prompt, get_num_audio_tokens, touching `get_generation_prompt, get_num_audio_tokens`.
- Code diff details:
  - `vllm/transformers_utils/tokenizer.py` modified +43/-48 (91 lines); hunks: -2,17 +2,10; -28,18 +21,54 @@ def __getattr__(name: str):; symbols: __getattr__, encode_tokens, cached_tokenizer_from_config, init_tokenizer_from_configs
  - `vllm/tokenizers/registry.py` modified +37/-1 (38 lines); hunks: -2,10 +2,12; -21,6 +23,9; symbols: get_tokenizer, cached_tokenizer_from_config, init_tokenizer_from_config
  - `vllm/tokenizers/__init__.py` modified +10/-1 (11 lines); hunks: -4,12 +4,21
  - `vllm/model_executor/models/granite_speech.py` modified +4/-4 (8 lines); hunks: -59,8 +59,8; -862,7 +862,7 @@ def get_generation_prompt(; symbols: get_generation_prompt, get_num_audio_tokens
  - `tests/models/multimodal/processing/test_common.py` modified +2/-5 (7 lines); hunks: -22,11 +22,8
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/tokenizer.py
@@ -2,17 +2,10 @@
-from functools import lru_cache
-from typing import TYPE_CHECKING, Any
-from typing_extensions import assert_never
+from typing import Any
-from vllm.tokenizers import TokenizerLike, get_tokenizer
-if TYPE_CHECKING:
diff -- vllm/tokenizers/registry.py
@@ -2,10 +2,12 @@
+from functools import lru_cache
-from typing import TypeVar, overload
+from typing import TYPE_CHECKING, TypeVar, overload
+from typing_extensions import assert_never
@@ -21,6 +23,9 @@
+if TYPE_CHECKING:
diff -- vllm/tokenizers/__init__.py
@@ -4,12 +4,21 @@
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/tokenizer.py` modified +43/-48; `vllm/tokenizers/registry.py` modified +37/-1; `vllm/tokenizers/__init__.py` modified +10/-1; `vllm/model_executor/models/granite_speech.py` modified +4/-4; `vllm/model_executor/models/nano_nemotron_vl.py` modified +2/-5; `vllm/model_executor/models/whisper.py` modified +3/-3
  - tests: `tests/models/multimodal/processing/test_common.py` modified +2/-5
- Risk and verification: The diff ships test coverage in `tests/compile/test_dynamic_shapes_compilation.py`, `tests/entrypoints/openai/test_chat_template.py`, `tests/entrypoints/openai/test_lora_resolvers.py`, `tests/entrypoints/openai/test_return_token_ids.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30035 - [Chore] Deprecate `merge_by_field_config` arg

- Link: https://github.com/vllm-project/vllm/pull/30035
- Status/date: merged / 2025-12-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +90/-302, 728 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Deprecate `merge_by_field_config` arg"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `vllm/multimodal/inputs.py`, `tests/multimodal/test_inputs.py`, `vllm/multimodal/utils.py`; technical summary: Covers "[Chore] Deprecate `merge_by_field_config` arg"; the main implementation surface is `vllm/multimodal/inputs.py`, `tests/multimodal/test_inputs.py`, `vllm/multimodal/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/multimodal/inputs.py` modified +46/-96 (142 lines); hunks: -3,7 +3,7; -201,8 +201,10 @@ def __eq__(self, other: object) -> bool:; symbols: __eq__, nested_tensors_equal, batched_tensors_equal, MultiModalFeatureSpec, touching `__eq__, nested_tensors_equal, batched_tensors_equal`; `tests/multimodal/test_inputs.py` removed +0/-91 (91 lines); hunks: -1,91 +0,0; symbols: assert_nested_tensors_equal, assert_multimodal_inputs_equal, test_multimodal_input_batch_single_tensor, test_multimodal_input_batch_multiple_tensors, touching `assert_nested_tensors_equal, assert_multimodal_inputs_equal, test_multimodal_input_batch_single_tensor`; `vllm/multimodal/utils.py` modified +12/-47 (59 lines); hunks: -19,7 +19,6; -427,59 +426,25 @@ def group_mm_kwargs_by_modality(; symbols: group_mm_kwargs_by_modality, fetch_audio, touching `group_mm_kwargs_by_modality, fetch_audio`; `vllm/model_executor/models/nano_nemotron_vl.py` modified +6/-6 (12 lines); hunks: -52,7 +52,6; -849,17 +848,18 @@ def _get_prompt_updates(; symbols: _get_prompt_updates, touching `_get_prompt_updates`.
- Code diff details:
  - `vllm/multimodal/inputs.py` modified +46/-96 (142 lines); hunks: -3,7 +3,7; -201,8 +201,10 @@ def __eq__(self, other: object) -> bool:; symbols: __eq__, nested_tensors_equal, batched_tensors_equal, MultiModalFeatureSpec
  - `tests/multimodal/test_inputs.py` removed +0/-91 (91 lines); hunks: -1,91 +0,0; symbols: assert_nested_tensors_equal, assert_multimodal_inputs_equal, test_multimodal_input_batch_single_tensor, test_multimodal_input_batch_multiple_tensors
  - `vllm/multimodal/utils.py` modified +12/-47 (59 lines); hunks: -19,7 +19,6; -427,59 +426,25 @@ def group_mm_kwargs_by_modality(; symbols: group_mm_kwargs_by_modality, fetch_audio
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +6/-6 (12 lines); hunks: -52,7 +52,6; -849,17 +848,18 @@ def _get_prompt_updates(; symbols: _get_prompt_updates
  - `tests/multimodal/test_cache.py` modified +3/-6 (9 lines); hunks: -85,12 +85,6 @@ def _dummy_items(; -107,6 +101,9 @@ def test_cache_item_size(item, expected_size):; symbols: _dummy_items, test_cache_item_size, _create_vllm_config
- Key code excerpts:

```diff
diff -- vllm/multimodal/inputs.py
@@ -3,7 +3,7 @@
-from collections.abc import Mapping, Sequence
+from collections.abc import Mapping, Sequence, Set
@@ -201,8 +201,10 @@ def __eq__(self, other: object) -> bool:
-    """Equality check between
-    [`NestedTensors`][vllm.multimodal.inputs.NestedTensors] objects."""
+    """
diff -- tests/multimodal/test_inputs.py
@@ -1,91 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import pytest
-import torch
-from vllm.multimodal.inputs import MultiModalKwargs, NestedTensors
-pytestmark = pytest.mark.cpu_test
diff -- vllm/multimodal/utils.py
@@ -19,7 +19,6 @@
```

- Reviewed files:
  - runtime: `vllm/multimodal/inputs.py` modified +46/-96; `vllm/multimodal/utils.py` modified +12/-47; `vllm/model_executor/models/nano_nemotron_vl.py` modified +6/-6; `vllm/multimodal/cache.py` modified +1/-8; `vllm/model_executor/models/paligemma.py` modified +3/-5
  - tests: `tests/multimodal/test_inputs.py` removed +0/-91; `tests/multimodal/test_cache.py` modified +3/-6; `tests/models/multimodal/processing/test_glm4_1v.py` modified +4/-3
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_glm4_1v.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/multimodal/test_cache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30170 - [Chore] Deprecate `SupportsMultiModal.merge_by_field_config`

- Link: https://github.com/vllm-project/vllm/pull/30170
- Status/date: merged / 2025-12-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 61 files, +23/-110, 568 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Deprecate `SupportsMultiModal.merge_by_field_config`"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/aya_vision.py`; technical summary: Covers "[Chore] Deprecate `SupportsMultiModal.merge_by_field_config`"; the main implementation surface is `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/aya_vision.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interfaces.py` modified +22/-3 (25 lines); hunks: -78,9 +78,9 @@ class SupportsMultiModal(Protocol):; -260,7 +260,26 @@ def supports_multimodal(model: object) -> TypeIs[SupportsMu...; symbols: SupportsMultiModal, supports_multimodal, supports_multimodal_raw_input_only, touching `SupportsMultiModal, supports_multimodal, supports_multimodal_raw_input_only`; `vllm/model_executor/models/aria.py` modified +0/-2 (2 lines); hunks: -499,8 +499,6 @@ class AriaForConditionalGeneration(nn.Module, SupportsMultiM...; symbols: AriaForConditionalGeneration, touching `AriaForConditionalGeneration`; `vllm/model_executor/models/aya_vision.py` modified +0/-2 (2 lines); hunks: -318,8 +318,6 @@ def _get_layer_index(feature_layer_index: int, num_hidden_la...; symbols: _get_layer_index, AyaVisionForConditionalGeneration, touching `_get_layer_index, AyaVisionForConditionalGeneration`; `vllm/model_executor/models/blip2.py` modified +0/-2 (2 lines); hunks: -523,8 +523,6 @@ def _get_prompt_updates(; symbols: _get_prompt_updates, Blip2ForConditionalGeneration, get_placeholder_str, touching `_get_prompt_updates, Blip2ForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/interfaces.py` modified +22/-3 (25 lines); hunks: -78,9 +78,9 @@ class SupportsMultiModal(Protocol):; -260,7 +260,26 @@ def supports_multimodal(model: object) -> TypeIs[SupportsMu...; symbols: SupportsMultiModal, supports_multimodal, supports_multimodal_raw_input_only
  - `vllm/model_executor/models/aria.py` modified +0/-2 (2 lines); hunks: -499,8 +499,6 @@ class AriaForConditionalGeneration(nn.Module, SupportsMultiM...; symbols: AriaForConditionalGeneration
  - `vllm/model_executor/models/aya_vision.py` modified +0/-2 (2 lines); hunks: -318,8 +318,6 @@ def _get_layer_index(feature_layer_index: int, num_hidden_la...; symbols: _get_layer_index, AyaVisionForConditionalGeneration
  - `vllm/model_executor/models/blip2.py` modified +0/-2 (2 lines); hunks: -523,8 +523,6 @@ def _get_prompt_updates(; symbols: _get_prompt_updates, Blip2ForConditionalGeneration, get_placeholder_str
  - `vllm/model_executor/models/chameleon.py` modified +0/-2 (2 lines); hunks: -918,8 +918,6 @@ def forward(; symbols: forward, ChameleonForConditionalGeneration
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -78,9 +78,9 @@ class SupportsMultiModal(Protocol):
-    merge_by_field_config: ClassVar[bool] = True
+    merge_by_field_config: ClassVar[bool | None] = None
-    A flag that indicates which implementation of
+    [DEPRECATED] A flag that indicates which implementation of
@@ -260,7 +260,26 @@ def supports_multimodal(model: object) -> TypeIs[SupportsMultiModal]: ...
-    return getattr(model, "supports_multimodal", False)
diff -- vllm/model_executor/models/aria.py
@@ -499,8 +499,6 @@ class AriaForConditionalGeneration(nn.Module, SupportsMultiModal):
-    merge_by_field_config = True
diff -- vllm/model_executor/models/aya_vision.py
@@ -318,8 +318,6 @@ def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
-    merge_by_field_config = True
diff -- vllm/model_executor/models/blip2.py
@@ -523,8 +523,6 @@ def _get_prompt_updates(
-    merge_by_field_config = True
diff -- vllm/model_executor/models/chameleon.py
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +22/-3; `vllm/model_executor/models/aria.py` modified +0/-2; `vllm/model_executor/models/aya_vision.py` modified +0/-2; `vllm/model_executor/models/blip2.py` modified +0/-2; `vllm/model_executor/models/chameleon.py` modified +0/-2; `vllm/model_executor/models/cohere2_vision.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/aya_vision.py`, `vllm/model_executor/models/blip2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30145 - [Renderer] Separate out `RendererConfig` from `ModelConfig`

- Link: https://github.com/vllm-project/vllm/pull/30145
- Status/date: merged / 2025-12-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 105 files, +971/-799, 4859 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Renderer] Separate out `RendererConfig` from `ModelConfig`"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `tests/entrypoints/test_chat_utils.py`, `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/registry.py`; technical summary: Covers "[Renderer] Separate out `RendererConfig` from `ModelConfig`"; the main implementation surface is `tests/entrypoints/test_chat_utils.py`, `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/entrypoints/test_chat_utils.py` modified +71/-123 (194 lines); hunks: -11,7 +11,7; -232,7 +232,7 @@ def test_parse_chat_messages_single_image(; symbols: test_parse_chat_messages_single_image, test_parse_chat_messages_single_image_with_uuid, test_parse_chat_messages_single_empty_image_with_uuid, test_parse_chat_messages_single_image_with_bad_uuid_format, touching `test_parse_chat_messages_single_image, test_parse_chat_messages_single_image_with_uuid, test_parse_chat_messages_single_empty_image_with_uuid`; `vllm/entrypoints/chat_utils.py` modified +45/-34 (79 lines); hunks: -44,7 +44,7; -452,9 +452,10 @@ def resolve_mistral_chat_template(; symbols: resolve_mistral_chat_template, _try_get_processor_chat_template, resolve_hf_chat_template, touching `resolve_mistral_chat_template, _try_get_processor_chat_template, resolve_hf_chat_template`; `vllm/multimodal/registry.py` modified +30/-34 (64 lines); hunks: -6,7 +6,7; -22,7 +22,7; symbols: _extract_mm_options, supports_multimodal_inputs, get_max_tokens_per_item_by_modality, touching `_extract_mm_options, supports_multimodal_inputs, get_max_tokens_per_item_by_modality`; `tests/v1/structured_output/test_reasoning_structured_output.py` modified +21/-14 (35 lines); hunks: -7,7 +7,7; -17,19 +17,26 @@ class TestReasoningStructuredOutput:; symbols: TestReasoningStructuredOutput, mock_model_config, mock_renderer_config, mock_scheduler_config, touching `TestReasoningStructuredOutput, mock_model_config, mock_renderer_config`.
- Code diff details:
  - `tests/entrypoints/test_chat_utils.py` modified +71/-123 (194 lines); hunks: -11,7 +11,7; -232,7 +232,7 @@ def test_parse_chat_messages_single_image(; symbols: test_parse_chat_messages_single_image, test_parse_chat_messages_single_image_with_uuid, test_parse_chat_messages_single_empty_image_with_uuid, test_parse_chat_messages_single_image_with_bad_uuid_format
  - `vllm/entrypoints/chat_utils.py` modified +45/-34 (79 lines); hunks: -44,7 +44,7; -452,9 +452,10 @@ def resolve_mistral_chat_template(; symbols: resolve_mistral_chat_template, _try_get_processor_chat_template, resolve_hf_chat_template
  - `vllm/multimodal/registry.py` modified +30/-34 (64 lines); hunks: -6,7 +6,7; -22,7 +22,7; symbols: _extract_mm_options, supports_multimodal_inputs, get_max_tokens_per_item_by_modality
  - `tests/v1/structured_output/test_reasoning_structured_output.py` modified +21/-14 (35 lines); hunks: -7,7 +7,7; -17,19 +17,26 @@ class TestReasoningStructuredOutput:; symbols: TestReasoningStructuredOutput, mock_model_config, mock_renderer_config, mock_scheduler_config
  - `tests/models/multimodal/test_mapping.py` modified +3/-30 (33 lines); hunks: -7,7 +7,6; -50,37 +49,11 @@ def test_hf_model_weights_mapper(model_arch: str):; symbols: test_hf_model_weights_mapper
- Key code excerpts:

```diff
diff -- tests/entrypoints/test_chat_utils.py
@@ -11,7 +11,7 @@
-from vllm.config import ModelConfig
+from vllm.config import ModelConfig, RendererConfig
@@ -232,7 +232,7 @@ def test_parse_chat_messages_single_image(
-        phi3v_model_config,
+        RendererConfig(model_config=phi3v_model_config),
@@ -264,7 +264,7 @@ def test_parse_chat_messages_single_image_with_uuid(
diff -- vllm/entrypoints/chat_utils.py
@@ -44,7 +44,7 @@
-from vllm.config import ModelConfig
+from vllm.config import ModelConfig, RendererConfig
@@ -452,9 +452,10 @@ def resolve_mistral_chat_template(
-    model_config: ModelConfig,
+    *,
+    trust_remote_code: bool,
diff -- vllm/multimodal/registry.py
@@ -6,7 +6,7 @@
```

- Reviewed files:
  - tests: `tests/entrypoints/test_chat_utils.py` modified +71/-123; `tests/v1/structured_output/test_reasoning_structured_output.py` modified +21/-14; `tests/models/multimodal/test_mapping.py` modified +3/-30; `tests/models/registry.py` modified +32/-1; `tests/entrypoints/openai/test_serving_chat.py` modified +21/-7
  - runtime: `vllm/entrypoints/chat_utils.py` modified +45/-34; `vllm/multimodal/registry.py` modified +30/-34; `vllm/multimodal/processing.py` modified +23/-5
- Risk and verification: The diff ships test coverage in `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`, `tests/compile/test_fusion.py`, `tests/compile/test_fusion_attn.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30199 - Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)"

- Link: https://github.com/vllm-project/vllm/pull/30199
- Status/date: merged / 2025-12-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 105 files, +799/-971, 4859 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)""; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `tests/entrypoints/test_chat_utils.py`, `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/registry.py`; technical summary: Covers "Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)""; the main implementation surface is `tests/entrypoints/test_chat_utils.py`, `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/entrypoints/test_chat_utils.py` modified +123/-71 (194 lines); hunks: -12,7 +12,7; -233,7 +233,7 @@ def test_parse_chat_messages_single_image(; symbols: test_parse_chat_messages_single_image, test_parse_chat_messages_single_image_with_uuid, test_parse_chat_messages_single_empty_image_with_uuid, test_parse_chat_messages_single_image_with_bad_uuid_format, touching `test_parse_chat_messages_single_image, test_parse_chat_messages_single_image_with_uuid, test_parse_chat_messages_single_empty_image_with_uuid`; `vllm/entrypoints/chat_utils.py` modified +34/-45 (79 lines); hunks: -44,7 +44,7; -452,10 +452,9 @@ def resolve_mistral_chat_template(; symbols: resolve_mistral_chat_template, _try_get_processor_chat_template, resolve_hf_chat_template, touching `resolve_mistral_chat_template, _try_get_processor_chat_template, resolve_hf_chat_template`; `vllm/multimodal/registry.py` modified +34/-30 (64 lines); hunks: -6,7 +6,7; -22,7 +22,7; symbols: _extract_mm_options, supports_multimodal_inputs, get_max_tokens_per_item_by_modality, touching `_extract_mm_options, supports_multimodal_inputs, get_max_tokens_per_item_by_modality`; `tests/v1/structured_output/test_reasoning_structured_output.py` modified +14/-21 (35 lines); hunks: -7,7 +7,7; -17,26 +17,19 @@ class TestReasoningStructuredOutput:; symbols: TestReasoningStructuredOutput, mock_renderer_config, mock_model_config, mock_scheduler_config, touching `TestReasoningStructuredOutput, mock_renderer_config, mock_model_config`.
- Code diff details:
  - `tests/entrypoints/test_chat_utils.py` modified +123/-71 (194 lines); hunks: -12,7 +12,7; -233,7 +233,7 @@ def test_parse_chat_messages_single_image(; symbols: test_parse_chat_messages_single_image, test_parse_chat_messages_single_image_with_uuid, test_parse_chat_messages_single_empty_image_with_uuid, test_parse_chat_messages_single_image_with_bad_uuid_format
  - `vllm/entrypoints/chat_utils.py` modified +34/-45 (79 lines); hunks: -44,7 +44,7; -452,10 +452,9 @@ def resolve_mistral_chat_template(; symbols: resolve_mistral_chat_template, _try_get_processor_chat_template, resolve_hf_chat_template
  - `vllm/multimodal/registry.py` modified +34/-30 (64 lines); hunks: -6,7 +6,7; -22,7 +22,7; symbols: _extract_mm_options, supports_multimodal_inputs, get_max_tokens_per_item_by_modality
  - `tests/v1/structured_output/test_reasoning_structured_output.py` modified +14/-21 (35 lines); hunks: -7,7 +7,7; -17,26 +17,19 @@ class TestReasoningStructuredOutput:; symbols: TestReasoningStructuredOutput, mock_renderer_config, mock_model_config, mock_scheduler_config
  - `tests/models/multimodal/test_mapping.py` modified +30/-3 (33 lines); hunks: -7,6 +7,7; -49,11 +50,37 @@ def test_hf_model_weights_mapper(model_arch: str):; symbols: test_hf_model_weights_mapper
- Key code excerpts:

```diff
diff -- tests/entrypoints/test_chat_utils.py
@@ -12,7 +12,7 @@
-from vllm.config import ModelConfig, RendererConfig
+from vllm.config import ModelConfig
@@ -233,7 +233,7 @@ def test_parse_chat_messages_single_image(
-        RendererConfig(model_config=phi3v_model_config),
+        phi3v_model_config,
@@ -265,7 +265,7 @@ def test_parse_chat_messages_single_image_with_uuid(
diff -- vllm/entrypoints/chat_utils.py
@@ -44,7 +44,7 @@
-from vllm.config import ModelConfig, RendererConfig
+from vllm.config import ModelConfig
@@ -452,10 +452,9 @@ def resolve_mistral_chat_template(
-    *,
-    trust_remote_code: bool,
+    model_config: ModelConfig,
diff -- vllm/multimodal/registry.py
@@ -6,7 +6,7 @@
```

- Reviewed files:
  - tests: `tests/entrypoints/test_chat_utils.py` modified +123/-71; `tests/v1/structured_output/test_reasoning_structured_output.py` modified +14/-21; `tests/models/multimodal/test_mapping.py` modified +30/-3; `tests/models/registry.py` modified +1/-32; `tests/entrypoints/openai/test_serving_chat.py` modified +7/-21
  - runtime: `vllm/entrypoints/chat_utils.py` modified +34/-45; `vllm/multimodal/registry.py` modified +34/-30; `vllm/multimodal/processing.py` modified +5/-23
- Risk and verification: The diff ships test coverage in `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`, `tests/compile/test_fusion.py`, `tests/compile/test_fusion_attn.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31569 - feat: support LoRA for DeepSeek-OCR(Language Model part)

- Link: https://github.com/vllm-project/vllm/pull/31569
- Status/date: merged / 2026-01-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-2, 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: support LoRA for DeepSeek-OCR(Language Model part)"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/models/supported_models.md`; technical summary: Covers "feat: support LoRA for DeepSeek-OCR(Language Model part)"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `docs/models/supported_models.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1 (14 lines); hunks: -14,9 +14,11; -343,7 +345,7 @@ def get_replacement_deepseek_vl2(item_idx: int):; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, load_weights, get_mm_mapping, touching `get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, load_weights`; `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -677,7 +677,7 @@ These models primarily accept the [`LLM.generate`](./generat....
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1 (14 lines); hunks: -14,9 +14,11; -343,7 +345,7 @@ def get_replacement_deepseek_vl2(item_idx: int):; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, load_weights, get_mm_mapping
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -677,7 +677,7 @@ These models primarily accept the [`LLM.generate`](./generat...
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -14,9 +14,11 @@
+    SupportsLoRA,
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -343,7 +345,7 @@ def get_replacement_deepseek_vl2(item_idx: int):
-class DeepseekOCRForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
+class DeepseekOCRForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
@@ -589,3 +591,13 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
diff -- docs/models/supported_models.md
@@ -677,7 +677,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
-| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | | ✅︎ |
+| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | ✅︎ | ✅︎ |
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +13/-1
  - docs: `docs/models/supported_models.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31947 - [Model] Standardize common vision encoders

- Link: https://github.com/vllm-project/vllm/pull/31947
- Status/date: merged / 2026-01-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +254/-174, 1287 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Standardize common vision encoders"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/siglip.py`, `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/phi3v.py`; technical summary: Covers "[Model] Standardize common vision encoders"; the main implementation surface is `vllm/model_executor/models/siglip.py`, `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/phi3v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/siglip.py` modified +73/-55 (128 lines); hunks: -1,7 +1,6; -19,7 +18,7; symbols: get_patch_grid_length, SiglipVisionEmbeddings, __init__, touching `get_patch_grid_length, SiglipVisionEmbeddings, __init__`; `vllm/model_executor/models/clip.py` modified +57/-12 (69 lines); hunks: -17,7 +17,7; -353,6 +353,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/phi3v.py` modified +31/-31 (62 lines); hunks: -29,7 +29,7; -96,6 +96,7; symbols: _init_img_processor, Phi3VImageEmbeddingInputs, Phi3ImageEmbeddingBase, __init__, touching `_init_img_processor, Phi3VImageEmbeddingInputs, Phi3ImageEmbeddingBase`; `vllm/model_executor/models/pixtral.py` modified +38/-7 (45 lines); hunks: -28,7 +28,7; -1043,25 +1043,34 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/siglip.py` modified +73/-55 (128 lines); hunks: -1,7 +1,6; -19,7 +18,7; symbols: get_patch_grid_length, SiglipVisionEmbeddings, __init__
  - `vllm/model_executor/models/clip.py` modified +57/-12 (69 lines); hunks: -17,7 +17,7; -353,6 +353,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/phi3v.py` modified +31/-31 (62 lines); hunks: -29,7 +29,7; -96,6 +96,7; symbols: _init_img_processor, Phi3VImageEmbeddingInputs, Phi3ImageEmbeddingBase, __init__
  - `vllm/model_executor/models/pixtral.py` modified +38/-7 (45 lines); hunks: -28,7 +28,7; -1043,25 +1043,34 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/paddleocr_vl.py` modified +2/-42 (44 lines); hunks: -38,10 +38,8; -77,6 +75,7; symbols: forward, SiglipMLP, __init__, SiglipEncoderLayer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/siglip.py
@@ -1,7 +1,6 @@
-import math
@@ -19,7 +18,7 @@
-from vllm.config.multimodal import BaseDummyOptions
+from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
@@ -276,7 +275,7 @@ def get_patch_grid_length(self) -> int:
-# Adapted from https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/siglip/modeling_siglip.py#L249 # noqa
diff -- vllm/model_executor/models/clip.py
@@ -17,7 +17,7 @@
-from vllm.config.multimodal import BaseDummyOptions
+from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
@@ -353,6 +353,7 @@ def __init__(
+        multimodal_config: MultiModalConfig | None = None,
@@ -365,36 +366,54 @@ def __init__(
-                "embed_dim must be divisible by num_heads "
diff -- vllm/model_executor/models/phi3v.py
@@ -29,7 +29,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/siglip.py` modified +73/-55; `vllm/model_executor/models/clip.py` modified +57/-12; `vllm/model_executor/models/phi3v.py` modified +31/-31; `vllm/model_executor/models/pixtral.py` modified +38/-7; `vllm/model_executor/models/paddleocr_vl.py` modified +2/-42; `vllm/model_executor/models/hyperclovax_vision.py` modified +9/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py`, `vllm/model_executor/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32016 - [Model] Remove redundant None check in DeepSeekOCR image input processing

- Link: https://github.com/vllm-project/vllm/pull/32016
- Status/date: merged / 2026-01-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-13, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove redundant None check in DeepSeekOCR image input processing"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_ocr.py`; technical summary: Covers "[Model] Remove redundant None check in DeepSeekOCR image input processing"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13 (23 lines); hunks: -437,19 +437,16 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _encode_global_features, touching `_parse_and_validate_image_input, _encode_global_features`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13 (23 lines); hunks: -437,19 +437,16 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _encode_global_features
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -437,19 +437,16 @@ def _parse_and_validate_image_input(
-        if pixel_values is not None:
-            base_size = self.vision_config.image_size
-            return DeepseekOCRImagePixelInputs(
-                type="pixel_values",
-                data=pixel_values,
-                images_crop=images_crop,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +10/-13
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32327 - [1/N] Reorganize multimodal processing code

- Link: https://github.com/vllm-project/vllm/pull/32327
- Status/date: merged / 2026-01-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 76 files, +717/-670, 2419 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N] Reorganize multimodal processing code"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/multimodal/processing/processor.py`, `vllm/multimodal/processing/context.py`, `vllm/multimodal/processing/__init__.py`; technical summary: Covers "[1/N] Reorganize multimodal processing code"; the main implementation surface is `vllm/multimodal/processing/processor.py`, `vllm/multimodal/processing/context.py`, `vllm/multimodal/processing/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/multimodal/processing/processor.py` renamed +16/-556 (572 lines); hunks: -1,24 +1,18; -27,173 +21,43; symbols: get_current_request_id, set_request_id, MultiModalProcessorTimingStats, to_dict, touching `get_current_request_id, set_request_id, MultiModalProcessorTimingStats`; `vllm/multimodal/processing/context.py` added +558/-0 (558 lines); hunks: -0,0 +1,558; symbols: get_current_request_id, set_request_id, MultiModalProcessorTimingStats, to_dict, touching `get_current_request_id, set_request_id, MultiModalProcessorTimingStats`; `vllm/multimodal/processing/__init__.py` added +27/-0 (27 lines); hunks: -0,0 +1,27; `vllm/multimodal/inputs.py` modified +9/-5 (14 lines); hunks: -33,8 +33,6; -979,9 +977,15 @@ def get_data(; symbols: get_data, MultiModalInputs, touching `get_data, MultiModalInputs`.
- Code diff details:
  - `vllm/multimodal/processing/processor.py` renamed +16/-556 (572 lines); hunks: -1,24 +1,18; -27,173 +21,43; symbols: get_current_request_id, set_request_id, MultiModalProcessorTimingStats, to_dict
  - `vllm/multimodal/processing/context.py` added +558/-0 (558 lines); hunks: -0,0 +1,558; symbols: get_current_request_id, set_request_id, MultiModalProcessorTimingStats, to_dict
  - `vllm/multimodal/processing/__init__.py` added +27/-0 (27 lines); hunks: -0,0 +1,27
  - `vllm/multimodal/inputs.py` modified +9/-5 (14 lines); hunks: -33,8 +33,6; -979,9 +977,15 @@ def get_data(; symbols: get_data, MultiModalInputs
  - `vllm/multimodal/processing/dummy_inputs.py` renamed +4/-8 (12 lines); hunks: -3,7 +3,7; -17,14 +17,10
- Key code excerpts:

```diff
diff -- vllm/multimodal/processing/processor.py
@@ -1,24 +1,18 @@
-import contextvars
-import threading
-import time
-from contextlib import contextmanager
-    Any,
-    overload,
diff -- vllm/multimodal/processing/context.py
@@ -0,0 +1,558 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import contextvars
+import threading
+import time
+from abc import abstractmethod
diff -- vllm/multimodal/processing/__init__.py
@@ -0,0 +1,27 @@
```

- Reviewed files:
  - runtime: `vllm/multimodal/processing/processor.py` renamed +16/-556; `vllm/multimodal/processing/context.py` added +558/-0; `vllm/multimodal/processing/__init__.py` added +27/-0; `vllm/multimodal/inputs.py` modified +9/-5; `vllm/multimodal/processing/dummy_inputs.py` renamed +4/-8; `vllm/model_executor/models/aya_vision.py` modified +6/-3
  - docs: `docs/contributing/model/multimodal.md` modified +4/-6
- Risk and verification: The diff ships test coverage in `tests/multimodal/test_processing.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32632 - [1/N] Initialize MM components in context managers (A-D)

- Link: https://github.com/vllm-project/vllm/pull/32632
- Status/date: merged / 2026-01-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +239/-267, 721 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N] Initialize MM components in context managers (A-D)"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/deepseek_vl2.py`; technical summary: Covers "[1/N] Initialize MM components in context managers (A-D)"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/deepseek_vl2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +40/-41 (81 lines); hunks: -383,46 +383,48 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -552,9 +554,6 @@ def _process_image_input(; symbols: __init__, _process_image_input, get_language_model, embed_multimodal, touching `__init__, _process_image_input, get_language_model`; `vllm/model_executor/models/bagel.py` modified +36/-43 (79 lines); hunks: -44,6 +44,7; -373,12 +374,13 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, embed_multimodal, get_language_model, forward, touching `__init__, embed_multimodal, get_language_model`; `vllm/model_executor/models/deepseek_vl2.py` modified +29/-30 (59 lines); hunks: -374,37 +374,39 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -603,9 +605,6 @@ def _process_image_input(; symbols: __init__, _process_image_input, get_language_model, embed_multimodal, touching `__init__, _process_image_input, get_language_model`; `vllm/model_executor/models/blip2.py` modified +24/-30 (54 lines); hunks: -549,31 +549,31 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -614,8 +614,6 @@ def _image_pixels_to_features(; symbols: __init__, token, _image_pixels_to_features, _process_image_pixels, touching `__init__, token, _image_pixels_to_features`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +40/-41 (81 lines); hunks: -383,46 +383,48 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -552,9 +554,6 @@ def _process_image_input(; symbols: __init__, _process_image_input, get_language_model, embed_multimodal
  - `vllm/model_executor/models/bagel.py` modified +36/-43 (79 lines); hunks: -44,6 +44,7; -373,12 +374,13 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, embed_multimodal, get_language_model, forward
  - `vllm/model_executor/models/deepseek_vl2.py` modified +29/-30 (59 lines); hunks: -374,37 +374,39 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -603,9 +605,6 @@ def _process_image_input(; symbols: __init__, _process_image_input, get_language_model, embed_multimodal
  - `vllm/model_executor/models/blip2.py` modified +24/-30 (54 lines); hunks: -549,31 +549,31 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -614,8 +614,6 @@ def _image_pixels_to_features(; symbols: __init__, token, _image_pixels_to_features, _process_image_pixels
  - `vllm/model_executor/models/aria.py` modified +21/-32 (53 lines); hunks: -15,9 +15,7; -539,30 +537,22 @@ def __init__(; symbols: __init__, _parse_and_validate_image_input, _process_image_input, get_language_model
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -383,46 +383,48 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.sam_model = build_sam_vit_b()
-        clip_vision_config = CLIPVisionConfig(
-            hidden_size=1024,
-            intermediate_size=4096,
-            num_attention_heads=16,
-            num_hidden_layers=24,
diff -- vllm/model_executor/models/bagel.py
@@ -44,6 +44,7 @@
+    TowerMissingLayer,
@@ -373,12 +374,13 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.language_model = init_vllm_registered_model(
-            vllm_config=vllm_config,
-            hf_config=config.llm_config,
-            prefix=maybe_prefix(prefix, "language_model"),
diff -- vllm/model_executor/models/deepseek_vl2.py
@@ -374,37 +374,39 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +40/-41; `vllm/model_executor/models/bagel.py` modified +36/-43; `vllm/model_executor/models/deepseek_vl2.py` modified +29/-30; `vllm/model_executor/models/blip2.py` modified +24/-30; `vllm/model_executor/models/aria.py` modified +21/-32; `vllm/model_executor/models/clip.py` modified +23/-24
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31972 - [Models]: Make Multimodal config implicit in ViT implementation

- Link: https://github.com/vllm-project/vllm/pull/31972
- Status/date: merged / 2026-01-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +118/-470, 2781 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models]: Make Multimodal config implicit in ViT implementation"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `vllm/model_executor/models/vision.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/dots_ocr.py`; technical summary: Covers "[Models]: Make Multimodal config implicit in ViT implementation"; the main implementation surface is `vllm/model_executor/models/vision.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/dots_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/vision.py` modified +48/-2 (50 lines); hunks: -10,7 +10,7; -79,7 +79,7 @@ def get_vision_encoder_info(hf_config: VisionLanguageConfig) -...; symbols: get_vision_encoder_info, get_vit_attn_backend, _get_vit_attn_backend, touching `get_vision_encoder_info, get_vit_attn_backend, _get_vit_attn_backend`; `vllm/model_executor/models/qwen2_vl.py` modified +6/-36 (42 lines); hunks: -43,7 +43,7; -106,6 +106,7; symbols: __init__, touching `__init__`; `vllm/model_executor/models/dots_ocr.py` modified +5/-34 (39 lines); hunks: -8,7 +8,7; -60,7 +60,7; symbols: __init__, touching `__init__`; `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-33 (38 lines); hunks: -43,7 +43,7; -109,6 +109,7; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/vision.py` modified +48/-2 (50 lines); hunks: -10,7 +10,7; -79,7 +79,7 @@ def get_vision_encoder_info(hf_config: VisionLanguageConfig) -...; symbols: get_vision_encoder_info, get_vit_attn_backend, _get_vit_attn_backend
  - `vllm/model_executor/models/qwen2_vl.py` modified +6/-36 (42 lines); hunks: -43,7 +43,7; -106,6 +106,7; symbols: __init__
  - `vllm/model_executor/models/dots_ocr.py` modified +5/-34 (39 lines); hunks: -8,7 +8,7; -60,7 +60,7; symbols: __init__
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-33 (38 lines); hunks: -43,7 +43,7; -109,6 +109,7; symbols: __init__
  - `vllm/model_executor/models/glm4_1v.py` modified +5/-30 (35 lines); hunks: -46,7 +46,7; -107,6 +107,7; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/vision.py
@@ -10,7 +10,7 @@
-from vllm.config import VllmConfig
+from vllm.config import MultiModalConfig, VllmConfig, get_current_vllm_config
@@ -79,7 +79,7 @@ def get_vision_encoder_info(hf_config: VisionLanguageConfig) -> VisionEncoderInf
-def get_vit_attn_backend(
+def _get_vit_attn_backend(
@@ -95,6 +95,52 @@ def get_vit_attn_backend(
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -43,7 +43,7 @@
-from vllm.config import MultiModalConfig, VllmConfig
+from vllm.config import VllmConfig
@@ -106,6 +106,7 @@
+    is_vit_use_data_parallel,
@@ -247,15 +248,10 @@ def __init__(
-        multimodal_config: MultiModalConfig | None = None,
diff -- vllm/model_executor/models/dots_ocr.py
@@ -8,7 +8,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/vision.py` modified +48/-2; `vllm/model_executor/models/qwen2_vl.py` modified +6/-36; `vllm/model_executor/models/dots_ocr.py` modified +5/-34; `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-33; `vllm/model_executor/models/glm4_1v.py` modified +5/-30; `vllm/model_executor/models/siglip.py` modified +4/-29
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/attention/mm_encoder_attention.py`, `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/deepencoder.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- Link: https://github.com/vllm-project/vllm/pull/33063
- Status/date: merged / 2026-01-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 164 files, +243/-241, 2158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Update type annotation of `input_ids` in model forward"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`; technical summary: Covers "[Chore] Update type annotation of `input_ids` in model forward"; the main implementation surface is `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention, touching `forward, ModernBertAttention`; `vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward, touching `altup_embed, forward, embed_input_ids`; `vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights, touching `embed_input_ids, forward, load_weights`; `vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__, touching `embed_input_ids, forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention
  - `vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward
  - `vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights
  - `vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__
  - `vllm/model_executor/models/opt.py` modified +3/-3 (6 lines); hunks: -267,7 +267,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -316,7 +316,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/modernbert.py
@@ -54,12 +54,11 @@ def forward(
-        if inputs_embeds is not None:
-            return self.norm(inputs_embeds)
-        else:
+        if inputs_embeds is None:
-            embeddings = self.norm(inputs_embeds)
-            return embeddings
diff -- vllm/model_executor/models/gemma3n.py
@@ -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -964,7 +964,7 @@ def fast_prefill_forward(
diff -- vllm/model_executor/models/gpt2.py
@@ -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/modernbert.py` modified +4/-5; `vllm/model_executor/models/gemma3n.py` modified +4/-4; `vllm/model_executor/models/gpt2.py` modified +3/-3; `vllm/model_executor/models/internlm2.py` modified +3/-3; `vllm/model_executor/models/opt.py` modified +3/-3; `vllm/model_executor/models/afmoe.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33165 - [Model] Support DeepSeek-OCR-2

- Link: https://github.com/vllm-project/vllm/pull/33165
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_ocr2.py`; associated commits `808dd87b3054`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1099/-1, 1159 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support DeepSeek-OCR-2"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_ocr2.py`, `vllm/transformers_utils/processors/deepseek_ocr2.py`; technical summary: Covers "[Model] Support DeepSeek-OCR-2"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr2.py`, `vllm/transformers_utils/processors/deepseek_ocr2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr2.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: DeepseekOCR2ProcessingInfo, get_hf_config, get_hf_processor, get_supported_mm_limits, touching `DeepseekOCR2ProcessingInfo, get_hf_config, get_hf_processor`; `vllm/transformers_utils/processors/deepseek_ocr2.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: DeepseekOCR2Processor, __init__, bos_id, eos_id, touching `DeepseekOCR2Processor, __init__, bos_id`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr2.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: DeepseekOCR2ProcessingInfo, get_hf_config, get_hf_processor, get_supported_mm_limits
  - `vllm/transformers_utils/processors/deepseek_ocr2.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: DeepseekOCR2Processor, __init__, bos_id, eos_id
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr2.py
@@ -0,0 +1,444 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""
+import math
+from collections.abc import Iterable, Mapping, Sequence
+from functools import partial
diff -- vllm/transformers_utils/processors/deepseek_ocr2.py
@@ -0,0 +1,320 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py
+import math
+import torch
+from PIL import Image, ImageOps
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr2.py` added +444/-0; `vllm/transformers_utils/processors/deepseek_ocr2.py` added +320/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33909 - [Models] Consolidate Deepseek-OCR2 processor

- Link: https://github.com/vllm-project/vllm/pull/33909
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_ocr2.py`; associated commits `87d0d17ab583`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +52/-336, 480 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Consolidate Deepseek-OCR2 processor"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/transformers_utils/processors/deepseek_ocr2.py`, `vllm/model_executor/models/deepseek_ocr2.py`; technical summary: Covers "[Models] Consolidate Deepseek-OCR2 processor"; the main implementation surface is `vllm/transformers_utils/processors/deepseek_ocr2.py`, `vllm/model_executor/models/deepseek_ocr2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/deepseek_ocr2.py` removed +0/-320 (320 lines); hunks: -1,320 +0,0; symbols: DeepseekOCR2Processor, __init__, bos_id, eos_id, touching `DeepseekOCR2Processor, __init__, bos_id`; `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4 (16 lines); hunks: -48,11 +48,10; -62,6 +61,7; symbols: get_hf_config, get_hf_processor, get_supported_mm_limits, touching `get_hf_config, get_hf_processor, get_supported_mm_limits`.
- Code diff details:
  - `vllm/transformers_utils/processors/deepseek_ocr2.py` removed +0/-320 (320 lines); hunks: -1,320 +0,0; symbols: DeepseekOCR2Processor, __init__, bos_id, eos_id
  - `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4 (16 lines); hunks: -48,11 +48,10; -62,6 +61,7; symbols: get_hf_config, get_hf_processor, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/deepseek_ocr2.py
@@ -1,320 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# adapted from https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py
-import math
-import torch
-from PIL import Image, ImageOps
diff -- vllm/model_executor/models/deepseek_ocr2.py
@@ -48,11 +48,10 @@
-from vllm.transformers_utils.processors.deepseek_ocr2 import (
+from vllm.transformers_utils.processors.deepseek_ocr import (
-    IMAGE_SIZE,
-    DeepseekOCR2Processor,
+    DeepseekOCRProcessor,
@@ -62,6 +61,7 @@
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/deepseek_ocr2.py` removed +0/-320; `vllm/model_executor/models/deepseek_ocr2.py` modified +12/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepencoder2.py`, `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34330 - [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`

- Link: https://github.com/vllm-project/vllm/pull/34330
- Status/date: merged / 2026-02-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 72 files, +131/-27, 784 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/idefics3.py`, `vllm/multimodal/processing/dummy_inputs.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "[Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`"; the main implementation surface is `vllm/model_executor/models/idefics3.py`, `vllm/multimodal/processing/dummy_inputs.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/idefics3.py` modified +3/-11 (14 lines); hunks: -42,7 +42,7; -285,15 +285,6 @@ def get_num_image_tokens(; symbols: get_num_image_tokens, get_image_size_with_most_features, Idefics3DummyInputsBuilder, get_dummy_text, touching `get_num_image_tokens, get_image_size_with_most_features, Idefics3DummyInputsBuilder`; `vllm/multimodal/processing/dummy_inputs.py` modified +11/-1 (12 lines); hunks: -63,6 +63,7 @@ def get_dummy_mm_data(; -83,6 +84,7 @@ def get_dummy_processor_inputs(; symbols: get_dummy_mm_data, get_dummy_processor_inputs, touching `get_dummy_mm_data, get_dummy_processor_inputs`; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +7/-2 (9 lines); hunks: -358,12 +358,14 @@ def get_dummy_mm_data(; -372,7 +374,10 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, touching `get_dummy_mm_data`; `vllm/model_executor/models/qwen3_vl.py` modified +6/-2 (8 lines); hunks: -796,14 +796,18 @@ def get_dummy_mm_data(; -828,7 +832,7 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, touching `get_dummy_mm_data`.
- Code diff details:
  - `vllm/model_executor/models/idefics3.py` modified +3/-11 (14 lines); hunks: -42,7 +42,7; -285,15 +285,6 @@ def get_num_image_tokens(; symbols: get_num_image_tokens, get_image_size_with_most_features, Idefics3DummyInputsBuilder, get_dummy_text
  - `vllm/multimodal/processing/dummy_inputs.py` modified +11/-1 (12 lines); hunks: -63,6 +63,7 @@ def get_dummy_mm_data(; -83,6 +84,7 @@ def get_dummy_processor_inputs(; symbols: get_dummy_mm_data, get_dummy_processor_inputs
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +7/-2 (9 lines); hunks: -358,12 +358,14 @@ def get_dummy_mm_data(; -372,7 +374,10 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data
  - `vllm/model_executor/models/qwen3_vl.py` modified +6/-2 (8 lines); hunks: -796,14 +796,18 @@ def get_dummy_mm_data(; -828,7 +832,7 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data
  - `vllm/model_executor/models/funaudiochat.py` modified +5/-2 (7 lines); hunks: -611,8 +611,11 @@ def get_dummy_mm_data(; -656,7 +659,7 @@ def _call_hf_processor(; symbols: get_dummy_mm_data, _call_hf_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/idefics3.py
@@ -42,7 +42,7 @@
-from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
+from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
@@ -285,15 +285,6 @@ def get_num_image_tokens(
-    def get_image_size_with_most_features(self) -> ImageSize:
-        processor = self.get_hf_processor()
-        image_processor: Idefics3ImageProcessor = processor.image_processor
diff -- vllm/multimodal/processing/dummy_inputs.py
@@ -63,6 +63,7 @@ def get_dummy_mm_data(
+        mm_processor_kwargs: Mapping[str, object] | None = None,
@@ -83,6 +84,7 @@ def get_dummy_processor_inputs(
+        mm_processor_kwargs: Mapping[str, object] | None = None,
@@ -92,16 +94,24 @@ def get_dummy_processor_inputs(
+            mm_processor_kwargs: Additional keyword arguments
+                                for hf_processor (optional)
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -358,12 +358,14 @@ def get_dummy_mm_data(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/idefics3.py` modified +3/-11; `vllm/multimodal/processing/dummy_inputs.py` modified +11/-1; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +7/-2; `vllm/model_executor/models/qwen3_vl.py` modified +6/-2; `vllm/model_executor/models/funaudiochat.py` modified +5/-2; `vllm/model_executor/models/qwen2_vl.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/aya_vision.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34085 - Fix DeepSeek-OCR tensor validation for all size variants

- Link: https://github.com/vllm-project/vllm/pull/34085
- Status/date: merged / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-1, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeek-OCR tensor validation for all size variants"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_ocr.py`; technical summary: Covers "Fix DeepSeek-OCR tensor validation for all size variants"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1 (12 lines); hunks: -448,14 +448,24 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, touching `_parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1 (12 lines); hunks: -448,14 +448,24 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -448,14 +448,24 @@ def _parse_and_validate_image_input(
-        base_size = self.vision_config.image_size
+        # Use actual tensor spatial dim instead of hardcoded
+        # vision_config.image_size (1024). The vision encoders (SAM & CLIP)
+        # support arbitrary resolutions via pos-encoding interpolation,
+        # so Tiny/Small/Base/Large variants all work with the same weights.
+        base_size = pixel_values.shape[-1]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +11/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35025 - [Refactor] Simplify dummy data generation

- Link: https://github.com/vllm-project/vllm/pull/35025
- Status/date: merged / 2026-02-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 78 files, +282/-367, 1791 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Simplify dummy data generation"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/config/multimodal.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/multimodal/registry.py`; technical summary: Covers "[Refactor] Simplify dummy data generation"; the main implementation surface is `vllm/config/multimodal.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/multimodal/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/config/multimodal.py` modified +36/-20 (56 lines); hunks: -2,7 +2,7; -43,11 +43,29 @@ class AudioDummyOptions(BaseDummyOptions):; symbols: AudioDummyOptions, MultiModalDummyOptionsBuiltins, MultiModalConfig, _validate_limit_per_prompt, touching `AudioDummyOptions, MultiModalDummyOptionsBuiltins, MultiModalConfig`; `vllm/model_executor/models/qwen3_vl.py` modified +23/-13 (36 lines); hunks: -703,11 +703,18 @@ def get_max_video_tokens(; -789,19 +796,15 @@ def get_dummy_mm_data(; symbols: get_max_video_tokens, get_dummy_mm_data, touching `get_max_video_tokens, get_dummy_mm_data`; `vllm/multimodal/registry.py` modified +1/-24 (25 lines); hunks: -5,7 +5,6; -99,27 +98,6 @@ class MultiModalRegistry:; symbols: MultiModalRegistry, _extract_mm_options, supports_multimodal_inputs, get_dummy_mm_inputs, touching `MultiModalRegistry, _extract_mm_options, supports_multimodal_inputs`; `docs/contributing/model/multimodal.md` modified +11/-11 (22 lines); hunks: -293,21 +293,22 @@ Assuming that the memory usage increases with the number o...; -479,17 +480,16 @@ Assuming that the memory usage increases with the number o....
- Code diff details:
  - `vllm/config/multimodal.py` modified +36/-20 (56 lines); hunks: -2,7 +2,7; -43,11 +43,29 @@ class AudioDummyOptions(BaseDummyOptions):; symbols: AudioDummyOptions, MultiModalDummyOptionsBuiltins, MultiModalConfig, _validate_limit_per_prompt
  - `vllm/model_executor/models/qwen3_vl.py` modified +23/-13 (36 lines); hunks: -703,11 +703,18 @@ def get_max_video_tokens(; -789,19 +796,15 @@ def get_dummy_mm_data(; symbols: get_max_video_tokens, get_dummy_mm_data
  - `vllm/multimodal/registry.py` modified +1/-24 (25 lines); hunks: -5,7 +5,6; -99,27 +98,6 @@ class MultiModalRegistry:; symbols: MultiModalRegistry, _extract_mm_options, supports_multimodal_inputs, get_dummy_mm_inputs
  - `docs/contributing/model/multimodal.md` modified +11/-11 (22 lines); hunks: -293,21 +293,22 @@ Assuming that the memory usage increases with the number o...; -479,17 +480,16 @@ Assuming that the memory usage increases with the number o...
  - `vllm/model_executor/models/qwen2_vl.py` modified +10/-9 (19 lines); hunks: -925,9 +925,14 @@ def get_image_size_with_most_features(; -1027,22 +1032,18 @@ def get_dummy_mm_data(; symbols: get_image_size_with_most_features, get_dummy_mm_data
- Key code excerpts:

```diff
diff -- vllm/config/multimodal.py
@@ -2,7 +2,7 @@
-from typing import Any, Literal, TypeAlias
+from typing import Any, Literal, TypeAlias, TypedDict, final
@@ -43,11 +43,29 @@ class AudioDummyOptions(BaseDummyOptions):
+@final
+class MultiModalDummyOptionsBuiltins(TypedDict, total=False):
+    """Type annotations for modality types predefined by vLLM."""
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -703,11 +703,18 @@ def get_max_video_tokens(
-        video_max_pixels = video_processor.size["longest_edge"]
+        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
+        video_size = mm_kwargs.get("size", video_processor.size)
+        temporal_patch_size = mm_kwargs.get(
+            "temporal_patch_size", video_processor.temporal_patch_size
+        )
diff -- vllm/multimodal/registry.py
@@ -5,7 +5,6 @@
```

- Reviewed files:
  - runtime: `vllm/config/multimodal.py` modified +36/-20; `vllm/model_executor/models/qwen3_vl.py` modified +23/-13; `vllm/multimodal/registry.py` modified +1/-24; `vllm/model_executor/models/qwen2_vl.py` modified +10/-9; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +6/-10; `vllm/multimodal/processing/dummy_inputs.py` modified +3/-13
  - docs: `docs/contributing/model/multimodal.md` modified +11/-11
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_audioflamingo3.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36024 - [Misc] Lazy import registered processors

- Link: https://github.com/vllm-project/vllm/pull/36024
- Status/date: merged / 2026-03-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +68/-51, 288 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Lazy import registered processors"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `vllm/transformers_utils/processors/__init__.py`, `tests/models/registry.py`, `vllm/transformers_utils/processors/deepseek_ocr.py`; technical summary: Covers "[Misc] Lazy import registered processors"; the main implementation surface is `vllm/transformers_utils/processors/__init__.py`, `tests/models/registry.py`, `vllm/transformers_utils/processors/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/__init__.py` modified +28/-10 (38 lines); hunks: -8,16 +8,20; -28,4 +32,18; symbols: __getattr__, __dir__, touching `__getattr__, __dir__`; `tests/models/registry.py` modified +2/-5 (7 lines); hunks: -1020,18 +1020,15 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +1/-4 (5 lines); hunks: -8,7 +8,7; -453,6 +453,3 @@ def tokenize_with_images(; symbols: tokenize_with_images, touching `tokenize_with_images`; `vllm/transformers_utils/processors/deepseek_vl2.py` modified +1/-4 (5 lines); hunks: -29,7 +29,7; -401,6 +401,3 @@ def tokenize_with_images(; symbols: tokenize_with_images, touching `tokenize_with_images`.
- Code diff details:
  - `vllm/transformers_utils/processors/__init__.py` modified +28/-10 (38 lines); hunks: -8,16 +8,20; -28,4 +32,18; symbols: __getattr__, __dir__
  - `tests/models/registry.py` modified +2/-5 (7 lines); hunks: -1020,18 +1020,15 @@ def check_available_online(; symbols: check_available_online
  - `vllm/transformers_utils/processors/deepseek_ocr.py` modified +1/-4 (5 lines); hunks: -8,7 +8,7; -453,6 +453,3 @@ def tokenize_with_images(; symbols: tokenize_with_images
  - `vllm/transformers_utils/processors/deepseek_vl2.py` modified +1/-4 (5 lines); hunks: -29,7 +29,7; -401,6 +401,3 @@ def tokenize_with_images(; symbols: tokenize_with_images
  - `vllm/transformers_utils/processors/ovis.py` modified +1/-4 (5 lines); hunks: -26,7 +26,7; -453,6 +453,3 @@ def model_input_names(self):; symbols: model_input_names
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/__init__.py
@@ -8,16 +8,20 @@
-from vllm.transformers_utils.processors.bagel import BagelProcessor
-from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
-from vllm.transformers_utils.processors.fireredasr2_processor import (
-    FireRedASR2Processor,
-)
-from vllm.transformers_utils.processors.funasr_processor import FunASRProcessor
diff -- tests/models/registry.py
@@ -1020,18 +1020,15 @@ def check_available_online(
-        "Qwen/Qwen3-ASR-1.7B",
+        "Qwen/Qwen3-ASR-0.6B",
-        is_available_online=False,
-        "Qwen/Qwen3-ASR-1.7B",
+        "Qwen/Qwen3-ASR-0.6B",
-        enforce_eager=True,
diff -- vllm/transformers_utils/processors/deepseek_ocr.py
@@ -8,7 +8,7 @@
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/__init__.py` modified +28/-10; `vllm/transformers_utils/processors/deepseek_ocr.py` modified +1/-4; `vllm/transformers_utils/processors/deepseek_vl2.py` modified +1/-4; `vllm/transformers_utils/processors/ovis.py` modified +1/-4; `vllm/transformers_utils/processors/ovis2_5.py` modified +1/-4; `vllm/transformers_utils/processors/bagel.py` modified +0/-4
  - tests: `tests/models/registry.py` modified +2/-5
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36670 - [Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop

- Link: https://github.com/vllm-project/vllm/pull/36670
- Status/date: merged / 2026-03-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +135/-4, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop"; model line: DeepSeek OCR 2; category: bug fix; main diff: `tests/models/multimodal/processing/test_deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr.py`; technical summary: Covers "[Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop"; the main implementation surface is `tests/models/multimodal/processing/test_deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: processor, TestDeepseekOCREmptyImagesCrop, test_empty_images_crop_small_image, test_populated_images_crop_large_image, touching `processor, TestDeepseekOCREmptyImagesCrop, test_empty_images_crop_small_image`; `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4 (5 lines); hunks: -452,10 +452,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, touching `_parse_and_validate_image_input`.
- Code diff details:
  - `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: processor, TestDeepseekOCREmptyImagesCrop, test_empty_images_crop_small_image, test_populated_images_crop_large_image
  - `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4 (5 lines); hunks: -452,10 +452,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_deepseek_ocr.py
@@ -0,0 +1,134 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Regression test for DeepSeek-OCR TensorSchema validation with empty images_crop.
+When using the Gundam preset (BASE_SIZE=1024, IMAGE_SIZE=640, CROP_MODE=True),
+images that are small enough to not require cropping produce an empty
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -452,10 +452,7 @@ def _parse_and_validate_image_input(
-        if images_crop is not None and images_crop.numel() > 0:
-            image_size = images_crop.shape[-1]
-        else:
-            image_size = base_size
+        image_size = images_crop.shape[-1] if images_crop is not None else base_size
```

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_deepseek_ocr.py` added +134/-0
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +1/-4
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_deepseek_ocr.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37289 - [Bugfix] Standardize custom HF Processor init

- Link: https://github.com/vllm-project/vllm/pull/37289
- Status/date: merged / 2026-03-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +39/-33, 152 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Standardize custom HF Processor init"; model line: DeepSeek OCR 2; category: bug fix; main diff: `vllm/transformers_utils/processors/qwen_vl.py`, `vllm/model_executor/models/glm4v.py`, `vllm/model_executor/models/qwen_vl.py`; technical summary: Covers "[Bugfix] Standardize custom HF Processor init"; the main implementation surface is `vllm/transformers_utils/processors/qwen_vl.py`, `vllm/model_executor/models/glm4v.py`, `vllm/model_executor/models/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/qwen_vl.py` modified +5/-18 (23 lines); hunks: -31,25 +31,12 @@ class QwenVLProcessor(ProcessorMixin):; symbols: QwenVLProcessor, __init__, image_start_tag, image_end_tag, touching `QwenVLProcessor, __init__, image_start_tag`; `vllm/model_executor/models/glm4v.py` modified +11/-3 (14 lines); hunks: -47,7 +47,10; -387,15 +390,20 @@ class GLM4VProcessingInfo(BaseProcessingInfo):; symbols: GLM4VProcessingInfo, get_hf_config, get_hf_processor, get_image_processor, touching `GLM4VProcessingInfo, get_hf_config, get_hf_processor`; `vllm/model_executor/models/qwen_vl.py` modified +11/-3 (14 lines); hunks: -44,7 +44,10; -432,15 +435,20 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, QwenVLProcessingInfo, get_hf_processor, get_image_processor, touching `__init__, QwenVLProcessingInfo, get_hf_processor`; `vllm/transformers_utils/processors/glm4v.py` modified +2/-7 (9 lines); hunks: -29,13 +29,8 @@ class GLM4VProcessor(ProcessorMixin):; symbols: GLM4VProcessor, __init__, touching `GLM4VProcessor, __init__`.
- Code diff details:
  - `vllm/transformers_utils/processors/qwen_vl.py` modified +5/-18 (23 lines); hunks: -31,25 +31,12 @@ class QwenVLProcessor(ProcessorMixin):; symbols: QwenVLProcessor, __init__, image_start_tag, image_end_tag
  - `vllm/model_executor/models/glm4v.py` modified +11/-3 (14 lines); hunks: -47,7 +47,10; -387,15 +390,20 @@ class GLM4VProcessingInfo(BaseProcessingInfo):; symbols: GLM4VProcessingInfo, get_hf_config, get_hf_processor, get_image_processor
  - `vllm/model_executor/models/qwen_vl.py` modified +11/-3 (14 lines); hunks: -44,7 +44,10; -432,15 +435,20 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, QwenVLProcessingInfo, get_hf_processor, get_image_processor
  - `vllm/transformers_utils/processors/glm4v.py` modified +2/-7 (9 lines); hunks: -29,13 +29,8 @@ class GLM4VProcessor(ProcessorMixin):; symbols: GLM4VProcessor, __init__
  - `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1 (4 lines); hunks: -196,8 +196,10 @@ def get_hf_processor(self, **kwargs: object):; symbols: get_hf_processor, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/qwen_vl.py
@@ -31,25 +31,12 @@ class QwenVLProcessor(ProcessorMixin):
+        image_processor: QwenVLImageProcessorFast,
-        image_size: int,
-        image_processor: QwenVLImageProcessorFast | None = None,
-        self.tokenizer = tokenizer
-        if image_processor is None:
-            image_processor = QwenVLImageProcessorFast(
diff -- vllm/model_executor/models/glm4v.py
@@ -47,7 +47,10 @@
-from vllm.transformers_utils.processors.glm4v import GLM4VProcessor
+from vllm.transformers_utils.processors.glm4v import (
+    GLM4VImageProcessorFast,
+    GLM4VProcessor,
+)
@@ -387,15 +390,20 @@ class GLM4VProcessingInfo(BaseProcessingInfo):
diff -- vllm/model_executor/models/qwen_vl.py
@@ -44,7 +44,10 @@
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/qwen_vl.py` modified +5/-18; `vllm/model_executor/models/glm4v.py` modified +11/-3; `vllm/model_executor/models/qwen_vl.py` modified +11/-3; `vllm/transformers_utils/processors/glm4v.py` modified +2/-7; `vllm/model_executor/models/deepseek_ocr.py` modified +3/-1; `vllm/model_executor/models/deepseek_ocr2.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_ocr.py`, `vllm/model_executor/models/deepseek_ocr2.py`, `vllm/model_executor/models/glm4v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35182 - [Misc] Reorganize inputs

- Link: https://github.com/vllm-project/vllm/pull/35182
- Status/date: merged / 2026-03-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 142 files, +1212/-1342, 6002 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Reorganize inputs"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`; technical summary: Covers "[Misc] Reorganize inputs"; the main implementation surface is `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/multimodal/inputs.py` modified +2/-162 (164 lines); hunks: -15,12 +15,11; -32,14 +31,9; symbols: VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins, PlaceholderRange, touching `VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins`; `vllm/entrypoints/pooling/score/serving.py` modified +36/-45 (81 lines); hunks: -35,7 +35,7; -110,12 +110,12 @@ async def _embedding_score(; symbols: _embedding_score, _preprocess_late_interaction_item, touching `_embedding_score, _preprocess_late_interaction_item`; `vllm/entrypoints/serve/render/serving.py` modified +38/-37 (75 lines); hunks: -34,9 +34,15; -127,22 +133,22 @@ async def render_chat_request(; symbols: render_chat_request, render_chat, touching `render_chat_request, render_chat`; `vllm/entrypoints/openai/responses/serving.py` modified +22/-26 (48 lines); hunks: -110,7 +110,7; -269,10 +269,10 @@ def __init__(; symbols: __init__, _validate_generator_input, create_responses, touching `__init__, _validate_generator_input, create_responses`.
- Code diff details:
  - `vllm/multimodal/inputs.py` modified +2/-162 (164 lines); hunks: -15,12 +15,11; -32,14 +31,9; symbols: VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins, PlaceholderRange
  - `vllm/entrypoints/pooling/score/serving.py` modified +36/-45 (81 lines); hunks: -35,7 +35,7; -110,12 +110,12 @@ async def _embedding_score(; symbols: _embedding_score, _preprocess_late_interaction_item
  - `vllm/entrypoints/serve/render/serving.py` modified +38/-37 (75 lines); hunks: -34,9 +34,15; -127,22 +133,22 @@ async def render_chat_request(; symbols: render_chat_request, render_chat
  - `vllm/entrypoints/openai/responses/serving.py` modified +22/-26 (48 lines); hunks: -110,7 +110,7; -269,10 +269,10 @@ def __init__(; symbols: __init__, _validate_generator_input, create_responses
  - `vllm/entrypoints/llm.py` modified +22/-22 (44 lines); hunks: -57,9 +57,9; -584,7 +584,7 @@ def wait_for_completion(; symbols: wait_for_completion, _resolve_mm_lora, beam_search
- Key code excerpts:

```diff
diff -- vllm/multimodal/inputs.py
@@ -15,12 +15,11 @@
-    final,
-from typing_extensions import NotRequired, TypeVar
+from typing_extensions import TypeVar
@@ -32,14 +31,9 @@
-    from vllm.inputs.data import _InputOptions
-    _InputOptions = dict
diff -- vllm/entrypoints/pooling/score/serving.py
@@ -35,7 +35,7 @@
-from vllm.inputs.data import ProcessorInputs, TokensPrompt, token_inputs
+from vllm.inputs import EngineInput, TokensPrompt, tokens_input
@@ -110,12 +110,12 @@ async def _embedding_score(
-        engine_prompts: list[ProcessorInputs] = []
+        engine_inputs: list[EngineInput] = []
-            engine_prompts.append(
diff -- vllm/entrypoints/serve/render/serving.py
@@ -34,9 +34,15 @@
```

- Reviewed files:
  - runtime: `vllm/multimodal/inputs.py` modified +2/-162; `vllm/entrypoints/pooling/score/serving.py` modified +36/-45; `vllm/entrypoints/serve/render/serving.py` modified +38/-37; `vllm/entrypoints/openai/responses/serving.py` modified +22/-26; `vllm/entrypoints/llm.py` modified +22/-22; `vllm/entrypoints/pooling/embed/io_processor.py` modified +20/-20
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/chat_completion/test_chat_error.py`, `tests/entrypoints/openai/chat_completion/test_serving_chat.py`, `tests/entrypoints/openai/responses/test_serving_responses.py`, `tests/entrypoints/serve/render/test_launch_render.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36464 - [Examples] Resettle generate examples.

- Link: https://github.com/vllm-project/vllm/pull/36464
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 36 files, +46/-50, 267 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Examples] Resettle generate examples."; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs/features/multimodal_inputs.md`, `examples/generate/multimodal/qwen2_5_omni/README.md`, `docs/features/reasoning_outputs.md`; technical summary: Covers "[Examples] Resettle generate examples."; the main implementation surface is `docs/features/multimodal_inputs.md`, `examples/generate/multimodal/qwen2_5_omni/README.md`, `docs/features/reasoning_outputs.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/features/multimodal_inputs.md` modified +7/-7 (14 lines); hunks: -68,7 +68,7 @@ You can pass a single image to the `'image'` field of the mult...; -101,7 +101,7 @@ To substitute multiple images inside the same text prompt, y...; `examples/generate/multimodal/qwen2_5_omni/README.md` renamed +6/-6 (12 lines); hunks: -6,15 +6,15 @@ This folder provides several example scripts on how to inferen...; -24,16 +24,16 @@ You can also test Qwen2.5-Omni on a single modality:; `docs/features/reasoning_outputs.md` modified +1/-1 (2 lines); hunks: -202,7 +202,7 @@ The reasoning content is also available when both tool calli...; `examples/generate/multimodal/vision_language_offline.py` renamed +1/-1 (2 lines); hunks: -1402,7 +1402,7 @@ def run_mantis(questions: list[str], modality: str) -> Mod...; symbols: run_mantis, run_minicpmv_base, touching `run_mantis, run_minicpmv_base`.
- Code diff details:
  - `docs/features/multimodal_inputs.md` modified +7/-7 (14 lines); hunks: -68,7 +68,7 @@ You can pass a single image to the `'image'` field of the mult...; -101,7 +101,7 @@ To substitute multiple images inside the same text prompt, y...
  - `examples/generate/multimodal/qwen2_5_omni/README.md` renamed +6/-6 (12 lines); hunks: -6,15 +6,15 @@ This folder provides several example scripts on how to inferen...; -24,16 +24,16 @@ You can also test Qwen2.5-Omni on a single modality:
  - `docs/features/reasoning_outputs.md` modified +1/-1 (2 lines); hunks: -202,7 +202,7 @@ The reasoning content is also available when both tool calli...
  - `examples/generate/multimodal/vision_language_offline.py` renamed +1/-1 (2 lines); hunks: -1402,7 +1402,7 @@ def run_mantis(questions: list[str], modality: str) -> Mod...; symbols: run_mantis, run_minicpmv_base
  - `examples/generate/multimodal/audio_language_offline.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- docs/features/multimodal_inputs.md
@@ -68,7 +68,7 @@ You can pass a single image to the `'image'` field of the multi-modal dictionary
-Full example: [examples/offline_inference/vision_language.py](../../examples/offline_inference/vision_language.py)
+Full example: [examples/generate/multimodal/vision_language_offline.py](../../examples/generate/multimodal/vision_language_offline.py)
@@ -101,7 +101,7 @@ To substitute multiple images inside the same text prompt, you can pass in a lis
-Full example: [examples/offline_inference/vision_language_multi_image.py](../../examples/offline_inference/vision_language_multi_image.py)
+Full example: [examples/generate/multimodal/vision_language_multi_image_offline.py](../../examples/generate/multimodal/vision_language_multi_image_offline.py)
@@ -287,13 +287,13 @@ Instead of NumPy arrays, you can also pass `'torch.Tensor'` instances, as shown
diff -- examples/generate/multimodal/qwen2_5_omni/README.md
@@ -6,15 +6,15 @@ This folder provides several example scripts on how to inference Qwen2.5-Omni of
-python examples/offline_inference/qwen2_5_omni/only_thinker.py \
+python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
-python examples/offline_inference/qwen2_5_omni/only_thinker.py \
+python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
-python examples/offline_inference/qwen2_5_omni/only_thinker.py \
+python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
diff -- docs/features/reasoning_outputs.md
@@ -202,7 +202,7 @@ The reasoning content is also available when both tool calling and the reasoning
```

- Reviewed files:
  - docs: `docs/features/multimodal_inputs.md` modified +7/-7; `examples/generate/multimodal/qwen2_5_omni/README.md` renamed +6/-6; `docs/features/reasoning_outputs.md` modified +1/-1; `examples/generate/multimodal/vision_language_offline.py` renamed +1/-1; `examples/generate/multimodal/audio_language_offline.py` renamed +0/-0; `examples/generate/multimodal/encoder_decoder_multimodal_offline.py` renamed +0/-0
- Risk and verification: This is mostly docs/examples in `docs/features/multimodal_inputs.md`, `docs/features/reasoning_outputs.md`, `docs/serving/openai_compatible_server.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #40830 - [MM][CG] Support ViT CG for Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/40830
- Status/date: merged / 2026-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +539/-22, 669 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Support ViT CG for Qwen2.5-VL"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; technical summary: Covers "[MM][CG] Support ViT CG for Qwen2.5-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21 (450 lines); hunks: -85,11 +85,13; -771,22 +773,54 @@ def invert_permutation(perm: torch.Tensor) -> torch.Tensor:; symbols: invert_permutation, forward, prepare_encoder_metadata, touching `invert_permutation, forward, prepare_encoder_metadata`; `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0 (95 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: qwen2_5_vl_chat_template, _window_attention_regression_image, _encoder_cudagraph_config, test_qwen2_5_vl_evs_batched_videos, touching `qwen2_5_vl_chat_template, _window_attention_regression_image, _encoder_cudagraph_config`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1 (13 lines); hunks: -54,7 +54,18 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, touching `qwen_vl_chat_template`; `docs/design/cuda_graphs_multimodal.md` modified +2/-0 (2 lines); hunks: -86,9 +86,11 @@ Models opt-in to encoder CUDA Graphs by implementing the [Sup....
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21 (450 lines); hunks: -85,11 +85,13; -771,22 +773,54 @@ def invert_permutation(perm: torch.Tensor) -> torch.Tensor:; symbols: invert_permutation, forward, prepare_encoder_metadata
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0 (95 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: qwen2_5_vl_chat_template, _window_attention_regression_image, _encoder_cudagraph_config, test_qwen2_5_vl_evs_batched_videos
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1 (13 lines); hunks: -54,7 +54,18 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0 (2 lines); hunks: -86,9 +86,11 @@ Models opt-in to encoder CUDA Graphs by implementing the [Sup...
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2466,6 +2466,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -85,11 +85,13 @@
+from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers
+    SupportsEncoderCudaGraph,
@@ -771,22 +773,54 @@ def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
-    def forward(
+    def prepare_encoder_metadata(
-        x: torch.Tensor,
diff -- tests/models/multimodal/generation/test_qwen2_5_vl.py
@@ -3,6 +3,7 @@
+from vllm.assets.image import ImageAsset
@@ -11,6 +12,7 @@
+IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
@@ -28,6 +30,25 @@ def qwen2_5_vl_chat_template(*query):
+WINDOW_ATTN_IMAGE_PROMPT = qwen2_5_vl_chat_template(
+    IMAGE_PLACEHOLDER,
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -54,7 +54,18 @@ def qwen_vl_chat_template(content: str) -> str:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21
  - tests: `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-1
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +2/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_qwen2_5_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42151 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/42151
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +112/-5, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5"; the main implementation surface is `examples/generate/multimodal/vision_language_offline.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/generate/multimodal/vision_language_offline.py` modified +93/-1 (94 lines); hunks: -2179,6 +2179,92 @@ def run_qwen3_vl_moe(questions: list[str], modality: str)...; -2442,6 +2528,8 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_qwen3_vl_moe, run_qwen3_5, run_qwen3_5_moe, run_r_vl, touching `run_qwen3_vl_moe, run_qwen3_5, run_qwen3_5_moe`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3 (18 lines); hunks: -42,6 +42,18 @@ def qwen_vl_chat_template(content: str) -> str:; -54,16 +66,16 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, touching `qwen_vl_chat_template`; `docs/design/cuda_graphs_multimodal.md` modified +2/-1 (3 lines); hunks: -85,8 +85,9 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...; `vllm/model_executor/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `examples/generate/multimodal/vision_language_offline.py` modified +93/-1 (94 lines); hunks: -2179,6 +2179,92 @@ def run_qwen3_vl_moe(questions: list[str], modality: str)...; -2442,6 +2528,8 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_qwen3_vl_moe, run_qwen3_5, run_qwen3_5_moe, run_r_vl
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3 (18 lines); hunks: -42,6 +42,18 @@ def qwen_vl_chat_template(content: str) -> str:; -54,16 +66,16 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1 (3 lines); hunks: -85,8 +85,9 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- examples/generate/multimodal/vision_language_offline.py
@@ -2179,6 +2179,92 @@ def run_qwen3_vl_moe(questions: list[str], modality: str) -> ModelRequestData:
+# Qwen3.5-Dense
+def run_qwen3_5(questions: list[str], modality: str) -> ModelRequestData:
+    model_name = "Qwen/Qwen3.5-4B"
+    mm_limit = {"image": 1, "video": 1} if modality == "image+video" else {modality: 1}
+    engine_args = EngineArgs(
+        model=model_name,
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -42,6 +42,18 @@ def qwen_vl_chat_template(content: str) -> str:
+    "qwen2_5_vl": VitCudagraphTestConfig(
+        model="Qwen/Qwen2.5-VL-3B-Instruct",
+        image_prompt=qwen_vl_chat_template(
+            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
+        ),
+        video_prompt=qwen_vl_chat_template(
diff -- docs/design/cuda_graphs_multimodal.md
@@ -85,8 +85,9 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
```

- Reviewed files:
  - docs: `examples/generate/multimodal/vision_language_offline.py` modified +93/-1; `docs/design/cuda_graphs_multimodal.md` modified +2/-1
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-3
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41736 - [MM][CG] Support ViT CG for Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/41736
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +315/-21, 415 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Support ViT CG for Qwen2-VL"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`; technical summary: Covers "[MM][CG] Support ViT CG for Qwen2-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +300/-20 (320 lines); hunks: -89,9 +89,11; -646,38 +648,84 @@ def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tenso...; symbols: compute_attn_mask_seqlen, prepare_encoder_metadata, forward, _get_mm_fields_config, touching `compute_attn_mask_seqlen, prepare_encoder_metadata, forward`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0 (12 lines); hunks: -78,6 +78,18 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, touching `qwen_vl_chat_template`; `docs/design/cuda_graphs_multimodal.md` modified +2/-1 (3 lines); hunks: -85,13 +85,14 @@ Models opt-in to encoder CUDA Graphs by implementing the [Su...; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2557,6 +2557,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2, touching `run_tarsier2`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +300/-20 (320 lines); hunks: -89,9 +89,11; -646,38 +648,84 @@ def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tenso...; symbols: compute_attn_mask_seqlen, prepare_encoder_metadata, forward, _get_mm_fields_config
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0 (12 lines); hunks: -78,6 +78,18 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-1 (3 lines); hunks: -85,13 +85,14 @@ Models opt-in to encoder CUDA Graphs by implementing the [Su...
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2557,6 +2557,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -89,9 +89,11 @@
+from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers
+    SupportsEncoderCudaGraph,
@@ -646,38 +648,84 @@ def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int | None:
+    def prepare_encoder_metadata(
+        self,
+        grid_thw: list[list[int]],
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -78,6 +78,18 @@ def qwen_vl_chat_template(content: str) -> str:
+    "qwen2_vl": VitCudagraphTestConfig(
+        model="Qwen/Qwen2-VL-2B-Instruct",
+        image_prompt=qwen_vl_chat_template(
+            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
+        ),
+        video_prompt=qwen_vl_chat_template(
diff -- docs/design/cuda_graphs_multimodal.md
@@ -85,13 +85,14 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +300/-20
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +2/-1; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42224 - [MM][CG] Enable encoder Cudagraph for Step3VL

- Link: https://github.com/vllm-project/vllm/pull/42224
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +384/-22, 534 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Enable encoder Cudagraph for Step3VL"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py`; technical summary: Covers "[MM][CG] Enable encoder Cudagraph for Step3VL"; the main implementation surface is `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #41759 - [MM][Perf][CG] Support ViT full CUDA graph for InternVL

- Link: https://github.com/vllm-project/vllm/pull/41759
- Status/date: merged / 2026-06-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +183/-2, 238 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for InternVL"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for InternVL"; the main implementation surface is `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/internvl.py` modified +166/-2 (168 lines); hunks: -10,7 +10,7; -55,6 +55,7; symbols: _get_prompt_updates, InternVLChatModel, get_num_mm_connector_tokens, get_encoder_cudagraph_config, touching `_get_prompt_updates, InternVLChatModel, get_num_mm_connector_tokens`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-0 (15 lines); hunks: -43,6 +43,10 @@ def qwen_vl_chat_template(content: str) -> str:; -51,6 +55,17 @@ def step3_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, internvl_chat_template, step3_vl_chat_template, touching `qwen_vl_chat_template, internvl_chat_template, step3_vl_chat_template`; `docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -82,6 +82,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2554,6 +2554,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2, touching `run_tarsier2`.
- Code diff details:
  - `vllm/model_executor/models/internvl.py` modified +166/-2 (168 lines); hunks: -10,7 +10,7; -55,6 +55,7; symbols: _get_prompt_updates, InternVLChatModel, get_num_mm_connector_tokens, get_encoder_cudagraph_config
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-0 (15 lines); hunks: -43,6 +43,10 @@ def qwen_vl_chat_template(content: str) -> str:; -51,6 +55,17 @@ def step3_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, internvl_chat_template, step3_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -82,6 +82,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2554,6 +2554,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -10,7 +10,7 @@
-from typing import Annotated, Literal, TypeAlias, TypeVar
+from typing import Annotated, Any, Literal, TypeAlias, TypeVar
@@ -55,6 +55,7 @@
+    SupportsEncoderCudaGraph,
@@ -543,7 +544,13 @@ def _get_prompt_updates(
-class InternVLChatModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -43,6 +43,10 @@ def qwen_vl_chat_template(content: str) -> str:
+def internvl_chat_template(content: str) -> str:
+    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
@@ -51,6 +55,17 @@ def step3_vl_chat_template(content: str) -> str:
+    "internvl": VitCudagraphTestConfig(
+        model="OpenGVLab/InternVL3-1B",
+        num_video_frames=8,
diff -- docs/design/cuda_graphs_multimodal.md
@@ -82,6 +82,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internvl.py` modified +166/-2
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +15/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +1/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40576 - [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference

- Link: https://github.com/vllm-project/vllm/pull/40576
- Status/date: merged / 2026-06-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +480/-25, 605 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/glm4_1v.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference"; the main implementation surface is `vllm/model_executor/models/glm4_1v.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/glm4_1v.py` modified +456/-25 (481 lines); hunks: -97,10 +97,12; -626,6 +628,11 @@ def __init__(; symbols: __init__, device, rot_pos_emb, compute_attn_mask_seqlen, touching `__init__, device, rot_pos_emb`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +22/-0 (22 lines); hunks: -137,6 +137,28 @@ def step3_vl_chat_template(content: str) -> str:; symbols: step3_vl_chat_template, touching `step3_vl_chat_template`; `docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2562,6 +2562,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2, touching `run_tarsier2`.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` modified +456/-25 (481 lines); hunks: -97,10 +97,12; -626,6 +628,11 @@ def __init__(; symbols: __init__, device, rot_pos_emb, compute_attn_mask_seqlen
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +22/-0 (22 lines); hunks: -137,6 +137,28 @@ def step3_vl_chat_template(content: str) -> str:; symbols: step3_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2562,6 +2562,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm4_1v.py
@@ -97,10 +97,12 @@
+from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers
+    SupportsEncoderCudaGraph,
@@ -626,6 +628,11 @@ def __init__(
+        use_data_parallel = is_vit_use_data_parallel()
+        self.tp_size = (
+            1 if use_data_parallel else get_tensor_model_parallel_world_size()
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -137,6 +137,28 @@ def step3_vl_chat_template(content: str) -> str:
+    "glm4_1v": VitCudagraphTestConfig(
+        model="zai-org/GLM-4.1V-9B-Thinking",
+        image_prompt=(
+            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
+            "<|begin_of_image|><|image|><|end_of_image|>"
+            "What is in this image?<|assistant|>assistant\n"
diff -- docs/design/cuda_graphs_multimodal.md
@@ -88,6 +88,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +456/-25
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +22/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +1/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45131 - Deprecated 1st generation Qwen and QwenVL models

- Link: https://github.com/vllm-project/vllm/pull/45131
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +6/-1349, 1585 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecated 1st generation Qwen and QwenVL models"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `vllm/model_executor/models/qwen_vl.py`, `vllm/model_executor/models/qwen.py`, `vllm/tokenizers/qwen_vl.py`; technical summary: Covers "Deprecated 1st generation Qwen and QwenVL models"; the main implementation surface is `vllm/model_executor/models/qwen_vl.py`, `vllm/model_executor/models/qwen.py`, `vllm/tokenizers/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen_vl.py` removed +0/-688 (688 lines); hunks: -1,688 +0,0; symbols: QwenImagePixelInputs, QwenImageEmbeddingInputs, VisualAttention, __init__, touching `QwenImagePixelInputs, QwenImageEmbeddingInputs, VisualAttention`; `vllm/model_executor/models/qwen.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: QWenMLP, __init__, forward, QWenAttention, touching `QWenMLP, __init__, forward`; `vllm/tokenizers/qwen_vl.py` removed +0/-71 (71 lines); hunks: -1,71 +0,0; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode, touching `get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize`; `examples/generate/multimodal/vision_language_multi_image_offline.py` modified +0/-44 (44 lines); hunks: -1042,49 +1042,6 @@ def load_phi4siglip(question: str, image_urls: list[str])...; -1544,7 +1501,6 @@ def load_molmo2(question: str, image_urls: list[str]) -> M...; symbols: load_phi4siglip, load_qwen_vl_chat, load_qwen2_vl, load_molmo2, touching `load_phi4siglip, load_qwen_vl_chat, load_qwen2_vl`.
- Code diff details:
  - `vllm/model_executor/models/qwen_vl.py` removed +0/-688 (688 lines); hunks: -1,688 +0,0; symbols: QwenImagePixelInputs, QwenImageEmbeddingInputs, VisualAttention, __init__
  - `vllm/model_executor/models/qwen.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: QWenMLP, __init__, forward, QWenAttention
  - `vllm/tokenizers/qwen_vl.py` removed +0/-71 (71 lines); hunks: -1,71 +0,0; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode
  - `examples/generate/multimodal/vision_language_multi_image_offline.py` modified +0/-44 (44 lines); hunks: -1042,49 +1042,6 @@ def load_phi4siglip(question: str, image_urls: list[str])...; -1544,7 +1501,6 @@ def load_molmo2(question: str, image_urls: list[str]) -> M...; symbols: load_phi4siglip, load_qwen_vl_chat, load_qwen2_vl, load_molmo2
  - `vllm/transformers_utils/processors/qwen_vl.py` removed +0/-42 (42 lines); hunks: -1,42 +0,0; symbols: QwenVLImageProcessorFast, QwenVLProcessor, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen_vl.py
@@ -1,688 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Adapted from
-# https://huggingface.co/Qwen/Qwen-VL/blob/main/modeling_qwen.py
-# Copyright (c) Alibaba Cloud.
-"""Inference-only Qwen-VL model compatible with HuggingFace weights."""
diff -- vllm/model_executor/models/qwen.py
@@ -1,377 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Adapted from
-# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
-# Copyright (c) Alibaba Cloud.
-# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
diff -- vllm/tokenizers/qwen_vl.py
@@ -1,71 +0,0 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen_vl.py` removed +0/-688; `vllm/model_executor/models/qwen.py` removed +0/-377; `vllm/tokenizers/qwen_vl.py` removed +0/-71; `vllm/transformers_utils/processors/qwen_vl.py` removed +0/-42
  - docs: `examples/generate/multimodal/vision_language_multi_image_offline.py` modified +0/-44; `examples/generate/multimodal/vision_language_offline.py` modified +0/-22
  - tests: `tests/models/registry.py` modified +0/-18; `tests/tokenizers_/conftest.py` removed +0/-14
- Risk and verification: The diff ships test coverage in `tests/distributed/test_pipeline_parallel.py`, `tests/models/multimodal/conftest.py`, `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40660 - [MM][Perf][CG] Support ViT full cudagraphs for mllama4

- Link: https://github.com/vllm-project/vllm/pull/40660
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +193/-14, 291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full cudagraphs for mllama4"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/mllama4.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`; technical summary: Covers "[MM][Perf][CG] Support ViT full cudagraphs for mllama4"; the main implementation surface is `vllm/model_executor/models/mllama4.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +160/-12 (172 lines); hunks: -19,7 +19,7; -78,6 +78,7; symbols: Llama4ImagePatchInputs, Llama4ForConditionalGeneration, update_physical_experts_metadata, get_image_patches_per_chunk, touching `Llama4ImagePatchInputs, Llama4ForConditionalGeneration, update_physical_experts_metadata`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +20/-0 (20 lines); hunks: -55,6 +55,26 @@ def step3_vl_chat_template(content: str) -> str:; symbols: step3_vl_chat_template, touching `step3_vl_chat_template`; `docs/design/cuda_graphs_multimodal.md` modified +9/-0 (9 lines); hunks: -82,6 +82,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...; -114,6 +115,14 @@ vllm serve Qwen/Qwen3-VL-32B \; `tests/models/utils.py` modified +3/-2 (5 lines); hunks: -506,12 +506,13 @@ class DummyConfig:; symbols: DummyConfig, touching `DummyConfig`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +160/-12 (172 lines); hunks: -19,7 +19,7; -78,6 +78,7; symbols: Llama4ImagePatchInputs, Llama4ForConditionalGeneration, update_physical_experts_metadata, get_image_patches_per_chunk
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +20/-0 (20 lines); hunks: -55,6 +55,26 @@ def step3_vl_chat_template(content: str) -> str:; symbols: step3_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +9/-0 (9 lines); hunks: -82,6 +82,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...; -114,6 +115,14 @@ vllm serve Qwen/Qwen3-VL-32B \
  - `tests/models/utils.py` modified +3/-2 (5 lines); hunks: -506,12 +506,13 @@ class DummyConfig:; symbols: DummyConfig
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2532,6 +2532,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -19,7 +19,7 @@
-from typing import Annotated, Literal
+from typing import Annotated, Any, Literal
@@ -78,6 +78,7 @@
+    SupportsEncoderCudaGraph,
@@ -105,7 +106,7 @@ class Llama4ImagePatchInputs(TensorSchema):
-    The number of total patches for each image in the batch.
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -55,6 +55,26 @@ def step3_vl_chat_template(content: str) -> str:
+    "llama4": VitCudagraphTestConfig(
+        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
+        modalities=["image"],
+        image_prompt=(
+            "<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
+            "<|image|>What is in this image?<|eot|>"
diff -- docs/design/cuda_graphs_multimodal.md
@@ -82,6 +82,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +160/-12
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +20/-0; `tests/models/utils.py` modified +3/-2
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +9/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/models/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43586 - [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR

- Link: https://github.com/vllm-project/vllm/pull/43586
- Status/date: merged / 2026-06-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +809/-69, 1559 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; technical summary: Covers "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #41992 - [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL

- Link: https://github.com/vllm-project/vllm/pull/41992
- Status/date: merged / 2026-06-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +498/-39, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL"; the main implementation surface is `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/moonvit.py` modified +266/-37 (303 lines); hunks: -45,7 +45,9; -110,23 +112,42 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds, touching `__init__, reset_parameters, forward`; `vllm/model_executor/models/kimi_vl.py` modified +195/-2 (197 lines); hunks: -56,7 +56,11; -79,6 +83,7; symbols: get_replacement, KimiVLForConditionalGeneration, __init__, get_encoder_cudagraph_config, touching `get_replacement, KimiVLForConditionalGeneration, __init__`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +35/-0 (35 lines); hunks: -48,6 +48,13 @@ def internvl_chat_template(content: str) -> str:; -100,6 +107,34 @@ def step3_vl_chat_template(content: str) -> str:; symbols: internvl_chat_template, kimi_vl_chat_template, step3_vl_chat_template, touching `internvl_chat_template, kimi_vl_chat_template, step3_vl_chat_template`; `docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -129,6 +129,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Su....
- Code diff details:
  - `vllm/model_executor/models/moonvit.py` modified +266/-37 (303 lines); hunks: -45,7 +45,9; -110,23 +112,42 @@ def __init__(; symbols: __init__, reset_parameters, forward, get_pos_embeds
  - `vllm/model_executor/models/kimi_vl.py` modified +195/-2 (197 lines); hunks: -56,7 +56,11; -79,6 +83,7; symbols: get_replacement, KimiVLForConditionalGeneration, __init__, get_encoder_cudagraph_config
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +35/-0 (35 lines); hunks: -48,6 +48,13 @@ def internvl_chat_template(content: str) -> str:; -100,6 +107,34 @@ def step3_vl_chat_template(content: str) -> str:; symbols: internvl_chat_template, kimi_vl_chat_template, step3_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -129,6 +129,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Su...
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2537,6 +2537,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -45,7 +45,9 @@
+from typing import Any
+import numpy as np
@@ -110,23 +112,42 @@ def __init__(
-    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
-        pos_embs = []
-        for shape in grid_hws.tolist():
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,7 +56,11 @@
-from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
+from vllm.model_executor.models.interfaces import (
+    SupportsEncoderCudaGraph,
+    SupportsMultiModal,
+    SupportsPP,
+)
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -48,6 +48,13 @@ def internvl_chat_template(content: str) -> str:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +266/-37; `vllm/model_executor/models/kimi_vl.py` modified +195/-2
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +35/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +1/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45993 - [Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/45993
- Status/date: merged / 2026-06-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +10/-3881, 4048 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `tests/tool_parsers/test_minimax_tool_parser.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/tool_parsers/minimax_tool_parser.py`; technical summary: Covers "[Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM"; the main implementation surface is `tests/tool_parsers/test_minimax_tool_parser.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/tool_parsers/minimax_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_minimax_tool_parser.py` removed +0/-1227 (1227 lines); hunks: -1,1227 +0,0; symbols: minimax_tokenizer, minimax_tool_parser, sample_tools, assert_tool_calls, touching `minimax_tokenizer, minimax_tool_parser, sample_tools`; `vllm/model_executor/models/minimax_text_01.py` removed +0/-1000 (1000 lines); hunks: -1,1000 +0,0; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func, touching `replace_weight_name, weight_loader_with_alias, wrapper`; `vllm/tool_parsers/minimax_tool_parser.py` removed +0/-852 (852 lines); hunks: -1,852 +0,0; symbols: MinimaxToolParser, __init__, preprocess_model_output, remove_tool_calls_from_think, touching `MinimaxToolParser, __init__, preprocess_model_output`; `vllm/model_executor/models/minimax_vl_01.py` removed +0/-385 (385 lines); hunks: -1,385 +0,0; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, MiniMaxVL01MultiModalProjector, __init__, touching `MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, MiniMaxVL01MultiModalProjector`.
- Code diff details:
  - `tests/tool_parsers/test_minimax_tool_parser.py` removed +0/-1227 (1227 lines); hunks: -1,1227 +0,0; symbols: minimax_tokenizer, minimax_tool_parser, sample_tools, assert_tool_calls
  - `vllm/model_executor/models/minimax_text_01.py` removed +0/-1000 (1000 lines); hunks: -1,1000 +0,0; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func
  - `vllm/tool_parsers/minimax_tool_parser.py` removed +0/-852 (852 lines); hunks: -1,852 +0,0; symbols: MinimaxToolParser, __init__, preprocess_model_output, remove_tool_calls_from_think
  - `vllm/model_executor/models/minimax_vl_01.py` removed +0/-385 (385 lines); hunks: -1,385 +0,0; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, MiniMaxVL01MultiModalProjector, __init__
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` removed +0/-113 (113 lines); hunks: -1,113 +0,0; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_minimax_tool_parser.py
@@ -1,1227 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# ruff: noqa: E501
-import json
-from typing import Any
-import pytest
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,1000 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""Inference-only MiniMaxText01 model."""
-from collections.abc import Iterable
-from itertools import islice
-from typing import TYPE_CHECKING
diff -- vllm/tool_parsers/minimax_tool_parser.py
@@ -1,852 +0,0 @@
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_minimax_tool_parser.py` removed +0/-1227; `tests/models/multimodal/processing/test_minimax_vl_01.py` removed +0/-113; `tests/models/multimodal/generation/test_common.py` modified +0/-23; `tests/models/multimodal/generation/vlm_utils/model_utils.py` modified +0/-18
  - runtime: `vllm/model_executor/models/minimax_text_01.py` removed +0/-1000; `vllm/tool_parsers/minimax_tool_parser.py` removed +0/-852; `vllm/model_executor/models/minimax_vl_01.py` removed +0/-385
  - docs: `examples/generate/multimodal/vision_language_offline.py` modified +0/-34
- Risk and verification: The diff ships test coverage in `rust/src/chat/tests/templates/vllm_examples/tool_chat_template_minimax_m1.jinja`, `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
