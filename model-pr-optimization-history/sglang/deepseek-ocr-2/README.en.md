# sglang DeepSeek OCR 2 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/deepseek_ocr.md` | [#17897](https://github.com/sgl-project/sglang/pull/17897) |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/deepseek-ocr-v2-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/configs/deepseek_ocr.py` | [#17897](https://github.com/sgl-project/sglang/pull/17897) |
| `python/sglang/srt/models/deepseek_ocr.py` | [#17897](https://github.com/sgl-project/sglang/pull/17897), [#19732](https://github.com/sgl-project/sglang/pull/19732) |
| `python/sglang/srt/multimodal/processors/deepseek_ocr.py` | [#17897](https://github.com/sgl-project/sglang/pull/17897) |
| `test/registered/xpu/test_deepseek_ocr.py` | no direct PR-number commit |
| `test/registered/xpu/test_deepseek_ocr_triton.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 29
- Total PRs in this document: 31
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-10-23 | [#11891](https://github.com/sgl-project/sglang/pull/11891) | merged | model: support deepseek-ocr | `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/deepseek_ocr.py` |
| 2025-10-31 | [#12384](https://github.com/sgl-project/sglang/pull/12384) | merged | [Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p… | `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py` |
| 2025-10-31 | [#12415](https://github.com/sgl-project/sglang/pull/12415) | merged | Feat: deepseek-ocr logits processor | `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py` |
| 2025-10-31 | [#12470](https://github.com/sgl-project/sglang/pull/12470) | merged | Fix lint in deepseek-ocr | `python/sglang/srt/configs/deepseek_ocr.py` |
| 2025-11-04 | [#12619](https://github.com/sgl-project/sglang/pull/12619) | open | [NPU] supports ds-ocr model on ascend | `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-01-30 | [#17897](https://github.com/sgl-project/sglang/pull/17897) | merged | Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833 | `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py` |
| 2026-02-05 | [#13561](https://github.com/sgl-project/sglang/pull/13561) | merged | [XPU] Integrate MoE and minor improvements in XPU attention backend | `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py` |
| 2026-02-15 | [#18860](https://github.com/sgl-project/sglang/pull/18860) | merged | update pre-commit config | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` |
| 2026-02-17 | [#18774](https://github.com/sgl-project/sglang/pull/18774) | merged | Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1 | `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-03-02 | [#19722](https://github.com/sgl-project/sglang/pull/19722) | open | fix: align DeepSeek OCR vision dtypes | `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-03-11 | [#19732](https://github.com/sgl-project/sglang/pull/19732) | merged | [AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test | `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-03-18 | [#20708](https://github.com/sgl-project/sglang/pull/20708) | merged | Add Mistral Small 4 (Pixtral) support | `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-03-19 | [#12555](https://github.com/sgl-project/sglang/pull/12555) | merged | [CPU] Fix MoE layer support for DeepSeek-OCR models | `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-04-03 | [#21738](https://github.com/sgl-project/sglang/pull/21738) | merged | refactor: replace mm_inputs dict with MultimodalProcessorOutput | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-04-04 | [#21735](https://github.com/sgl-project/sglang/pull/21735) | merged | fix ut test_moe | `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-21 | [#23044](https://github.com/sgl-project/sglang/pull/23044) | merged | [XPU] Fix DeepSeek-OCR tests under transformers 5.x | `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` |
| 2026-04-29 | [#23820](https://github.com/sgl-project/sglang/pull/23820) | merged | Update XPU Docker runtime stack & hf_home config | `test/srt/xpu/test_intel_xpu_backend.py`, `test/srt/xpu/test_deepseek_ocr.py`, `docker/xpu.Dockerfile` |
| 2026-05-13 | [#25182](https://github.com/sgl-project/sglang/pull/25182) | merged | chore: add vLLM SPDX copyright headers to ported files | `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py` |
| 2026-05-15 | [#25364](https://github.com/sgl-project/sglang/pull/25364) | open | Add Accuracy Benchmark for OCR models | `benchmark/ocr/bench_sglang.py`, `benchmark/ocr/eval_utils.py`, `benchmark/ocr/generate_report.py` |
| 2026-05-21 | [#25257](https://github.com/sgl-project/sglang/pull/25257) | merged | [NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2 | `python/sglang/srt/models/deepseek.py` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-05-22 | [#25589](https://github.com/sgl-project/sglang/pull/25589) | closed | [Fix] DeepSeek-OCR-2 bench_serving: fix processor loading and GPU JPEG tensor conversion | `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/benchmark/utils.py` |
| 2026-05-22 | [#24701](https://github.com/sgl-project/sglang/pull/24701) | merged | [FIX][1/2] fix step3-vl/deepseek-ocr image processor error | `python/sglang/srt/multimodal/processors/step3_vl.py` |
| 2026-05-23 | [#25403](https://github.com/sgl-project/sglang/pull/25403) | merged | [FIX][2/2] fix step3-vl/deepseek-ocr image processor error | `python/sglang/srt/configs/deepseek_ocr.py` |
| 2026-05-27 | [#25405](https://github.com/sgl-project/sglang/pull/25405) | merged | [XPU] Add registry mechanism for XPU CI tests | `.github/workflows/pr-test-xpu.yml`, `test/registered/xpu/test_xpu_basic.py`, `test/srt/run_suite.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-06 | [#27248](https://github.com/sgl-project/sglang/pull/27248) | merged | [Doc][CPU]Update Cookbook with Xeon support info | `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` |
| 2026-06-23 | [#27527](https://github.com/sgl-project/sglang/pull/27527) | merged | Vectorize _create_custom_4d_mask in CustomQwen2Decoder | `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py` |
| 2026-06-23 | [#28988](https://github.com/sgl-project/sglang/pull/28988) | merged | [CI] Fix lint brought by #27527 | `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py` |

## Per-PR Diff Audit Cards

### PR #11891 - model: support deepseek-ocr

- Link: https://github.com/sgl-project/sglang/pull/11891
- Status/date: merged / 2025-10-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +2125/-117, 2504 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: support deepseek-ocr"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/deepseek_ocr.py`; technical summary: Covers "model: support deepseek-ocr"; the main implementation surface is `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_ocr.py` added +1516/-0 (1516 lines); hunks: -0,0 +1,1516; symbols: _flatten_embeddings, _embedding_count_expression, _merge_multimodal_embeddings, isin_list, touching `_flatten_embeddings, _embedding_count_expression, _merge_multimodal_embeddings`; `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95 (289 lines); hunks: -11,6 +11,8; -61,6 +63,7 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__, touching `select_best_resolution, __setitem__, VLChatProcessorOutput`; `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0 (262 lines); hunks: -0,0 +1,262; symbols: ImageTransform, __init__, __call__, VisionEncoderConfig, touching `ImageTransform, __init__, __call__`; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` added +37/-0 (37 lines); hunks: -0,0 +1,37; symbols: DeepseekOCRProcessor, __init__, process_mm_data_async, touching `DeepseekOCRProcessor, __init__, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_ocr.py` added +1516/-0 (1516 lines); hunks: -0,0 +1,1516; symbols: _flatten_embeddings, _embedding_count_expression, _merge_multimodal_embeddings, isin_list
  - `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95 (289 lines); hunks: -11,6 +11,8; -61,6 +63,7 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__
  - `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0 (262 lines); hunks: -0,0 +1,262; symbols: ImageTransform, __init__, __call__, VisionEncoderConfig
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` added +37/-0 (37 lines); hunks: -0,0 +1,37; symbols: DeepseekOCRProcessor, __init__, process_mm_data_async
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -921,6 +921,7 @@ def is_generation_model(model_architectures: List[str], is_e...; symbols: is_generation_model
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -0,0 +1,1516 @@
+# Copyright 2025 The SwissAI Initiative
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/configs/deepseekvl2.py
@@ -11,6 +11,8 @@
+from sglang.srt.configs.deepseek_ocr import BASE_SIZE, IMAGE_SIZE, MAX_CROPS, MIN_CROPS
@@ -61,6 +63,7 @@ def __setitem__(self, key, value):
+    images_crop: torch.LongTensor
@@ -104,6 +107,68 @@ def __call__(self, pil_img: Image.Image):
+def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
+    best_ratio_diff = float("inf")
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -0,0 +1,262 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` added +1516/-0; `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95; `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` added +37/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0; `python/sglang/srt/models/deepseek_v2.py` modified +0/-1
  - tests: `test/srt/test_vision_openai_server_a.py` modified +56/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/srt/test_vision_openai_server_a.py`, `test/srt/test_vision_openai_server_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12384 - [Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…

- Link: https://github.com/sgl-project/sglang/pull/12384
- Status/date: merged / 2025-10-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +683/-216, 1133 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py`; technical summary: Covers "[Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…"; the main implementation surface is `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10 (531 lines); hunks: -1,8 +1,19; -18,18 +29,59; symbols: ImageTransform, DictOutput, items, keys, touching `ImageTransform, DictOutput, items`; `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194 (289 lines); hunks: -11,8 +11,6; -63,7 +61,6 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__, touching `select_best_resolution, __setitem__, VLChatProcessorOutput`; `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0 (35 lines); hunks: -0,0 +1,35; symbols: register_customized_processor, that, MyModelConfig, decorator, touching `register_customized_processor, that, MyModelConfig`; `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12 (44 lines); hunks: -54,6 +54,7; -172,6 +173,16 @@ def _load_deepseek_v32_model(; symbols: _load_deepseek_v32_model, _is_deepseek_ocr_model, get_config, get_processor, touching `_load_deepseek_v32_model, _is_deepseek_ocr_model, get_config`.
- Code diff details:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10 (531 lines); hunks: -1,8 +1,19; -18,18 +29,59; symbols: ImageTransform, DictOutput, items, keys
  - `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194 (289 lines); hunks: -11,8 +11,6; -63,7 +61,6 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__
  - `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0 (35 lines); hunks: -0,0 +1,35; symbols: register_customized_processor, that, MyModelConfig, decorator
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12 (44 lines); hunks: -54,6 +54,7; -172,6 +173,16 @@ def _load_deepseek_v32_model(; symbols: _load_deepseek_v32_model, _is_deepseek_ocr_model, get_config, get_processor
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -1,8 +1,19 @@
-from typing import Tuple
-import torchvision.transforms as T
-from PIL import Image
-from transformers import PretrainedConfig
+import math
+from dataclasses import dataclass
diff -- python/sglang/srt/configs/deepseekvl2.py
@@ -11,8 +11,6 @@
-from sglang.srt.configs.deepseek_ocr import BASE_SIZE, IMAGE_SIZE, MAX_CROPS, MIN_CROPS
@@ -63,7 +61,6 @@ def __setitem__(self, key, value):
-    images_crop: torch.LongTensor
@@ -107,68 +104,6 @@ def __call__(self, pil_img: Image.Image):
-def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
-    best_ratio_diff = float("inf")
diff -- python/sglang/srt/multimodal/customized_mm_processor_utils.py
@@ -0,0 +1,35 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10; `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194; `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12415 - Feat: deepseek-ocr logits processor

- Link: https://github.com/sgl-project/sglang/pull/12415
- Status/date: merged / 2025-10-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +89/-1, 115 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Feat: deepseek-ocr logits processor"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py`; technical summary: Covers "Feat: deepseek-ocr logits processor"; the main implementation surface is `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0 (22 lines); hunks: -15,6 +15,10; -26,6 +30,24; symbols: get_default_ngram_custom_params, touching `get_default_ngram_custom_params`; `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1 (68 lines); hunks: -1,7 +1,7; -126,3 +126,69 @@ class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudget...; symbols: DeepSeekR1ThinkingBudgetLogitProcessor, DeepseekOCRNoRepeatNGramLogitProcessor, __call__, touching `DeepSeekR1ThinkingBudgetLogitProcessor, DeepseekOCRNoRepeatNGramLogitProcessor, __call__`.
- Code diff details:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0 (22 lines); hunks: -15,6 +15,10; -26,6 +30,24; symbols: get_default_ngram_custom_params
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1 (68 lines); hunks: -1,7 +1,7; -126,3 +126,69 @@ class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudget...; symbols: DeepSeekR1ThinkingBudgetLogitProcessor, DeepseekOCRNoRepeatNGramLogitProcessor, __call__
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -15,6 +15,10 @@
+from sglang.srt.sampling.custom_logit_processor import (
+    DeepseekOCRNoRepeatNGramLogitProcessor,
+)
@@ -26,6 +30,24 @@
+NGRAM_NO_REPEAT_SIZE = 30
+NGRAM_NO_REPEAT_WINDOW = 90
diff -- python/sglang/srt/sampling/custom_logit_processor.py
@@ -1,7 +1,7 @@
-from typing import TYPE_CHECKING, Any, Dict, List, Optional
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
@@ -126,3 +126,69 @@ class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
+# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
+class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
+    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0; `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12470 - Fix lint in deepseek-ocr

- Link: https://github.com/sgl-project/sglang/pull/12470
- Status/date: merged / 2025-10-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix lint in deepseek-ocr"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/configs/deepseek_ocr.py`; technical summary: Covers "Fix lint in deepseek-ocr"; the main implementation surface is `python/sglang/srt/configs/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1 (1 lines); hunks: -14,7 +14,6.
- Code diff details:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1 (1 lines); hunks: -14,7 +14,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -14,7 +14,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12619 - [NPU] supports ds-ocr model on ascend

- Link: https://github.com/sgl-project/sglang/pull/12619
- Status/date: open / 2025-11-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +200/-60, 389 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] supports ds-ocr model on ascend"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`; technical summary: Covers "[NPU] supports ds-ocr model on ascend"; the main implementation surface is `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek.py` modified +142/-49 (191 lines); hunks: -23,11 +23,14; -36,19 +39,21; symbols: DeepseekMLP, __init__, get_moe_weights, touching `DeepseekMLP, __init__, get_moe_weights`; `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11 (69 lines); hunks: -30,6 +30,7; -1770,6 +1771,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek.py` modified +142/-49 (191 lines); hunks: -23,11 +23,14; -36,19 +39,21; symbols: DeepseekMLP, __init__, get_moe_weights
  - `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11 (69 lines); hunks: -30,6 +30,7; -1770,6 +1771,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek.py
@@ -23,11 +23,14 @@
+    get_pp_group,
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
+from sglang.srt.layers.dp_attention import is_dp_attention_enabled
@@ -36,19 +39,21 @@
-from sglang.srt.layers.moe.fused_moe_triton import fused_moe
-from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -30,6 +30,7 @@
+from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
@@ -1770,6 +1771,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        expert_params_mapping = FusedMoE.make_expert_params_mapping(
+            ckpt_gate_proj_name="gate_proj",
+            ckpt_down_proj_name="down_proj",
+            ckpt_up_proj_name="up_proj",
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +142/-49; `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17897 - Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833

- Link: https://github.com/sgl-project/sglang/pull/17897
- Status/date: merged / 2026-01-30
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_ocr.md`, `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`; associated commits `84ab611af8b7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +618/-140, 1057 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`; technical summary: Covers "Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833"; the main implementation surface is `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_ocr.py` modified +446/-116 (562 lines); hunks: -24,6 +24,7; -702,6 +703,7 @@ def __init__(; symbols: __init__, forward, _build_sam, touching `__init__, forward, _build_sam`; `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9 (41 lines); hunks: -196,6 +196,7 @@ def __init__(; -243,6 +244,7 @@ def __init__(; symbols: __init__, process_one, tokenize_with_images, touching `__init__, process_one, tokenize_with_images`; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +8/-0 (8 lines); hunks: -12,6 +12,14 @@ class DeepseekOCRProcessor(BaseMultimodalProcessor):; symbols: DeepseekOCRProcessor, __init__, touching `DeepseekOCRProcessor, __init__`; `docs/basic_usage/deepseek_ocr.md` added +54/-0 (54 lines); hunks: -0,0 +1,54.
- Code diff details:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +446/-116 (562 lines); hunks: -24,6 +24,7; -702,6 +703,7 @@ def __init__(; symbols: __init__, forward, _build_sam
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9 (41 lines); hunks: -196,6 +196,7 @@ def __init__(; -243,6 +244,7 @@ def __init__(; symbols: __init__, process_one, tokenize_with_images
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +8/-0 (8 lines); hunks: -12,6 +12,14 @@ class DeepseekOCRProcessor(BaseMultimodalProcessor):; symbols: DeepseekOCRProcessor, __init__
  - `docs/basic_usage/deepseek_ocr.md` added +54/-0 (54 lines); hunks: -0,0 +1,54
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -24,6 +24,7 @@
+import transformers
@@ -702,6 +703,7 @@ def __init__(
+        net_3_out_channels: int = 1024,
@@ -776,7 +778,7 @@ def __init__(
-            512, 1024, kernel_size=3, stride=2, padding=1, bias=False
+            512, net_3_out_channels, kernel_size=3, stride=2, padding=1, bias=False
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -196,6 +196,7 @@ def __init__(
+        ocr2_mode: bool = False,
@@ -243,6 +244,7 @@ def __init__(
+        self.ocr2_mode = ocr2_mode
@@ -359,6 +361,13 @@ def process_one(
+        has_images = len(images_list) > 0
+        has_local_crops = False
diff -- python/sglang/srt/multimodal/processors/deepseek_ocr.py
@@ -12,6 +12,14 @@ class DeepseekOCRProcessor(BaseMultimodalProcessor):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +446/-116; `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +8/-0
  - docs: `docs/basic_usage/deepseek_ocr.md` added +54/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_loader/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13561 - [XPU] Integrate MoE and minor improvements in XPU attention backend

- Link: https://github.com/sgl-project/sglang/pull/13561
- Status/date: merged / 2026-02-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +233/-7, 372 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU] Integrate MoE and minor improvements in XPU attention backend"; model line: DeepSeek OCR 2; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py`; technical summary: Covers "[XPU] Integrate MoE and minor improvements in XPU attention backend"; the main implementation surface is `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/unquant.py` modified +49/-0 (49 lines); hunks: -32,6 +32,7; -470,6 +471,54 @@ def forward_cpu(; symbols: forward_cpu, forward_xpu, forward_npu, touching `forward_cpu, forward_xpu, forward_npu`; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1 (35 lines); hunks: -20,6 +20,8; -40,6 +42,8; symbols: fused_experts_impl, fused_moe, touching `fused_experts_impl, fused_moe`; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3 (16 lines); hunks: -19,7 +19,7; -33,6 +33,7; symbols: run, touching `run`; `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2 (5 lines); hunks: -5,12 +5,13.
- Code diff details:
  - `python/sglang/srt/layers/quantization/unquant.py` modified +49/-0 (49 lines); hunks: -32,6 +32,7; -470,6 +471,54 @@ def forward_cpu(; symbols: forward_cpu, forward_xpu, forward_npu
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1 (35 lines); hunks: -20,6 +20,8; -40,6 +42,8; symbols: fused_experts_impl, fused_moe
  - `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3 (16 lines); hunks: -19,7 +19,7; -33,6 +33,7; symbols: run
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2 (5 lines); hunks: -5,12 +5,13
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-0 (1 lines); hunks: -72,6 +72,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/unquant.py
@@ -32,6 +32,7 @@
+    use_intel_xpu_backend,
@@ -470,6 +471,54 @@ def forward_cpu(
+    def forward_xpu(
+        self,
+        layer: torch.nn.Module,
+        dispatch_output: StandardDispatchOutput,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -20,6 +20,8 @@
+    is_xpu,
+    use_intel_xpu_backend,
@@ -40,6 +42,8 @@
+_is_xpu = is_xpu()
+_use_sgl_xpu = use_intel_xpu_backend()
@@ -55,6 +59,8 @@
diff -- python/sglang/srt/layers/moe/moe_runner/triton.py
@@ -19,7 +19,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/unquant.py` modified +49/-0; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3; `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2; `python/sglang/srt/layers/moe/topk.py` modified +1/-0; `python/sglang/srt/utils/common.py` modified +11/-1
  - tests: `test/srt/xpu/test_deepseek_ocr.py` added +121/-0; `test/srt/run_suite.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/xpu/test_deepseek_ocr.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18860 - update pre-commit config

- Link: https://github.com/sgl-project/sglang/pull/18860
- Status/date: merged / 2026-02-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 135 files, +239/-198, 1632 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update pre-commit config"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`; technical summary: Covers "update pre-commit config"; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend, touching `forward_decode, forward_extend`; `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method, touching `get_moe_method`; `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend
  - `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15
  - `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2 (4 lines); hunks: -1,6 +1,6
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6; `python/sglang/srt/models/pixtral.py` modified +6/-2; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2; `python/sglang/srt/multimodal/processors/ernie45_vl.py` modified +3/-1
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `test/manual/test_vlm_accuracy.py`, `test/registered/attention/test_triton_sliding_window.py`, `test/registered/layers/test_fla_layernorm_guard.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18774 - Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1

- Link: https://github.com/sgl-project/sglang/pull/18774
- Status/date: merged / 2026-02-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-1, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_ocr.py`; technical summary: Covers "Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1"; the main implementation surface is `python/sglang/srt/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1 (11 lines); hunks: -1216,9 +1216,18 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1 (11 lines); hunks: -1216,9 +1216,18 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -1216,9 +1216,18 @@ def forward(
+                causal_mask_mapping = {
+                    "full_attention": self._update_causal_mask(
+                        attention_mask,
+                        inputs_embeds,
+                        cache_position,
+                        past_key_values,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19722 - fix: align DeepSeek OCR vision dtypes

- Link: https://github.com/sgl-project/sglang/pull/19722
- Status/date: open / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-6, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: align DeepSeek OCR vision dtypes"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/models/deepseek_ocr.py`; technical summary: Covers "fix: align DeepSeek OCR vision dtypes"; the main implementation surface is `python/sglang/srt/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6 (20 lines); hunks: -1509,18 +1509,26 @@ def _collect_mm_flag(; -1612,7 +1620,7 @@ def _pixel_values_to_embedding(; symbols: _collect_mm_flag, _encode_ocr2_features, _encode_ocr1_features, _pixel_values_to_embedding, touching `_collect_mm_flag, _encode_ocr2_features, _encode_ocr1_features`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6 (20 lines); hunks: -1509,18 +1509,26 @@ def _collect_mm_flag(; -1612,7 +1620,7 @@ def _pixel_values_to_embedding(; symbols: _collect_mm_flag, _encode_ocr2_features, _encode_ocr1_features, _pixel_values_to_embedding
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -1509,18 +1509,26 @@ def _collect_mm_flag(
+        sam_dtype = next(self.sam_model.parameters()).dtype
+        projector_dtype = next(self.projector.parameters()).dtype
+        images = images.to(dtype=sam_dtype)
+        features = features.to(dtype=projector_dtype)
+        sam_dtype = next(self.sam_model.parameters()).dtype
+        vision_dtype = self.vision_model.dtype
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19732 - [AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test

- Link: https://github.com/sgl-project/sglang/pull/19732
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_ocr.py`; associated commits `dc4380e33ac9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +23/-5, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `python/sglang/srt/models/deepseek_ocr.py`; technical summary: Covers "[AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test"; the main implementation surface is `python/sglang/srt/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3 (7 lines); hunks: -125,8 +125,9 @@ def isin_list(; -1685,7 +1686,7 @@ def _process_image_input(self, mm_items: List[MultimodalDa...; symbols: isin_list, _process_image_input, touching `isin_list, _process_image_input`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3 (7 lines); hunks: -125,8 +125,9 @@ def isin_list(; -1685,7 +1686,7 @@ def _process_image_input(self, mm_items: List[MultimodalDa...; symbols: isin_list, _process_image_input
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -125,8 +125,9 @@ def isin_list(
-    test_elements = torch.tensor(test_elements_list, pin_memory=True).to(
-        device=elements.device, non_blocking=True
+    use_pin = torch.cuda.is_available() and not getattr(torch.version, "hip", None)
+    test_elements = torch.tensor(test_elements_list, pin_memory=use_pin).to(
+        device=elements.device, non_blocking=use_pin
@@ -1685,7 +1686,7 @@ def _process_image_input(self, mm_items: List[MultimodalDataItem]) -> torch.Tens
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20708 - Add Mistral Small 4 (Pixtral) support

- Link: https://github.com/sgl-project/sglang/pull/20708
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +360/-124, 868 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Mistral Small 4 (Pixtral) support"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`; technical summary: Covers "Add Mistral Small 4 (Pixtral) support"; the main implementation surface is `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44 (115 lines); hunks: -1,11 +1,12; -20,63 +21,47 @@ class PixtralProcessor(BaseMultimodalProcessor):; symbols: PixtralProcessor, get_patch_grid_size, __init__, defined, touching `PixtralProcessor, get_patch_grid_size, __init__`; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13 (45 lines); hunks: -333,6 +333,8 @@ def _process_messages(; -469,19 +471,20 @@ def _apply_jinja_template(; symbols: _process_messages, _apply_jinja_template, _get_history_tool_calls_cnt, _patch_mistral_skip_special_tokens, touching `_process_messages, _apply_jinja_template, _get_history_tool_calls_cnt`; `python/sglang/srt/parser/reasoning_parser.py` modified +28/-0 (28 lines); hunks: -450,6 +450,33 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:; -474,6 +501,7 @@ class ReasoningParser:; symbols: detect_and_parse, MistralDetector, __init__, ReasoningParser, touching `detect_and_parse, MistralDetector, __init__`; `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment, touching `detect_and_parse, parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44 (115 lines); hunks: -1,11 +1,12; -20,63 +21,47 @@ class PixtralProcessor(BaseMultimodalProcessor):; symbols: PixtralProcessor, get_patch_grid_size, __init__, defined
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13 (45 lines); hunks: -333,6 +333,8 @@ def _process_messages(; -469,19 +471,20 @@ def _apply_jinja_template(; symbols: _process_messages, _apply_jinja_template, _get_history_tool_calls_cnt, _patch_mistral_skip_special_tokens
  - `python/sglang/srt/parser/reasoning_parser.py` modified +28/-0 (28 lines); hunks: -450,6 +450,33 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:; -474,6 +501,7 @@ class ReasoningParser:; symbols: detect_and_parse, MistralDetector, __init__, ReasoningParser
  - `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment
  - `python/sglang/srt/configs/janus_pro.py` modified +12/-12 (24 lines); hunks: -123,14 +123,14 @@ class SigLIPVisionCfg:; -595,12 +595,12 @@ def batchify(; symbols: SigLIPVisionCfg, MultiModalityConfig, __init__, batchify
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/pixtral.py
@@ -1,11 +1,12 @@
-import asyncio
+from transformers import PreTrainedTokenizerBase
+from sglang.srt.managers.schedule_batch import Modality
@@ -20,63 +21,47 @@ class PixtralProcessor(BaseMultimodalProcessor):
-    IMG_BREAK_TOKEN_ID = 12
-    IMG_END_TOKEN_ID = 13
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -333,6 +333,8 @@ def _process_messages(
+        self._patch_mistral_skip_special_tokens(request)
@@ -469,19 +471,20 @@ def _apply_jinja_template(
+            extra_template_kwargs = {}
+            if request.reasoning_effort is not None:
+                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
+            if request.chat_template_kwargs:
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -450,6 +450,33 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13; `python/sglang/srt/parser/reasoning_parser.py` modified +28/-0; `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9; `python/sglang/srt/configs/janus_pro.py` modified +12/-12; `python/sglang/srt/configs/jet_nemotron.py` modified +12/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/janus_pro.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12555 - [CPU] Fix MoE layer support for DeepSeek-OCR models

- Link: https://github.com/sgl-project/sglang/pull/12555
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +65/-10, 110 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Fix MoE layer support for DeepSeek-OCR models"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`; technical summary: Covers "[CPU] Fix MoE layer support for DeepSeek-OCR models"; the main implementation surface is `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek.py` modified +31/-9 (40 lines); hunks: -48,7 +48,12; -176,14 +181,31 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: DeepseekMLP, forward, touching `DeepseekMLP, forward`; `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1 (35 lines); hunks: -41,6 +41,10; -1772,7 +1776,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, post_load_weights, touching `load_weights, post_load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek.py` modified +31/-9 (40 lines); hunks: -48,7 +48,12; -176,14 +181,31 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: DeepseekMLP, forward
  - `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1 (35 lines); hunks: -41,6 +41,10; -1772,7 +1776,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, post_load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek.py
@@ -48,7 +48,12 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu
+_is_cpu_amx_available = cpu_has_amx_support()
+_is_cpu = is_cpu()
+if _is_cpu and _is_cpu_amx_available:
+    import sgl_kernel  # noqa: F401
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -41,6 +41,10 @@
+from sglang.srt.utils import cpu_has_amx_support, is_cpu
+_is_cpu_amx_available = cpu_has_amx_support()
+_is_cpu = is_cpu()
@@ -1772,7 +1776,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
@@ -1852,6 +1855,36 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        self.post_load_weights()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +31/-9; `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21738 - refactor: replace mm_inputs dict with MultimodalProcessorOutput

- Link: https://github.com/sgl-project/sglang/pull/21738
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 40 files, +408/-314, 1321 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor: replace mm_inputs dict with MultimodalProcessorOutput"; model line: DeepSeek OCR 2; category: model implementation change; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`; technical summary: Covers "refactor: replace mm_inputs dict with MultimodalProcessorOutput"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async, touching `get_mm_data, process_mm_data_async`; `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async, touching `_process_special_format, process_mm_data_async, process_qwen_mm_data_async`; `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, touching `_process_special_format, process_mm_data_async`; `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async, touching `get_mm_data, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15 (34 lines); hunks: -1,6 +1,10; -69,13 +73,13 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -12,7 +12,11 @@
-from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+    MultimodalProcessorOutput,
+)
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -11,6 +11,7 @@
+    MultimodalProcessorOutput,
@@ -337,14 +338,14 @@ async def _process_special_format(
-        return {
-            "input_ids": input_ids_tensor.flatten().tolist(),
-            "mm_items": mm_items,
-            "im_start_id": self.img_start_token_id,
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -5,6 +5,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23; `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24; `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22; `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19; `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15; `python/sglang/srt/multimodal/processors/transformers_auto.py` modified +18/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/managers/io_struct.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21735 - fix ut test_moe

- Link: https://github.com/sgl-project/sglang/pull/21735
- Status/date: merged / 2026-04-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +105/-32, 214 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix ut test_moe"; model line: DeepSeek OCR 2; category: bug fix; main diff: `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`; technical summary: Covers "fix ut test_moe"; the main implementation surface is `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26 (54 lines); hunks: -2,9 +2,11; -19,11 +21,32; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass, touching `TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass`; `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: TestDeepSeekOCRTriton, setUpClass, touching `TestDeepSeekOCRTriton, setUpClass`; `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5 (29 lines); hunks: -3,6 +3,7; -15,26 +16,44; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper, touching `_cleanup_xpu_memory, intel_xpu_benchmark, decorator`; `test/srt/run_suite.py` modified +2/-1 (3 lines); hunks: -77,7 +77,8.
- Code diff details:
  - `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26 (54 lines); hunks: -2,9 +2,11; -19,11 +21,32; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass
  - `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: TestDeepSeekOCRTriton, setUpClass
  - `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5 (29 lines); hunks: -3,6 +3,7; -15,26 +16,44; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper
  - `test/srt/run_suite.py` modified +2/-1 (3 lines); hunks: -77,7 +77,8
- Key code excerpts:

```diff
diff -- test/srt/xpu/test_deepseek_ocr.py
@@ -2,9 +2,11 @@
+import gc
+from pathlib import Path
@@ -19,11 +21,32 @@
+    @classmethod
+    def _cleanup_xpu_memory(cls):
+        gc.collect()
diff -- test/srt/xpu/test_deepseek_ocr_triton.py
@@ -0,0 +1,51 @@
+"""
+python3 -m unittest test_deepseek_ocr_triton.py
+"""
+import os
+import unittest
+from pathlib import Path
diff -- test/srt/xpu/test_intel_xpu_backend.py
@@ -3,6 +3,7 @@
```

- Reviewed files:
  - tests: `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26; `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0; `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5; `test/srt/run_suite.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- Link: https://github.com/sgl-project/sglang/pull/23001
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 330 files, +80364/-0, 68714 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`; technical summary: Covers "Add new Mintlify documentation site (docs_new/)"; the main implementation surface is `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23044 - [XPU] Fix DeepSeek-OCR tests under transformers 5.x

- Link: https://github.com/sgl-project/sglang/pull/23044
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-8, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU] Fix DeepSeek-OCR tests under transformers 5.x"; model line: DeepSeek OCR 2; category: bug fix; main diff: `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`; technical summary: Covers "[XPU] Fix DeepSeek-OCR tests under transformers 5.x"; the main implementation surface is `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4 (6 lines); hunks: -9,9 +9,9; -38,9 +38,7 @@ def _cleanup_xpu_memory(cls):; symbols: _cleanup_xpu_memory, setUpClass, touching `_cleanup_xpu_memory, setUpClass`; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4 (6 lines); hunks: -7,8 +7,8; -21,9 +21,7 @@ class TestDeepSeekOCRTriton(deepseek_ocr.TestDeepSeekOCR):; symbols: TestDeepSeekOCRTriton, setUpClass, touching `TestDeepSeekOCRTriton, setUpClass`.
- Code diff details:
  - `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4 (6 lines); hunks: -9,9 +9,9; -38,9 +38,7 @@ def _cleanup_xpu_memory(cls):; symbols: _cleanup_xpu_memory, setUpClass
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4 (6 lines); hunks: -7,8 +7,8; -21,9 +21,7 @@ class TestDeepSeekOCRTriton(deepseek_ocr.TestDeepSeekOCR):; symbols: TestDeepSeekOCRTriton, setUpClass
- Key code excerpts:

```diff
diff -- test/srt/xpu/test_deepseek_ocr.py
@@ -9,9 +9,9 @@
-from transformers import AutoTokenizer
+from sglang.srt.utils.hf_transformers import get_tokenizer
@@ -38,9 +38,7 @@ def _cleanup_xpu_memory(cls):
-        cls.tokenizer = AutoTokenizer.from_pretrained(
-            cls.model, use_fast=False, trust_remote_code=True
-        )
diff -- test/srt/xpu/test_deepseek_ocr_triton.py
@@ -7,8 +7,8 @@
-from transformers import AutoTokenizer
+from sglang.srt.utils.hf_transformers import get_tokenizer
@@ -21,9 +21,7 @@ class TestDeepSeekOCRTriton(deepseek_ocr.TestDeepSeekOCR):
-        cls.tokenizer = AutoTokenizer.from_pretrained(
-            cls.model, use_fast=False, trust_remote_code=True
-        )
```

- Reviewed files:
  - tests: `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4
- Risk and verification: The diff ships test coverage in `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- Link: https://github.com/sgl-project/sglang/pull/23337
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 179 files, +16004/-8152, 23604 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Sync docs_new with legacy docs and update migration redirects"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`; technical summary: Covers "[Docs] Sync docs_new with legacy docs and update migration redirects"; the main implementation surface is `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23820 - Update XPU Docker runtime stack & hf_home config

- Link: https://github.com/sgl-project/sglang/pull/23820
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +38/-54, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update XPU Docker runtime stack & hf_home config"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `test/srt/xpu/test_intel_xpu_backend.py`, `test/srt/xpu/test_deepseek_ocr.py`, `docker/xpu.Dockerfile`; technical summary: Covers "Update XPU Docker runtime stack & hf_home config"; the main implementation surface is `test/srt/xpu/test_intel_xpu_backend.py`, `test/srt/xpu/test_deepseek_ocr.py`, `docker/xpu.Dockerfile`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/xpu/test_intel_xpu_backend.py` modified +12/-25 (37 lines); hunks: -3,7 +3,6; -16,29 +15,17; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper, touching `_cleanup_xpu_memory, intel_xpu_benchmark, decorator`; `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17 (23 lines); hunks: -2,7 +2,6; -21,22 +20,8; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass, touching `TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass`; `docker/xpu.Dockerfile` modified +10/-6 (16 lines); hunks: -20,6 +20,16 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +48,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda...; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3 (10 lines); hunks: -6,7 +6,7; -16,10 +16,11; symbols: TestDeepSeekOCRTriton, setUpClass, here, touching `TestDeepSeekOCRTriton, setUpClass, here`.
- Code diff details:
  - `test/srt/xpu/test_intel_xpu_backend.py` modified +12/-25 (37 lines); hunks: -3,7 +3,6; -16,29 +15,17; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper
  - `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17 (23 lines); hunks: -2,7 +2,6; -21,22 +20,8; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass
  - `docker/xpu.Dockerfile` modified +10/-6 (16 lines); hunks: -20,6 +20,16 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +48,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda...
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3 (10 lines); hunks: -6,7 +6,7; -16,10 +16,11; symbols: TestDeepSeekOCRTriton, setUpClass, here
  - `.github/workflows/pr-test-xpu.yml` modified +3/-3 (6 lines); hunks: -72,8 +72,6 @@ jobs:; -99,8 +97,10 @@ jobs:
- Key code excerpts:

```diff
diff -- test/srt/xpu/test_intel_xpu_backend.py
@@ -3,7 +3,6 @@
-import gc
@@ -16,29 +15,17 @@
-def _cleanup_xpu_memory():
-    gc.collect()
-    try:
-        import torch
diff -- test/srt/xpu/test_deepseek_ocr.py
@@ -2,7 +2,6 @@
-import gc
@@ -21,22 +20,8 @@
-    @classmethod
-    def _cleanup_xpu_memory(cls):
-        gc.collect()
-        try:
diff -- docker/xpu.Dockerfile
@@ -20,6 +20,16 @@ ARG SG_LANG_KERNEL_BRANCH=main
```

- Reviewed files:
  - tests: `test/srt/xpu/test_intel_xpu_backend.py` modified +12/-25; `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3
  - other: `docker/xpu.Dockerfile` modified +10/-6
  - ci: `.github/workflows/pr-test-xpu.yml` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- Link: https://github.com/sgl-project/sglang/pull/25182
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 136 files, +255/-0, 872 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "chore: add vLLM SPDX copyright headers to ported files"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`; technical summary: Covers "chore: add vLLM SPDX copyright headers to ported files"; the main implementation surface is `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6; `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6.
- Code diff details:
  - `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma2.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/baichuan.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/baichuan.py
diff -- python/sglang/srt/models/commandr.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/commandr.py
diff -- python/sglang/srt/models/dbrx.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/gemma.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +4/-0; `python/sglang/srt/models/commandr.py` modified +4/-0; `python/sglang/srt/models/dbrx.py` modified +3/-0; `python/sglang/srt/models/gemma.py` modified +3/-0; `python/sglang/srt/models/gemma2.py` modified +3/-0; `python/sglang/srt/models/gpt_bigcode.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_custom_ops.py`, `python/sglang/test/test_marlin_utils.py`, `sgl-kernel/tests/test_causal_conv1d.py`, `test/registered/layers/mamba/test_causal_conv1d.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25364 - Add Accuracy Benchmark for OCR models

- Link: https://github.com/sgl-project/sglang/pull/25364
- Status/date: open / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +1928/-0, 1946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Accuracy Benchmark for OCR models"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `benchmark/ocr/bench_sglang.py`, `benchmark/ocr/eval_utils.py`, `benchmark/ocr/generate_report.py`; technical summary: Covers "Add Accuracy Benchmark for OCR models"; the main implementation surface is `benchmark/ocr/bench_sglang.py`, `benchmark/ocr/eval_utils.py`, `benchmark/ocr/generate_report.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `benchmark/ocr/bench_sglang.py` added +735/-0 (735 lines); hunks: -0,0 +1,735; symbols: BenchArgs, add_cli_args, preflight_check, pdf_page_to_base64_png, touching `BenchArgs, add_cli_args, preflight_check`; `benchmark/ocr/eval_utils.py` added +631/-0 (631 lines); hunks: -0,0 +1,631; symbols: normalize_text, strip_markdown, fuzzy_contains, exact_contains, touching `normalize_text, strip_markdown, fuzzy_contains`; `benchmark/ocr/generate_report.py` added +382/-0 (382 lines); hunks: -0,0 +1,382; symbols: _ocr_to_html, _latex_to_display, _render_sample, generate_report, touching `_ocr_to_html, _latex_to_display, _render_sample`; `benchmark/ocr/README.md` added +178/-0 (178 lines); hunks: -0,0 +1,178.
- Code diff details:
  - `benchmark/ocr/bench_sglang.py` added +735/-0 (735 lines); hunks: -0,0 +1,735; symbols: BenchArgs, add_cli_args, preflight_check, pdf_page_to_base64_png
  - `benchmark/ocr/eval_utils.py` added +631/-0 (631 lines); hunks: -0,0 +1,631; symbols: normalize_text, strip_markdown, fuzzy_contains, exact_contains
  - `benchmark/ocr/generate_report.py` added +382/-0 (382 lines); hunks: -0,0 +1,382; symbols: _ocr_to_html, _latex_to_display, _render_sample, generate_report
  - `benchmark/ocr/README.md` added +178/-0 (178 lines); hunks: -0,0 +1,178
  - `python/pyproject.toml` modified +1/-0 (1 lines); hunks: -85,6 +85,7 @@ dependencies = [
- Key code excerpts:

```diff
diff -- benchmark/ocr/bench_sglang.py
@@ -0,0 +1,735 @@
+"""
+Benchmark DeepSeek-OCR-2 (and similar OCR VLMs) on olmOCR-bench via a running sglang server.
+Usage:
+    # 0. Download the dataset (one-time, ~2 GB with PDFs via Git LFS)
+    hf download --repo-type dataset \\
+        allenai/olmOCR-bench --local-dir ./olmOCR-bench
diff -- benchmark/ocr/eval_utils.py
@@ -0,0 +1,631 @@
+"""
+Evaluation utilities for the OCR benchmark (olmOCR-bench test classes).
+Implements:
+  - text_presence      : short text segment must be present in OCR output
+  - text_absence       : text (headers/footers/page numbers) must NOT appear
+  - natural_reading_order : two text spans must appear in correct relative order
diff -- benchmark/ocr/generate_report.py
@@ -0,0 +1,382 @@
```

- Reviewed files:
  - other: `benchmark/ocr/bench_sglang.py` added +735/-0; `benchmark/ocr/eval_utils.py` added +631/-0; `benchmark/ocr/generate_report.py` added +382/-0; `benchmark/ocr/README.md` added +178/-0
  - runtime: `python/pyproject.toml` modified +1/-0; `python/pyproject_xpu.toml` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/pyproject_xpu.toml`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25257 - [NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2

- Link: https://github.com/sgl-project/sglang/pull/25257
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-3, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2"; model line: DeepSeek OCR 2; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek.py`; technical summary: Covers "[NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2"; the main implementation surface is `python/sglang/srt/models/deepseek.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek.py` modified +11/-3 (14 lines); hunks: -39,7 +39,6; -50,14 +49,23; symbols: DeepseekMLP, forward, touching `DeepseekMLP, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek.py` modified +11/-3 (14 lines); hunks: -39,7 +39,6; -50,14 +49,23; symbols: DeepseekMLP, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek.py
@@ -39,7 +39,6 @@
-from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
@@ -50,14 +49,23 @@
-from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu
+from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu, is_npu
+_is_npu = is_npu()
+if _is_npu:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +11/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- Link: https://github.com/sgl-project/sglang/pull/24751
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +45/-44, 401 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`; technical summary: Covers "fix(mm): make multimodal data loading non-blocking to prevent health check stalls"; the main implementation surface is `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data, touching `_process_loaded_mm_data, load_mm_data`; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async, touching `_process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async`; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async, touching `_process_special_format, process_mm_data_async`; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async, touching `__init__, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -1,3 +1,4 @@
+import asyncio
@@ -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):
-    def load_mm_data(
+    async def load_mm_data(
@@ -772,7 +773,7 @@ def load_mm_data(
-            return self.legacy_load_mm_data(
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -310,7 +310,7 @@ async def _process_special_format(
-            base_output = self.load_mm_data(
+            base_output = await self.load_mm_data(
@@ -423,7 +423,7 @@ async def process_qwen_mm_data_async(
-        base_output = self.load_mm_data(
+        base_output = await self.load_mm_data(
@@ -644,7 +644,7 @@ async def process_internlm2_mm_data_async(
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -118,7 +118,7 @@ async def _process_special_format(
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/clip.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25589 - [Fix] DeepSeek-OCR-2 bench_serving: fix processor loading and GPU JPEG tensor conversion

- Link: https://github.com/sgl-project/sglang/pull/25589
- Status/date: closed / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-11, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] DeepSeek-OCR-2 bench_serving: fix processor loading and GPU JPEG tensor conversion"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/benchmark/utils.py`; technical summary: Covers "[Fix] DeepSeek-OCR-2 bench_serving: fix processor loading and GPU JPEG tensor conversion"; the main implementation surface is `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/benchmark/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/deepseek_ocr.py` modified +5/-0 (5 lines); hunks: -454,6 +454,11 @@ def tokenize_with_images(; symbols: tokenize_with_images, touching `tokenize_with_images`; `python/sglang/benchmark/utils.py` modified +7/-11 (18 lines); hunks: -71,20 +71,16 @@ def get_processor(; symbols: get_processor, download_and_cache_hf_file, touching `get_processor, download_and_cache_hf_file`.
- Code diff details:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +5/-0 (5 lines); hunks: -454,6 +454,11 @@ def tokenize_with_images(; symbols: tokenize_with_images
  - `python/sglang/benchmark/utils.py` modified +7/-11 (18 lines); hunks: -71,20 +71,16 @@ def get_processor(; symbols: get_processor, download_and_cache_hf_file
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -454,6 +454,11 @@ def tokenize_with_images(
+            # GPU JPEG decode returns a (C, H, W) uint8 torch.Tensor;
+            # convert to PIL Image so that PIL-specific operations below work.
+            if isinstance(image, torch.Tensor):
+                image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
diff -- python/sglang/benchmark/utils.py
@@ -71,20 +71,16 @@ def get_processor(
-    if pretrained_model_name_or_path.endswith(
-        ".json"
-    ) or pretrained_model_name_or_path.endswith(".model"):
-        from sglang.srt.utils.hf_transformers_utils import get_processor
-        return get_processor(pretrained_model_name_or_path)
+    from sglang.srt.utils.hf_transformers_utils import (
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +5/-0; `python/sglang/benchmark/utils.py` modified +7/-11
- Risk and verification: Runtime changes concentrate in `python/sglang/benchmark/utils.py`, `python/sglang/srt/configs/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24701 - [FIX][1/2] fix step3-vl/deepseek-ocr image processor error

- Link: https://github.com/sgl-project/sglang/pull/24701
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +67/-20, 160 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[FIX][1/2] fix step3-vl/deepseek-ocr image processor error"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/step3_vl.py`; technical summary: Covers "[FIX][1/2] fix step3-vl/deepseek-ocr image processor error"; the main implementation surface is `python/sglang/srt/multimodal/processors/step3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20 (87 lines); hunks: -8,6 +8,7; -20,14 +21,37; symbols: GPUToTensor, forward, __call__, ImagePatcher, touching `GPUToTensor, forward, __call__`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20 (87 lines); hunks: -8,6 +8,7; -20,14 +21,37; symbols: GPUToTensor, forward, __call__, ImagePatcher
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/step3_vl.py
@@ -8,6 +8,7 @@
+from torchvision.transforms import functional as F
@@ -20,14 +21,37 @@
-ImageWithPatches = tuple[Image.Image, list[Image.Image], list[int] | None]
+Step3Image = Union[Image.Image, torch.Tensor]
+ImageWithPatches = tuple[Step3Image, list[Step3Image], list[int] | None]
-    def forward(self, raw_image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/step3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25403 - [FIX][2/2] fix step3-vl/deepseek-ocr image processor error

- Link: https://github.com/sgl-project/sglang/pull/25403
- Status/date: merged / 2026-05-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +100/-12, 184 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[FIX][2/2] fix step3-vl/deepseek-ocr image processor error"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/configs/deepseek_ocr.py`; technical summary: Covers "[FIX][2/2] fix step3-vl/deepseek-ocr image processor error"; the main implementation surface is `python/sglang/srt/configs/deepseek_ocr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/deepseek_ocr.py` modified +100/-12 (112 lines); hunks: -1,9 +1,11; -18,6 +20,8; symbols: get_default_ngram_custom_params, get_image_size, resize_image, crop_image, touching `get_default_ngram_custom_params, get_image_size, resize_image`.
- Code diff details:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +100/-12 (112 lines); hunks: -1,9 +1,11; -18,6 +20,8; symbols: get_default_ngram_custom_params, get_image_size, resize_image, crop_image
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -1,9 +1,11 @@
-from typing import Any, Dict, List, Optional, Tuple
+from typing import Any, Dict, List, Optional, Tuple, Union
+from torchvision.transforms import InterpolationMode
+from torchvision.transforms import functional as TF
@@ -18,6 +20,8 @@
+DeepseekOCRImage = Union[Image.Image, torch.Tensor]
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +100/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/deepseek_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25405 - [XPU] Add registry mechanism for XPU CI tests

- Link: https://github.com/sgl-project/sglang/pull/25405
- Status/date: merged / 2026-05-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +156/-22, 310 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU] Add registry mechanism for XPU CI tests"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `.github/workflows/pr-test-xpu.yml`, `test/registered/xpu/test_xpu_basic.py`, `test/srt/run_suite.py`; technical summary: Covers "[XPU] Add registry mechanism for XPU CI tests"; the main implementation surface is `.github/workflows/pr-test-xpu.yml`, `test/registered/xpu/test_xpu_basic.py`, `test/srt/run_suite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/pr-test-xpu.yml` modified +79/-9 (88 lines); hunks: -68,7 +68,8 @@ jobs:; -108,16 +109,80 @@ jobs:; `test/registered/xpu/test_xpu_basic.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: TestXPUBasic, test_basic_generation, touching `TestXPUBasic, test_basic_generation`; `test/srt/run_suite.py` modified +4/-10 (14 lines); hunks: -87,16 +87,10; `test/run_suite.py` modified +7/-1 (8 lines); hunks: -20,6 +20,7; -77,6 +78,10; symbols: _valid_suites_by_backend, touching `_valid_suites_by_backend`.
- Code diff details:
  - `.github/workflows/pr-test-xpu.yml` modified +79/-9 (88 lines); hunks: -68,7 +68,8 @@ jobs:; -108,16 +109,80 @@ jobs:
  - `test/registered/xpu/test_xpu_basic.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: TestXPUBasic, test_basic_generation
  - `test/srt/run_suite.py` modified +4/-10 (14 lines); hunks: -87,16 +87,10
  - `test/run_suite.py` modified +7/-1 (8 lines); hunks: -20,6 +20,7; -77,6 +78,10; symbols: _valid_suites_by_backend
  - `test/registered/xpu/test_deepseek_ocr_triton.py` renamed +7/-0 (7 lines); hunks: -9,12 +9,19
- Key code excerpts:

```diff
diff -- .github/workflows/pr-test-xpu.yml
@@ -68,7 +68,8 @@ jobs:
-  build-and-test:
+  # ==================== Stage A ==================== #
+  stage-a-test-1-gpu-xpu:
@@ -108,16 +109,80 @@ jobs:
-          docker exec "$cid" /home/sdp/miniforge3/envs/py3.12/bin/python3 -m pip install pytest expecttest ray huggingface_hub
+          docker exec "$cid" /home/sdp/miniforge3/envs/py3.12/bin/python3 -m pip install pytest expecttest ray huggingface_hub tabulate
diff -- test/registered/xpu/test_xpu_basic.py
@@ -0,0 +1,47 @@
+"""
+Basic XPU test: verifies the server starts and produces a non-empty
+response on Intel XPU with the default attention backend.
+Assigned to stage-a so it gates stage-b before the heavier tests run.
+Usage:
+python3 -m unittest test_xpu_basic.TestXPUBasic.test_basic_generation
diff -- test/srt/run_suite.py
@@ -87,16 +87,10 @@
```

- Reviewed files:
  - ci: `.github/workflows/pr-test-xpu.yml` modified +79/-9
  - tests: `test/registered/xpu/test_xpu_basic.py` added +47/-0; `test/srt/run_suite.py` modified +4/-10; `test/run_suite.py` modified +7/-1; `test/registered/xpu/test_deepseek_ocr_triton.py` renamed +7/-0; `python/sglang/test/ci/ci_register.py` modified +5/-1; `test/registered/xpu/test_deepseek_ocr.py` renamed +3/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/ci/ci_register.py`, `test/registered/attention/test_chunk_gated_delta_rule.py`, `test/registered/xpu/test_deepseek_ocr.py`, `test/registered/xpu/test_deepseek_ocr_triton.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- Link: https://github.com/sgl-project/sglang/pull/25813
- Status/date: merged / 2026-06-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 47 files, +1262/-2154, 4187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): port popular model usage guides into cookbook pages"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`; technical summary: Covers "docs(cookbook): port popular model usage guides into cookbook pages"; the main implementation surface is `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64, touching `image_to_base64`.
- Code diff details:
  - `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0
  - `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64
  - `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181 (181 lines); hunks: -1,181 +0,0
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26; `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181; `docs_new/docs/basic_usage/glmv.mdx` removed +0/-139
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27248 - [Doc][CPU]Update Cookbook with Xeon support info

- Link: https://github.com/sgl-project/sglang/pull/27248
- Status/date: merged / 2026-06-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +443/-121, 1524 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc][CPU]Update Cookbook with Xeon support info"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`; technical summary: Covers "[Doc][CPU]Update Cookbook with Xeon support info"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #27527 - Vectorize _create_custom_4d_mask in CustomQwen2Decoder

- Link: https://github.com/sgl-project/sglang/pull/27527
- Status/date: merged / 2026-06-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +536/-20, 569 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Vectorize _create_custom_4d_mask in CustomQwen2Decoder"; model line: DeepSeek OCR 2; category: docs/tests/CI; main diff: `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`; technical summary: Covers "Vectorize _create_custom_4d_mask in CustomQwen2Decoder"; the main implementation surface is `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_ocr.py` modified +23/-20 (43 lines); hunks: -1295,31 +1295,34 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask, touching `_create_custom_4d_mask`; `test/manual/test_create_custom_4d_mask.py` added +513/-0 (513 lines); hunks: -0,0 +1,513; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new, _make_token_type_ids, _make_random_token_type_ids, touching `_create_custom_4d_mask_reference, _create_custom_4d_mask_new, _make_token_type_ids`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +23/-20 (43 lines); hunks: -1295,31 +1295,34 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask
  - `test/manual/test_create_custom_4d_mask.py` added +513/-0 (513 lines); hunks: -0,0 +1,513; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new, _make_token_type_ids, _make_random_token_type_ids
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -1295,31 +1295,34 @@ def _create_custom_4d_mask(
-                masks = []
-                for b in range(batch_size):
-                    mask = torch.full(
-                        (sequence_length, sequence_length),
-                        fill_value=min_dtype,
-                        dtype=dtype,
diff -- test/manual/test_create_custom_4d_mask.py
@@ -0,0 +1,513 @@
+"""
+Unit tests for _create_custom_4d_mask (commit a475156d).
+Verifies:
+  1. Numerical accuracy of the new vectorised implementation against the
+     original loop-based reference.
+  2. Wall-clock performance improvement on a range of (batch, seq_len) sizes.
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +23/-20
  - tests: `test/manual/test_create_custom_4d_mask.py` added +513/-0
- Risk and verification: The diff ships test coverage in `test/manual/test_create_custom_4d_mask.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28988 - [CI] Fix lint brought by #27527

- Link: https://github.com/sgl-project/sglang/pull/28988
- Status/date: merged / 2026-06-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-49, 277 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Fix lint brought by #27527"; model line: DeepSeek OCR 2; category: bug fix; main diff: `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`; technical summary: Covers "[CI] Fix lint brought by #27527"; the main implementation surface is `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_ocr.py` modified +3/-3 (6 lines); hunks: -1297,7 +1297,7 @@ def _create_custom_4d_mask(; -1312,8 +1312,8 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask, touching `_create_custom_4d_mask`; `test/manual/test_create_custom_4d_mask.py` modified +65/-46 (111 lines); hunks: -43,6 +43,7; -58,7 +59,7 @@ def _create_custom_4d_mask_reference(; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new, touching `_create_custom_4d_mask_reference, _create_custom_4d_mask_new`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +3/-3 (6 lines); hunks: -1297,7 +1297,7 @@ def _create_custom_4d_mask(; -1312,8 +1312,8 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask
  - `test/manual/test_create_custom_4d_mask.py` modified +65/-46 (111 lines); hunks: -43,6 +43,7; -58,7 +59,7 @@ def _create_custom_4d_mask_reference(; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -1297,7 +1297,7 @@ def _create_custom_4d_mask(
-                is_text  = token_type_ids == 1  # [B, S]
+                is_text = token_type_ids == 1  # [B, S]
@@ -1312,8 +1312,8 @@ def _create_custom_4d_mask(
-                    is_text.unsqueeze(2)   # [B, S, 1]
-                    & is_text.unsqueeze(1) # [B, 1, S]
+                    is_text.unsqueeze(2)  # [B, S, 1]
diff -- test/manual/test_create_custom_4d_mask.py
@@ -43,6 +43,7 @@
@@ -58,7 +59,7 @@ def _create_custom_4d_mask_reference(
-        text_positions  = (type_ids == 1).nonzero(as_tuple=True)[0]
+        text_positions = (type_ids == 1).nonzero(as_tuple=True)[0]
@@ -79,13 +80,14 @@ def _create_custom_4d_mask_reference(
-    is_text  = token_type_ids == 1  # [B, S]
+    is_text = token_type_ids == 1  # [B, S]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +3/-3
  - tests: `test/manual/test_create_custom_4d_mask.py` modified +65/-46
- Risk and verification: The diff ships test coverage in `test/manual/test_create_custom_4d_mask.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
