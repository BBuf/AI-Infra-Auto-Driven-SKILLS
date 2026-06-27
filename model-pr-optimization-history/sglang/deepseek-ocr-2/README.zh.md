# sglang DeepSeek OCR 2 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/basic_usage/deepseek_ocr.md` | [#17897](https://github.com/sgl-project/sglang/pull/17897) |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/deepseek-ocr-v2-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/deepseek_ocr.py` | [#17897](https://github.com/sgl-project/sglang/pull/17897) |
| `python/sglang/srt/models/deepseek_ocr.py` | [#17897](https://github.com/sgl-project/sglang/pull/17897), [#19732](https://github.com/sgl-project/sglang/pull/19732) |
| `python/sglang/srt/multimodal/processors/deepseek_ocr.py` | [#17897](https://github.com/sgl-project/sglang/pull/17897) |
| `test/registered/xpu/test_deepseek_ocr.py` | 无直接 PR 号提交 |
| `test/registered/xpu/test_deepseek_ocr_triton.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 29
- 当前文档总 PR 数: 31
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #11891 - model: support deepseek-ocr

- 链接: https://github.com/sgl-project/sglang/pull/11891
- 状态/时间: merged / 2025-10-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+2125/-117，可读 patch 2504 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support deepseek-ocr」；模型线: DeepSeek OCR 2；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/deepseek_ocr.py`；技术摘要: 覆盖「model: support deepseek-ocr」；主要实现面是 `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_ocr.py` added +1516/-0 (1516 lines); hunks: -0,0 +1,1516; symbols: _flatten_embeddings, _embedding_count_expression, _merge_multimodal_embeddings, isin_list，涉及 `_flatten_embeddings, _embedding_count_expression, _merge_multimodal_embeddings`；`python/sglang/srt/configs/deepseekvl2.py` modified +194/-95 (289 lines); hunks: -11,6 +11,8; -61,6 +63,7 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__，涉及 `select_best_resolution, __setitem__, VLChatProcessorOutput`；`python/sglang/srt/configs/deepseek_ocr.py` added +262/-0 (262 lines); hunks: -0,0 +1,262; symbols: ImageTransform, __init__, __call__, VisionEncoderConfig，涉及 `ImageTransform, __init__, __call__`；`python/sglang/srt/multimodal/processors/deepseek_ocr.py` added +37/-0 (37 lines); hunks: -0,0 +1,37; symbols: DeepseekOCRProcessor, __init__, process_mm_data_async，涉及 `DeepseekOCRProcessor, __init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` added +1516/-0 (1516 lines); hunks: -0,0 +1,1516; symbols: _flatten_embeddings, _embedding_count_expression, _merge_multimodal_embeddings, isin_list
  - `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95 (289 lines); hunks: -11,6 +11,8; -61,6 +63,7 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__
  - `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0 (262 lines); hunks: -0,0 +1,262; symbols: ImageTransform, __init__, __call__, VisionEncoderConfig
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` added +37/-0 (37 lines); hunks: -0,0 +1,37; symbols: DeepseekOCRProcessor, __init__, process_mm_data_async
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -921,6 +921,7 @@ def is_generation_model(model_architectures: List[str], is_e...; symbols: is_generation_model
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` added +1516/-0; `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95; `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` added +37/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0; `python/sglang/srt/models/deepseek_v2.py` modified +0/-1
  - tests: `test/srt/test_vision_openai_server_a.py` modified +56/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/test_vision_openai_server_a.py`, `test/srt/test_vision_openai_server_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12384 - [Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…

- 链接: https://github.com/sgl-project/sglang/pull/12384
- 状态/时间: merged / 2025-10-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+683/-216，可读 patch 1133 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py`；技术摘要: 覆盖「[Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…」；主要实现面是 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10 (531 lines); hunks: -1,8 +1,19; -18,18 +29,59; symbols: ImageTransform, DictOutput, items, keys，涉及 `ImageTransform, DictOutput, items`；`python/sglang/srt/configs/deepseekvl2.py` modified +95/-194 (289 lines); hunks: -11,8 +11,6; -63,7 +61,6 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__，涉及 `select_best_resolution, __setitem__, VLChatProcessorOutput`；`python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0 (35 lines); hunks: -0,0 +1,35; symbols: register_customized_processor, that, MyModelConfig, decorator，涉及 `register_customized_processor, that, MyModelConfig`；`python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12 (44 lines); hunks: -54,6 +54,7; -172,6 +173,16 @@ def _load_deepseek_v32_model(; symbols: _load_deepseek_v32_model, _is_deepseek_ocr_model, get_config, get_processor，涉及 `_load_deepseek_v32_model, _is_deepseek_ocr_model, get_config`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10 (531 lines); hunks: -1,8 +1,19; -18,18 +29,59; symbols: ImageTransform, DictOutput, items, keys
  - `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194 (289 lines); hunks: -11,8 +11,6; -63,7 +61,6 @@ def __setitem__(self, key, value):; symbols: select_best_resolution, __setitem__, VLChatProcessorOutput, __call__
  - `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0 (35 lines); hunks: -0,0 +1,35; symbols: register_customized_processor, that, MyModelConfig, decorator
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12 (44 lines); hunks: -54,6 +54,7; -172,6 +173,16 @@ def _load_deepseek_v32_model(; symbols: _load_deepseek_v32_model, _is_deepseek_ocr_model, get_config, get_processor
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10; `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194; `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12415 - Feat: deepseek-ocr logits processor

- 链接: https://github.com/sgl-project/sglang/pull/12415
- 状态/时间: merged / 2025-10-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+89/-1，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat: deepseek-ocr logits processor」；模型线: DeepSeek OCR 2；类别: 模型实现调整；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py`；技术摘要: 覆盖「Feat: deepseek-ocr logits processor」；主要实现面是 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0 (22 lines); hunks: -15,6 +15,10; -26,6 +30,24; symbols: get_default_ngram_custom_params，涉及 `get_default_ngram_custom_params`；`python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1 (68 lines); hunks: -1,7 +1,7; -126,3 +126,69 @@ class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudget...; symbols: DeepSeekR1ThinkingBudgetLogitProcessor, DeepseekOCRNoRepeatNGramLogitProcessor, __call__，涉及 `DeepSeekR1ThinkingBudgetLogitProcessor, DeepseekOCRNoRepeatNGramLogitProcessor, __call__`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0 (22 lines); hunks: -15,6 +15,10; -26,6 +30,24; symbols: get_default_ngram_custom_params
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1 (68 lines); hunks: -1,7 +1,7; -126,3 +126,69 @@ class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudget...; symbols: DeepSeekR1ThinkingBudgetLogitProcessor, DeepseekOCRNoRepeatNGramLogitProcessor, __call__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0; `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12470 - Fix lint in deepseek-ocr

- 链接: https://github.com/sgl-project/sglang/pull/12470
- 状态/时间: merged / 2025-10-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix lint in deepseek-ocr」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`；技术摘要: 覆盖「Fix lint in deepseek-ocr」；主要实现面是 `python/sglang/srt/configs/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1 (1 lines); hunks: -14,7 +14,6。
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1 (1 lines); hunks: -14,7 +14,6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -14,7 +14,6 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12619 - [NPU] supports ds-ocr model on ascend

- 链接: https://github.com/sgl-project/sglang/pull/12619
- 状态/时间: open / 2025-11-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+200/-60，可读 patch 389 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] supports ds-ocr model on ascend」；模型线: DeepSeek OCR 2；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`；技术摘要: 覆盖「[NPU] supports ds-ocr model on ascend」；主要实现面是 `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek.py` modified +142/-49 (191 lines); hunks: -23,11 +23,14; -36,19 +39,21; symbols: DeepseekMLP, __init__, get_moe_weights，涉及 `DeepseekMLP, __init__, get_moe_weights`；`python/sglang/srt/models/deepseek_ocr.py` modified +58/-11 (69 lines); hunks: -30,6 +30,7; -1770,6 +1771,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek.py` modified +142/-49 (191 lines); hunks: -23,11 +23,14; -36,19 +39,21; symbols: DeepseekMLP, __init__, get_moe_weights
  - `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11 (69 lines); hunks: -30,6 +30,7; -1770,6 +1771,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +142/-49; `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17897 - Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833

- 链接: https://github.com/sgl-project/sglang/pull/17897
- 状态/时间: merged / 2026-01-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/deepseek_ocr.md`, `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`；关联提交 `84ab611af8b7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+618/-140，可读 patch 1057 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`；技术摘要: 覆盖「Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833」；主要实现面是 `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_ocr.py` modified +446/-116 (562 lines); hunks: -24,6 +24,7; -702,6 +703,7 @@ def __init__(; symbols: __init__, forward, _build_sam，涉及 `__init__, forward, _build_sam`；`python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9 (41 lines); hunks: -196,6 +196,7 @@ def __init__(; -243,6 +244,7 @@ def __init__(; symbols: __init__, process_one, tokenize_with_images，涉及 `__init__, process_one, tokenize_with_images`；`python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +8/-0 (8 lines); hunks: -12,6 +12,14 @@ class DeepseekOCRProcessor(BaseMultimodalProcessor):; symbols: DeepseekOCRProcessor, __init__，涉及 `DeepseekOCRProcessor, __init__`；`docs/basic_usage/deepseek_ocr.md` added +54/-0 (54 lines); hunks: -0,0 +1,54。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +446/-116 (562 lines); hunks: -24,6 +24,7; -702,6 +703,7 @@ def __init__(; symbols: __init__, forward, _build_sam
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9 (41 lines); hunks: -196,6 +196,7 @@ def __init__(; -243,6 +244,7 @@ def __init__(; symbols: __init__, process_one, tokenize_with_images
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +8/-0 (8 lines); hunks: -12,6 +12,14 @@ class DeepseekOCRProcessor(BaseMultimodalProcessor):; symbols: DeepseekOCRProcessor, __init__
  - `docs/basic_usage/deepseek_ocr.md` added +54/-0 (54 lines); hunks: -0,0 +1,54
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +446/-116; `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +8/-0
  - docs: `docs/basic_usage/deepseek_ocr.md` added +54/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_loader/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13561 - [XPU] Integrate MoE and minor improvements in XPU attention backend

- 链接: https://github.com/sgl-project/sglang/pull/13561
- 状态/时间: merged / 2026-02-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+233/-7，可读 patch 372 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Integrate MoE and minor improvements in XPU attention backend」；模型线: DeepSeek OCR 2；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py`；技术摘要: 覆盖「[XPU] Integrate MoE and minor improvements in XPU attention backend」；主要实现面是 `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/unquant.py` modified +49/-0 (49 lines); hunks: -32,6 +32,7; -470,6 +471,54 @@ def forward_cpu(; symbols: forward_cpu, forward_xpu, forward_npu，涉及 `forward_cpu, forward_xpu, forward_npu`；`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1 (35 lines); hunks: -20,6 +20,8; -40,6 +42,8; symbols: fused_experts_impl, fused_moe，涉及 `fused_experts_impl, fused_moe`；`python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3 (16 lines); hunks: -19,7 +19,7; -33,6 +33,7; symbols: run，涉及 `run`；`python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2 (5 lines); hunks: -5,12 +5,13。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/unquant.py` modified +49/-0 (49 lines); hunks: -32,6 +32,7; -470,6 +471,54 @@ def forward_cpu(; symbols: forward_cpu, forward_xpu, forward_npu
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1 (35 lines); hunks: -20,6 +20,8; -40,6 +42,8; symbols: fused_experts_impl, fused_moe
  - `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3 (16 lines); hunks: -19,7 +19,7; -33,6 +33,7; symbols: run
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2 (5 lines); hunks: -5,12 +5,13
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-0 (1 lines); hunks: -72,6 +72,7
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/unquant.py` modified +49/-0; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3; `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2; `python/sglang/srt/layers/moe/topk.py` modified +1/-0; `python/sglang/srt/utils/common.py` modified +11/-1
  - tests: `test/srt/xpu/test_deepseek_ocr.py` added +121/-0; `test/srt/run_suite.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/xpu/test_deepseek_ocr.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18860 - update pre-commit config

- 链接: https://github.com/sgl-project/sglang/pull/18860
- 状态/时间: merged / 2026-02-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 135 个文件，+239/-198，可读 patch 1632 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update pre-commit config」；模型线: DeepSeek OCR 2；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`；技术摘要: 覆盖「update pre-commit config」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend，涉及 `forward_decode, forward_extend`；`python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15；`python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method，涉及 `get_moe_method`；`test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend
  - `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15
  - `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2 (4 lines); hunks: -1,6 +1,6
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6; `python/sglang/srt/models/pixtral.py` modified +6/-2; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2; `python/sglang/srt/multimodal/processors/ernie45_vl.py` modified +3/-1
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1
- 验证与风险: diff 自带测试面 `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `test/manual/test_vlm_accuracy.py`, `test/registered/attention/test_triton_sliding_window.py`, `test/registered/layers/test_fla_layernorm_guard.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18774 - Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1

- 链接: https://github.com/sgl-project/sglang/pull/18774
- 状态/时间: merged / 2026-02-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-1，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1」；模型线: DeepSeek OCR 2；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`；技术摘要: 覆盖「Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1」；主要实现面是 `python/sglang/srt/models/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1 (11 lines); hunks: -1216,9 +1216,18 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1 (11 lines); hunks: -1216,9 +1216,18 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19722 - fix: align DeepSeek OCR vision dtypes

- 链接: https://github.com/sgl-project/sglang/pull/19722
- 状态/时间: open / 2026-03-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-6，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: align DeepSeek OCR vision dtypes」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`；技术摘要: 覆盖「fix: align DeepSeek OCR vision dtypes」；主要实现面是 `python/sglang/srt/models/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6 (20 lines); hunks: -1509,18 +1509,26 @@ def _collect_mm_flag(; -1612,7 +1620,7 @@ def _pixel_values_to_embedding(; symbols: _collect_mm_flag, _encode_ocr2_features, _encode_ocr1_features, _pixel_values_to_embedding，涉及 `_collect_mm_flag, _encode_ocr2_features, _encode_ocr1_features`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6 (20 lines); hunks: -1509,18 +1509,26 @@ def _collect_mm_flag(; -1612,7 +1620,7 @@ def _pixel_values_to_embedding(; symbols: _collect_mm_flag, _encode_ocr2_features, _encode_ocr1_features, _pixel_values_to_embedding
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19732 - [AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test

- 链接: https://github.com/sgl-project/sglang/pull/19732
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_ocr.py`；关联提交 `dc4380e33ac9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+23/-5，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`；技术摘要: 覆盖「[AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test」；主要实现面是 `python/sglang/srt/models/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3 (7 lines); hunks: -125,8 +125,9 @@ def isin_list(; -1685,7 +1686,7 @@ def _process_image_input(self, mm_items: List[MultimodalDa...; symbols: isin_list, _process_image_input，涉及 `isin_list, _process_image_input`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3 (7 lines); hunks: -125,8 +125,9 @@ def isin_list(; -1685,7 +1686,7 @@ def _process_image_input(self, mm_items: List[MultimodalDa...; symbols: isin_list, _process_image_input
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20708 - Add Mistral Small 4 (Pixtral) support

- 链接: https://github.com/sgl-project/sglang/pull/20708
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+360/-124，可读 patch 868 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Mistral Small 4 (Pixtral) support」；模型线: DeepSeek OCR 2；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`；技术摘要: 覆盖「Add Mistral Small 4 (Pixtral) support」；主要实现面是 `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44 (115 lines); hunks: -1,11 +1,12; -20,63 +21,47 @@ class PixtralProcessor(BaseMultimodalProcessor):; symbols: PixtralProcessor, get_patch_grid_size, __init__, defined，涉及 `PixtralProcessor, get_patch_grid_size, __init__`；`python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13 (45 lines); hunks: -333,6 +333,8 @@ def _process_messages(; -469,19 +471,20 @@ def _apply_jinja_template(; symbols: _process_messages, _apply_jinja_template, _get_history_tool_calls_cnt, _patch_mistral_skip_special_tokens，涉及 `_process_messages, _apply_jinja_template, _get_history_tool_calls_cnt`；`python/sglang/srt/parser/reasoning_parser.py` modified +28/-0 (28 lines); hunks: -450,6 +450,33 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:; -474,6 +501,7 @@ class ReasoningParser:; symbols: detect_and_parse, MistralDetector, __init__, ReasoningParser，涉及 `detect_and_parse, MistralDetector, __init__`；`python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment，涉及 `detect_and_parse, parse_streaming_increment`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44 (115 lines); hunks: -1,11 +1,12; -20,63 +21,47 @@ class PixtralProcessor(BaseMultimodalProcessor):; symbols: PixtralProcessor, get_patch_grid_size, __init__, defined
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13 (45 lines); hunks: -333,6 +333,8 @@ def _process_messages(; -469,19 +471,20 @@ def _apply_jinja_template(; symbols: _process_messages, _apply_jinja_template, _get_history_tool_calls_cnt, _patch_mistral_skip_special_tokens
  - `python/sglang/srt/parser/reasoning_parser.py` modified +28/-0 (28 lines); hunks: -450,6 +450,33 @@ def detect_and_parse(self, text: str) -> StreamingParseResult:; -474,6 +501,7 @@ class ReasoningParser:; symbols: detect_and_parse, MistralDetector, __init__, ReasoningParser
  - `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment
  - `python/sglang/srt/configs/janus_pro.py` modified +12/-12 (24 lines); hunks: -123,14 +123,14 @@ class SigLIPVisionCfg:; -595,12 +595,12 @@ def batchify(; symbols: SigLIPVisionCfg, MultiModalityConfig, __init__, batchify
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13; `python/sglang/srt/parser/reasoning_parser.py` modified +28/-0; `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9; `python/sglang/srt/configs/janus_pro.py` modified +12/-12; `python/sglang/srt/configs/jet_nemotron.py` modified +12/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/janus_pro.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12555 - [CPU] Fix MoE layer support for DeepSeek-OCR models

- 链接: https://github.com/sgl-project/sglang/pull/12555
- 状态/时间: merged / 2026-03-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+65/-10，可读 patch 110 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Fix MoE layer support for DeepSeek-OCR models」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`；技术摘要: 覆盖「[CPU] Fix MoE layer support for DeepSeek-OCR models」；主要实现面是 `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek.py` modified +31/-9 (40 lines); hunks: -48,7 +48,12; -176,14 +181,31 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: DeepseekMLP, forward，涉及 `DeepseekMLP, forward`；`python/sglang/srt/models/deepseek_ocr.py` modified +34/-1 (35 lines); hunks: -41,6 +41,10; -1772,7 +1776,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, post_load_weights，涉及 `load_weights, post_load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek.py` modified +31/-9 (40 lines); hunks: -48,7 +48,12; -176,14 +181,31 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: DeepseekMLP, forward
  - `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1 (35 lines); hunks: -41,6 +41,10; -1772,7 +1776,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, post_load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +31/-9; `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21738 - refactor: replace mm_inputs dict with MultimodalProcessorOutput

- 链接: https://github.com/sgl-project/sglang/pull/21738
- 状态/时间: merged / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 40 个文件，+408/-314，可读 patch 1321 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor: replace mm_inputs dict with MultimodalProcessorOutput」；模型线: DeepSeek OCR 2；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`；技术摘要: 覆盖「refactor: replace mm_inputs dict with MultimodalProcessorOutput」；主要实现面是 `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async，涉及 `get_mm_data, process_mm_data_async`；`python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `_process_special_format, process_mm_data_async, process_qwen_mm_data_async`；`python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async，涉及 `_process_special_format, process_mm_data_async`；`python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async，涉及 `get_mm_data, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15 (34 lines); hunks: -1,6 +1,10; -69,13 +73,13 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23; `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24; `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22; `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19; `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15; `python/sglang/srt/multimodal/processors/transformers_auto.py` modified +18/-14
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/managers/io_struct.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21735 - fix ut test_moe

- 链接: https://github.com/sgl-project/sglang/pull/21735
- 状态/时间: merged / 2026-04-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+105/-32，可读 patch 214 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix ut test_moe」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`；技术摘要: 覆盖「fix ut test_moe」；主要实现面是 `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26 (54 lines); hunks: -2,9 +2,11; -19,11 +21,32; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass，涉及 `TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass`；`test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: TestDeepSeekOCRTriton, setUpClass，涉及 `TestDeepSeekOCRTriton, setUpClass`；`test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5 (29 lines); hunks: -3,6 +3,7; -15,26 +16,44; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper，涉及 `_cleanup_xpu_memory, intel_xpu_benchmark, decorator`；`test/srt/run_suite.py` modified +2/-1 (3 lines); hunks: -77,7 +77,8。
- 代码 diff 细节:
  - `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26 (54 lines); hunks: -2,9 +2,11; -19,11 +21,32; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass
  - `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: TestDeepSeekOCRTriton, setUpClass
  - `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5 (29 lines); hunks: -3,6 +3,7; -15,26 +16,44; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper
  - `test/srt/run_suite.py` modified +2/-1 (3 lines); hunks: -77,7 +77,8
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26; `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0; `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5; `test/srt/run_suite.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- 链接: https://github.com/sgl-project/sglang/pull/23001
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 330 个文件，+80364/-0，可读 patch 68714 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add new Mintlify documentation site (docs_new/)」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`；技术摘要: 覆盖「Add new Mintlify documentation site (docs_new/)」；主要实现面是 `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23044 - [XPU] Fix DeepSeek-OCR tests under transformers 5.x

- 链接: https://github.com/sgl-project/sglang/pull/23044
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-8，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Fix DeepSeek-OCR tests under transformers 5.x」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`；技术摘要: 覆盖「[XPU] Fix DeepSeek-OCR tests under transformers 5.x」；主要实现面是 `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4 (6 lines); hunks: -9,9 +9,9; -38,9 +38,7 @@ def _cleanup_xpu_memory(cls):; symbols: _cleanup_xpu_memory, setUpClass，涉及 `_cleanup_xpu_memory, setUpClass`；`test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4 (6 lines); hunks: -7,8 +7,8; -21,9 +21,7 @@ class TestDeepSeekOCRTriton(deepseek_ocr.TestDeepSeekOCR):; symbols: TestDeepSeekOCRTriton, setUpClass，涉及 `TestDeepSeekOCRTriton, setUpClass`。
- 代码 diff 细节:
  - `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4 (6 lines); hunks: -9,9 +9,9; -38,9 +38,7 @@ def _cleanup_xpu_memory(cls):; symbols: _cleanup_xpu_memory, setUpClass
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4 (6 lines); hunks: -7,8 +7,8; -21,9 +21,7 @@ class TestDeepSeekOCRTriton(deepseek_ocr.TestDeepSeekOCR):; symbols: TestDeepSeekOCRTriton, setUpClass
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4
- 验证与风险: diff 自带测试面 `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- 链接: https://github.com/sgl-project/sglang/pull/23337
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 179 个文件，+16004/-8152，可读 patch 23604 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Sync docs_new with legacy docs and update migration redirects」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`；技术摘要: 覆盖「[Docs] Sync docs_new with legacy docs and update migration redirects」；主要实现面是 `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23820 - Update XPU Docker runtime stack & hf_home config

- 链接: https://github.com/sgl-project/sglang/pull/23820
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+38/-54，可读 patch 204 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update XPU Docker runtime stack & hf_home config」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `test/srt/xpu/test_intel_xpu_backend.py`, `test/srt/xpu/test_deepseek_ocr.py`, `docker/xpu.Dockerfile`；技术摘要: 覆盖「Update XPU Docker runtime stack & hf_home config」；主要实现面是 `test/srt/xpu/test_intel_xpu_backend.py`, `test/srt/xpu/test_deepseek_ocr.py`, `docker/xpu.Dockerfile`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/xpu/test_intel_xpu_backend.py` modified +12/-25 (37 lines); hunks: -3,7 +3,6; -16,29 +15,17; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper，涉及 `_cleanup_xpu_memory, intel_xpu_benchmark, decorator`；`test/srt/xpu/test_deepseek_ocr.py` modified +6/-17 (23 lines); hunks: -2,7 +2,6; -21,22 +20,8; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass，涉及 `TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass`；`docker/xpu.Dockerfile` modified +10/-6 (16 lines); hunks: -20,6 +20,16 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +48,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda...；`test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3 (10 lines); hunks: -6,7 +6,7; -16,10 +16,11; symbols: TestDeepSeekOCRTriton, setUpClass, here，涉及 `TestDeepSeekOCRTriton, setUpClass, here`。
- 代码 diff 细节:
  - `test/srt/xpu/test_intel_xpu_backend.py` modified +12/-25 (37 lines); hunks: -3,7 +3,6; -16,29 +15,17; symbols: _cleanup_xpu_memory, intel_xpu_benchmark, decorator, wrapper
  - `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17 (23 lines); hunks: -2,7 +2,6; -21,22 +20,8; symbols: TestDeepSeekOCR, _cleanup_xpu_memory, setUpClass, tearDownClass
  - `docker/xpu.Dockerfile` modified +10/-6 (16 lines); hunks: -20,6 +20,16 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +48,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda...
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3 (10 lines); hunks: -6,7 +6,7; -16,10 +16,11; symbols: TestDeepSeekOCRTriton, setUpClass, here
  - `.github/workflows/pr-test-xpu.yml` modified +3/-3 (6 lines); hunks: -72,8 +72,6 @@ jobs:; -99,8 +97,10 @@ jobs:
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/srt/xpu/test_intel_xpu_backend.py` modified +12/-25; `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3
  - other: `docker/xpu.Dockerfile` modified +10/-6
  - ci: `.github/workflows/pr-test-xpu.yml` modified +3/-3
- 验证与风险: diff 自带测试面 `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`, `test/srt/xpu/test_intel_xpu_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- 链接: https://github.com/sgl-project/sglang/pull/25182
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 136 个文件，+255/-0，可读 patch 872 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「chore: add vLLM SPDX copyright headers to ported files」；模型线: DeepSeek OCR 2；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`；技术摘要: 覆盖「chore: add vLLM SPDX copyright headers to ported files」；主要实现面是 `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7；`python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7；`python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6；`python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6。
- 代码 diff 细节:
  - `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma2.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +4/-0; `python/sglang/srt/models/commandr.py` modified +4/-0; `python/sglang/srt/models/dbrx.py` modified +3/-0; `python/sglang/srt/models/gemma.py` modified +3/-0; `python/sglang/srt/models/gemma2.py` modified +3/-0; `python/sglang/srt/models/gpt_bigcode.py` modified +3/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_custom_ops.py`, `python/sglang/test/test_marlin_utils.py`, `sgl-kernel/tests/test_causal_conv1d.py`, `test/registered/layers/mamba/test_causal_conv1d.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25364 - Add Accuracy Benchmark for OCR models

- 链接: https://github.com/sgl-project/sglang/pull/25364
- 状态/时间: open / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+1928/-0，可读 patch 1946 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Accuracy Benchmark for OCR models」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `benchmark/ocr/bench_sglang.py`, `benchmark/ocr/eval_utils.py`, `benchmark/ocr/generate_report.py`；技术摘要: 覆盖「Add Accuracy Benchmark for OCR models」；主要实现面是 `benchmark/ocr/bench_sglang.py`, `benchmark/ocr/eval_utils.py`, `benchmark/ocr/generate_report.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `benchmark/ocr/bench_sglang.py` added +735/-0 (735 lines); hunks: -0,0 +1,735; symbols: BenchArgs, add_cli_args, preflight_check, pdf_page_to_base64_png，涉及 `BenchArgs, add_cli_args, preflight_check`；`benchmark/ocr/eval_utils.py` added +631/-0 (631 lines); hunks: -0,0 +1,631; symbols: normalize_text, strip_markdown, fuzzy_contains, exact_contains，涉及 `normalize_text, strip_markdown, fuzzy_contains`；`benchmark/ocr/generate_report.py` added +382/-0 (382 lines); hunks: -0,0 +1,382; symbols: _ocr_to_html, _latex_to_display, _render_sample, generate_report，涉及 `_ocr_to_html, _latex_to_display, _render_sample`；`benchmark/ocr/README.md` added +178/-0 (178 lines); hunks: -0,0 +1,178。
- 代码 diff 细节:
  - `benchmark/ocr/bench_sglang.py` added +735/-0 (735 lines); hunks: -0,0 +1,735; symbols: BenchArgs, add_cli_args, preflight_check, pdf_page_to_base64_png
  - `benchmark/ocr/eval_utils.py` added +631/-0 (631 lines); hunks: -0,0 +1,631; symbols: normalize_text, strip_markdown, fuzzy_contains, exact_contains
  - `benchmark/ocr/generate_report.py` added +382/-0 (382 lines); hunks: -0,0 +1,382; symbols: _ocr_to_html, _latex_to_display, _render_sample, generate_report
  - `benchmark/ocr/README.md` added +178/-0 (178 lines); hunks: -0,0 +1,178
  - `python/pyproject.toml` modified +1/-0 (1 lines); hunks: -85,6 +85,7 @@ dependencies = [
- 关键代码摘录:

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

- 已读文件:
  - other: `benchmark/ocr/bench_sglang.py` added +735/-0; `benchmark/ocr/eval_utils.py` added +631/-0; `benchmark/ocr/generate_report.py` added +382/-0; `benchmark/ocr/README.md` added +178/-0
  - runtime: `python/pyproject.toml` modified +1/-0; `python/pyproject_xpu.toml` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/pyproject.toml`, `python/pyproject_xpu.toml`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25257 - [NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2

- 链接: https://github.com/sgl-project/sglang/pull/25257
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-3，可读 patch 42 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2」；模型线: DeepSeek OCR 2；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek.py`；技术摘要: 覆盖「[NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2」；主要实现面是 `python/sglang/srt/models/deepseek.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek.py` modified +11/-3 (14 lines); hunks: -39,7 +39,6; -50,14 +49,23; symbols: DeepseekMLP, forward，涉及 `DeepseekMLP, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek.py` modified +11/-3 (14 lines); hunks: -39,7 +39,6; -50,14 +49,23; symbols: DeepseekMLP, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +11/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- 链接: https://github.com/sgl-project/sglang/pull/24751
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+45/-44，可读 patch 401 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`；技术摘要: 覆盖「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；主要实现面是 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data，涉及 `_process_loaded_mm_data, load_mm_data`；`python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `_process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async`；`python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async，涉及 `_process_special_format, process_mm_data_async`；`python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async，涉及 `__init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/clip.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25589 - [Fix] DeepSeek-OCR-2 bench_serving: fix processor loading and GPU JPEG tensor conversion

- 链接: https://github.com/sgl-project/sglang/pull/25589
- 状态/时间: closed / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-11，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] DeepSeek-OCR-2 bench_serving: fix processor loading and GPU JPEG tensor conversion」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/benchmark/utils.py`；技术摘要: 覆盖「[Fix] DeepSeek-OCR-2 bench_serving: fix processor loading and GPU JPEG tensor conversion」；主要实现面是 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/benchmark/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/deepseek_ocr.py` modified +5/-0 (5 lines); hunks: -454,6 +454,11 @@ def tokenize_with_images(; symbols: tokenize_with_images，涉及 `tokenize_with_images`；`python/sglang/benchmark/utils.py` modified +7/-11 (18 lines); hunks: -71,20 +71,16 @@ def get_processor(; symbols: get_processor, download_and_cache_hf_file，涉及 `get_processor, download_and_cache_hf_file`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +5/-0 (5 lines); hunks: -454,6 +454,11 @@ def tokenize_with_images(; symbols: tokenize_with_images
  - `python/sglang/benchmark/utils.py` modified +7/-11 (18 lines); hunks: -71,20 +71,16 @@ def get_processor(; symbols: get_processor, download_and_cache_hf_file
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +5/-0; `python/sglang/benchmark/utils.py` modified +7/-11
- 验证与风险: runtime 路径改动集中在 `python/sglang/benchmark/utils.py`, `python/sglang/srt/configs/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24701 - [FIX][1/2] fix step3-vl/deepseek-ocr image processor error

- 链接: https://github.com/sgl-project/sglang/pull/24701
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+67/-20，可读 patch 160 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FIX][1/2] fix step3-vl/deepseek-ocr image processor error」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/step3_vl.py`；技术摘要: 覆盖「[FIX][1/2] fix step3-vl/deepseek-ocr image processor error」；主要实现面是 `python/sglang/srt/multimodal/processors/step3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20 (87 lines); hunks: -8,6 +8,7; -20,14 +21,37; symbols: GPUToTensor, forward, __call__, ImagePatcher，涉及 `GPUToTensor, forward, __call__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20 (87 lines); hunks: -8,6 +8,7; -20,14 +21,37; symbols: GPUToTensor, forward, __call__, ImagePatcher
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/step3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25403 - [FIX][2/2] fix step3-vl/deepseek-ocr image processor error

- 链接: https://github.com/sgl-project/sglang/pull/25403
- 状态/时间: merged / 2026-05-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+100/-12，可读 patch 184 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FIX][2/2] fix step3-vl/deepseek-ocr image processor error」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`；技术摘要: 覆盖「[FIX][2/2] fix step3-vl/deepseek-ocr image processor error」；主要实现面是 `python/sglang/srt/configs/deepseek_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/deepseek_ocr.py` modified +100/-12 (112 lines); hunks: -1,9 +1,11; -18,6 +20,8; symbols: get_default_ngram_custom_params, get_image_size, resize_image, crop_image，涉及 `get_default_ngram_custom_params, get_image_size, resize_image`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +100/-12 (112 lines); hunks: -1,9 +1,11; -18,6 +20,8; symbols: get_default_ngram_custom_params, get_image_size, resize_image, crop_image
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +100/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25405 - [XPU] Add registry mechanism for XPU CI tests

- 链接: https://github.com/sgl-project/sglang/pull/25405
- 状态/时间: merged / 2026-05-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+156/-22，可读 patch 310 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Add registry mechanism for XPU CI tests」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test-xpu.yml`, `test/registered/xpu/test_xpu_basic.py`, `test/srt/run_suite.py`；技术摘要: 覆盖「[XPU] Add registry mechanism for XPU CI tests」；主要实现面是 `.github/workflows/pr-test-xpu.yml`, `test/registered/xpu/test_xpu_basic.py`, `test/srt/run_suite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/pr-test-xpu.yml` modified +79/-9 (88 lines); hunks: -68,7 +68,8 @@ jobs:; -108,16 +109,80 @@ jobs:；`test/registered/xpu/test_xpu_basic.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: TestXPUBasic, test_basic_generation，涉及 `TestXPUBasic, test_basic_generation`；`test/srt/run_suite.py` modified +4/-10 (14 lines); hunks: -87,16 +87,10；`test/run_suite.py` modified +7/-1 (8 lines); hunks: -20,6 +20,7; -77,6 +78,10; symbols: _valid_suites_by_backend，涉及 `_valid_suites_by_backend`。
- 代码 diff 细节:
  - `.github/workflows/pr-test-xpu.yml` modified +79/-9 (88 lines); hunks: -68,7 +68,8 @@ jobs:; -108,16 +109,80 @@ jobs:
  - `test/registered/xpu/test_xpu_basic.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: TestXPUBasic, test_basic_generation
  - `test/srt/run_suite.py` modified +4/-10 (14 lines); hunks: -87,16 +87,10
  - `test/run_suite.py` modified +7/-1 (8 lines); hunks: -20,6 +20,7; -77,6 +78,10; symbols: _valid_suites_by_backend
  - `test/registered/xpu/test_deepseek_ocr_triton.py` renamed +7/-0 (7 lines); hunks: -9,12 +9,19
- 关键代码摘录:

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

- 已读文件:
  - ci: `.github/workflows/pr-test-xpu.yml` modified +79/-9
  - tests: `test/registered/xpu/test_xpu_basic.py` added +47/-0; `test/srt/run_suite.py` modified +4/-10; `test/run_suite.py` modified +7/-1; `test/registered/xpu/test_deepseek_ocr_triton.py` renamed +7/-0; `python/sglang/test/ci/ci_register.py` modified +5/-1; `test/registered/xpu/test_deepseek_ocr.py` renamed +3/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/ci/ci_register.py`, `test/registered/attention/test_chunk_gated_delta_rule.py`, `test/registered/xpu/test_deepseek_ocr.py`, `test/registered/xpu/test_deepseek_ocr_triton.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- 链接: https://github.com/sgl-project/sglang/pull/25813
- 状态/时间: merged / 2026-06-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+1262/-2154，可读 patch 4187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): port popular model usage guides into cookbook pages」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`；技术摘要: 覆盖「docs(cookbook): port popular model usage guides into cookbook pages」；主要实现面是 `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27248 - [Doc][CPU]Update Cookbook with Xeon support info

- 链接: https://github.com/sgl-project/sglang/pull/27248
- 状态/时间: merged / 2026-06-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+443/-121，可读 patch 1524 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc][CPU]Update Cookbook with Xeon support info」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`；技术摘要: 覆盖「[Doc][CPU]Update Cookbook with Xeon support info」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27527 - Vectorize _create_custom_4d_mask in CustomQwen2Decoder

- 链接: https://github.com/sgl-project/sglang/pull/27527
- 状态/时间: merged / 2026-06-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+536/-20，可读 patch 569 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Vectorize _create_custom_4d_mask in CustomQwen2Decoder」；模型线: DeepSeek OCR 2；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`；技术摘要: 覆盖「Vectorize _create_custom_4d_mask in CustomQwen2Decoder」；主要实现面是 `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_ocr.py` modified +23/-20 (43 lines); hunks: -1295,31 +1295,34 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask，涉及 `_create_custom_4d_mask`；`test/manual/test_create_custom_4d_mask.py` added +513/-0 (513 lines); hunks: -0,0 +1,513; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new, _make_token_type_ids, _make_random_token_type_ids，涉及 `_create_custom_4d_mask_reference, _create_custom_4d_mask_new, _make_token_type_ids`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +23/-20 (43 lines); hunks: -1295,31 +1295,34 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask
  - `test/manual/test_create_custom_4d_mask.py` added +513/-0 (513 lines); hunks: -0,0 +1,513; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new, _make_token_type_ids, _make_random_token_type_ids
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +23/-20
  - tests: `test/manual/test_create_custom_4d_mask.py` added +513/-0
- 验证与风险: diff 自带测试面 `test/manual/test_create_custom_4d_mask.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28988 - [CI] Fix lint brought by #27527

- 链接: https://github.com/sgl-project/sglang/pull/28988
- 状态/时间: merged / 2026-06-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-49，可读 patch 277 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Fix lint brought by #27527」；模型线: DeepSeek OCR 2；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`；技术摘要: 覆盖「[CI] Fix lint brought by #27527」；主要实现面是 `python/sglang/srt/models/deepseek_ocr.py`, `test/manual/test_create_custom_4d_mask.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_ocr.py` modified +3/-3 (6 lines); hunks: -1297,7 +1297,7 @@ def _create_custom_4d_mask(; -1312,8 +1312,8 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask，涉及 `_create_custom_4d_mask`；`test/manual/test_create_custom_4d_mask.py` modified +65/-46 (111 lines); hunks: -43,6 +43,7; -58,7 +59,7 @@ def _create_custom_4d_mask_reference(; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new，涉及 `_create_custom_4d_mask_reference, _create_custom_4d_mask_new`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +3/-3 (6 lines); hunks: -1297,7 +1297,7 @@ def _create_custom_4d_mask(; -1312,8 +1312,8 @@ def _create_custom_4d_mask(; symbols: _create_custom_4d_mask
  - `test/manual/test_create_custom_4d_mask.py` modified +65/-46 (111 lines); hunks: -43,6 +43,7; -58,7 +59,7 @@ def _create_custom_4d_mask_reference(; symbols: _create_custom_4d_mask_reference, _create_custom_4d_mask_new
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +3/-3
  - tests: `test/manual/test_create_custom_4d_mask.py` modified +65/-46
- 验证与风险: diff 自带测试面 `test/manual/test_create_custom_4d_mask.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
