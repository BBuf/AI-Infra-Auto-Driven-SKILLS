# vllm InternVL 3.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/pooling/test_intern_vit.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/processing/test_internvl.py` | [#12553](https://github.com/vllm-project/vllm/pull/12553), [#37260](https://github.com/vllm-project/vllm/pull/37260) |
| `vllm/model_executor/models/intern_vit.py` | [#6514](https://github.com/vllm-project/vllm/pull/6514), [#7067](https://github.com/vllm-project/vllm/pull/7067), [#9528](https://github.com/vllm-project/vllm/pull/9528), [#23909](https://github.com/vllm-project/vllm/pull/23909), [#38049](https://github.com/vllm-project/vllm/pull/38049) |
| `vllm/model_executor/models/internvl.py` | [#6514](https://github.com/vllm-project/vllm/pull/6514), [#7067](https://github.com/vllm-project/vllm/pull/7067), [#7164](https://github.com/vllm-project/vllm/pull/7164), [#7860](https://github.com/vllm-project/vllm/pull/7860), [#8201](https://github.com/vllm-project/vllm/pull/8201), [#8250](https://github.com/vllm-project/vllm/pull/8250), [#8299](https://github.com/vllm-project/vllm/pull/8299), [#8375](https://github.com/vllm-project/vllm/pull/8375), [#8614](https://github.com/vllm-project/vllm/pull/8614), [#8946](https://github.com/vllm-project/vllm/pull/8946), [#9351](https://github.com/vllm-project/vllm/pull/9351), [#9528](https://github.com/vllm-project/vllm/pull/9528), ... (29 total) |
| `vllm/transformers_utils/processors/internvl.py` | [#37260](https://github.com/vllm-project/vllm/pull/37260), [#37324](https://github.com/vllm-project/vllm/pull/37324) |

## PR 覆盖总览

- git 追溯 PR 数: 30
- 原文档显式引用补充 PR 数: 7
- 当前文档总 PR 数: 37
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-07-29 | [#6514](https://github.com/vllm-project/vllm/pull/6514) | merged | [Model] Initialize support for InternVL2 series models | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py` |
| 2024-08-03 | [#7067](https://github.com/vllm-project/vllm/pull/7067) | merged | [Model] Refactor and decouple weight loading logic for InternVL2 model | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py` |
| 2024-08-07 | [#7164](https://github.com/vllm-project/vllm/pull/7164) | merged | [Bugfix] Fix input processor for InternVL2 model | `vllm/model_executor/models/internvl.py`, `tests/models/test_internvl.py` |
| 2024-09-05 | [#7860](https://github.com/vllm-project/vllm/pull/7860) | merged | Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism) | `vllm/model_executor/models/internvl.py` |
| 2024-09-07 | [#8201](https://github.com/vllm-project/vllm/pull/8201) | merged | [Model][VLM] Support multi-images inputs for InternVL2 models | `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-09-11 | [#8299](https://github.com/vllm-project/vllm/pull/8299) | merged | [Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel | `vllm/model_executor/models/internvl.py` |
| 2024-09-12 | [#8375](https://github.com/vllm-project/vllm/pull/8375) | merged | [Bugfix] Fix InternVL2 inference with various num_patches | `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-09-25 | [#8250](https://github.com/vllm-project/vllm/pull/8250) | merged | [BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv | `vllm/model_executor/models/internvl.py` |
| 2024-09-25 | [#8614](https://github.com/vllm-project/vllm/pull/8614) | merged | [VLM][Bugfix] enable internvl running with num_scheduler_steps > 1 | `vllm/model_executor/models/internvl.py` |
| 2024-09-30 | [#8946](https://github.com/vllm-project/vllm/pull/8946) | merged | [Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg | `vllm/model_executor/models/internvl.py` |
| 2024-10-15 | [#9351](https://github.com/vllm-project/vllm/pull/9351) | merged | [Bugfix] Update InternVL input mapper to support image embeds | `vllm/model_executor/models/internvl.py` |
| 2024-10-22 | [#9528](https://github.com/vllm-project/vllm/pull/9528) | merged | [Model][VLM] Initialize support for Mono-InternVL model | `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/decoder_only/vision_language/test_internvl.py` |
| 2024-11-21 | [#10518](https://github.com/vllm-project/vllm/pull/10518) | merged | [Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models | `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2024-12-13 | [#11165](https://github.com/vllm-project/vllm/pull/11165) | merged | [V1][VLM] Fix edge case bug for InternVL2 | `vllm/model_executor/models/internvl.py` |
| 2025-02-04 | [#12553](https://github.com/vllm-project/vllm/pull/12553) | merged | [VLM] Merged multi-modal processor for InternVL-based models | `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py` |
| 2025-03-13 | [#14738](https://github.com/vllm-project/vllm/pull/14738) | merged | [VLM] Support loading InternVideo2.5 models as original InternVLChatModel | `vllm/model_executor/models/internvl.py` |
| 2025-03-20 | [#15086](https://github.com/vllm-project/vllm/pull/15086) | merged | [Bugfix] Fix embedding assignment for InternVL-based models | `vllm/model_executor/models/internvl.py` |
| 2025-05-25 | [#18499](https://github.com/vllm-project/vllm/pull/18499) | merged | [VLM] Initialize video input support for InternVL models | `vllm/model_executor/models/internvl.py` |
| 2025-05-29 | [#18842](https://github.com/vllm-project/vllm/pull/18842) | merged | [LoRA] Add LoRA support for InternVL | `vllm/model_executor/models/internvl.py` |
| 2025-07-29 | [#21684](https://github.com/vllm-project/vllm/pull/21684) | merged | Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema | `vllm/model_executor/models/internvl.py` |
| 2025-08-26 | [#23658](https://github.com/vllm-project/vllm/pull/23658) | merged | [Model] Enable video support for InternVL3.5 models | `vllm/model_executor/models/internvl.py` |
| 2025-08-27 | [#23742](https://github.com/vllm-project/vllm/pull/23742) | merged | [Model] Enable native HF format InternVL support | `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2025-09-10 | [#24519](https://github.com/vllm-project/vllm/pull/24519) | merged | [Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention. | `vllm/model_executor/models/internvl.py` |
| 2025-09-18 | [#23909](https://github.com/vllm-project/vllm/pull/23909) | merged | [Model] enable data parallel for InternVL vision encoder | `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py` |
| 2025-10-03 | [#26153](https://github.com/vllm-project/vllm/pull/26153) | merged | [Model] Use `merge_by_field_config` for MM models (InternVL family) | `vllm/model_executor/models/internvl.py` |
| 2026-01-23 | [#32397](https://github.com/vllm-project/vllm/pull/32397) | merged | [Model] Enable LoRA support for internvl2 | `vllm/model_executor/models/internvl.py` |
| 2026-03-17 | [#37260](https://github.com/vllm-project/vllm/pull/37260) | merged | [1/2] Move InternVL-based processors | `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py` |
| 2026-03-18 | [#37324](https://github.com/vllm-project/vllm/pull/37324) | merged | [2/3] Refactor InternVL-based processors | `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py` |
| 2026-03-25 | [#35182](https://github.com/vllm-project/vllm/pull/35182) | merged | [Misc] Reorganize inputs | `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py` |
| 2026-03-26 | [#38049](https://github.com/vllm-project/vllm/pull/38049) | merged | [Model] Add torch.compile support for InternVL vision encoder | `vllm/model_executor/models/intern_vit.py` |
| 2026-04-15 | [#38901](https://github.com/vllm-project/vllm/pull/38901) | merged | refactor hard coded device string in test files under tests/compile tests/quantization tests/models and tests/model_executor | `tests/models/multimodal/pooling/test_intern_vit.py`, `tests/models/multimodal/pooling/test_radio.py`, `tests/models/test_utils.py` |
| 2026-04-15 | [#30566](https://github.com/vllm-project/vllm/pull/30566) | merged | Update to transformers v5 | `tests/models/registry.py`, `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/generation/test_common.py` |
| 2026-05-19 | [#42347](https://github.com/vllm-project/vllm/pull/42347) | merged | [Perf][4/n] Eliminate various GPU CPU syncs | `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py` |
| 2026-06-04 | [#41759](https://github.com/vllm-project/vllm/pull/41759) | merged | [MM][Perf][CG] Support ViT full CUDA graph for InternVL | `vllm/model_executor/models/internvl.py` |
| 2026-06-12 | [#45129](https://github.com/vllm-project/vllm/pull/45129) | merged | [Model] Remove Mono-InternVL (InternLM2VEForCausalLM) | `vllm/model_executor/models/internvl.py` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | merged | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |
| 2026-06-18 | [#42727](https://github.com/vllm-project/vllm/pull/42727) | merged | fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config | `vllm/model_executor/layers/quantization/auto_awq.py`, `vllm/model_executor/layers/quantization/awq.py`, `vllm/model_executor/layers/quantization/moe_wna16.py` |

## 逐 PR diff 审计卡

### PR #6514 - [Model] Initialize support for InternVL2 series models

- 链接: https://github.com/vllm-project/vllm/pull/6514
- 状态/时间: merged / 2024-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `7cbd9ec7a9bf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1042/-6，可读 patch 1164 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Initialize support for InternVL2 series models」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py`；技术摘要: 覆盖「[Model] Initialize support for InternVL2 series models」；主要实现面是 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/test_internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks，涉及 `InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio`；`vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward，涉及 `InternVisionEmbeddings, __init__, _get_pos_embed`；`tests/models/test_internvl.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: InternVLProcessor, __init__, __call__, generate，涉及 `InternVLProcessor, __init__, __call__`；`vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: InternVLChatConfig, __init__，涉及 `InternVLChatConfig, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: InternVLImagePixelInputs, build_transform, find_closest_aspect_ratio, calculate_num_blocks
  - `vllm/model_executor/models/intern_vit.py` added +270/-0 (270 lines); hunks: -0,0 +1,270; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward
  - `tests/models/test_internvl.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: InternVLProcessor, __init__, __call__, generate
  - `vllm/transformers_utils/configs/internvl.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: InternVLChatConfig, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -0,0 +1,471 @@
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
+# Licensed under The MIT License [see LICENSE for details]
+# --------------------------------------------------------
diff -- vllm/model_executor/models/intern_vit.py
@@ -0,0 +1,270 @@
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
+# Licensed under The MIT License [see LICENSE for details]
+# --------------------------------------------------------
diff -- tests/models/test_internvl.py
@@ -0,0 +1,201 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` added +471/-0; `vllm/model_executor/models/intern_vit.py` added +270/-0; `vllm/transformers_utils/configs/internvl.py` added +51/-0
  - tests: `tests/models/test_internvl.py` added +201/-0
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7067 - [Model] Refactor and decouple weight loading logic for InternVL2 model

- 链接: https://github.com/vllm-project/vllm/pull/7067
- 状态/时间: merged / 2024-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `0c25435daa0a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+38/-55，可读 patch 123 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Refactor and decouple weight loading logic for InternVL2 model」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`；技术摘要: 覆盖「[Model] Refactor and decouple weight loading logic for InternVL2 model」；主要实现面是 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +28/-54 (82 lines); hunks: -4,6 +4,7; -414,58 +415,31 @@ def sample(; symbols: sample, load_weights, _filter_weights，涉及 `sample, load_weights, _filter_weights`；`vllm/model_executor/models/intern_vit.py` modified +10/-1 (11 lines); hunks: -4,7 +4,7; -16,6 +16,7; symbols: forward, load_weights，涉及 `forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +28/-54 (82 lines); hunks: -4,6 +4,7; -414,58 +415,31 @@ def sample(; symbols: sample, load_weights, _filter_weights
  - `vllm/model_executor/models/intern_vit.py` modified +10/-1 (11 lines); hunks: -4,7 +4,7; -16,6 +16,7; symbols: forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -4,6 +4,7 @@
+import itertools
@@ -414,58 +415,31 @@ def sample(
-    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        stacked_params_mapping = [
-            # (param_name, shard_name, shard_id)
-            (".qkv_proj", ".q_proj", "q"),
diff -- vllm/model_executor/models/intern_vit.py
@@ -4,7 +4,7 @@
-from typing import Optional
+from typing import Iterable, Optional, Tuple
@@ -16,6 +16,7 @@
+from vllm.model_executor.model_loader.weight_utils import default_weight_loader
@@ -268,3 +269,11 @@ def forward(
+    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +28/-54; `vllm/model_executor/models/intern_vit.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7164 - [Bugfix] Fix input processor for InternVL2 model

- 链接: https://github.com/vllm-project/vllm/pull/7164
- 状态/时间: merged / 2024-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `b764547616e6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+73/-34，可读 patch 211 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix input processor for InternVL2 model」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/internvl.py`, `tests/models/test_internvl.py`；技术摘要: 覆盖「[Bugfix] Fix input processor for InternVL2 model」；主要实现面是 `vllm/model_executor/models/internvl.py`, `tests/models/test_internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +54/-30 (84 lines); hunks: -38,9 +38,6; -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, wi...; symbols: InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks，涉及 `InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks`；`tests/models/test_internvl.py` modified +19/-4 (23 lines); hunks: -5,6 +5,7; -26,10 +27,15; symbols: __init__, __call__，涉及 `__init__, __call__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +54/-30 (84 lines); hunks: -38,9 +38,6; -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, wi...; symbols: InternVLImagePixelInputs, find_closest_aspect_ratio, calculate_num_blocks
  - `tests/models/test_internvl.py` modified +19/-4 (23 lines); hunks: -5,6 +5,7; -26,10 +27,15; symbols: __init__, __call__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -38,9 +38,6 @@
-MAX_IMAGE_FEATURE_SIZE_WIDTH = 3000
-MAX_IMAGE_FEATURE_SIZE_HEIGHT = 500
@@ -84,11 +81,9 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
-def calculate_num_blocks(orig_width: int,
-                         orig_height: int,
-                         min_num=1,
diff -- tests/models/test_internvl.py
@@ -5,6 +5,7 @@
+from transformers import AutoConfig
@@ -26,10 +27,15 @@
+DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]
-    snapshot_download("OpenGVLab/InternVL2-1B"),
-    snapshot_download("OpenGVLab/InternVL2-2B"),
-    # snapshot_download("OpenGVLab/InternVL2-4B"),  # broken
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +54/-30
  - tests: `tests/models/test_internvl.py` modified +19/-4
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7860 - Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism)

- 链接: https://github.com/vllm-project/vllm/pull/7860
- 状态/时间: merged / 2024-09-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `8685ba1a1ec0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+90/-35，可读 patch 266 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism)」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「Inclusion of InternVLChatModel In PP_SUPPORTED_MODELS(Pipeline Parallelism)」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +3/-1 (4 lines); hunks: -341,6 +341,8 @@ def __init__(self,; -461,7 +463,7 @@ def forward(; symbols: __init__, pixel_shuffle, forward，涉及 `__init__, pixel_shuffle, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +3/-1 (4 lines); hunks: -341,6 +341,8 @@ def __init__(self,; -461,7 +463,7 @@ def forward(; symbols: __init__, pixel_shuffle, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -341,6 +341,8 @@ def __init__(self,
+        self.make_empty_intermediate_tensors = (
+            self.language_model.make_empty_intermediate_tensors)
@@ -461,7 +463,7 @@ def forward(
-                                                  None,
+                                                  intermediate_tensors,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +3/-1
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`, `tests/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8201 - [Model][VLM] Support multi-images inputs for InternVL2 models

- 链接: https://github.com/vllm-project/vllm/pull/8201
- 状态/时间: merged / 2024-09-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `e807125936a9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+199/-57，可读 patch 482 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][VLM] Support multi-images inputs for InternVL2 models」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model][VLM] Support multi-images inputs for InternVL2 models」；主要实现面是 `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/test_internvl.py` modified +73/-19 (92 lines); hunks: -1,5 +1,5; -9,7 +9,8; symbols: generate, run_test, __init__, __call__，涉及 `generate, run_test, __init__`；`vllm/model_executor/models/internvl.py` modified +46/-14 (60 lines); hunks: -5,6 +5,7; -26,6 +27,7; symbols: find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess, input_processor_for_internvl，涉及 `find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess`。
- 代码 diff 细节:
  - `tests/models/test_internvl.py` modified +73/-19 (92 lines); hunks: -1,5 +1,5; -9,7 +9,8; symbols: generate, run_test, __init__, __call__
  - `vllm/model_executor/models/internvl.py` modified +46/-14 (60 lines); hunks: -5,6 +5,7; -26,6 +27,7; symbols: find_closest_aspect_ratio, calculate_num_blocks, dynamic_preprocess, input_processor_for_internvl
- 关键代码摘录:

```diff
diff -- tests/models/test_internvl.py
@@ -1,5 +1,5 @@
-from typing import List, Optional, Tuple, Type
+from typing import List, Optional, Tuple, Type, Union
@@ -9,7 +9,8 @@
-from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
+from ..conftest import (IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner,
+                        _ImageAssets)
diff -- vllm/model_executor/models/internvl.py
@@ -5,6 +5,7 @@
+import re
@@ -26,6 +27,7 @@
+from vllm.utils import is_list_of
@@ -95,8 +97,8 @@ def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
-                         max_num: int,
-                         image_size: int) -> Tuple[int, int, int]:
```

- 已读文件:
  - tests: `tests/models/test_internvl.py` modified +73/-19
  - runtime: `vllm/model_executor/models/internvl.py` modified +46/-14
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`, `tests/models/test_phi3v.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8299 - [Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel

- 链接: https://github.com/vllm-project/vllm/pull/8299
- 状态/时间: merged / 2024-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `1230263e161c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-3，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Bugfix] Fix InternVL2 vision embeddings process with pipeline parallel」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +2/-1 (3 lines); hunks: -17,6 +17,7; -480,7 +481,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +2/-1 (3 lines); hunks: -17,6 +17,7; -480,7 +481,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -17,6 +17,7 @@
+from vllm.distributed import get_pp_group
@@ -480,7 +481,7 @@ def forward(
-        if image_input is not None:
+        if image_input is not None and get_pp_group().is_first_rank:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8375 - [Bugfix] Fix InternVL2 inference with various num_patches

- 链接: https://github.com/vllm-project/vllm/pull/8375
- 状态/时间: merged / 2024-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `e56bf2774158`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+39/-3，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix InternVL2 inference with various num_patches」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Bugfix] Fix InternVL2 inference with various num_patches」；主要实现面是 `tests/models/test_internvl.py`, `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/test_internvl.py` modified +35/-0 (35 lines); hunks: -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_...; symbols: test_multi_images_models, test_different_num_patches，涉及 `test_multi_images_models, test_different_num_patches`；`vllm/model_executor/models/internvl.py` modified +4/-3 (7 lines); hunks: -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: obj...; -449,11 +449,12 @@ def _parse_and_validate_image_input(; symbols: input_mapper_for_internvl, _parse_and_validate_image_input，涉及 `input_mapper_for_internvl, _parse_and_validate_image_input`。
- 代码 diff 细节:
  - `tests/models/test_internvl.py` modified +35/-0 (35 lines); hunks: -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_...; symbols: test_multi_images_models, test_different_num_patches
  - `vllm/model_executor/models/internvl.py` modified +4/-3 (7 lines); hunks: -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: obj...; -449,11 +449,12 @@ def _parse_and_validate_image_input(; symbols: input_mapper_for_internvl, _parse_and_validate_image_input
- 关键代码摘录:

```diff
diff -- tests/models/test_internvl.py
@@ -331,6 +331,41 @@ def test_multi_images_models(hf_runner, vllm_runner, image_assets, model,
+@pytest.mark.parametrize("model", ["OpenGVLab/InternVL2-2B"])
+@pytest.mark.parametrize("size_factors", [[0.5, 1.0]])
+@pytest.mark.parametrize("dtype", [target_dtype])
+@pytest.mark.parametrize("max_tokens", [128])
+@pytest.mark.parametrize("num_logprobs", [5])
+@torch.inference_mode()
diff -- vllm/model_executor/models/internvl.py
@@ -270,14 +270,14 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):
+        # we can't stack here because the images may have different num_patches
-        data = torch.stack(data)
@@ -449,11 +449,12 @@ def _parse_and_validate_image_input(
+            # We need to flatten (B, N, P) to (B*N*P),
+            # so we call flatten_bn twice.
-                    flatten_bn(pixel_values, concat=True).flatten(0, 1)),
```

- 已读文件:
  - tests: `tests/models/test_internvl.py` modified +35/-0
  - runtime: `vllm/model_executor/models/internvl.py` modified +4/-3
- 验证与风险: diff 自带测试面 `tests/models/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8250 - [BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv

- 链接: https://github.com/vllm-project/vllm/pull/8250
- 状态/时间: merged / 2024-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `e3dd0692fa2c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+126/-41，可读 patch 343 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[BugFix] Propagate 'trust_remote_code' setting in internvl and minicpmv」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +9/-6 (15 lines); hunks: -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inpu...; -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):; symbols: input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl，涉及 `input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +9/-6 (15 lines); hunks: -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inpu...; -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):; symbols: input_processor_for_internvl, input_mapper_for_internvl, dummy_data_for_internvl
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -230,8 +230,9 @@ def input_processor_for_internvl(ctx: InputContext, llm_inputs: LLMInputs):
-    tokenizer = cached_get_tokenizer(model_config.tokenizer,
-                                     trust_remote_code=True)
+    tokenizer = cached_get_tokenizer(
+        model_config.tokenizer,
+        trust_remote_code=model_config.trust_remote_code)
@@ -278,8 +279,9 @@ def input_mapper_for_internvl(ctx: InputContext, data: object):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +9/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/minicpmv.py`, `vllm/model_executor/models/qwen.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8614 - [VLM][Bugfix] enable internvl running with num_scheduler_steps > 1

- 链接: https://github.com/vllm-project/vllm/pull/8614
- 状态/时间: merged / 2024-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `0c4d2ad5e641`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM][Bugfix] enable internvl running with num_scheduler_steps > 1」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[VLM][Bugfix] enable internvl running with num_scheduler_steps > 1」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +6/-1 (7 lines); hunks: -19,7 +19,7; -376,6 +376,11 @@ def __init__(self,; symbols: __init__, pixel_shuffle，涉及 `__init__, pixel_shuffle`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +6/-1 (7 lines); hunks: -19,7 +19,7; -376,6 +376,11 @@ def __init__(self,; symbols: __init__, pixel_shuffle
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -19,7 +19,7 @@
-from vllm.model_executor.layers.sampler import SamplerOutput
+from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
@@ -376,6 +376,11 @@ def __init__(self,
+        if hasattr(self.language_model, "sampler"):
+            self.sampler = self.language_model.sampler
+        else:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8946 - [Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg

- 链接: https://github.com/vllm-project/vllm/pull/8946
- 状态/时间: merged / 2024-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `2ae25f79cf1e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+90/-61，可读 patch 252 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] Expose InternVL2 max_dynamic_patch as a mm_processor_kwarg」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +89/-61 (150 lines); hunks: -5,8 +5,9; -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess, image_to_pixel_values，涉及 `calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +89/-61 (150 lines); hunks: -5,8 +5,9; -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, dynamic_preprocess, image_to_pixel_values
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -5,8 +5,9 @@
-from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
-                    TypedDict, Union)
+from functools import partial
+from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional,
+                    Tuple, TypedDict, Union)
@@ -122,6 +123,20 @@ def calculate_num_blocks(orig_width: int, orig_height: int, min_num: int,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +89/-61
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9351 - [Bugfix] Update InternVL input mapper to support image embeds

- 链接: https://github.com/vllm-project/vllm/pull/9351
- 状态/时间: merged / 2024-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `55e081fbad29`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Update InternVL input mapper to support image embeds」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Bugfix] Update InternVL input mapper to support image embeds」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +2/-0 (2 lines); hunks: -342,6 +342,8 @@ def input_mapper(; symbols: input_mapper，涉及 `input_mapper`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +2/-0 (2 lines); hunks: -342,6 +342,8 @@ def input_mapper(; symbols: input_mapper
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -342,6 +342,8 @@ def input_mapper(
+        else:
+            return MultiModalInputs({"image_embeds": data})
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9528 - [Model][VLM] Initialize support for Mono-InternVL model

- 链接: https://github.com/vllm-project/vllm/pull/9528
- 状态/时间: merged / 2024-10-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `bb392ea2d2bf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+254/-28，可读 patch 387 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][VLM] Initialize support for Mono-InternVL model」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/decoder_only/vision_language/test_internvl.py`；技术摘要: 覆盖「[Model][VLM] Initialize support for Mono-InternVL model」；主要实现面是 `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/intern_vit.py`, `tests/models/decoder_only/vision_language/test_internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +42/-19 (61 lines); hunks: -21,7 +21,8; -427,13 +428,9 @@ def __init__(self,; symbols: __init__, sampler, _init_vision_model, _init_mlp1，涉及 `__init__, sampler, _init_vision_model`；`vllm/model_executor/models/intern_vit.py` modified +31/-0 (31 lines); hunks: -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.T...; symbols: forward, InternVisionPatchModel, __init__, get_input_embeddings，涉及 `forward, InternVisionPatchModel, __init__`；`tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8 (21 lines); hunks: -7,7 +7,6; -19,15 +18,20; symbols: generate, run_awq_test，涉及 `generate, run_awq_test`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +42/-19 (61 lines); hunks: -21,7 +21,8; -427,13 +428,9 @@ def __init__(self,; symbols: __init__, sampler, _init_vision_model, _init_mlp1
  - `vllm/model_executor/models/intern_vit.py` modified +31/-0 (31 lines); hunks: -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.T...; symbols: forward, InternVisionPatchModel, __init__, get_input_embeddings
  - `tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8 (21 lines); hunks: -7,7 +7,6; -19,15 +18,20; symbols: generate, run_awq_test
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -21,7 +21,8 @@
-from vllm.model_executor.models.intern_vit import InternVisionModel
+from vllm.model_executor.models.intern_vit import (InternVisionModel,
+                                                   InternVisionPatchModel)
@@ -427,13 +428,9 @@ def __init__(self,
-        vision_feature_layer = self.select_layer
-        if vision_feature_layer < 0:
diff -- vllm/model_executor/models/intern_vit.py
@@ -97,6 +97,37 @@ def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
+class InternVisionPatchModel(nn.Module):
+    def __init__(self, config: PretrainedConfig):
+        super().__init__()
+        self.config = config
+        self.embeddings = InternVisionEmbeddings(config)
+    def get_input_embeddings(self):
diff -- tests/models/decoder_only/vision_language/test_internvl.py
@@ -7,7 +7,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +42/-19; `vllm/model_executor/models/intern_vit.py` modified +31/-0
  - tests: `tests/models/decoder_only/vision_language/test_internvl.py` modified +13/-8
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10518 - [Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models

- 链接: https://github.com/vllm-project/vllm/pull/10518
- 状态/时间: merged / 2024-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `d5ec121f95f5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+255/-14，可读 patch 350 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`, `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] Expose `dynamic_image_size` as mm_processor_kwargs for InternVL2 models」；主要实现面是 `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`, `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override，涉及 `input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens`；`vllm/model_executor/models/internvl.py` modified +49/-14 (63 lines); hunks: -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; -183,10 +190,17 @@ def image_to_pixel_values(image: Image.Image, input_size:...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values, image_to_pixel_values_wrapper，涉及 `calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values`。
- 代码 diff 细节:
  - `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override
  - `vllm/model_executor/models/internvl.py` modified +49/-14 (63 lines); hunks: -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int,...; -183,10 +190,17 @@ def image_to_pixel_values(image: Image.Image, input_size:...; symbols: calculate_num_blocks, calculate_num_blocks_wrapper, image_to_pixel_values, image_to_pixel_values_wrapper
- 关键代码摘录:

```diff
diff -- tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py
@@ -0,0 +1,206 @@
+"""Tests for InternVL's multimodal preprocessing kwargs."""
+from typing import Callable, Optional
+import pytest
+from transformers import AutoTokenizer
+from vllm.inputs import InputContext, token_inputs
+from vllm.multimodal import MultiModalRegistry
diff -- vllm/model_executor/models/internvl.py
@@ -123,8 +123,15 @@ def calculate_num_blocks(orig_width: int, orig_height: int, min_num: int,
-def calculate_num_blocks_wrapper(hf_config: PretrainedConfig,
-                                 max_dynamic_patch: Optional[int] = None):
+def calculate_num_blocks_wrapper(
+    hf_config: PretrainedConfig,
+    max_dynamic_patch: Optional[int] = None,
+    dynamic_image_size: Optional[bool] = None,
```

- 已读文件:
  - tests: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py` added +206/-0
  - runtime: `vllm/model_executor/models/internvl.py` modified +49/-14
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11165 - [V1][VLM] Fix edge case bug for InternVL2

- 链接: https://github.com/vllm-project/vllm/pull/11165
- 状态/时间: merged / 2024-12-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `969da7d70bc0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1][VLM] Fix edge case bug for InternVL2」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[V1][VLM] Fix edge case bug for InternVL2」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -669,8 +669,11 @@ def _process_image_input(; symbols: _process_image_input，涉及 `_process_image_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -669,8 +669,11 @@ def _process_image_input(; symbols: _process_image_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -669,8 +669,11 @@ def _process_image_input(
+        # Only one image in the current batch
-            image_embeds = image_embeds.unsqueeze(0)
+            image_embeds = image_embeds.view(
+                -1, self.config.text_config.hidden_size).unsqueeze(0)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12553 - [VLM] Merged multi-modal processor for InternVL-based models

- 链接: https://github.com/vllm-project/vllm/pull/12553
- 状态/时间: merged / 2025-02-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_internvl.py`, `vllm/model_executor/models/internvl.py`；关联提交 `d1ca7df84d9f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+1434/-986，可读 patch 3135 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Merged multi-modal processor for InternVL-based models」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`；技术摘要: 覆盖「[VLM] Merged multi-modal processor for InternVL-based models」；主要实现面是 `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +503/-320 (823 lines); hunks: -6,35 +6,37; -75,22 +77,27 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio，涉及 `InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio`；`tests/models/multimodal/processing/test_internvl.py` modified +32/-175 (207 lines); hunks: -1,207 +1,64; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override，涉及 `input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +503/-320 (823 lines); hunks: -6,35 +6,37; -75,22 +77,27 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, build_transform, find_closest_aspect_ratio
  - `tests/models/multimodal/processing/test_internvl.py` modified +32/-175 (207 lines); hunks: -1,207 +1,64; symbols: input_processor_for_internvl, dummy_data_for_internvl, get_max_internvl_image_tokens, test_input_mapper_override
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -6,35 +6,37 @@
-import re
-from functools import cached_property, partial
+from abc import ABC, abstractmethod
+from functools import cached_property
-                    TypedDict, Union)
+                    TypedDict, TypeVar, Union)
diff -- tests/models/multimodal/processing/test_internvl.py
@@ -1,207 +1,64 @@
-from typing import Callable, Optional
+from typing import Optional
-from transformers import AutoTokenizer
-from vllm.inputs import InputContext, token_inputs
-from vllm.multimodal import MultiModalRegistry
+from vllm.multimodal import MULTIMODAL_REGISTRY
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +503/-320
  - tests: `tests/models/multimodal/processing/test_internvl.py` modified +32/-175
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_h2ovl.py`, `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14738 - [VLM] Support loading InternVideo2.5 models as original InternVLChatModel

- 链接: https://github.com/vllm-project/vllm/pull/14738
- 状态/时间: merged / 2025-03-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `b1cc4dfef57a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-3，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support loading InternVideo2.5 models as original InternVLChatModel」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[VLM] Support loading InternVideo2.5 models as original InternVLChatModel」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +8/-1 (9 lines); hunks: -981,5 +981,12 @@ def sample(; symbols: sample, load_weights，涉及 `sample, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +8/-1 (9 lines); hunks: -981,5 +981,12 @@ def sample(; symbols: sample, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -981,5 +981,12 @@ def sample(
-        loader = AutoWeightsLoader(self)
+        # unused modules appear in OpenGVLab/InternVideo2_5_Chat_8B
+        skip_prefixes = [
+            "action_embed", "temporal_embed", "track_embed",
+            "track_embed_decoder", "box_token", "cg_criterion", "cg_model",
+            "loc_encoder", "loc_decoder", "sam", "temporal_token",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +8/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15086 - [Bugfix] Fix embedding assignment for InternVL-based models

- 链接: https://github.com/vllm-project/vllm/pull/15086
- 状态/时间: merged / 2025-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `ffa443afedd3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+123/-106，可读 patch 488 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix embedding assignment for InternVL-based models」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Bugfix] Fix embedding assignment for InternVL-based models」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +104/-69 (173 lines); hunks: -9,14 +9,13; -36,10 +35,12; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id, get_image_repl_features，涉及 `InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +104/-69 (173 lines); hunks: -9,14 +9,13; -36,10 +35,12; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, image_token_id, get_image_repl_features
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -9,14 +9,13 @@
-from typing import (List, Literal, Optional, Set, Tuple, TypedDict, TypeVar,
-                    Union)
+from typing import Literal, Optional, Set, Tuple, TypedDict, TypeVar, Union
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchEncoding, PretrainedConfig, TensorType
@@ -36,10 +35,12 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +104/-69
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma3_mm.py`, `vllm/model_executor/models/h2ovl.py`, `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18499 - [VLM] Initialize video input support for InternVL models

- 链接: https://github.com/vllm-project/vllm/pull/18499
- 状态/时间: merged / 2025-05-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `75f81750f3a9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+596/-62，可读 patch 940 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Initialize video input support for InternVL models」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[VLM] Initialize video input support for InternVL models」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunks: -8,8 +8,9; -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform，涉及 `InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +485/-26 (511 lines); hunks: -8,8 +8,9; -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):; symbols: InternVLImageEmbeddingInputs, InternVLVideoPixelInputs, InternVLVideoEmbeddingInputs, build_transform
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -8,8 +8,9 @@
-from typing import Literal, Optional, TypedDict, TypeVar, Union
+from typing import Any, Literal, Optional, TypedDict, TypeVar, Union
+import numpy.typing as npt
@@ -74,6 +75,33 @@ class InternVLImageEmbeddingInputs(TypedDict):
+class InternVLVideoPixelInputs(TypedDict):
+    type: Literal["pixel_values_videos"]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +485/-26
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18842 - [LoRA] Add LoRA support for InternVL

- 链接: https://github.com/vllm-project/vllm/pull/18842
- 状态/时间: merged / 2025-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `34d6c447c4b9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-2，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA] Add LoRA support for InternVL」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[LoRA] Add LoRA support for InternVL」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +23/-2 (25 lines); hunks: -22,6 +22,7; -36,7 +37,8; symbols: get_video_replacement_internvl, InternVLChatModel, __init__, load_weights，涉及 `get_video_replacement_internvl, InternVLChatModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +23/-2 (25 lines); hunks: -22,6 +22,7; -36,7 +37,8; symbols: get_video_replacement_internvl, InternVLChatModel, __init__, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -22,6 +22,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -36,7 +37,8 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
+                         SupportsMultiModal, SupportsPP)
@@ -1014,7 +1016,17 @@ def get_video_replacement_internvl(item_idx: int):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +23/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21684 - Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/21684
- 状态/时间: merged / 2025-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `f1e2c095ecee`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+49/-62，可读 patch 184 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「Migrate InternVLImageInputs and InternVLVideoInputs to TensorSchema」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +49/-62 (111 lines); hunks: -9,7 +9,7; -37,6 +37,7; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs，涉及 `InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +49/-62 (111 lines); hunks: -9,7 +9,7; -37,6 +37,7; symbols: InternVLImagePixelInputs, InternVLImageEmbeddingInputs, InternVLVideoPixelInputs
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -9,7 +9,7 @@
-from typing import Any, Literal, Optional, TypedDict, TypeVar, Union
+from typing import Annotated, Any, Literal, Optional, TypeVar, Union
@@ -37,6 +37,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -51,54 +52,60 @@
-class InternVLImagePixelInputs(TypedDict):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +49/-62
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23658 - [Model] Enable video support for InternVL3.5 models

- 链接: https://github.com/vllm-project/vllm/pull/23658
- 状态/时间: merged / 2025-08-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `9816b81f5f9f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+22/-7，可读 patch 71 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable video support for InternVL3.5 models」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] Enable video support for InternVL3.5 models」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunks: -855,9 +855,13 @@ def get_supported_mm_limits(self):; symbols: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features，涉及 `get_supported_mm_limits, get_video_token, get_num_frames_with_most_features`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +7/-3 (10 lines); hunks: -855,9 +855,13 @@ def get_supported_mm_limits(self):; symbols: get_supported_mm_limits, get_video_token, get_num_frames_with_most_features
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -855,9 +855,13 @@ def get_supported_mm_limits(self):
-        if text_model_type == "qwen2":
-            return "<|video_pad|>"
-        return None
+        video_token_map = {
+            "qwen2": "<|video_pad|>",
+            "qwen3": "<|video_pad|>",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +7/-3
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23742 - [Model] Enable native HF format InternVL support

- 链接: https://github.com/vllm-project/vllm/pull/23742
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+18/-16，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable native HF format InternVL support」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md`；技术摘要: 覆盖「[Model] Enable native HF format InternVL support」；主要实现面是 `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`, `docs/models/supported_models.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunks: -222,21 +222,6; -461,6 +446,20；`tests/models/registry.py` modified +2/-1 (3 lines); hunks: -429,6 +429,7 @@ def check_available_online(; -584,7 +585,7 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generat...；`vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -217,6 +217,7。
- 代码 diff 细节:
  - `tests/models/multimodal/generation/test_common.py` modified +14/-15 (29 lines); hunks: -222,21 +222,6; -461,6 +446,20
  - `tests/models/registry.py` modified +2/-1 (3 lines); hunks: -429,6 +429,7 @@ def check_available_online(; -584,7 +585,7 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generat...
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -217,6 +217,7
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/generation/test_common.py
@@ -222,21 +222,6 @@
-    # Check "auto" with fallback to transformers
-    "internvl-transformers": VLMTestInfo(
-        models=["OpenGVLab/InternVL3-1B-hf"],
-        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
-        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
-        img_idx_to_prompt=lambda idx: "<IMG_CONTEXT>",
diff -- tests/models/registry.py
@@ -429,6 +429,7 @@ def check_available_online(
+    "InternVLForConditionalGeneration": _HfExamplesInfo("OpenGVLab/InternVL3-1B-hf"),    # noqa: E501
@@ -584,7 +585,7 @@ def check_available_online(
-    "TransformersForMultimodalLM": _HfExamplesInfo("OpenGVLab/InternVL3-1B-hf"),
+    "TransformersForMultimodalLM": _HfExamplesInfo("BAAI/Emu3-Chat-hf"),
diff -- docs/models/supported_models.md
@@ -629,6 +629,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
+| `InternVLForConditionalGeneration` | InternVL 3.0 (HF format) | T + I<sup>E+</sup> + V<sup>E+</sup> | `OpenGVLab/InternVL3-1B-hf`, etc. | ✅︎ | ✅︎ | ✅︎ |
diff -- vllm/model_executor/models/registry.py
```

- 已读文件:
  - tests: `tests/models/multimodal/generation/test_common.py` modified +14/-15; `tests/models/registry.py` modified +2/-1
  - docs: `docs/models/supported_models.md` modified +1/-0
  - runtime: `vllm/model_executor/models/registry.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24519 - [Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention.

- 链接: https://github.com/vllm-project/vllm/pull/24519
- 状态/时间: merged / 2025-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `267c80d31f6b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-1，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention.」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention.」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -37,6 +38,7; symbols: InternVLVideoEmbeddingInputs, build_transform, apply，涉及 `InternVLVideoEmbeddingInputs, build_transform, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -37,6 +38,7; symbols: InternVLVideoEmbeddingInputs, build_transform, apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -7,6 +7,7 @@
+import os
@@ -37,6 +38,7 @@
+from vllm.utils import set_default_torch_num_threads
@@ -115,13 +117,26 @@ class InternVLVideoEmbeddingInputs(TensorSchema):
-    return T.Compose([
+    transform = T.Compose([
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +16/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23909 - [Model] enable data parallel for InternVL vision encoder

- 链接: https://github.com/vllm-project/vllm/pull/23909
- 状态/时间: merged / 2025-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；关联提交 `52bc9d5b3edb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+80/-33，可读 patch 262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] enable data parallel for InternVL vision encoder」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] enable data parallel for InternVL vision encoder」；主要实现面是 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/intern_vit.py` modified +75/-32 (107 lines); hunks: -25,9 +25,11; -137,6 +139,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):; -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: get_video_replacement_internvl, InternVLChatModel, get_placeholder_str, __init__，涉及 `get_video_replacement_internvl, InternVLChatModel, get_placeholder_str`。
- 代码 diff 细节:
  - `vllm/model_executor/models/intern_vit.py` modified +75/-32 (107 lines); hunks: -25,9 +25,11; -137,6 +139,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/internvl.py` modified +4/-1 (5 lines); hunks: -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):; -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: get_video_replacement_internvl, InternVLChatModel, get_placeholder_str, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/intern_vit.py
@@ -25,9 +25,11 @@
+                                               ReplicatedLinear,
+from vllm.multimodal.utils import run_dp_sharded_vision_model
@@ -137,6 +139,7 @@ def __init__(
+        use_data_parallel: bool = False,
@@ -150,23 +153,34 @@ def __init__(
-        self.tp_size = get_tensor_model_parallel_world_size()
diff -- vllm/model_executor/models/internvl.py
@@ -1020,6 +1020,8 @@ def get_video_replacement_internvl(item_idx: int):
+    supports_encoder_tp_data = True
@@ -1038,6 +1040,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
@@ -1105,7 +1108,7 @@ def _init_vision_model(
-            )
+                use_data_parallel=self.use_data_parallel)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/intern_vit.py` modified +75/-32; `vllm/model_executor/models/internvl.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26153 - [Model] Use `merge_by_field_config` for MM models (InternVL family)

- 链接: https://github.com/vllm-project/vllm/pull/26153
- 状态/时间: merged / 2025-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `f9a8084e4879`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+84/-182，可读 patch 785 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Use `merge_by_field_config` for MM models (InternVL family)」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] Use `merge_by_field_config` for MM models (InternVL family)」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +27/-54 (81 lines); hunks: -17,7 +17,7; -28,7 +28,7; symbols: _preprocess_image, __call__, InternVLProcessor, _preprocess_video，涉及 `_preprocess_image, __call__, InternVLProcessor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +27/-54 (81 lines); hunks: -17,7 +17,7; -28,7 +28,7; symbols: _preprocess_image, __call__, InternVLProcessor, _preprocess_video
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -17,7 +17,7 @@
-from transformers import BatchEncoding, PretrainedConfig, TensorType
+from transformers import BatchFeature, PretrainedConfig, TensorType
@@ -28,7 +28,7 @@
-                                    MultiModalKwargsItems, NestedTensors)
+                                    MultiModalKwargsItems)
@@ -42,8 +42,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +27/-54
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32397 - [Model] Enable LoRA support for internvl2

- 链接: https://github.com/vllm-project/vllm/pull/32397
- 状态/时间: merged / 2026-01-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `fec9da0af48d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-3，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable LoRA support for internvl2」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] Enable LoRA support for internvl2」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +16/-3 (19 lines); hunks: -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: __init__, get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens，涉及 `__init__, get_mm_mapping, get_num_mm_encoder_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +16/-3 (19 lines); hunks: -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: __init__, get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -1086,9 +1086,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        self.num_image_token = int(
-            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
-        )
+        self.patch_tokens = (image_size // patch_size) ** 2
+        self.num_image_token = int(self.patch_tokens * (config.downsample_ratio**2))
@@ -1430,3 +1429,17 @@ def get_mm_mapping(self) -> MultiModelKeys:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +16/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37260 - [1/2] Move InternVL-based processors

- 链接: https://github.com/vllm-project/vllm/pull/37260
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_internvl.py`, `vllm/model_executor/models/internvl.py`, `vllm/transformers_utils/processors/internvl.py`；关联提交 `f34032433573`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+3252/-3099，可读 patch 6681 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/2] Move InternVL-based processors」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`；技术摘要: 覆盖「[1/2] Move InternVL-based processors」；主要实现面是 `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`, `tests/models/multimodal/processing/test_internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/internvl.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num, get_internvl_target_ratios，涉及 `build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num`；`vllm/model_executor/models/internvl.py` modified +7/-578 (585 lines); hunks: -7,16 +7,13; -28,7 +25,6; symbols: InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform, find_closest_aspect_ratio，涉及 `InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform`；`tests/models/multimodal/processing/test_internvl.py` modified +1/-1 (2 lines); hunks: -23,7 +23,7 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches，涉及 `_get_expected_num_patches`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/internvl.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: build_transform, find_closest_aspect_ratio, resolve_internvl_min_max_num, get_internvl_target_ratios
  - `vllm/model_executor/models/internvl.py` modified +7/-578 (585 lines); hunks: -7,16 +7,13; -28,7 +25,6; symbols: InternVLImagePixelInputs, InternVLVideoEmbeddingInputs, build_transform, find_closest_aspect_ratio
  - `tests/models/multimodal/processing/test_internvl.py` modified +1/-1 (2 lines); hunks: -23,7 +23,7 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/internvl.py
@@ -0,0 +1,603 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
diff -- vllm/model_executor/models/internvl.py
@@ -7,16 +7,13 @@
-from abc import ABC, abstractmethod
+from abc import abstractmethod
-from typing import Annotated, Any, Literal, TypeAlias, TypeVar
+from typing import Annotated, Literal, TypeAlias, TypeVar
-import numpy.typing as npt
-import torchvision.transforms as T
diff -- tests/models/multimodal/processing/test_internvl.py
@@ -23,7 +23,7 @@ def _get_expected_num_patches(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/internvl.py` added +603/-0; `vllm/model_executor/models/internvl.py` modified +7/-578
  - tests: `tests/models/multimodal/processing/test_internvl.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_h2ovl.py`, `tests/models/multimodal/processing/test_internvl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37324 - [2/3] Refactor InternVL-based processors

- 链接: https://github.com/vllm-project/vllm/pull/37324
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`, `vllm/transformers_utils/processors/internvl.py`；关联提交 `99267c23ca51`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+762/-1146，可读 patch 2597 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[2/3] Refactor InternVL-based processors」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[2/3] Refactor InternVL-based processors」；主要实现面是 `vllm/transformers_utils/processors/internvl.py`, `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/internvl.py` modified +233/-273 (506 lines); hunks: -7,24 +7,17; -33,7 +26,7; symbols: build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor, InternVLImageProcessor，涉及 `build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor`；`vllm/model_executor/models/internvl.py` modified +85/-46 (131 lines); hunks: -9,6 +9,7; -45,8 +46,9; symbols: BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits, get_num_image_tokens，涉及 `BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/internvl.py` modified +233/-273 (506 lines); hunks: -7,24 +7,17; -33,7 +26,7; symbols: build_transform, video_to_pixel_values_internvl, BaseInternVLProcessor, InternVLImageProcessor
  - `vllm/model_executor/models/internvl.py` modified +85/-46 (131 lines); hunks: -9,6 +9,7; -45,8 +46,9; symbols: BaseInternVLProcessingInfo, get_hf_processor, get_supported_mm_limits, get_num_image_tokens
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/internvl.py
@@ -7,24 +7,17 @@
-from abc import ABC, abstractmethod
-from typing import Any, TypeVar
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchFeature, TensorType
+from transformers.processing_utils import ProcessorMixin
-from vllm.tokenizers import TokenizerLike
diff -- vllm/model_executor/models/internvl.py
@@ -9,6 +9,7 @@
+from functools import cached_property
@@ -45,8 +46,9 @@
-    BaseInternVLProcessor,
+    InternVLImageProcessor,
+    InternVLVideoProcessor,
@@ -123,7 +125,7 @@ class BaseInternVLProcessingInfo(BaseProcessingInfo):
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/internvl.py` modified +233/-273; `vllm/model_executor/models/internvl.py` modified +85/-46
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35182 - [Misc] Reorganize inputs

- 链接: https://github.com/vllm-project/vllm/pull/35182
- 状态/时间: merged / 2026-03-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 142 个文件，+1212/-1342，可读 patch 6002 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Reorganize inputs」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`；技术摘要: 覆盖「[Misc] Reorganize inputs」；主要实现面是 `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/multimodal/inputs.py` modified +2/-162 (164 lines); hunks: -15,12 +15,11; -32,14 +31,9; symbols: VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins, PlaceholderRange，涉及 `VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins`；`vllm/entrypoints/pooling/score/serving.py` modified +36/-45 (81 lines); hunks: -35,7 +35,7; -110,12 +110,12 @@ async def _embedding_score(; symbols: _embedding_score, _preprocess_late_interaction_item，涉及 `_embedding_score, _preprocess_late_interaction_item`；`vllm/entrypoints/serve/render/serving.py` modified +38/-37 (75 lines); hunks: -34,9 +34,15; -127,22 +133,22 @@ async def render_chat_request(; symbols: render_chat_request, render_chat，涉及 `render_chat_request, render_chat`；`vllm/entrypoints/openai/responses/serving.py` modified +22/-26 (48 lines); hunks: -110,7 +110,7; -269,10 +269,10 @@ def __init__(; symbols: __init__, _validate_generator_input, create_responses，涉及 `__init__, _validate_generator_input, create_responses`。
- 代码 diff 细节:
  - `vllm/multimodal/inputs.py` modified +2/-162 (164 lines); hunks: -15,12 +15,11; -32,14 +31,9; symbols: VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins, PlaceholderRange
  - `vllm/entrypoints/pooling/score/serving.py` modified +36/-45 (81 lines); hunks: -35,7 +35,7; -110,12 +110,12 @@ async def _embedding_score(; symbols: _embedding_score, _preprocess_late_interaction_item
  - `vllm/entrypoints/serve/render/serving.py` modified +38/-37 (75 lines); hunks: -34,9 +34,15; -127,22 +133,22 @@ async def render_chat_request(; symbols: render_chat_request, render_chat
  - `vllm/entrypoints/openai/responses/serving.py` modified +22/-26 (48 lines); hunks: -110,7 +110,7; -269,10 +269,10 @@ def __init__(; symbols: __init__, _validate_generator_input, create_responses
  - `vllm/entrypoints/llm.py` modified +22/-22 (44 lines); hunks: -57,9 +57,9; -584,7 +584,7 @@ def wait_for_completion(; symbols: wait_for_completion, _resolve_mm_lora, beam_search
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/multimodal/inputs.py` modified +2/-162; `vllm/entrypoints/pooling/score/serving.py` modified +36/-45; `vllm/entrypoints/serve/render/serving.py` modified +38/-37; `vllm/entrypoints/openai/responses/serving.py` modified +22/-26; `vllm/entrypoints/llm.py` modified +22/-22; `vllm/entrypoints/pooling/embed/io_processor.py` modified +20/-20
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/chat_completion/test_chat_error.py`, `tests/entrypoints/openai/chat_completion/test_serving_chat.py`, `tests/entrypoints/openai/responses/test_serving_responses.py`, `tests/entrypoints/serve/render/test_launch_render.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38049 - [Model] Add torch.compile support for InternVL vision encoder

- 链接: https://github.com/vllm-project/vllm/pull/38049
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/intern_vit.py`；关联提交 `38de82231023`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+20/-3，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add torch.compile support for InternVL vision encoder」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/intern_vit.py`；技术摘要: 覆盖「[Model] Add torch.compile support for InternVL vision encoder」；主要实现面是 `vllm/model_executor/models/intern_vit.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunks: -15,6 +15,10; -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: forward, InternVisionEncoderLayer, __init__，涉及 `forward, InternVisionEncoderLayer, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/intern_vit.py` modified +11/-2 (13 lines); hunks: -15,6 +15,10; -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: forward, InternVisionEncoderLayer, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/intern_vit.py
@@ -15,6 +15,10 @@
+from vllm.compilation.decorators import (
+    should_torch_compile_mm_encoder,
+    support_torch_compile,
+)
@@ -280,6 +284,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+@support_torch_compile(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/intern_vit.py` modified +11/-2
- 验证与风险: runtime 路径改动集中在 `vllm/config/utils.py`, `vllm/model_executor/models/intern_vit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38901 - refactor hard coded device string in test files under tests/compile tests/quantization tests/models and tests/model_executor

- 链接: https://github.com/vllm-project/vllm/pull/38901
- 状态/时间: merged / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+122/-66，可读 patch 760 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor hard coded device string in test files under tests/compile tests/quantization tests/models and tests/model_executor」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/pooling/test_intern_vit.py`, `tests/models/multimodal/pooling/test_radio.py`, `tests/models/test_utils.py`；技术摘要: 覆盖「refactor hard coded device string in test files under tests/compile tests/quantization tests/models and tests/model_executor」；主要实现面是 `tests/models/multimodal/pooling/test_intern_vit.py`, `tests/models/multimodal/pooling/test_radio.py`, `tests/models/test_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_intern_vit.py` modified +8/-4 (12 lines); hunks: -7,6 +7,7; -15,6 +16,8; symbols: run_intern_vit_test，涉及 `run_intern_vit_test`；`tests/models/multimodal/pooling/test_radio.py` modified +8/-4 (12 lines); hunks: -8,6 +8,7; -17,6 +18,8; symbols: run_radio_test，涉及 `run_radio_test`；`tests/models/test_utils.py` modified +8/-2 (10 lines); hunks: -10,6 +10,8; -174,8 +176,12 @@ def __exit__(self, exception_type, exception_value, traceba...; symbols: ModuleWithBatchNorm, __init__, __exit__, test_merge_multimodal_embeddings_no_sync，涉及 `ModuleWithBatchNorm, __init__, __exit__`；`tests/model_executor/test_eagle_quantization.py` modified +3/-2 (5 lines); hunks: -10,9 +10,10。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_intern_vit.py` modified +8/-4 (12 lines); hunks: -7,6 +7,7; -15,6 +16,8; symbols: run_intern_vit_test
  - `tests/models/multimodal/pooling/test_radio.py` modified +8/-4 (12 lines); hunks: -8,6 +8,7; -17,6 +18,8; symbols: run_radio_test
  - `tests/models/test_utils.py` modified +8/-2 (10 lines); hunks: -10,6 +10,8; -174,8 +176,12 @@ def __exit__(self, exception_type, exception_value, traceba...; symbols: ModuleWithBatchNorm, __init__, __exit__, test_merge_multimodal_embeddings_no_sync
  - `tests/model_executor/test_eagle_quantization.py` modified +3/-2 (5 lines); hunks: -10,9 +10,10
  - `tests/basic_correctness/test_cumem.py` modified +10/-8 (18 lines); hunks: -13,6 +13,8; -26,13 +28,13 @@ def test_python_error():; symbols: test_python_error, test_basic_cumem
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_intern_vit.py
@@ -7,6 +7,7 @@
+from vllm.platforms import current_platform
@@ -15,6 +16,8 @@
+DEVICE_TYPE = current_platform.device_type
@@ -39,9 +42,9 @@ def run_intern_vit_test(
-    ).to("cuda")
+    ).to(DEVICE_TYPE)
diff -- tests/models/multimodal/pooling/test_radio.py
@@ -8,6 +8,7 @@
+from vllm.platforms import current_platform
@@ -17,6 +18,8 @@
+DEVICE_TYPE = current_platform.device_type
@@ -51,7 +54,7 @@ def run_radio_test(
-    ).to("cuda")
+    ).to(DEVICE_TYPE)
diff -- tests/models/test_utils.py
@@ -10,6 +10,8 @@
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_intern_vit.py` modified +8/-4; `tests/models/multimodal/pooling/test_radio.py` modified +8/-4; `tests/models/test_utils.py` modified +8/-2; `tests/model_executor/test_eagle_quantization.py` modified +3/-2; `tests/basic_correctness/test_cumem.py` modified +10/-8; `tests/quantization/test_torchao.py` modified +9/-8
- 验证与风险: diff 自带测试面 `tests/basic_correctness/test_cumem.py`, `tests/compile/passes/distributed/test_async_tp.py`, `tests/compile/passes/distributed/test_fusion_all_reduce.py`, `tests/compile/passes/distributed/test_sequence_parallelism.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30566 - Update to transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/30566
- 状态/时间: merged / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 41 个文件，+445/-115，可读 patch 1409 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update to transformers v5」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `tests/models/registry.py`, `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/generation/test_common.py`；技术摘要: 覆盖「Update to transformers v5」；主要实现面是 `tests/models/registry.py`, `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/generation/test_common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/registry.py` modified +130/-9 (139 lines); hunks: -335,7 +335,15 @@ def check_available_online(; -475,6 +483,13 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`vllm/model_executor/models/gemma4_mm.py` modified +36/-15 (51 lines); hunks: -125,8 +125,12 @@ class Gemma4AudioInputs(TensorSchema):; -505,6 +509,8 @@ def _call_hf_processor(; symbols: Gemma4AudioInputs, _call_hf_processor，涉及 `Gemma4AudioInputs, _call_hf_processor`；`tests/models/multimodal/generation/test_common.py` modified +38/-6 (44 lines); hunks: -186,7 +186,14; -397,14 +404,14；`vllm/tokenizers/registry.py` modified +34/-1 (35 lines); hunks: -1,5 +1,6; -10,6 +11,7; symbols: get_tokenizer，涉及 `get_tokenizer`。
- 代码 diff 细节:
  - `tests/models/registry.py` modified +130/-9 (139 lines); hunks: -335,7 +335,15 @@ def check_available_online(; -475,6 +483,13 @@ def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/gemma4_mm.py` modified +36/-15 (51 lines); hunks: -125,8 +125,12 @@ class Gemma4AudioInputs(TensorSchema):; -505,6 +509,8 @@ def _call_hf_processor(; symbols: Gemma4AudioInputs, _call_hf_processor
  - `tests/models/multimodal/generation/test_common.py` modified +38/-6 (44 lines); hunks: -186,7 +186,14; -397,14 +404,14
  - `vllm/tokenizers/registry.py` modified +34/-1 (35 lines); hunks: -1,5 +1,6; -10,6 +11,7; symbols: get_tokenizer
  - `tests/model_executor/test_weight_utils.py` modified +0/-18 (18 lines); hunks: -1,7 +1,6; -10,26 +9,10; symbols: test_hf_transfer_auto_activation, test_download_weights_from_hf, test_missing_target_returns_none
- 关键代码摘录:

```diff
diff -- tests/models/registry.py
@@ -335,7 +335,15 @@ def check_available_online(
-        "OpenGVLab/Mono-InternVL-2B", trust_remote_code=True
+        "OpenGVLab/Mono-InternVL-2B",
+        trust_remote_code=True,
+        max_transformers_version="4.57",
+        transformers_version_reason={
+            "vllm": (
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -125,8 +125,12 @@ class Gemma4AudioInputs(TensorSchema):
-    input_features_padded: Annotated[torch.Tensor, TensorShape("bn", "s", "f")]
-    input_features_mask: Annotated[torch.Tensor, TensorShape("bn", "s")]
+    input_features_padded: Annotated[
+        torch.Tensor, TensorShape("bn", "s", "f", dynamic_dims={"s"})
+    ]
+    input_features_mask: Annotated[
diff -- tests/models/multimodal/generation/test_common.py
@@ -186,7 +186,14 @@
```

- 已读文件:
  - tests: `tests/models/registry.py` modified +130/-9; `tests/models/multimodal/generation/test_common.py` modified +38/-6; `tests/model_executor/test_weight_utils.py` modified +0/-18; `tests/models/multimodal/generation/test_phi4siglip.py` modified +11/-0; `tests/models/utils.py` modified +10/-1
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +36/-15; `vllm/tokenizers/registry.py` modified +34/-1; `vllm/model_executor/model_loader/gguf_loader.py` modified +12/-0
- 验证与风险: diff 自带测试面 `requirements/test/cuda.in`, `requirements/test/cuda.txt`, `requirements/test/nightly-torch.txt`, `requirements/test/rocm.in`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42347 - [Perf][4/n] Eliminate various GPU CPU syncs

- 链接: https://github.com/vllm-project/vllm/pull/42347
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+129/-108，可读 patch 606 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][4/n] Eliminate various GPU CPU syncs」；模型线: InternVL 3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py`；技术摘要: 覆盖「[Perf][4/n] Eliminate various GPU CPU syncs」；主要实现面是 `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/utils.py` modified +7/-15 (22 lines); hunks: -30,10 +30,8; -498,10 +496,9 @@ def isin_list(; symbols: isin_list, extract_layer_index, cast_overflow_tensors, fast_topk，涉及 `isin_list, extract_layer_index, cast_overflow_tensors`；`vllm/model_executor/models/qwen2_5_vl.py` modified +12/-7 (19 lines); hunks: -84,6 +84,7; -679,6 +680,7 @@ def rotary_pos_emb_thw(self, t, h, w):; symbols: rotary_pos_emb_thw, get_rope_by_thw, _get_mm_fields_config, _call_hf_processor，涉及 `rotary_pos_emb_thw, get_rope_by_thw, _get_mm_fields_config`；`vllm/model_executor/models/granite_speech.py` modified +7/-7 (14 lines); hunks: -143,7 +143,7 @@ def _get_mm_fields_config(; -717,13 +717,13 @@ def _build_input_features_mask(; symbols: _get_mm_fields_config, _get_prompt_updates, _build_input_features_mask, _pad_and_stack_input_features，涉及 `_get_mm_fields_config, _get_prompt_updates, _build_input_features_mask`；`vllm/model_executor/models/phi4mm_audio.py` modified +9/-3 (12 lines); hunks: -586,7 +586,9 @@ def forward_embeddings(; -605,7 +607,9 @@ def forward_embeddings(; symbols: forward_embeddings, calculate_hs_mask，涉及 `forward_embeddings, calculate_hs_mask`。
- 代码 diff 细节:
  - `vllm/model_executor/models/utils.py` modified +7/-15 (22 lines); hunks: -30,10 +30,8; -498,10 +496,9 @@ def isin_list(; symbols: isin_list, extract_layer_index, cast_overflow_tensors, fast_topk
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +12/-7 (19 lines); hunks: -84,6 +84,7; -679,6 +680,7 @@ def rotary_pos_emb_thw(self, t, h, w):; symbols: rotary_pos_emb_thw, get_rope_by_thw, _get_mm_fields_config, _call_hf_processor
  - `vllm/model_executor/models/granite_speech.py` modified +7/-7 (14 lines); hunks: -143,7 +143,7 @@ def _get_mm_fields_config(; -717,13 +717,13 @@ def _build_input_features_mask(; symbols: _get_mm_fields_config, _get_prompt_updates, _build_input_features_mask, _pad_and_stack_input_features
  - `vllm/model_executor/models/phi4mm_audio.py` modified +9/-3 (12 lines); hunks: -586,7 +586,9 @@ def forward_embeddings(; -605,7 +607,9 @@ def forward_embeddings(; symbols: forward_embeddings, calculate_hs_mask
  - `vllm/model_executor/models/bert.py` modified +3/-6 (9 lines); hunks: -559,13 +559,10 @@ def _encode_token_type_ids(; symbols: _encode_token_type_ids, _decode_token_type_ids
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/utils.py
@@ -30,10 +30,8 @@
-from vllm.utils.platform_utils import (
-    is_pin_memory_available,
-)
+    async_tensor_h2d,
@@ -498,10 +496,9 @@ def isin_list(
-    test_elements = torch.tensor(
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -84,6 +84,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -679,6 +680,7 @@ def rotary_pos_emb_thw(self, t, h, w):
+        pos_ids = pos_ids.to(cos.device, non_blocking=True)
@@ -737,9 +739,10 @@ def get_rope_by_thw(self, t, h, w):
-        cos_thw = cos_thw[window_index_thw, :, :]
+        window_index_thw_dev = window_index_thw.to(cos_thw.device, non_blocking=True)
diff -- vllm/model_executor/models/granite_speech.py
@@ -143,7 +143,7 @@ def _get_mm_fields_config(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/utils.py` modified +7/-15; `vllm/model_executor/models/qwen2_5_vl.py` modified +12/-7; `vllm/model_executor/models/granite_speech.py` modified +7/-7; `vllm/model_executor/models/phi4mm_audio.py` modified +9/-3; `vllm/model_executor/models/bert.py` modified +3/-6; `vllm/model_executor/models/qwen3_vl.py` modified +6/-3
- 验证与风险: diff 自带测试面 `tests/v1/logits_processors/test_correctness.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41759 - [MM][Perf][CG] Support ViT full CUDA graph for InternVL

- 链接: https://github.com/vllm-project/vllm/pull/41759
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `f25952e59b4a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+183/-2，可读 patch 238 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support ViT full CUDA graph for InternVL」；模型线: InternVL 3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[MM][Perf][CG] Support ViT full CUDA graph for InternVL」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +166/-2 (168 lines); hunks: -10,7 +10,7; -55,6 +55,7; symbols: _get_prompt_updates, InternVLChatModel, get_num_mm_connector_tokens, get_encoder_cudagraph_config，涉及 `_get_prompt_updates, InternVLChatModel, get_num_mm_connector_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +166/-2 (168 lines); hunks: -10,7 +10,7; -55,6 +55,7; symbols: _get_prompt_updates, InternVLChatModel, get_num_mm_connector_tokens, get_encoder_cudagraph_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -10,7 +10,7 @@
-from typing import Annotated, Literal, TypeAlias, TypeVar
+from typing import Annotated, Any, Literal, TypeAlias, TypeVar
@@ -55,6 +55,7 @@
+    SupportsEncoderCudaGraph,
@@ -543,7 +544,13 @@ def _get_prompt_updates(
-class InternVLChatModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +166/-2
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_vit_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45129 - [Model] Remove Mono-InternVL (InternLM2VEForCausalLM)

- 链接: https://github.com/vllm-project/vllm/pull/45129
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/internvl.py`；关联提交 `f1e13f7df9ad`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+53/-262，可读 patch 470 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Remove Mono-InternVL (InternLM2VEForCausalLM)」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/internvl.py`；技术摘要: 覆盖「[Model] Remove Mono-InternVL (InternLM2VEForCausalLM)」；主要实现面是 `vllm/model_executor/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/internvl.py` modified +12/-39 (51 lines); hunks: -23,7 +23,6; -582,14 +581,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _init_vision_model, _init_mlp1, _parse_and_validate_multimodal_inputs，涉及 `__init__, _init_vision_model, _init_mlp1`。
- 代码 diff 细节:
  - `vllm/model_executor/models/internvl.py` modified +12/-39 (51 lines); hunks: -23,7 +23,6; -582,14 +581,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _init_vision_model, _init_mlp1, _parse_and_validate_multimodal_inputs
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/internvl.py
@@ -23,7 +23,6 @@
-    InternVisionPatchModel,
@@ -582,14 +581,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        llm_arch_name = config.text_config.architectures[0]
-        self.is_mono = llm_arch_name == "InternLM2VEForCausalLM"
-                is_mono=self.is_mono,
@@ -604,7 +599,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/internvl.py` modified +12/-39
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43586 - [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR

- 链接: https://github.com/vllm-project/vllm/pull/43586
- 状态/时间: merged / 2026-06-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+809/-69，可读 patch 1559 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR」；模型线: InternVL 3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`；技术摘要: 覆盖「[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR」；主要实现面是 `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features，涉及 `get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__`；`docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata，涉及 `BudgetGraphMetadata`；`tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template，涉及 `VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template`；`examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2，涉及 `run_tarsier2`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features
  - `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template
  - `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2
  - `vllm/model_executor/models/interfaces.py` modified +5/-0 (5 lines); hunks: -1623,6 +1623,7 @@ def postprocess_encoder_output(; -1643,6 +1644,7 @@ def prepare_encoder_cudagraph_capture_inputs(; symbols: postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs, prepare_encoder_cudagraph_replay_buffers, encoder_cudagraph_forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5; `vllm/model_executor/models/interfaces.py` modified +5/-0; `vllm/model_executor/models/step3_vl.py` modified +5/-0; `vllm/model_executor/models/glm4_1v.py` modified +4/-0; `vllm/model_executor/models/internvl.py` modified +4/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +63/-16; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42727 - fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config

- 链接: https://github.com/vllm-project/vllm/pull/42727
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+579/-428，可读 patch 1485 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/auto_awq.py`, `vllm/model_executor/layers/quantization/awq.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`；技术摘要: 覆盖「fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config」；主要实现面是 `vllm/model_executor/layers/quantization/auto_awq.py`, `vllm/model_executor/layers/quantization/awq.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/auto_awq.py` renamed +285/-71 (356 lines); hunks: -1,14 +1,15; -36,7 +37,6; symbols: _noop_loader, AWQMarlinConfig, for, AutoAWQConfig，涉及 `_noop_loader, AWQMarlinConfig, for`；`vllm/model_executor/layers/quantization/awq.py` removed +0/-286 (286 lines); hunks: -1,286 +0,0; symbols: AWQConfig, for, __init__, __repr__，涉及 `AWQConfig, for, __init__`；`vllm/model_executor/layers/quantization/moe_wna16.py` modified +8/-24 (32 lines); hunks: -27,9 +27,6; -55,10 +52,8 @@ def __init__(; symbols: __init__, is_moe_wna16_compatible, get_quant_method，涉及 `__init__, is_moe_wna16_compatible, get_quant_method`；`vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +10/-10 (20 lines); hunks: -728,18 +728,18 @@ def _process_weights_cpu(; -753,7 +753,7 @@ def _process_weights_cpu(; symbols: _process_weights_cpu, convert_to_wna16_moe_kernel_format，涉及 `_process_weights_cpu, convert_to_wna16_moe_kernel_format`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/auto_awq.py` renamed +285/-71 (356 lines); hunks: -1,14 +1,15; -36,7 +37,6; symbols: _noop_loader, AWQMarlinConfig, for, AutoAWQConfig
  - `vllm/model_executor/layers/quantization/awq.py` removed +0/-286 (286 lines); hunks: -1,286 +0,0; symbols: AWQConfig, for, __init__, __repr__
  - `vllm/model_executor/layers/quantization/moe_wna16.py` modified +8/-24 (32 lines); hunks: -27,9 +27,6; -55,10 +52,8 @@ def __init__(; symbols: __init__, is_moe_wna16_compatible, get_quant_method
  - `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +10/-10 (20 lines); hunks: -728,18 +728,18 @@ def _process_weights_cpu(; -753,7 +753,7 @@ def _process_weights_cpu(; symbols: _process_weights_cpu, convert_to_wna16_moe_kernel_format
  - `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py` modified +11/-9 (20 lines); hunks: -8,9 +8,8; -125,12 +124,12 @@ def _build_awq_method(self):; symbols: _build_awq_method
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/auto_awq.py
@@ -1,14 +1,15 @@
-from typing import TYPE_CHECKING, Any
+from typing import TYPE_CHECKING, Any, Union
+from vllm import _custom_ops as ops
@@ -36,7 +37,6 @@
-from vllm.model_executor.layers.quantization.awq import AWQConfig
@@ -55,7 +55,10 @@
diff -- vllm/model_executor/layers/quantization/awq.py
@@ -1,286 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from typing import TYPE_CHECKING, Any, Union
-import torch
-from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
-from transformers import PretrainedConfig
diff -- vllm/model_executor/layers/quantization/moe_wna16.py
@@ -27,9 +27,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/auto_awq.py` renamed +285/-71; `vllm/model_executor/layers/quantization/awq.py` removed +0/-286; `vllm/model_executor/layers/quantization/moe_wna16.py` modified +8/-24; `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +10/-10; `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py` modified +11/-9; `vllm/model_executor/layers/quantization/__init__.py` modified +5/-4
- 验证与风险: diff 自带测试面 `tests/quantization/test_auto_awq.py`, `tests/quantization/test_auto_round.py`, `tests/quantization/test_configs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
