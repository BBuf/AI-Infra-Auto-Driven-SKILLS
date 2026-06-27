# vllm GLM VLM/OCR 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/processing/test_glm4_1v.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/glm4_1v.py` | [#21678](https://github.com/vllm-project/vllm/pull/21678), [#22751](https://github.com/vllm-project/vllm/pull/22751), [#33005](https://github.com/vllm-project/vllm/pull/33005), [#34483](https://github.com/vllm-project/vllm/pull/34483), [#37962](https://github.com/vllm-project/vllm/pull/37962) |
| `vllm/model_executor/models/glm4v.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/glm_ocr.py` | [#33005](https://github.com/vllm-project/vllm/pull/33005), [#33350](https://github.com/vllm-project/vllm/pull/33350), [#37962](https://github.com/vllm-project/vllm/pull/37962) |
| `vllm/model_executor/models/glm_ocr_mtp.py` | [#33005](https://github.com/vllm-project/vllm/pull/33005) |
| `vllm/transformers_utils/processors/glm4v.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 6
- 原文档显式引用补充 PR 数: 10
- 当前文档总 PR 数: 16
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-10-11 | [#9242](https://github.com/vllm-project/vllm/pull/9242) | merged | [Model] Add GLM-4v support and meet vllm==0.6.2 | `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py` |
| 2025-07-01 | [#19331](https://github.com/vllm-project/vllm/pull/19331) | merged | Add GLM-4.1V model | `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py` |
| 2025-07-28 | [#21678](https://github.com/vllm-project/vllm/pull/21678) | merged | Migrate Glm4vImageInputs, Glm4vVideoInputs to TensorSchema | `vllm/model_executor/models/glm4_1v.py` |
| 2025-08-13 | [#22751](https://github.com/vllm-project/vllm/pull/22751) | merged | [Model] Decouple glm4v | `vllm/model_executor/models/glm4_1v.py` |
| 2025-10-31 | [#27860](https://github.com/vllm-project/vllm/pull/27860) | merged | [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V | `vllm/model_executor/models/glm4_1v.py` |
| 2026-01-26 | [#33005](https://github.com/vllm-project/vllm/pull/33005) | merged | [GLM-OCR] GLM-OCR with MTP Support | `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4_1v.py` |
| 2026-01-29 | [#33350](https://github.com/vllm-project/vllm/pull/33350) | merged | [Bugfix] Fix broken GLM-OCR initialization | `vllm/model_executor/models/glm_ocr.py` |
| 2026-02-13 | [#34483](https://github.com/vllm-project/vllm/pull/34483) | merged | [Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image | `vllm/model_executor/models/glm4_1v.py` |
| 2026-03-25 | [#35182](https://github.com/vllm-project/vllm/pull/35182) | merged | [Misc] Reorganize inputs | `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py` |
| 2026-03-26 | [#37962](https://github.com/vllm-project/vllm/pull/37962) | merged | [bug-fix] GLM OCR Patch Merger context_dim | `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py` |
| 2026-04-22 | [#39986](https://github.com/vllm-project/vllm/pull/39986) | merged | [Multimodal] Add PyAV video backend for concurrent video decoding | `vllm/multimodal/video.py`, `tests/multimodal/test_video.py`, `tests/models/multimodal/processing/test_glm4_1v.py` |
| 2026-05-19 | [#42347](https://github.com/vllm-project/vllm/pull/42347) | merged | [Perf][4/n] Eliminate various GPU CPU syncs | `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py` |
| 2026-05-29 | [#43575](https://github.com/vllm-project/vllm/pull/43575) | merged | [feat] add GlmgaProcessor specific logits in `glm4_1v.py` | `vllm/model_executor/models/glm4_1v.py`, `vllm/multimodal/video.py`, `tests/models/registry.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-09 | [#40576](https://github.com/vllm-project/vllm/pull/40576) | merged | [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference | `vllm/model_executor/models/glm4_1v.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | merged | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |

## 逐 PR diff 审计卡

### PR #9242 - [Model] Add GLM-4v support and meet vllm==0.6.2

- 链接: https://github.com/vllm-project/vllm/pull/9242
- 状态/时间: merged / 2024-10-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+776/-72，可读 patch 1059 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add GLM-4v support and meet vllm==0.6.2」；模型线: GLM VLM/OCR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py`；技术摘要: 覆盖「[Model] Add GLM-4v support and meet vllm==0.6.2」；主要实现面是 `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/chatglm.py` modified +298/-52 (350 lines); hunks: -1,42 +1,229; -127,7 +314,7 @@ class GLMMLP(nn.Module):; symbols: calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings, GLMImagePixelInputs，涉及 `calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings`；`vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0 (298 lines); hunks: -0,0 +1,298; symbols: PatchEmbedding, __init__, forward, Attention，涉及 `PatchEmbedding, __init__, forward`；`tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: run_test, processor, test_models，涉及 `run_test, processor, test_models`；`vllm/transformers_utils/tokenizer.py` modified +21/-18 (39 lines); hunks: -59,6 +59,26 @@ def __len__(self):; -143,24 +163,7 @@ def get_tokenizer(; symbols: __len__, patch_padding_side, _pad, get_tokenizer，涉及 `__len__, patch_padding_side, _pad`。
- 代码 diff 细节:
  - `vllm/model_executor/models/chatglm.py` modified +298/-52 (350 lines); hunks: -1,42 +1,229; -127,7 +314,7 @@ class GLMMLP(nn.Module):; symbols: calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings, GLMImagePixelInputs
  - `vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0 (298 lines); hunks: -0,0 +1,298; symbols: PatchEmbedding, __init__, forward, Attention
  - `tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: run_test, processor, test_models
  - `vllm/transformers_utils/tokenizer.py` modified +21/-18 (39 lines); hunks: -59,6 +59,26 @@ def __len__(self):; -143,24 +163,7 @@ def get_tokenizer(; symbols: __len__, patch_padding_side, _pad, get_tokenizer
  - `docs/source/models/supported_models.rst` modified +6/-0 (6 lines); hunks: -346,6 +346,12 @@ Text Generation
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/chatglm.py
@@ -1,42 +1,229 @@
-# https://github.com/THUDM/ChatGLM2-6B
+# https://github.com/THUDM/GLM-4
-from typing import Iterable, List, Optional, Tuple, Union
+from argparse import Namespace
+from array import array
+from typing import Dict, Iterable, List, Mapping, Optional, Tuple, TypedDict
diff -- vllm/model_executor/models/glm4_vision_encoder.py
@@ -0,0 +1,298 @@
+# coding=utf-8
+# Adapted from
+# https://github.com/THUDM/GLM-4
+"""Inference-only GLM-4v model visual encoder compatible with THUDM weights."""
+from argparse import Namespace
+from typing import Optional
diff -- tests/models/decoder_only/vision_language/test_glm4.py
@@ -0,0 +1,133 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/chatglm.py` modified +298/-52; `vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0; `vllm/transformers_utils/tokenizer.py` modified +21/-18; `vllm/model_executor/models/registry.py` modified +4/-2
  - tests: `tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0
  - docs: `docs/source/models/supported_models.rst` modified +6/-0; `examples/offline_inference_vision_language.py` modified +16/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_glm4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19331 - Add GLM-4.1V model

- 链接: https://github.com/vllm-project/vllm/pull/19331
- 状态/时间: merged / 2025-07-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1946/-16，可读 patch 2230 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add GLM-4.1V model」；模型线: GLM VLM/OCR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py`；技术摘要: 覆盖「Add GLM-4.1V model」；主要实现面是 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_1v.py` added +1589/-0 (1589 lines); hunks: -0,0 +1,1589; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs, Glm4vVideoEmbeddingInputs，涉及 `Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs`；`vllm/model_executor/layers/rotary_embedding.py` modified +119/-0 (119 lines); hunks: -23,6 +23,7; -1118,6 +1119,15 @@ def get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor，涉及 `get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor`；`vllm/multimodal/parse.py` modified +40/-2 (42 lines); hunks: -224,8 +224,14 @@ def __init__(self, data: Union[torch.Tensor, list[torch.Ten...; -320,13 +326,15 @@ def __init__(; symbols: __init__, VideoProcessorItems, get_num_frames，涉及 `__init__, VideoProcessorItems, get_num_frames`；`tests/models/multimodal/generation/test_common.py` modified +28/-0 (28 lines); hunks: -309,6 +309,34。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_1v.py` added +1589/-0 (1589 lines); hunks: -0,0 +1,1589; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs, Glm4vVideoEmbeddingInputs
  - `vllm/model_executor/layers/rotary_embedding.py` modified +119/-0 (119 lines); hunks: -23,6 +23,7; -1118,6 +1119,15 @@ def get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/multimodal/parse.py` modified +40/-2 (42 lines); hunks: -224,8 +224,14 @@ def __init__(self, data: Union[torch.Tensor, list[torch.Ten...; -320,13 +326,15 @@ def __init__(; symbols: __init__, VideoProcessorItems, get_num_frames
  - `tests/models/multimodal/generation/test_common.py` modified +28/-0 (28 lines); hunks: -309,6 +309,34
  - `vllm/multimodal/video.py` modified +21/-6 (27 lines); hunks: -24,6 +24,7 @@ def resize_video(frames: npt.NDArray, size: tuple[int, int]) -...; -92,14 +93,16 @@ def get_cv2_video_api(self):; symbols: resize_video, get_cv2_video_api, load_bytes
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_1v.py
@@ -0,0 +1,1589 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from
+# https://github.com/huggingface/transformers/blob/main/src/transformers/models/Glm4v/modeling_Glm4v.py
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The ZhipuAI Team.
+# Copyright 2025 The HuggingFace Inc. team.
diff -- vllm/model_executor/layers/rotary_embedding.py
@@ -23,6 +23,7 @@
+import itertools
@@ -1118,6 +1119,15 @@ def get_input_positions_tensor(
+        elif "glm4v" in hf_config.model_type:
+            return cls._glm4v_get_input_positions_tensor(
+                input_tokens=input_tokens,
+                hf_config=hf_config,
diff -- vllm/multimodal/parse.py
@@ -224,8 +224,14 @@ def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_1v.py` added +1589/-0; `vllm/model_executor/layers/rotary_embedding.py` modified +119/-0; `vllm/multimodal/parse.py` modified +40/-2; `vllm/multimodal/video.py` modified +21/-6
  - tests: `tests/models/multimodal/generation/test_common.py` modified +28/-0; `tests/models/multimodal/generation/vlm_utils/model_utils.py` modified +24/-0; `tests/models/multimodal/processing/test_common.py` modified +24/-0; `tests/models/multimodal/generation/vlm_utils/custom_inputs.py` modified +20/-0
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/test_video.py`, `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/vlm_utils/custom_inputs.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21678 - Migrate Glm4vImageInputs, Glm4vVideoInputs to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/21678
- 状态/时间: merged / 2025-07-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_1v.py`；关联提交 `88e46c7c8dfa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+69/-66，可读 patch 218 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Migrate Glm4vImageInputs, Glm4vVideoInputs to TensorSchema」；模型线: GLM VLM/OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_1v.py`；技术摘要: 覆盖「Migrate Glm4vImageInputs, Glm4vVideoInputs to TensorSchema」；主要实现面是 `vllm/model_executor/models/glm4_1v.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_1v.py` modified +46/-65 (111 lines); hunks: -29,7 +29,7; -70,6 +70,7; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs，涉及 `Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_1v.py` modified +46/-65 (111 lines); hunks: -29,7 +29,7; -70,6 +70,7; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_1v.py
@@ -29,7 +29,7 @@
-from typing import Any, Callable, Literal, Optional, TypedDict, Union
+from typing import Annotated, Any, Callable, Literal, Optional, Union
@@ -70,6 +70,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -88,80 +89,68 @@
-class Glm4vImagePixelInputs(TypedDict):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +46/-65
- 验证与风险: diff 自带测试面 `tests/standalone_tests/test_tensor_schema.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22751 - [Model] Decouple glm4v

- 链接: https://github.com/vllm-project/vllm/pull/22751
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_1v.py`；关联提交 `fde0b611a37e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+23/-7，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Decouple glm4v」；模型线: GLM VLM/OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_1v.py`；技术摘要: 覆盖「[Model] Decouple glm4v」；主要实现面是 `vllm/model_executor/models/glm4_1v.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_1v.py` modified +21/-5 (26 lines); hunks: -1227,10 +1227,7 @@ class Glm4vForConditionalGeneration(nn.Module, SupportsMu...; -1567,7 +1564,26 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: Glm4vForConditionalGeneration, get_mm_mapping, Glm4vMoeForConditionalGeneration，涉及 `Glm4vForConditionalGeneration, get_mm_mapping, Glm4vMoeForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_1v.py` modified +21/-5 (26 lines); hunks: -1227,10 +1227,7 @@ class Glm4vForConditionalGeneration(nn.Module, SupportsMu...; -1567,7 +1564,26 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: Glm4vForConditionalGeneration, get_mm_mapping, Glm4vMoeForConditionalGeneration
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_1v.py
@@ -1227,10 +1227,7 @@ class Glm4vForConditionalGeneration(nn.Module, SupportsMultiModal,
-        "gate_up_proj": [
-            "gate_proj",
-            "up_proj",
-        ],
+        "gate_up_proj": ["gate_up_proj"]
@@ -1567,7 +1564,26 @@ def get_mm_mapping(self) -> MultiModelKeys:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +21/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/registry.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27860 - [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V

- 链接: https://github.com/vllm-project/vllm/pull/27860
- 状态/时间: merged / 2025-10-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+147/-2，可读 patch 184 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_1v.py`；技术摘要: 覆盖「[Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V」；主要实现面是 `vllm/model_executor/models/glm4_1v.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_1v.py` modified +147/-2 (149 lines); hunks: -26,6 +26,7; -36,7 +37,7; symbols: get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings, get_mrope_input_positions，涉及 `get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_1v.py` modified +147/-2 (149 lines); hunks: -26,6 +26,7; -36,7 +37,7; symbols: get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings, get_mrope_input_positions
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_1v.py
@@ -26,6 +26,7 @@
+import itertools
@@ -36,7 +37,7 @@
-from transformers import BatchFeature
+from transformers import BatchFeature, PretrainedConfig
@@ -89,6 +90,7 @@
+    SupportsMRoPE,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +147/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_1v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33005 - [GLM-OCR] GLM-OCR with MTP Support

- 链接: https://github.com/vllm-project/vllm/pull/33005
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`；关联提交 `bb17e8f11c38`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+873/-8，可读 patch 1048 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-OCR] GLM-OCR with MTP Support」；模型线: GLM VLM/OCR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4_1v.py`；技术摘要: 覆盖「[GLM-OCR] GLM-OCR with MTP Support」；主要实现面是 `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4_1v.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm_ocr.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: GlmOcrVisionMLP, GlmOcrVisionAttention, __init__, split_qkv，涉及 `GlmOcrVisionMLP, GlmOcrVisionAttention, __init__`；`vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: GlmOcrMultiTokenPredictorLayer, __init__, forward, GlmOcrMultiTokenPredictor，涉及 `GlmOcrMultiTokenPredictorLayer, __init__, forward`；`vllm/model_executor/models/glm4_1v.py` modified +3/-2 (5 lines); hunks: -24,7 +24,8; -1418,7 +1419,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm_ocr.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: GlmOcrVisionMLP, GlmOcrVisionAttention, __init__, split_qkv
  - `vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: GlmOcrMultiTokenPredictorLayer, __init__, forward, GlmOcrMultiTokenPredictor
  - `vllm/model_executor/models/glm4_1v.py` modified +3/-2 (5 lines); hunks: -24,7 +24,8; -1418,7 +1419,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm_ocr.py
@@ -0,0 +1,389 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/huggingface/transformers/blob/main/src/transformers/models/Glm4v/modeling_Glm4v.py
+# Copyright 2026 The ZhipuAI Team.
+# Copyright 2026 The vLLM team.
diff -- vllm/model_executor/models/glm_ocr_mtp.py
@@ -0,0 +1,285 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2026 The ZhipuAI Team.
+# Copyright 2026 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/glm4_1v.py
@@ -24,7 +24,8 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm_ocr.py` added +389/-0; `vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0; `vllm/model_executor/models/glm4_1v.py` modified +3/-2
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33350 - [Bugfix] Fix broken GLM-OCR initialization

- 链接: https://github.com/vllm-project/vllm/pull/33350
- 状态/时间: merged / 2026-01-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm_ocr.py`；关联提交 `5e73e4900c80`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix broken GLM-OCR initialization」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm_ocr.py`；技术摘要: 覆盖「[Bugfix] Fix broken GLM-OCR initialization」；主要实现面是 `vllm/model_executor/models/glm_ocr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm_ocr.py` modified +1/-1 (2 lines); hunks: -249,7 +249,7 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__，涉及 `GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm_ocr.py` modified +1/-1 (2 lines); hunks: -249,7 +249,7 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm_ocr.py
@@ -249,7 +249,7 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):
-        vision_config: GlmOcrVisionConfig,
+        vision_config: "GlmOcrVisionConfig",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm_ocr.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34483 - [Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image

- 链接: https://github.com/vllm-project/vllm/pull/34483
- 状态/时间: merged / 2026-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_1v.py`；关联提交 `dcf6ee8592b4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-2，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_1v.py`；技术摘要: 覆盖「[Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image」；主要实现面是 `vllm/model_executor/models/glm4_1v.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_1v.py` modified +22/-2 (24 lines); hunks: -869,9 +869,28 @@ def _get_vision_info(; -884,7 +903,8 @@ def get_num_image_tokens(; symbols: _get_vision_info, _get_image_max_pixels, get_image_size_with_most_features, get_num_image_tokens，涉及 `_get_vision_info, _get_image_max_pixels, get_image_size_with_most_features`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_1v.py` modified +22/-2 (24 lines); hunks: -869,9 +869,28 @@ def _get_vision_info(; -884,7 +903,8 @@ def get_num_image_tokens(; symbols: _get_vision_info, _get_image_max_pixels, get_image_size_with_most_features, get_num_image_tokens
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_1v.py
@@ -869,9 +869,28 @@ def _get_vision_info(
+    def _get_image_max_pixels(self) -> int:
+        """Read max_pixels from the HF image processor config.
+        Despite the name, ``longest_edge`` is a pixel **area** (total pixel
+        count), not an edge length.  The HF processor passes it directly to
+        ``smart_resize`` as the ``max_pixels`` argument, which constrains
+        ``t_bar * h_bar * w_bar <= max_pixels``.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +22/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_1v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35182 - [Misc] Reorganize inputs

- 链接: https://github.com/vllm-project/vllm/pull/35182
- 状态/时间: merged / 2026-03-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 142 个文件，+1212/-1342，可读 patch 6002 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Reorganize inputs」；模型线: GLM VLM/OCR；类别: 模型实现调整；主要 diff: `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`；技术摘要: 覆盖「[Misc] Reorganize inputs」；主要实现面是 `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #37962 - [bug-fix] GLM OCR Patch Merger context_dim

- 链接: https://github.com/vllm-project/vllm/pull/37962
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/glm_ocr.py`；关联提交 `757eafcf37ba`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-4，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bug-fix] GLM OCR Patch Merger context_dim」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py`；技术摘要: 覆盖「[bug-fix] GLM OCR Patch Merger context_dim」；主要实现面是 `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm_ocr.py` modified +8/-3 (11 lines); hunks: -35,7 +35,10; -250,12 +253,13 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__，涉及 `GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__`；`vllm/model_executor/models/glm4_1v.py` modified +6/-1 (7 lines); hunks: -38,7 +38,10; -604,6 +607,7 @@ def forward(; symbols: forward, Glm4vVisionTransformer, __init__，涉及 `forward, Glm4vVisionTransformer, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm_ocr.py` modified +8/-3 (11 lines); hunks: -35,7 +35,10; -250,12 +253,13 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__
  - `vllm/model_executor/models/glm4_1v.py` modified +6/-1 (7 lines); hunks: -38,7 +38,10; -604,6 +607,7 @@ def forward(; symbols: forward, Glm4vVisionTransformer, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm_ocr.py
@@ -35,7 +35,10 @@
-    from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig
+    from transformers.models.glm_ocr.configuration_glm_ocr import (
+        GlmOcrTextConfig,
+        GlmOcrVisionConfig,
+    )
@@ -250,12 +253,13 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):
diff -- vllm/model_executor/models/glm4_1v.py
@@ -38,7 +38,10 @@
-from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig
+from transformers.models.glm4v.configuration_glm4v import (
+    Glm4vTextConfig,
+    Glm4vVisionConfig,
+)
@@ -604,6 +607,7 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm_ocr.py` modified +8/-3; `vllm/model_executor/models/glm4_1v.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/glm_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39986 - [Multimodal] Add PyAV video backend for concurrent video decoding

- 链接: https://github.com/vllm-project/vllm/pull/39986
- 状态/时间: merged / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+290/-118，可读 patch 622 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Multimodal] Add PyAV video backend for concurrent video decoding」；模型线: GLM VLM/OCR；类别: 文档/测试/CI；主要 diff: `vllm/multimodal/video.py`, `tests/multimodal/test_video.py`, `tests/models/multimodal/processing/test_glm4_1v.py`；技术摘要: 覆盖「[Multimodal] Add PyAV video backend for concurrent video decoding」；主要实现面是 `vllm/multimodal/video.py`, `tests/multimodal/test_video.py`, `tests/models/multimodal/processing/test_glm4_1v.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/multimodal/video.py` modified +174/-94 (268 lines); hunks: -3,7 +3,7; -19,6 +19,11; symbols: read_frames, PyAVVideoBackendMixin, get_metadata, decode_frames，涉及 `read_frames, PyAVVideoBackendMixin, get_metadata`；`tests/multimodal/test_video.py` modified +104/-17 (121 lines); hunks: -71,7 +71,9 @@ def test_video_backend_handles_broken_frames(monkeypatch: pyte...; -158,12 +160,12 @@ def release(self):; symbols: test_video_backend_handles_broken_frames, release, test_video_recovery_with_corrupted_file, test_video_recovery_dynamic_backend，涉及 `test_video_backend_handles_broken_frames, release, test_video_recovery_with_corrupted_file`；`tests/models/multimodal/processing/test_glm4_1v.py` modified +8/-4 (12 lines); hunks: -6,7 +6,7; -70,9 +70,11 @@ def test_processor_override(; symbols: test_processor_override, test_video_loader_consistency，涉及 `test_processor_override, test_video_loader_consistency`；`vllm/envs.py` modified +4/-3 (7 lines); hunks: -829,9 +829,10 @@ def _get_or_set_default() -> str:; symbols: _get_or_set_default，涉及 `_get_or_set_default`。
- 代码 diff 细节:
  - `vllm/multimodal/video.py` modified +174/-94 (268 lines); hunks: -3,7 +3,7; -19,6 +19,11; symbols: read_frames, PyAVVideoBackendMixin, get_metadata, decode_frames
  - `tests/multimodal/test_video.py` modified +104/-17 (121 lines); hunks: -71,7 +71,9 @@ def test_video_backend_handles_broken_frames(monkeypatch: pyte...; -158,12 +160,12 @@ def release(self):; symbols: test_video_backend_handles_broken_frames, release, test_video_recovery_with_corrupted_file, test_video_recovery_dynamic_backend
  - `tests/models/multimodal/processing/test_glm4_1v.py` modified +8/-4 (12 lines); hunks: -6,7 +6,7; -70,9 +70,11 @@ def test_processor_override(; symbols: test_processor_override, test_video_loader_consistency
  - `vllm/envs.py` modified +4/-3 (7 lines); hunks: -829,9 +829,10 @@ def _get_or_set_default() -> str:; symbols: _get_or_set_default
- 关键代码摘录:

```diff
diff -- vllm/multimodal/video.py
@@ -3,7 +3,7 @@
-from typing import Any, NamedTuple, cast
+from typing import Any, ClassVar, Literal, NamedTuple, cast
@@ -19,6 +19,11 @@
+try:
+    import av
+except ImportError:
diff -- tests/multimodal/test_video.py
@@ -71,7 +71,9 @@ def test_video_backend_handles_broken_frames(monkeypatch: pytest.MonkeyPatch):
-        frames, metadata = loader.load_bytes(video_data, num_frames=-1)
+        frames, metadata = loader.load_bytes(
+            video_data, num_frames=-1, backend="opencv"
+        )
@@ -158,12 +160,12 @@ def release(self):
-            video_data, num_frames=8, frame_recovery=False
diff -- tests/models/multimodal/processing/test_glm4_1v.py
@@ -6,7 +6,7 @@
```

- 已读文件:
  - runtime: `vllm/multimodal/video.py` modified +174/-94; `vllm/envs.py` modified +4/-3
  - tests: `tests/multimodal/test_video.py` modified +104/-17; `tests/models/multimodal/processing/test_glm4_1v.py` modified +8/-4
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_glm4_1v.py`, `tests/multimodal/test_video.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42347 - [Perf][4/n] Eliminate various GPU CPU syncs

- 链接: https://github.com/vllm-project/vllm/pull/42347
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+129/-108，可读 patch 606 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][4/n] Eliminate various GPU CPU syncs」；模型线: GLM VLM/OCR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py`；技术摘要: 覆盖「[Perf][4/n] Eliminate various GPU CPU syncs」；主要实现面是 `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #43575 - [feat] add GlmgaProcessor specific logits in `glm4_1v.py`

- 链接: https://github.com/vllm-project/vllm/pull/43575
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+346/-33，可读 patch 500 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feat] add GlmgaProcessor specific logits in `glm4_1v.py`」；模型线: GLM VLM/OCR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/glm4_1v.py`, `vllm/multimodal/video.py`, `tests/models/registry.py`；技术摘要: 覆盖「[feat] add GlmgaProcessor specific logits in `glm4_1v.py`」；主要实现面是 `vllm/model_executor/models/glm4_1v.py`, `vllm/multimodal/video.py`, `tests/models/registry.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_1v.py` modified +241/-32 (273 lines); hunks: -36,7 +36,9; -122,6 +124,15; symbols: _to_video_metadata, get_image_processor, get_video_processor, get_mm_max_tokens_per_item，涉及 `_to_video_metadata, get_image_processor, get_video_processor`；`vllm/multimodal/video.py` modified +101/-0 (101 lines); hunks: -639,6 +639,107 @@ def load_bytes(; symbols: load_bytes, GLM4_6VVideoBackend, _prepare_source, compute_frames_index_to_sample，涉及 `load_bytes, GLM4_6VVideoBackend, _prepare_source`；`tests/models/registry.py` modified +4/-1 (5 lines); hunks: -935,7 +935,10 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_1v.py` modified +241/-32 (273 lines); hunks: -36,7 +36,9; -122,6 +124,15; symbols: _to_video_metadata, get_image_processor, get_video_processor, get_mm_max_tokens_per_item
  - `vllm/multimodal/video.py` modified +101/-0 (101 lines); hunks: -639,6 +639,107 @@ def load_bytes(; symbols: load_bytes, GLM4_6VVideoBackend, _prepare_source, compute_frames_index_to_sample
  - `tests/models/registry.py` modified +4/-1 (5 lines); hunks: -935,7 +935,10 @@ def check_available_online(; symbols: check_available_online
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_1v.py
@@ -36,7 +36,9 @@
+import transformers
+from packaging.version import Version
@@ -122,6 +124,15 @@
+TRANSFORMERS_WITH_GA = Version(transformers.__version__) >= Version("5.10.0.dev0")
+def _to_video_metadata(metadata: Mapping[str, Any]) -> VideoMetadata:
+    return VideoMetadata(
diff -- vllm/multimodal/video.py
@@ -639,6 +639,107 @@ def load_bytes(
+@VIDEO_LOADER_REGISTRY.register("glm4_6v")
+class GLM4_6VVideoBackend(VideoBackend):
+    @classmethod
+    def _prepare_source(cls, source: VideoSourceMetadata) -> VideoSourceMetadata:
+        # Estimate duration from frame count and fps when the container
+        # does not report it (common for WebM/streaming inputs).
diff -- tests/models/registry.py
@@ -935,7 +935,10 @@ def check_available_online(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +241/-32; `vllm/multimodal/video.py` modified +101/-0
  - tests: `tests/models/registry.py` modified +4/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- 链接: https://github.com/vllm-project/vllm/pull/43167
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 56 个文件，+88/-731，可读 patch 1251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove KV cache scale boilerplate from model weight loading methods」；模型线: GLM VLM/OCR；类别: 文档/测试/CI；主要 diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`；技术摘要: 覆盖「Remove KV cache scale boilerplate from model weight loading methods」；主要实现面是 `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name，涉及 `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`；`vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader，涉及 `_get_moe_weight_dtype, kv_cache_scale_loader`；`vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod，涉及 `KVCacheScaleParameter, __new__, weight_loader`；`vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter，涉及 `get_quant_method, get_cache_scale, get_cache_scale_mapper`。
- 代码 diff 细节:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- 关键代码摘录:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- 已读文件:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- 验证与风险: diff 自带测试面 `tests/model_executor/test_eagle_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40576 - [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference

- 链接: https://github.com/vllm-project/vllm/pull/40576
- 状态/时间: merged / 2026-06-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+480/-25，可读 patch 605 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference」；模型线: GLM VLM/OCR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/glm4_1v.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`；技术摘要: 覆盖「[MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference」；主要实现面是 `vllm/model_executor/models/glm4_1v.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`, `docs/design/cuda_graphs_multimodal.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_1v.py` modified +456/-25 (481 lines); hunks: -97,10 +97,12; -626,6 +628,11 @@ def __init__(; symbols: __init__, device, rot_pos_emb, compute_attn_mask_seqlen，涉及 `__init__, device, rot_pos_emb`；`tests/models/multimodal/generation/test_vit_cudagraph.py` modified +22/-0 (22 lines); hunks: -137,6 +137,28 @@ def step3_vl_chat_template(content: str) -> str:; symbols: step3_vl_chat_template，涉及 `step3_vl_chat_template`；`docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...；`examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2562,6 +2562,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2，涉及 `run_tarsier2`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_1v.py` modified +456/-25 (481 lines); hunks: -97,10 +97,12; -626,6 +628,11 @@ def __init__(; symbols: __init__, device, rot_pos_emb, compute_attn_mask_seqlen
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +22/-0 (22 lines); hunks: -137,6 +137,28 @@ def step3_vl_chat_template(content: str) -> str:; symbols: step3_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...
  - `examples/generate/multimodal/vision_language_offline.py` modified +1/-0 (1 lines); hunks: -2562,6 +2562,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_tarsier2
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +456/-25
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +22/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +1/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_vit_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43586 - [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR

- 链接: https://github.com/vllm-project/vllm/pull/43586
- 状态/时间: merged / 2026-06-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+809/-69，可读 patch 1559 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR」；模型线: GLM VLM/OCR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`；技术摘要: 覆盖「[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR」；主要实现面是 `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`。下方保留文件级证据、代码摘录和验证风险。
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

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
