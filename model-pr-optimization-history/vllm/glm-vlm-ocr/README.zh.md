# vllm GLM VLM/OCR 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

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
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 9
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
| 2026-03-26 | [#37962](https://github.com/vllm-project/vllm/pull/37962) | merged | [bug-fix] GLM OCR Patch Merger context_dim | `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py` |

## 逐 PR diff 审计卡

### PR #9242 - [Model] Add GLM-4v support and meet vllm==0.6.2

- 链接: https://github.com/vllm-project/vllm/pull/9242
- 状态/时间: merged / 2024-10-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+776/-72，可读 patch 1059 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add GLM-4v support and meet vllm==0.6.2」；模型线: GLM VLM/OCR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py`；PR 正文摘要: Overview This PR support the glm-4v-9b multimodal model while maintaining compatibility with `chatglm`. This PR was inspired and reused some code here #5358 This PR is updated f...。
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
- 动机: 标题「Add GLM-4.1V model」；模型线: GLM VLM/OCR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py`；PR 正文摘要: This PR aims to add support for the GLM-4.1V model. Due to the upgrade of the implementation in the transformers library, some interfaces have been changed. This model definitel...。
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
- 动机: 标题「Migrate Glm4vImageInputs, Glm4vVideoInputs to TensorSchema」；模型线: GLM VLM/OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_1v.py`；PR 正文摘要: This PR migrates Glm4vImageInputs and Glm4vVideoInputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This brings it in line...。
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
- 动机: 标题「[Model] Decouple glm4v」；模型线: GLM VLM/OCR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_1v.py`；PR 正文未提供可用摘要。
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
- 动机: 标题「[Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_1v.py`；PR 正文摘要: - Fix #27854 Example should work now.。
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
- 动机: 标题「[GLM-OCR] GLM-OCR with MTP Support」；模型线: GLM VLM/OCR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4_1v.py`；PR 正文摘要: A dense model using the GLM-4-0414 architecture with bias, featuring a completely new VIT structure and MTP implementation.。
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
- 动机: 标题「[Bugfix] Fix broken GLM-OCR initialization」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm_ocr.py`；PR 正文摘要: - The GLM-OCR model is broken actually, because `GlmOcrVisionConfig` is only imported at type checking https://github.com/vllm-project/vllm/blob/c6e7404cc5713a926e8b6c187b5f197a...。
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
- 动机: 标题「[Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_1v.py`；PR 正文摘要: Fixes #34040 `get_image_size_with_most_features()` and `get_num_image_tokens()` called `_get_vision_info()` with the default `num_frames=16` (video), causing `smart_resize` to c...。
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

### PR #37962 - [bug-fix] GLM OCR Patch Merger context_dim

- 链接: https://github.com/vllm-project/vllm/pull/37962
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/glm_ocr.py`；关联提交 `757eafcf37ba`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-4，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bug-fix] GLM OCR Patch Merger context_dim」；模型线: GLM VLM/OCR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py`；PR 正文摘要: Modified the reading logic in GLM-OCR, the original algorithm implementation error, although the numbers just want to wait, but in fact, it should be text_config intermediate_si...。
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

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
