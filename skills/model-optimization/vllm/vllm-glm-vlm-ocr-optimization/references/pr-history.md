# vllm GLM VLM/OCR PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/models/multimodal/processing/test_glm4_1v.py` | no direct PR-number commit |
| `vllm/model_executor/models/glm4_1v.py` | [#21678](https://github.com/vllm-project/vllm/pull/21678), [#22751](https://github.com/vllm-project/vllm/pull/22751), [#33005](https://github.com/vllm-project/vllm/pull/33005), [#34483](https://github.com/vllm-project/vllm/pull/34483), [#37962](https://github.com/vllm-project/vllm/pull/37962) |
| `vllm/model_executor/models/glm4v.py` | no direct PR-number commit |
| `vllm/model_executor/models/glm_ocr.py` | [#33005](https://github.com/vllm-project/vllm/pull/33005), [#33350](https://github.com/vllm-project/vllm/pull/33350), [#37962](https://github.com/vllm-project/vllm/pull/37962) |
| `vllm/model_executor/models/glm_ocr_mtp.py` | [#33005](https://github.com/vllm-project/vllm/pull/33005) |
| `vllm/transformers_utils/processors/glm4v.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #9242 - [Model] Add GLM-4v support and meet vllm==0.6.2

- Link: https://github.com/vllm-project/vllm/pull/9242
- Status/date: merged / 2024-10-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +776/-72, 1059 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR adds or enables a model support/runtime surface. Title: "[Model] Add GLM-4v support and meet vllm==0.6.2". The diff centers on `vllm/model_executor/models/chatglm.py`, `vllm/model_executor/models/glm4_vision_encoder.py`, `tests/models/decoder_only/vision_language/test_glm4.py`. PR body context: Overview --- This PR support the glm-4v-9b multimodal model while maintaining compatibility with `chatglm`. This PR was inspired and reused some code here #5358 This PR is updat...
- Key implementation: `vllm/model_executor/models/chatglm.py` modified +298/-52 (350 lines); hunks: -1,42 +1,229; -127,7 +314,7 @@ class GLMMLP(nn.Module):; symbols: calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings, GLMImagePixelInputs, touching `calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings`; `vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0 (298 lines); hunks: -0,0 +1,298; symbols: PatchEmbedding, __init__, forward, Attention, touching `PatchEmbedding, __init__, forward`; `tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: run_test, processor, test_models, touching `run_test, processor, test_models`; `vllm/transformers_utils/tokenizer.py` modified +21/-18 (39 lines); hunks: -59,6 +59,26 @@ def __len__(self):; -143,24 +163,7 @@ def get_tokenizer(; symbols: __len__, patch_padding_side, _pad, get_tokenizer, touching `__len__, patch_padding_side, _pad`.
- Code diff details:
  - `vllm/model_executor/models/chatglm.py` modified +298/-52 (350 lines); hunks: -1,42 +1,229; -127,7 +314,7 @@ class GLMMLP(nn.Module):; symbols: calculate_image_placeholder, mm_input_mapper_for_glmv, merge_glm_vision_embeddings, GLMImagePixelInputs
  - `vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0 (298 lines); hunks: -0,0 +1,298; symbols: PatchEmbedding, __init__, forward, Attention
  - `tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: run_test, processor, test_models
  - `vllm/transformers_utils/tokenizer.py` modified +21/-18 (39 lines); hunks: -59,6 +59,26 @@ def __len__(self):; -143,24 +163,7 @@ def get_tokenizer(; symbols: __len__, patch_padding_side, _pad, get_tokenizer
  - `docs/source/models/supported_models.rst` modified +6/-0 (6 lines); hunks: -346,6 +346,12 @@ Text Generation
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/chatglm.py` modified +298/-52; `vllm/model_executor/models/glm4_vision_encoder.py` added +298/-0; `vllm/transformers_utils/tokenizer.py` modified +21/-18; `vllm/model_executor/models/registry.py` modified +4/-2
  - tests: `tests/models/decoder_only/vision_language/test_glm4.py` added +133/-0
  - docs: `docs/source/models/supported_models.rst` modified +6/-0; `examples/offline_inference_vision_language.py` modified +16/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_glm4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19331 - Add GLM-4.1V model

- Link: https://github.com/vllm-project/vllm/pull/19331
- Status/date: merged / 2025-07-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +1946/-16, 2230 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR adds or enables a model support/runtime surface. Title: "Add GLM-4.1V model". The diff centers on `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/layers/rotary_embedding.py`, `vllm/multimodal/parse.py`. PR body context: This PR aims to add support for the GLM-4.1V model. Due to the upgrade of the implementation in the transformers library, some interfaces have been changed. This model definitel...
- Key implementation: `vllm/model_executor/models/glm4_1v.py` added +1589/-0 (1589 lines); hunks: -0,0 +1,1589; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs, Glm4vVideoEmbeddingInputs, touching `Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs`; `vllm/model_executor/layers/rotary_embedding.py` modified +119/-0 (119 lines); hunks: -23,6 +23,7; -1118,6 +1119,15 @@ def get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor, touching `get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor`; `vllm/multimodal/parse.py` modified +40/-2 (42 lines); hunks: -224,8 +224,14 @@ def __init__(self, data: Union[torch.Tensor, list[torch.Ten...; -320,13 +326,15 @@ def __init__(; symbols: __init__, VideoProcessorItems, get_num_frames, touching `__init__, VideoProcessorItems, get_num_frames`; `tests/models/multimodal/generation/test_common.py` modified +28/-0 (28 lines); hunks: -309,6 +309,34.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` added +1589/-0 (1589 lines); hunks: -0,0 +1,1589; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs, Glm4vVideoEmbeddingInputs
  - `vllm/model_executor/layers/rotary_embedding.py` modified +119/-0 (119 lines); hunks: -23,6 +23,7; -1118,6 +1119,15 @@ def get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/multimodal/parse.py` modified +40/-2 (42 lines); hunks: -224,8 +224,14 @@ def __init__(self, data: Union[torch.Tensor, list[torch.Ten...; -320,13 +326,15 @@ def __init__(; symbols: __init__, VideoProcessorItems, get_num_frames
  - `tests/models/multimodal/generation/test_common.py` modified +28/-0 (28 lines); hunks: -309,6 +309,34
  - `vllm/multimodal/video.py` modified +21/-6 (27 lines); hunks: -24,6 +24,7 @@ def resize_video(frames: npt.NDArray, size: tuple[int, int]) -...; -92,14 +93,16 @@ def get_cv2_video_api(self):; symbols: resize_video, get_cv2_video_api, load_bytes
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_1v.py` added +1589/-0; `vllm/model_executor/layers/rotary_embedding.py` modified +119/-0; `vllm/multimodal/parse.py` modified +40/-2; `vllm/multimodal/video.py` modified +21/-6
  - tests: `tests/models/multimodal/generation/test_common.py` modified +28/-0; `tests/models/multimodal/generation/vlm_utils/model_utils.py` modified +24/-0; `tests/models/multimodal/processing/test_common.py` modified +24/-0; `tests/models/multimodal/generation/vlm_utils/custom_inputs.py` modified +20/-0
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/test_video.py`, `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/vlm_utils/custom_inputs.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21678 - Migrate Glm4vImageInputs, Glm4vVideoInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/21678
- Status/date: merged / 2025-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_1v.py`; associated commits `88e46c7c8dfa`
- Diff scope read: GitHub Pull Request files API returned 2 files, +69/-66, 218 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR changes model-related implementation. Title: "Migrate Glm4vImageInputs, Glm4vVideoInputs to TensorSchema". The diff centers on `vllm/model_executor/models/glm4_1v.py`. PR body context: ## Purpose This PR migrates Glm4vImageInputs and Glm4vVideoInputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This brings...
- Key implementation: `vllm/model_executor/models/glm4_1v.py` modified +46/-65 (111 lines); hunks: -29,7 +29,7; -70,6 +70,7; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs, touching `Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs`.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` modified +46/-65 (111 lines); hunks: -29,7 +29,7; -70,6 +70,7; symbols: Glm4vImagePixelInputs, Glm4vImageEmbeddingInputs, Glm4vVideoPixelInputs
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +46/-65
- Risk and verification: The diff ships test coverage in `tests/standalone_tests/test_tensor_schema.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22751 - [Model] Decouple glm4v

- Link: https://github.com/vllm-project/vllm/pull/22751
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_1v.py`; associated commits `fde0b611a37e`
- Diff scope read: GitHub Pull Request files API returned 3 files, +23/-7, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Model] Decouple glm4v". The diff centers on `vllm/model_executor/models/glm4_1v.py`. PR body context: # Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, FIX #22077 - [ ] The test plan, such as providing test command. - [ ] The test results...
- Key implementation: `vllm/model_executor/models/glm4_1v.py` modified +21/-5 (26 lines); hunks: -1227,10 +1227,7 @@ class Glm4vForConditionalGeneration(nn.Module, SupportsMu...; -1567,7 +1564,26 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: Glm4vForConditionalGeneration, get_mm_mapping, Glm4vMoeForConditionalGeneration, touching `Glm4vForConditionalGeneration, get_mm_mapping, Glm4vMoeForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` modified +21/-5 (26 lines); hunks: -1227,10 +1227,7 @@ class Glm4vForConditionalGeneration(nn.Module, SupportsMu...; -1567,7 +1564,26 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: Glm4vForConditionalGeneration, get_mm_mapping, Glm4vMoeForConditionalGeneration
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +21/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/registry.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27860 - [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V

- Link: https://github.com/vllm-project/vllm/pull/27860
- Status/date: merged / 2025-10-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +147/-2, 184 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V". The diff centers on `vllm/model_executor/models/glm4_1v.py`. PR body context: ## Purpose - Fix #27854 ## Test Plan ## Test Result Example should work now. --- Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as...
- Key implementation: `vllm/model_executor/models/glm4_1v.py` modified +147/-2 (149 lines); hunks: -26,6 +26,7; -36,7 +37,7; symbols: get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings, get_mrope_input_positions, touching `get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings`.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` modified +147/-2 (149 lines); hunks: -26,6 +26,7; -36,7 +37,7; symbols: get_video_replacement_glm4v, Glm4vForConditionalGeneration, get_multimodal_embeddings, get_mrope_input_positions
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +147/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_1v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33005 - [GLM-OCR] GLM-OCR with MTP Support

- Link: https://github.com/vllm-project/vllm/pull/33005
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`; associated commits `bb17e8f11c38`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +873/-8, 1048 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR adds or enables a model support/runtime surface. Title: "[GLM-OCR] GLM-OCR with MTP Support". The diff centers on `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm_ocr_mtp.py`, `vllm/model_executor/models/glm4_1v.py`. PR body context: A dense model using the GLM-4-0414 architecture with bias, featuring a completely new VIT structure and MTP implementation.
- Key implementation: `vllm/model_executor/models/glm_ocr.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: GlmOcrVisionMLP, GlmOcrVisionAttention, __init__, split_qkv, touching `GlmOcrVisionMLP, GlmOcrVisionAttention, __init__`; `vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: GlmOcrMultiTokenPredictorLayer, __init__, forward, GlmOcrMultiTokenPredictor, touching `GlmOcrMultiTokenPredictorLayer, __init__, forward`; `vllm/model_executor/models/glm4_1v.py` modified +3/-2 (5 lines); hunks: -24,7 +24,8; -1418,7 +1419,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/glm_ocr.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: GlmOcrVisionMLP, GlmOcrVisionAttention, __init__, split_qkv
  - `vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: GlmOcrMultiTokenPredictorLayer, __init__, forward, GlmOcrMultiTokenPredictor
  - `vllm/model_executor/models/glm4_1v.py` modified +3/-2 (5 lines); hunks: -24,7 +24,8; -1418,7 +1419,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm_ocr.py` added +389/-0; `vllm/model_executor/models/glm_ocr_mtp.py` added +285/-0; `vllm/model_executor/models/glm4_1v.py` modified +3/-2
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/test_vit_backend_functionality.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33350 - [Bugfix] Fix broken GLM-OCR initialization

- Link: https://github.com/vllm-project/vllm/pull/33350
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm_ocr.py`; associated commits `5e73e4900c80`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Fix broken GLM-OCR initialization". The diff centers on `vllm/model_executor/models/glm_ocr.py`. PR body context: ## Purpose - The GLM-OCR model is broken actually, because `GlmOcrVisionConfig` is only imported at type checking https://github.com/vllm-project/vllm/blob/c6e7404cc5713a926e8b6...
- Key implementation: `vllm/model_executor/models/glm_ocr.py` modified +1/-1 (2 lines); hunks: -249,7 +249,7 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__, touching `GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__`.
- Code diff details:
  - `vllm/model_executor/models/glm_ocr.py` modified +1/-1 (2 lines); hunks: -249,7 +249,7 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glm_ocr.py
@@ -249,7 +249,7 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):
-        vision_config: GlmOcrVisionConfig,
+        vision_config: "GlmOcrVisionConfig",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm_ocr.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34483 - [Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image

- Link: https://github.com/vllm-project/vllm/pull/34483
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_1v.py`; associated commits `dcf6ee8592b4`
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-2, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image". The diff centers on `vllm/model_executor/models/glm4_1v.py`. PR body context: ## Purpose Fixes #34040 `get_image_size_with_most_features()` and `get_num_image_tokens()` called `_get_vision_info()` with the default `num_frames=16` (video), causing `smart_r...
- Key implementation: `vllm/model_executor/models/glm4_1v.py` modified +22/-2 (24 lines); hunks: -869,9 +869,28 @@ def _get_vision_info(; -884,7 +903,8 @@ def get_num_image_tokens(; symbols: _get_vision_info, _get_image_max_pixels, get_image_size_with_most_features, get_num_image_tokens, touching `_get_vision_info, _get_image_max_pixels, get_image_size_with_most_features`.
- Code diff details:
  - `vllm/model_executor/models/glm4_1v.py` modified +22/-2 (24 lines); hunks: -869,9 +869,28 @@ def _get_vision_info(; -884,7 +903,8 @@ def get_num_image_tokens(; symbols: _get_vision_info, _get_image_max_pixels, get_image_size_with_most_features, get_num_image_tokens
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm4_1v.py` modified +22/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_1v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37962 - [bug-fix] GLM OCR Patch Merger context_dim

- Link: https://github.com/vllm-project/vllm/pull/37962
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/glm_ocr.py`; associated commits `757eafcf37ba`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-4, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For GLM VLM/OCR, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[bug-fix] GLM OCR Patch Merger context_dim". The diff centers on `vllm/model_executor/models/glm_ocr.py`, `vllm/model_executor/models/glm4_1v.py`. PR body context: ## Purpose Modified the reading logic in GLM-OCR, the original algorithm implementation error, although the numbers just want to wait, but in fact, it should be text_config inte...
- Key implementation: `vllm/model_executor/models/glm_ocr.py` modified +8/-3 (11 lines); hunks: -35,7 +35,10; -250,12 +253,13 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__, touching `GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__`; `vllm/model_executor/models/glm4_1v.py` modified +6/-1 (7 lines); hunks: -38,7 +38,10; -604,6 +607,7 @@ def forward(; symbols: forward, Glm4vVisionTransformer, __init__, touching `forward, Glm4vVisionTransformer, __init__`.
- Code diff details:
  - `vllm/model_executor/models/glm_ocr.py` modified +8/-3 (11 lines); hunks: -35,7 +35,10; -250,12 +253,13 @@ class GlmOcrPatchMerger(Glm4vPatchMerger):; symbols: GlmOcrPatchMerger, GlmOcrVisionTransformer, __init__
  - `vllm/model_executor/models/glm4_1v.py` modified +6/-1 (7 lines); hunks: -38,7 +38,10; -604,6 +607,7 @@ def forward(; symbols: forward, Glm4vVisionTransformer, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/glm_ocr.py` modified +8/-3; `vllm/model_executor/models/glm4_1v.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glm4_1v.py`, `vllm/model_executor/models/glm_ocr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.
