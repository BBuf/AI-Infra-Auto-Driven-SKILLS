# vllm Intern-S1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/tool_chat_template_internlm2_tool.jinja` | no direct PR-number commit |
| `tests/tool_parsers/test_internlm2_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/models/internlm2.py` | no direct PR-number commit |
| `vllm/model_executor/models/interns1.py` | [#21628](https://github.com/vllm-project/vllm/pull/21628), [#21671](https://github.com/vllm-project/vllm/pull/21671), [#22417](https://github.com/vllm-project/vllm/pull/22417), [#23510](https://github.com/vllm-project/vllm/pull/23510), [#25644](https://github.com/vllm-project/vllm/pull/25644) |
| `vllm/model_executor/models/interns1_pro.py` | [#33636](https://github.com/vllm-project/vllm/pull/33636), [#33793](https://github.com/vllm-project/vllm/pull/33793) |
| `vllm/model_executor/models/interns1_vit.py` | [#21628](https://github.com/vllm-project/vllm/pull/21628), [#27480](https://github.com/vllm-project/vllm/pull/27480) |
| `vllm/tool_parsers/internlm2_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 8
- Extra PRs preserved from existing docs: 16
- Total PRs in this document: 24
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-26 | [#21628](https://github.com/vllm-project/vllm/pull/21628) | merged | Support Intern-S1 | `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py` |
| 2025-07-27 | [#21671](https://github.com/vllm-project/vllm/pull/21671) | merged | [VLM] Add video support for Intern-S1 | `vllm/model_executor/models/interns1.py` |
| 2025-08-07 | [#22417](https://github.com/vllm-project/vllm/pull/22417) | merged | [Bugfix] Fix wrong method name in Intern-S1 image processor | `vllm/model_executor/models/interns1.py` |
| 2025-09-02 | [#23510](https://github.com/vllm-project/vllm/pull/23510) | merged | Migrate Interns1 inputs to TensorSchema | `vllm/model_executor/models/interns1.py` |
| 2025-09-25 | [#25644](https://github.com/vllm-project/vllm/pull/25644) | merged | [Bugfix] Fix InternS1 video processing after Transformers v4.56 | `vllm/model_executor/models/interns1.py` |
| 2025-10-24 | [#27480](https://github.com/vllm-project/vllm/pull/27480) | merged | [Bugfix] Fix interns1-vit qk norm code path | `vllm/model_executor/models/interns1_vit.py` |
| 2026-02-03 | [#33636](https://github.com/vllm-project/vllm/pull/33636) | merged | [Models] Intern-S1-Pro | `vllm/model_executor/models/interns1_pro.py` |
| 2026-02-04 | [#33750](https://github.com/vllm-project/vllm/pull/33750) | merged | [MM] Align the prefix of MMEncoderAttention with Attention | `vllm/model_executor/models/aimv2.py`, `vllm/model_executor/models/blip.py`, `vllm/model_executor/models/glm4_1v.py` |
| 2026-02-04 | [#33793](https://github.com/vllm-project/vllm/pull/33793) | merged | [Bugfix] Fix interns1-pro initialization and PP | `vllm/model_executor/models/interns1_pro.py` |
| 2026-02-11 | [#34330](https://github.com/vllm-project/vllm/pull/34330) | merged | [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder` | `vllm/model_executor/models/idefics3.py`, `vllm/multimodal/processing/dummy_inputs.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2026-02-13 | [#34358](https://github.com/vllm-project/vllm/pull/34358) | merged | [Bugfix] Standardize getting number of image patches/tokens | `vllm/model_executor/models/gemma3_mm.py`, `vllm/model_executor/models/lfm2_vl.py`, `vllm/model_executor/models/idefics3.py` |
| 2026-02-16 | [#34585](https://github.com/vllm-project/vllm/pull/34585) | merged | [CI/Build] Enable tests for recent day-0 new models | `vllm/model_executor/models/interns1_pro.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py` |
| 2026-02-23 | [#35025](https://github.com/vllm-project/vllm/pull/35025) | merged | [Refactor] Simplify dummy data generation | `vllm/config/multimodal.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/multimodal/registry.py` |
| 2026-03-09 | [#34858](https://github.com/vllm-project/vllm/pull/34858) | merged | Increase Flexibility for OOV Multimodal Token Handling | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2026-03-16 | [#27599](https://github.com/vllm-project/vllm/pull/27599) | merged | [CI/Build] Add common tool call parser test suite | `tests/tool_parsers/common_tests.py`, `tests/tool_parsers/test_internlm2_tool_parser.py`, `tests/tool_parsers/test_granite_tool_parser.py` |
| 2026-03-19 | [#37545](https://github.com/vllm-project/vllm/pull/37545) | merged | [Model] Remove unnecessary `get_language_model` | `vllm/model_executor/models/kimi_audio.py`, `vllm/model_executor/models/lightonocr.py`, `vllm/model_executor/models/hyperclovax_vision_v2.py` |
| 2026-03-25 | [#35182](https://github.com/vllm-project/vllm/pull/35182) | merged | [Misc] Reorganize inputs | `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py` |
| 2026-03-26 | [#38029](https://github.com/vllm-project/vllm/pull/38029) | merged | [Tool Parser][1/3] Pass tools to ToolParser constructor | `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py` |
| 2026-03-31 | [#38264](https://github.com/vllm-project/vllm/pull/38264) | merged | [Mypy] Fix adjust_request typing | `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py` |
| 2026-03-31 | [#38189](https://github.com/vllm-project/vllm/pull/38189) | merged | [Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers | `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-05-26 | [#38278](https://github.com/vllm-project/vllm/pull/38278) | merged | [Model] Use AutoWeightsLoader for InternLM2 | `vllm/model_executor/models/internlm2.py` |
| 2026-06-12 | [#45129](https://github.com/vllm-project/vllm/pull/45129) | merged | [Model] Remove Mono-InternVL (InternLM2VEForCausalLM) | `vllm/model_executor/models/internlm2_ve.py`, `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/skyworkr1v.py` |

## Per-PR Diff Audit Cards

### PR #21628 - Support Intern-S1

- Link: https://github.com/vllm-project/vllm/pull/21628
- Status/date: merged / 2025-07-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`; associated commits `875af38e0121`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1196/-0, 1247 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Intern-S1"; model line: Intern-S1; category: model support/runtime entry; main diff: `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`; technical summary: Covers "Support Intern-S1"; the main implementation surface is `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1.py` added +711/-0 (711 lines); hunks: -0,0 +1,711; symbols: InternS1MultiModalProjector, __init__, forward, InternS1ImagePixelInputs, touching `InternS1MultiModalProjector, __init__, forward`; `vllm/model_executor/models/interns1_vit.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: InternS1VisionPatchEmbeddings, __init__, forward, InternS1VisionEmbeddings, touching `InternS1VisionPatchEmbeddings, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` added +711/-0 (711 lines); hunks: -0,0 +1,711; symbols: InternS1MultiModalProjector, __init__, forward, InternS1ImagePixelInputs
  - `vllm/model_executor/models/interns1_vit.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: InternS1VisionPatchEmbeddings, __init__, forward, InternS1VisionEmbeddings
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -0,0 +1,711 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# --------------------------------------------------------
+# InternS1
+# Copyright (c) 2025 Shanghai AI Lab
+# Licensed under The MIT License [see LICENSE for details]
diff -- vllm/model_executor/models/interns1_vit.py
@@ -0,0 +1,421 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1.py` added +711/-0; `vllm/model_executor/models/interns1_vit.py` added +421/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21671 - [VLM] Add video support for Intern-S1

- Link: https://github.com/vllm-project/vllm/pull/21671
- Status/date: merged / 2025-07-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1.py`; associated commits `3d847a3125cd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +173/-50, 375 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Add video support for Intern-S1"; model line: Intern-S1; category: model support/runtime entry; main diff: `vllm/model_executor/models/interns1.py`; technical summary: Covers "[VLM] Add video support for Intern-S1"; the main implementation surface is `vllm/model_executor/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1.py` modified +166/-45 (211 lines); hunks: -9,9 +9,10; -139,13 +140,13 @@ def get_interns1_target_ratios(; symbols: get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits, touching `get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` modified +166/-45 (211 lines); hunks: -9,9 +9,10; -139,13 +140,13 @@ def get_interns1_target_ratios(; symbols: get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -9,9 +9,10 @@
+import regex as re
-from transformers import InternVLProcessor, PretrainedConfig
+from transformers import BatchFeature, InternVLProcessor, PretrainedConfig
@@ -139,13 +140,13 @@ def get_interns1_target_ratios(
-    """Basic image-only ProcessingInfo for InternS1-style models."""
+    """ProcessingInfo for InternS1-style models."""
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1.py` modified +166/-45
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22417 - [Bugfix] Fix wrong method name in Intern-S1 image processor

- Link: https://github.com/vllm-project/vllm/pull/22417
- Status/date: merged / 2025-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1.py`; associated commits `04cf435d95fe`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix wrong method name in Intern-S1 image processor"; model line: Intern-S1; category: bug fix; main diff: `vllm/model_executor/models/interns1.py`; technical summary: Covers "[Bugfix] Fix wrong method name in Intern-S1 image processor"; the main implementation surface is `vllm/model_executor/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens, touching `get_num_image_tokens`.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -161,7 +161,7 @@ def get_num_image_tokens(
-        num_image_patches = processor.get_number_of_image_tokens(
+        num_image_patches = processor.get_number_of_image_patches(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/interns1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23510 - Migrate Interns1 inputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/23510
- Status/date: merged / 2025-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1.py`; associated commits `56d04089ef50`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +50/-51, 167 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate Interns1 inputs to TensorSchema"; model line: Intern-S1; category: model implementation change; main diff: `vllm/model_executor/models/interns1.py`; technical summary: Covers "Migrate Interns1 inputs to TensorSchema"; the main implementation surface is `vllm/model_executor/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1.py` modified +50/-51 (101 lines); hunks: -7,7 +7,7; -32,6 +32,7; symbols: forward, InternS1ImagePixelInputs, InternS1ImageEmbeddingInputs, touching `forward, InternS1ImagePixelInputs, InternS1ImageEmbeddingInputs`.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` modified +50/-51 (101 lines); hunks: -7,7 +7,7; -32,6 +32,7; symbols: forward, InternS1ImagePixelInputs, InternS1ImageEmbeddingInputs
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -7,7 +7,7 @@
-from typing import Literal, Optional, TypedDict, Union
+from typing import Annotated, Literal, Optional, Union
@@ -32,6 +32,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -62,51 +63,60 @@ def forward(self, image_features):
-class InternS1ImagePixelInputs(TypedDict):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1.py` modified +50/-51
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/interns1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25644 - [Bugfix] Fix InternS1 video processing after Transformers v4.56

- Link: https://github.com/vllm-project/vllm/pull/25644
- Status/date: merged / 2025-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1.py`; associated commits `03858e6d1c85`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +68/-3, 128 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix InternS1 video processing after Transformers v4.56"; model line: Intern-S1; category: bug fix; main diff: `vllm/model_executor/models/interns1.py`; technical summary: Covers "[Bugfix] Fix InternS1 video processing after Transformers v4.56"; the main implementation surface is `vllm/model_executor/models/interns1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1.py` modified +10/-1 (11 lines); hunks: -16,6 +16,8; -31,6 +33,8; symbols: InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits, touching `InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits`.
- Code diff details:
  - `vllm/model_executor/models/interns1.py` modified +10/-1 (11 lines); hunks: -16,6 +16,8; -31,6 +33,8; symbols: InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -16,6 +16,8 @@
+from transformers.models.internvl.video_processing_internvl import (
+    InternVLVideoProcessor)
@@ -31,6 +33,8 @@
+from vllm.transformers_utils.processor import (
+    cached_video_processor_from_config)
@@ -152,7 +156,12 @@ class InternS1ProcessingInfo(BaseProcessingInfo):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1.py` modified +10/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27480 - [Bugfix] Fix interns1-vit qk norm code path

- Link: https://github.com/vllm-project/vllm/pull/27480
- Status/date: merged / 2025-10-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1_vit.py`; associated commits `acc78aeb88c8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-4, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix interns1-vit qk norm code path"; model line: Intern-S1; category: bug fix; main diff: `vllm/model_executor/models/interns1_vit.py`; technical summary: Covers "[Bugfix] Fix interns1-vit qk norm code path"; the main implementation surface is `vllm/model_executor/models/interns1_vit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1_vit.py` modified +3/-4 (7 lines); hunks: -217,16 +217,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/interns1_vit.py` modified +3/-4 (7 lines); hunks: -217,16 +217,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1_vit.py
@@ -217,16 +217,15 @@ def __init__(
-        B, N, C = x.shape
+        """x shape: (B, N, C)"""
-            B_, N_, H_, D_ = q.shape
-            q = self.q_norm(q.flatten(-2, -1)).view(B_, N_, H_, D_)
-            k = self.k_norm(k.flatten(-2, -1)).view(B_, N_, H_, D_)
+            q = self.q_norm(q)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1_vit.py` modified +3/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/interns1_vit.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33636 - [Models] Intern-S1-Pro

- Link: https://github.com/vllm-project/vllm/pull/33636
- Status/date: merged / 2026-02-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1_pro.py`; associated commits `a3acfa10719a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +942/-11, 1062 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Intern-S1-Pro"; model line: Intern-S1; category: model implementation change; main diff: `vllm/model_executor/models/interns1_pro.py`; technical summary: Covers "[Models] Intern-S1-Pro"; the main implementation surface is `vllm/model_executor/models/interns1_pro.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1_pro.py` added +633/-0 (633 lines); hunks: -0,0 +1,633; symbols: InternS1ProProcessingInfo, get_hf_config, get_hf_processor, InternS1ProMoeMLP, touching `InternS1ProProcessingInfo, get_hf_config, get_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/interns1_pro.py` added +633/-0 (633 lines); hunks: -0,0 +1,633; symbols: InternS1ProProcessingInfo, get_hf_config, get_hf_processor, InternS1ProMoeMLP
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1_pro.py
@@ -0,0 +1,633 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The Qwen Team.
+# Copyright 2025 The HuggingFace Inc. team.
+# All rights reserved.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1_pro.py` added +633/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33750 - [MM] Align the prefix of MMEncoderAttention with Attention

- Link: https://github.com/vllm-project/vllm/pull/33750
- Status/date: merged / 2026-02-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +17/-15, 151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM] Align the prefix of MMEncoderAttention with Attention"; model line: Intern-S1; category: bug fix; main diff: `vllm/model_executor/models/aimv2.py`, `vllm/model_executor/models/blip.py`, `vllm/model_executor/models/glm4_1v.py`; technical summary: Covers "[MM] Align the prefix of MMEncoderAttention with Attention"; the main implementation surface is `vllm/model_executor/models/aimv2.py`, `vllm/model_executor/models/blip.py`, `vllm/model_executor/models/glm4_1v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/aimv2.py` modified +1/-1 (2 lines); hunks: -130,7 +130,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/blip.py` modified +1/-1 (2 lines); hunks: -126,7 +126,7 @@ def __init__(; symbols: __init__, _shape, touching `__init__, _shape`; `vllm/model_executor/models/glm4_1v.py` modified +1/-1 (2 lines); hunks: -296,7 +296,7 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/glm4v.py` modified +1/-1 (2 lines); hunks: -139,7 +139,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/aimv2.py` modified +1/-1 (2 lines); hunks: -130,7 +130,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/blip.py` modified +1/-1 (2 lines); hunks: -126,7 +126,7 @@ def __init__(; symbols: __init__, _shape
  - `vllm/model_executor/models/glm4_1v.py` modified +1/-1 (2 lines); hunks: -296,7 +296,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/glm4v.py` modified +1/-1 (2 lines); hunks: -139,7 +139,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/idefics2_vision_model.py` modified +1/-1 (2 lines); hunks: -166,7 +166,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/aimv2.py
@@ -130,7 +130,7 @@ def __init__(
-            prefix=prefix,
+            prefix=f"{prefix}.attn",
diff -- vllm/model_executor/models/blip.py
@@ -126,7 +126,7 @@ def __init__(
-            prefix=prefix,
+            prefix=f"{prefix}.attn",
diff -- vllm/model_executor/models/glm4_1v.py
@@ -296,7 +296,7 @@ def __init__(
-            prefix=prefix,
+            prefix=f"{prefix}.attn",
diff -- vllm/model_executor/models/glm4v.py
@@ -139,7 +139,7 @@ def __init__(
-            prefix=prefix,
+            prefix=f"{prefix}.attn",
diff -- vllm/model_executor/models/idefics2_vision_model.py
@@ -166,7 +166,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/aimv2.py` modified +1/-1; `vllm/model_executor/models/blip.py` modified +1/-1; `vllm/model_executor/models/glm4_1v.py` modified +1/-1; `vllm/model_executor/models/glm4v.py` modified +1/-1; `vllm/model_executor/models/idefics2_vision_model.py` modified +1/-1; `vllm/model_executor/models/intern_vit.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/aimv2.py`, `vllm/model_executor/models/blip.py`, `vllm/model_executor/models/glm4_1v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33793 - [Bugfix] Fix interns1-pro initialization and PP

- Link: https://github.com/vllm-project/vllm/pull/33793
- Status/date: merged / 2026-02-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/interns1_pro.py`; associated commits `192ad4648b20`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +43/-22, 163 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix interns1-pro initialization and PP"; model line: Intern-S1; category: bug fix; main diff: `vllm/model_executor/models/interns1_pro.py`; technical summary: Covers "[Bugfix] Fix interns1-pro initialization and PP"; the main implementation surface is `vllm/model_executor/models/interns1_pro.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1_pro.py` modified +26/-12 (38 lines); hunks: -32,7 +32,6; -41,8 +40,8; symbols: __init__, InternS1ProMoeLLMForCausalLM, InternS1ProForConditionalGeneration, touching `__init__, InternS1ProMoeLLMForCausalLM, InternS1ProForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/interns1_pro.py` modified +26/-12 (38 lines); hunks: -32,7 +32,6; -41,8 +40,8; symbols: __init__, InternS1ProMoeLLMForCausalLM, InternS1ProForConditionalGeneration
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1_pro.py
@@ -32,7 +32,6 @@
-from vllm.attention.layer import Attention
@@ -41,8 +40,8 @@
+from vllm.model_executor.layers.attention import Attention
-from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
@@ -188,7 +187,6 @@ def __init__(
-            routing_method_type=RoutingMethodType.Renormalize,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1_pro.py` modified +26/-12
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34330 - [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`

- Link: https://github.com/vllm-project/vllm/pull/34330
- Status/date: merged / 2026-02-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 72 files, +131/-27, 784 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`"; model line: Intern-S1; category: model implementation change; main diff: `vllm/model_executor/models/idefics3.py`, `vllm/multimodal/processing/dummy_inputs.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "[Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder`"; the main implementation surface is `vllm/model_executor/models/idefics3.py`, `vllm/multimodal/processing/dummy_inputs.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #34358 - [Bugfix] Standardize getting number of image patches/tokens

- Link: https://github.com/vllm-project/vllm/pull/34358
- Status/date: merged / 2026-02-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +320/-332, 1617 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Standardize getting number of image patches/tokens"; model line: Intern-S1; category: bug fix; main diff: `vllm/model_executor/models/gemma3_mm.py`, `vllm/model_executor/models/lfm2_vl.py`, `vllm/model_executor/models/idefics3.py`; technical summary: Covers "[Bugfix] Standardize getting number of image patches/tokens"; the main implementation surface is `vllm/model_executor/models/gemma3_mm.py`, `vllm/model_executor/models/lfm2_vl.py`, `vllm/model_executor/models/idefics3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma3_mm.py` modified +40/-52 (92 lines); hunks: -7,6 +7,7; -84,54 +85,35 @@ def get_hf_processor(self, **kwargs: object):; symbols: get_hf_processor, get_supported_mm_limits, _resolve_image_kwargs, _resolve_kw, touching `get_hf_processor, get_supported_mm_limits, _resolve_image_kwargs`; `vllm/model_executor/models/lfm2_vl.py` modified +44/-21 (65 lines); hunks: -176,7 +176,7 @@ def _get_grid_layout(; -190,18 +190,27 @@ def _get_image_feature_grid_size(; symbols: _get_grid_layout, _get_image_feature_grid_size, get_num_patches, get_image_repl, touching `_get_grid_layout, _get_image_feature_grid_size, get_num_patches`; `vllm/model_executor/models/idefics3.py` modified +22/-42 (64 lines); hunks: -16,7 +16,6; -168,54 +167,35 @@ def _get_image_feature_grid_size(; symbols: _get_image_feature_grid_size, get_num_patches, _get_image_token, get_image_repl, touching `_get_image_feature_grid_size, get_num_patches, _get_image_token`; `vllm/model_executor/models/keye.py` modified +28/-16 (44 lines); hunks: -10,7 +10,7; -1011,24 +1011,25 @@ def _get_vision_info(; symbols: _get_vision_info, get_num_image_tokens, get_num_video_tokens, get_image_size_with_most_features, touching `_get_vision_info, get_num_image_tokens, get_num_video_tokens`.
- Code diff details:
  - `vllm/model_executor/models/gemma3_mm.py` modified +40/-52 (92 lines); hunks: -7,6 +7,7; -84,54 +85,35 @@ def get_hf_processor(self, **kwargs: object):; symbols: get_hf_processor, get_supported_mm_limits, _resolve_image_kwargs, _resolve_kw
  - `vllm/model_executor/models/lfm2_vl.py` modified +44/-21 (65 lines); hunks: -176,7 +176,7 @@ def _get_grid_layout(; -190,18 +190,27 @@ def _get_image_feature_grid_size(; symbols: _get_grid_layout, _get_image_feature_grid_size, get_num_patches, get_image_repl
  - `vllm/model_executor/models/idefics3.py` modified +22/-42 (64 lines); hunks: -16,7 +16,6; -168,54 +167,35 @@ def _get_image_feature_grid_size(; symbols: _get_image_feature_grid_size, get_num_patches, _get_image_token, get_image_repl
  - `vllm/model_executor/models/keye.py` modified +28/-16 (44 lines); hunks: -10,7 +10,7; -1011,24 +1011,25 @@ def _get_vision_info(; symbols: _get_vision_info, get_num_image_tokens, get_num_video_tokens, get_image_size_with_most_features
  - `vllm/model_executor/models/cohere2_vision.py` modified +10/-31 (41 lines); hunks: -11,7 +11,7; -166,43 +166,20 @@ def get_num_patches(; symbols: get_num_patches, Cohere2VisionDummyInputsBuilder, _call_hf_processor, get_replacement
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gemma3_mm.py
@@ -7,6 +7,7 @@
+from transformers.models.gemma3.image_processing_gemma3 import Gemma3ImageProcessor
@@ -84,54 +85,35 @@ def get_hf_processor(self, **kwargs: object):
-    def _resolve_image_kwargs(
-        self,
-        processor: Gemma3Processor,
-        keys: set[str],
diff -- vllm/model_executor/models/lfm2_vl.py
@@ -176,7 +176,7 @@ def _get_grid_layout(
-    ) -> tuple[int, int]:
+    ) -> tuple[int, int, int]:
@@ -190,18 +190,27 @@ def _get_image_feature_grid_size(
-        processor: Lfm2VlProcessor | None,
-    ) -> tuple[int, int]:
-        if processor is None:
diff -- vllm/model_executor/models/idefics3.py
@@ -16,7 +16,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma3_mm.py` modified +40/-52; `vllm/model_executor/models/lfm2_vl.py` modified +44/-21; `vllm/model_executor/models/idefics3.py` modified +22/-42; `vllm/model_executor/models/keye.py` modified +28/-16; `vllm/model_executor/models/cohere2_vision.py` modified +10/-31; `vllm/model_executor/models/ernie45_vl.py` modified +27/-12
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_mrope.py`, `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/processing/test_gemma3.py`, `tests/models/multimodal/processing/test_idefics3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34585 - [CI/Build] Enable tests for recent day-0 new models

- Link: https://github.com/vllm-project/vllm/pull/34585
- Status/date: merged / 2026-02-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +6/-16, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI/Build] Enable tests for recent day-0 new models"; model line: Intern-S1; category: docs/tests/CI; main diff: `vllm/model_executor/models/interns1_pro.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`; technical summary: Covers "[CI/Build] Enable tests for recent day-0 new models"; the main implementation surface is `vllm/model_executor/models/interns1_pro.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interns1_pro.py` modified +3/-7 (10 lines); hunks: -85,11 +85,7 @@ def get_hf_config(self):; -497,7 +493,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: get_hf_config, get_hf_processor, InternS1ProMoeMLP, __init__, touching `get_hf_config, get_hf_processor, InternS1ProMoeMLP`; `tests/models/multimodal/processing/test_common.py` modified +3/-4 (7 lines); hunks: -102,13 +102,13 @@ def glmasr_patch_mm_data(mm_data: MultiModalDataDict) -> M...; -450,6 +450,8 @@ def test_processing_correctness(; symbols: glmasr_patch_mm_data, test_processing_correctness, touching `glmasr_patch_mm_data, test_processing_correctness`; `tests/models/multimodal/processing/test_tensor_schema.py` modified +0/-3 (3 lines); hunks: -160,9 +160,6 @@ def test_model_tensor_schema(model_id: str):; symbols: test_model_tensor_schema, touching `test_model_tensor_schema`; `tests/models/registry.py` modified +0/-2 (2 lines); hunks: -730,7 +730,6 @@ def check_available_online(; -755,7 +754,6 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`.
- Code diff details:
  - `vllm/model_executor/models/interns1_pro.py` modified +3/-7 (10 lines); hunks: -85,11 +85,7 @@ def get_hf_config(self):; -497,7 +493,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: get_hf_config, get_hf_processor, InternS1ProMoeMLP, __init__
  - `tests/models/multimodal/processing/test_common.py` modified +3/-4 (7 lines); hunks: -102,13 +102,13 @@ def glmasr_patch_mm_data(mm_data: MultiModalDataDict) -> M...; -450,6 +450,8 @@ def test_processing_correctness(; symbols: glmasr_patch_mm_data, test_processing_correctness
  - `tests/models/multimodal/processing/test_tensor_schema.py` modified +0/-3 (3 lines); hunks: -160,9 +160,6 @@ def test_model_tensor_schema(model_id: str):; symbols: test_model_tensor_schema
  - `tests/models/registry.py` modified +0/-2 (2 lines); hunks: -730,7 +730,6 @@ def check_available_online(; -755,7 +754,6 @@ def check_available_online(; symbols: check_available_online
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interns1_pro.py
@@ -85,11 +85,7 @@ def get_hf_config(self):
-        return AutoProcessor.from_pretrained(
-            self.ctx.model_config.model,
-            trust_remote_code=True,
-            **kwargs,
-        )
+        return self.ctx.get_hf_processor(**kwargs)
diff -- tests/models/multimodal/processing/test_common.py
@@ -102,13 +102,13 @@ def glmasr_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:
+    "lfm2_vl": False,
-    "lfm2_vl": False,
@@ -450,6 +450,8 @@ def test_processing_correctness(
+    if model_id == "allendou/Fun-ASR-Nano-2512-vllm":
+        pytest.skip("Cached audio `input_features` not matched. Fix later.")
@@ -468,9 +470,6 @@ def test_processing_correctness(
diff -- tests/models/multimodal/processing/test_tensor_schema.py
@@ -160,9 +160,6 @@ def test_model_tensor_schema(model_id: str):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interns1_pro.py` modified +3/-7
  - tests: `tests/models/multimodal/processing/test_common.py` modified +3/-4; `tests/models/multimodal/processing/test_tensor_schema.py` modified +0/-3; `tests/models/registry.py` modified +0/-2
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35025 - [Refactor] Simplify dummy data generation

- Link: https://github.com/vllm-project/vllm/pull/35025
- Status/date: merged / 2026-02-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 78 files, +282/-367, 1791 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Simplify dummy data generation"; model line: Intern-S1; category: model implementation change; main diff: `vllm/config/multimodal.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/multimodal/registry.py`; technical summary: Covers "[Refactor] Simplify dummy data generation"; the main implementation surface is `vllm/config/multimodal.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/multimodal/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #34858 - Increase Flexibility for OOV Multimodal Token Handling

- Link: https://github.com/vllm-project/vllm/pull/34858
- Status/date: merged / 2026-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 28 files, +79/-77, 619 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Increase Flexibility for OOV Multimodal Token Handling"; model line: Intern-S1; category: model implementation change; main diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "Increase Flexibility for OOV Multimodal Token Handling"; the main implementation surface is `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interfaces.py` modified +31/-18 (49 lines); hunks: -130,6 +130,13 @@ class SupportsMultiModal(Protocol):; -149,6 +156,17 @@ def embed_multimodal(self, **kwargs: object) -> MultiModalE...; symbols: SupportsMultiModal, get_placeholder_str, embed_multimodal, configure_mm_token_handling, touching `SupportsMultiModal, get_placeholder_str, embed_multimodal`; `vllm/model_executor/models/step3_vl.py` modified +13/-4 (17 lines); hunks: -937,14 +937,26 @@ def get_placeholder_str(cls, modality: str, i: int) -> str...; -1080,8 +1092,6 @@ def embed_input_ids(; symbols: get_placeholder_str, __init__, embed_input_ids, forward, touching `get_placeholder_str, __init__, embed_input_ids`; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3 (12 lines); hunks: -1428,11 +1428,19 @@ def embed_input_ids(; -1450,7 +1458,6 @@ def embed_input_ids(; symbols: embed_input_ids, forward, touching `embed_input_ids, forward`; `vllm/model_executor/models/granite_speech.py` modified +6/-3 (9 lines); hunks: -600,6 +600,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -793,8 +799,6 @@ def embed_input_ids(; symbols: __init__, embed_input_ids, forward, touching `__init__, embed_input_ids, forward`.
- Code diff details:
  - `vllm/model_executor/models/interfaces.py` modified +31/-18 (49 lines); hunks: -130,6 +130,13 @@ class SupportsMultiModal(Protocol):; -149,6 +156,17 @@ def embed_multimodal(self, **kwargs: object) -> MultiModalE...; symbols: SupportsMultiModal, get_placeholder_str, embed_multimodal, configure_mm_token_handling
  - `vllm/model_executor/models/step3_vl.py` modified +13/-4 (17 lines); hunks: -937,14 +937,26 @@ def get_placeholder_str(cls, modality: str, i: int) -> str...; -1080,8 +1092,6 @@ def embed_input_ids(; symbols: get_placeholder_str, __init__, embed_input_ids, forward
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3 (12 lines); hunks: -1428,11 +1428,19 @@ def embed_input_ids(; -1450,7 +1458,6 @@ def embed_input_ids(; symbols: embed_input_ids, forward
  - `vllm/model_executor/models/granite_speech.py` modified +6/-3 (9 lines); hunks: -600,6 +600,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -793,8 +799,6 @@ def embed_input_ids(; symbols: __init__, embed_input_ids, forward
  - `vllm/model_executor/models/llava_next.py` modified +5/-3 (8 lines); hunks: -270,6 +270,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -497,8 +502,6 @@ def embed_input_ids(; symbols: __init__, embed_input_ids, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -130,6 +130,13 @@ class SupportsMultiModal(Protocol):
+    _has_oov_mm_tokens: bool = False
+    """
+    In general, this should be set at init time by invoking
+    `configure_mm_token_handling` models & passing all potentially
+    OOV multimodal tokens.
+    """
diff -- vllm/model_executor/models/step3_vl.py
@@ -937,14 +937,26 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | None:
+        # NOTE: This behavior is consistent with the previous OOV handling,
+        # but does not currently handle the start/stop toks around the
+        # image features (<patch_start> <patch_end> <im_start> <im_end>)
+        # See: https://huggingface.co/stepfun-ai/step3/blob/main/processing_step3v.py#L323
+        #
+        # If this becomes an issue or we refactor to handle this using the
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -1428,11 +1428,19 @@ def embed_input_ids(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +31/-18; `vllm/model_executor/models/step3_vl.py` modified +13/-4; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3; `vllm/model_executor/models/granite_speech.py` modified +6/-3; `vllm/model_executor/models/llava_next.py` modified +5/-3; `vllm/model_executor/models/ultravox.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/clip.py`, `vllm/model_executor/models/eagle2_5_vl.py`, `vllm/model_executor/models/ernie45_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27599 - [CI/Build] Add common tool call parser test suite

- Link: https://github.com/vllm-project/vllm/pull/27599
- Status/date: merged / 2026-03-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1201/-5, 1251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI/Build] Add common tool call parser test suite"; model line: Intern-S1; category: docs/tests/CI; main diff: `tests/tool_parsers/common_tests.py`, `tests/tool_parsers/test_internlm2_tool_parser.py`, `tests/tool_parsers/test_granite_tool_parser.py`; technical summary: Covers "[CI/Build] Add common tool call parser test suite"; the main implementation surface is `tests/tool_parsers/common_tests.py`, `tests/tool_parsers/test_internlm2_tool_parser.py`, `tests/tool_parsers/test_granite_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/common_tests.py` added +378/-0 (378 lines); hunks: -0,0 +1,378; symbols: ToolParserTestConfig, ToolParserTests, providing, TestMistralToolParser, touching `ToolParserTestConfig, ToolParserTests, providing`; `tests/tool_parsers/test_internlm2_tool_parser.py` added +122/-0 (122 lines); hunks: -0,0 +1,122; symbols: TestInternLM2ToolParser, tokenizer, test_config, touching `TestInternLM2ToolParser, tokenizer, test_config`; `tests/tool_parsers/test_granite_tool_parser.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: TestGraniteToolParser, test_config, test_granite_token_prefix_format, test_granite_string_prefix_format, touching `TestGraniteToolParser, test_config, test_granite_token_prefix_format`; `tests/tool_parsers/test_step3_tool_parser.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: TestStep3ToolParser, tokenizer, test_config, touching `TestStep3ToolParser, tokenizer, test_config`.
- Code diff details:
  - `tests/tool_parsers/common_tests.py` added +378/-0 (378 lines); hunks: -0,0 +1,378; symbols: ToolParserTestConfig, ToolParserTests, providing, TestMistralToolParser
  - `tests/tool_parsers/test_internlm2_tool_parser.py` added +122/-0 (122 lines); hunks: -0,0 +1,122; symbols: TestInternLM2ToolParser, tokenizer, test_config
  - `tests/tool_parsers/test_granite_tool_parser.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: TestGraniteToolParser, test_config, test_granite_token_prefix_format, test_granite_string_prefix_format
  - `tests/tool_parsers/test_step3_tool_parser.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: TestStep3ToolParser, tokenizer, test_config
  - `tests/tool_parsers/test_phi4mini_tool_parser.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: TestPhi4MiniToolParser, tokenizer, test_config
- Key code excerpts:

```diff
diff -- tests/tool_parsers/common_tests.py
@@ -0,0 +1,378 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+from dataclasses import dataclass, field
+from types import NoneType
+from typing import Any
diff -- tests/tool_parsers/test_internlm2_tool_parser.py
@@ -0,0 +1,122 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from unittest.mock import MagicMock
+import pytest
+from tests.tool_parsers.common_tests import (
+    ToolParserTestConfig,
diff -- tests/tool_parsers/test_granite_tool_parser.py
@@ -0,0 +1,118 @@
```

- Reviewed files:
  - tests: `tests/tool_parsers/common_tests.py` added +378/-0; `tests/tool_parsers/test_internlm2_tool_parser.py` added +122/-0; `tests/tool_parsers/test_granite_tool_parser.py` added +118/-0; `tests/tool_parsers/test_step3_tool_parser.py` added +112/-0; `tests/tool_parsers/test_phi4mini_tool_parser.py` added +110/-0; `tests/tool_parsers/test_longcat_tool_parser.py` added +101/-0
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/tool_parsers/test_gigachat3_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_hunyuan_a13b_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_olmo3_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37545 - [Model] Remove unnecessary `get_language_model`

- Link: https://github.com/vllm-project/vllm/pull/37545
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +93/-95, 302 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove unnecessary `get_language_model`"; model line: Intern-S1; category: model implementation change; main diff: `vllm/model_executor/models/kimi_audio.py`, `vllm/model_executor/models/lightonocr.py`, `vllm/model_executor/models/hyperclovax_vision_v2.py`; technical summary: Covers "[Model] Remove unnecessary `get_language_model`"; the main implementation surface is `vllm/model_executor/models/kimi_audio.py`, `vllm/model_executor/models/lightonocr.py`, `vllm/model_executor/models/hyperclovax_vision_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_audio.py` modified +18/-28 (46 lines); hunks: -15,7 +15,6; -54,7 +53,6; symbols: __init__, forward, compute_logits, load_weights, touching `__init__, forward, compute_logits`; `vllm/model_executor/models/lightonocr.py` modified +23/-22 (45 lines); hunks: -163,29 +163,30 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/hyperclovax_vision_v2.py` modified +14/-23 (37 lines); hunks: -470,15 +470,6 @@ def __init__(; -492,18 +483,21 @@ def __init__(; symbols: __init__, _parse_and_validate_multimodal_inputs, get_language_model, embed_multimodal, touching `__init__, _parse_and_validate_multimodal_inputs, get_language_model`; `vllm/model_executor/models/cohere_asr.py` modified +21/-8 (29 lines); hunks: -1704,6 +1704,12 @@ def _calc_context_sizes(; -1714,7 +1720,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: _calc_context_sizes, CohereASRProjector, CohereASRModel, __init__, touching `_calc_context_sizes, CohereASRProjector, CohereASRModel`.
- Code diff details:
  - `vllm/model_executor/models/kimi_audio.py` modified +18/-28 (46 lines); hunks: -15,7 +15,6; -54,7 +53,6; symbols: __init__, forward, compute_logits, load_weights
  - `vllm/model_executor/models/lightonocr.py` modified +23/-22 (45 lines); hunks: -163,29 +163,30 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/hyperclovax_vision_v2.py` modified +14/-23 (37 lines); hunks: -470,15 +470,6 @@ def __init__(; -492,18 +483,21 @@ def __init__(; symbols: __init__, _parse_and_validate_multimodal_inputs, get_language_model, embed_multimodal
  - `vllm/model_executor/models/cohere_asr.py` modified +21/-8 (29 lines); hunks: -1704,6 +1704,12 @@ def _calc_context_sizes(; -1714,7 +1720,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: _calc_context_sizes, CohereASRProjector, CohereASRModel, __init__
  - `vllm/model_executor/models/fireredasr2.py` modified +10/-5 (15 lines); hunks: -754,12 +754,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_audio.py
@@ -15,7 +15,6 @@
-from vllm.model_executor.layers.logits_processor import LogitsProcessor
@@ -54,7 +53,6 @@
-from vllm.v1.sample.metadata import SamplingMetadata
@@ -431,28 +429,24 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.audio_tower = KimiAudioWhisperEncoder(
-            vllm_config=vllm_config,
diff -- vllm/model_executor/models/lightonocr.py
@@ -163,29 +163,30 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        self.vision_tower = init_vision_tower_for_llava(
-            config,
-            quant_config=quant_config,
-            require_post_norm=False,
-            prefix=maybe_prefix(prefix, "vision_tower"),
-        )
diff -- vllm/model_executor/models/hyperclovax_vision_v2.py
@@ -470,15 +470,6 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_audio.py` modified +18/-28; `vllm/model_executor/models/lightonocr.py` modified +23/-22; `vllm/model_executor/models/hyperclovax_vision_v2.py` modified +14/-23; `vllm/model_executor/models/cohere_asr.py` modified +21/-8; `vllm/model_executor/models/fireredasr2.py` modified +10/-5; `vllm/model_executor/models/interns1_pro.py` modified +7/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/fireredasr2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35182 - [Misc] Reorganize inputs

- Link: https://github.com/vllm-project/vllm/pull/35182
- Status/date: merged / 2026-03-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 142 files, +1212/-1342, 6002 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Reorganize inputs"; model line: Intern-S1; category: model implementation change; main diff: `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`; technical summary: Covers "[Misc] Reorganize inputs"; the main implementation surface is `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #38029 - [Tool Parser][1/3] Pass tools to ToolParser constructor

- Link: https://github.com/vllm-project/vllm/pull/38029
- Status/date: merged / 2026-03-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +147/-92, 858 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Parser][1/3] Pass tools to ToolParser constructor"; model line: Intern-S1; category: model implementation change; main diff: `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`; technical summary: Covers "[Tool Parser][1/3] Pass tools to ToolParser constructor"; the main implementation surface is `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab, touching `ToolParser, __init__, vocab`; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config, touching `Qwen3CoderToolParser, __init__, _reset_streaming_state`; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call, touching `__init__, setup_parser, set_tools`; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call, touching `__init__, setup_parser, set_tools`.
- Code diff details:
  - `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2 (9 lines); hunks: -17,6 +17,7; -47,8 +48,12 @@ class Llama4PythonicToolParser(ToolParser):; symbols: Llama4PythonicToolParser, __init__
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/abstract_tool_parser.py
@@ -5,13 +5,18 @@
+from typing import TypeAlias
+from openai.types.responses.tool import Tool as ResponsesTool
-from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
+    ChatCompletionToolsParam,
diff -- vllm/tool_parsers/qwen3coder_tool_parser.py
@@ -10,7 +10,6 @@
-    ChatCompletionToolsParam,
@@ -23,15 +22,16 @@
+    Tool,
-    def __init__(self, tokenizer: TokenizerLike):
-        super().__init__(tokenizer)
+    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -11,7 +11,6 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5; `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2; `vllm/tool_parsers/llama_tool_parser.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/entrypoints/openai/parser/responses_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38264 - [Mypy] Fix adjust_request typing

- Link: https://github.com/vllm-project/vllm/pull/38264
- Status/date: merged / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +49/-17, 241 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Mypy] Fix adjust_request typing"; model line: Intern-S1; category: bug fix; main diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py`; technical summary: Covers "[Mypy] Fix adjust_request typing"; the main implementation surface is `vllm/tool_parsers/deepseekv32_tool_parser.py`, `vllm/tool_parsers/functiongemma_tool_parser.py`, `vllm/tool_parsers/gigachat3_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -78,7 +79,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request, touching `__init__, adjust_request`; `vllm/tool_parsers/functiongemma_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -86,7 +87,9 @@ def _parse_arguments(self, args_str: str) -> dict:; symbols: _parse_arguments, adjust_request, touching `_parse_arguments, adjust_request`; `vllm/tool_parsers/gigachat3_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -55,7 +56,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request, touching `__init__, adjust_request`; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-1 (5 lines); hunks: -30,6 +30,7; -151,7 +152,9 @@ def _tools_enabled(request: ChatCompletionRequest) -> bool:; symbols: _tools_enabled, adjust_request, touching `_tools_enabled, adjust_request`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -78,7 +79,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request
  - `vllm/tool_parsers/functiongemma_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -86,7 +87,9 @@ def _parse_arguments(self, args_str: str) -> dict:; symbols: _parse_arguments, adjust_request
  - `vllm/tool_parsers/gigachat3_tool_parser.py` modified +4/-1 (5 lines); hunks: -18,6 +18,7; -55,7 +56,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-1 (5 lines); hunks: -30,6 +30,7; -151,7 +152,9 @@ def _tools_enabled(request: ChatCompletionRequest) -> bool:; symbols: _tools_enabled, adjust_request
  - `vllm/tool_parsers/granite4_tool_parser.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -59,7 +60,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, adjust_request
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -19,6 +19,7 @@
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
@@ -78,7 +79,9 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-    def adjust_request(self, request):
+    def adjust_request(
+        self, request: ChatCompletionRequest | ResponsesRequest
+    ) -> ChatCompletionRequest | ResponsesRequest:
diff -- vllm/tool_parsers/functiongemma_tool_parser.py
@@ -18,6 +18,7 @@
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
@@ -86,7 +87,9 @@ def _parse_arguments(self, args_str: str) -> dict:
-    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
+    def adjust_request(
+        self, request: ChatCompletionRequest | ResponsesRequest
+    ) -> ChatCompletionRequest | ResponsesRequest:
diff -- vllm/tool_parsers/gigachat3_tool_parser.py
@@ -18,6 +18,7 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +4/-1; `vllm/tool_parsers/functiongemma_tool_parser.py` modified +4/-1; `vllm/tool_parsers/gigachat3_tool_parser.py` modified +4/-1; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +4/-1; `vllm/tool_parsers/granite4_tool_parser.py` modified +4/-1; `vllm/tool_parsers/hermes_tool_parser.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/serve/render/serving.py`, `vllm/parser/abstract_parser.py`, `vllm/tool_parsers/abstract_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38189 - [Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers

- Link: https://github.com/vllm-project/vllm/pull/38189
- Status/date: merged / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +113/-105, 532 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers"; model line: Intern-S1; category: docs/tests/CI; main diff: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`; technical summary: Covers "[Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers"; the main implementation surface is `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +32/-27 (59 lines); hunks: -27,21 +27,26 @@ def glm4_moe_tokenizer():; -671,14 +676,13 @@ def test_streaming_json_escape_in_string(glm4_moe_tool_par...; symbols: glm4_moe_tokenizer, glm4_moe_tool_parser, mock_request, sample_tools, touching `glm4_moe_tokenizer, glm4_moe_tool_parser, mock_request`; `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +19/-13 (32 lines); hunks: -11,6 +11,10; -24,8 +28,8; symbols: make_parser, make_tool_param, test_content_before_tool_call_streaming, test_type_conversion_in_streaming, touching `make_parser, make_tool_param, test_content_before_tool_call_streaming`; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +10/-12 (22 lines); hunks: -31,13 +31,13 @@ def qwen3_tokenizer():; -376,7 +376,7 @@ def test_extract_tool_calls_fallback_no_tags(; symbols: qwen3_tokenizer, qwen3_tool_parser, qwen3_xml_tool_parser, touching `qwen3_tokenizer, qwen3_tool_parser, qwen3_xml_tool_parser`; `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +13/-8 (21 lines); hunks: -25,14 +25,8 @@ def glm47_tokenizer():; -49,6 +43,17 @@ def mock_request() -> ChatCompletionRequest:; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, sample_tools, touching `glm47_tokenizer, glm47_tool_parser, mock_request`.
- Code diff details:
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +32/-27 (59 lines); hunks: -27,21 +27,26 @@ def glm4_moe_tokenizer():; -671,14 +676,13 @@ def test_streaming_json_escape_in_string(glm4_moe_tool_par...; symbols: glm4_moe_tokenizer, glm4_moe_tool_parser, mock_request, sample_tools
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +19/-13 (32 lines); hunks: -11,6 +11,10; -24,8 +28,8; symbols: make_parser, make_tool_param, test_content_before_tool_call_streaming, test_type_conversion_in_streaming
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +10/-12 (22 lines); hunks: -31,13 +31,13 @@ def qwen3_tokenizer():; -376,7 +376,7 @@ def test_extract_tool_calls_fallback_no_tags(; symbols: qwen3_tokenizer, qwen3_tool_parser, qwen3_xml_tool_parser
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +13/-8 (21 lines); hunks: -25,14 +25,8 @@ def glm47_tokenizer():; -49,6 +43,17 @@ def mock_request() -> ChatCompletionRequest:; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, sample_tools
  - `tests/tool_parsers/test_step3p5_tool_parser.py` modified +8/-10 (18 lines); hunks: -28,8 +28,8 @@ def step3p5_tokenizer():; -386,7 +386,7 @@ def test_extract_tool_calls_fallback_no_tags(step3p5_tool_pa...; symbols: step3p5_tokenizer, step3p5_tool_parser, test_extract_tool_calls_fallback_no_tags, test_extract_tool_calls_type_conversion
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_glm4_moe_tool_parser.py
@@ -27,21 +27,26 @@ def glm4_moe_tokenizer():
-def glm4_moe_tool_parser(glm4_moe_tokenizer):
-    return Glm4MoeModelToolParser(glm4_moe_tokenizer)
-@pytest.fixture
-def mock_request() -> ChatCompletionRequest:
-    request = Mock(spec=ChatCompletionRequest)
-    request.tools = [  # GLM45 parser needs this attribute to enable tool parsing.
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -11,6 +11,10 @@
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionToolsParam,
+    FunctionDefinition,
+)
@@ -24,8 +28,8 @@
-def make_parser() -> DeepSeekV32ToolParser:
diff -- tests/tool_parsers/test_qwen3coder_tool_parser.py
@@ -31,13 +31,13 @@ def qwen3_tokenizer():
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +32/-27; `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +19/-13; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +10/-12; `tests/tool_parsers/test_glm47_moe_tool_parser.py` modified +13/-8; `tests/tool_parsers/test_step3p5_tool_parser.py` modified +8/-10
  - runtime: `vllm/tool_parsers/abstract_tool_parser.py` modified +10/-1; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +3/-6; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +3/-5
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- Link: https://github.com/vllm-project/vllm/pull/35949
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +325/-702, 2430 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; model line: Intern-S1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`; technical summary: Covers "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; the main implementation surface is `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake, touching `_resolve_layer_name, _moe_forward, _moe_forward_shared`; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__, touching `FusedMoE, __init__`; `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__
  - `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +5/-30 (35 lines); hunks: -100,7 +100,7 @@ def __init__(; -170,7 +170,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py
@@ -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:
-# the runner's 'forward_dispatch' method.
+# the runner's '_forward_dispatch' method.
+# These functions should never be called directly since they do not
+# include all the functionality of the MoE layer.
-    return layer.runner.forward_dispatch(
+    return layer.runner._forward_dispatch(
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -230,11 +230,18 @@ class FusedMoE(PluggableLayer):
-        reduce_results: Whether to all_reduce on the output of the layer
+        routed_scaling_factor: A scaling factor that is applied to the topk_weights
+                               by the router or the output of the layer depending
+                               on the value of `apply_routed_scale_to_output`
+        apply_routed_scale_to_output: Determine whether or not `routed_scaling_factor`
+                                      is applied to the topk_weights or to the experts
diff -- vllm/model_executor/models/exaone_moe.py
@@ -31,6 +31,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/models/exaone_moe.py` modified +18/-28; `vllm/model_executor/models/kimi_linear.py` modified +20/-26; `vllm/model_executor/models/AXK1.py` modified +5/-30; `vllm/model_executor/models/ernie45_vl_moe.py` modified +5/-30
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: Intern-S1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping, touching `extra_repr, fused_moe_make_expert_params_mapping`; `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits, touching `make_empty_intermediate_tensors, get_expert_mapping, load_weights`; `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights, touching `compute_logits, get_expert_mapping, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -414,7 +417,7 @@ def load_moe_expert_weights(
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -41,7 +41,9 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38278 - [Model] Use AutoWeightsLoader for InternLM2

- Link: https://github.com/vllm-project/vllm/pull/38278
- Status/date: merged / 2026-05-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +42/-34, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use AutoWeightsLoader for InternLM2"; model line: Intern-S1; category: model implementation change; main diff: `vllm/model_executor/models/internlm2.py`; technical summary: Covers "[Model] Use AutoWeightsLoader for InternLM2"; the main implementation surface is `vllm/model_executor/models/internlm2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/internlm2.py` modified +42/-34 (76 lines); hunks: -41,6 +41,7; -308,6 +309,42 @@ def forward(; symbols: forward, load_weights, InternLM2ForCausalLM, compute_logits, touching `forward, load_weights, InternLM2ForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/internlm2.py` modified +42/-34 (76 lines); hunks: -41,6 +41,7; -308,6 +309,42 @@ def forward(; symbols: forward, load_weights, InternLM2ForCausalLM, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internlm2.py
@@ -41,6 +41,7 @@
+    AutoWeightsLoader,
@@ -308,6 +309,42 @@ def forward(
+    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        stacked_params_mapping = [
+            # (param_name, shard_name, shard_id)
+            ("gate_up_proj", "w1", 0),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internlm2.py` modified +42/-34
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/internlm2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45129 - [Model] Remove Mono-InternVL (InternLM2VEForCausalLM)

- Link: https://github.com/vllm-project/vllm/pull/45129
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +53/-262, 470 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove Mono-InternVL (InternLM2VEForCausalLM)"; model line: Intern-S1; category: model implementation change; main diff: `vllm/model_executor/models/internlm2_ve.py`, `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/skyworkr1v.py`; technical summary: Covers "[Model] Remove Mono-InternVL (InternLM2VEForCausalLM)"; the main implementation surface is `vllm/model_executor/models/internlm2_ve.py`, `vllm/model_executor/models/internvl.py`, `vllm/model_executor/models/skyworkr1v.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/internlm2_ve.py` removed +0/-139 (139 lines); hunks: -1,139 +0,0; symbols: InternLM2VEDecoderLayer, __init__, forward, InternLM2VEModel, touching `InternLM2VEDecoderLayer, __init__, forward`; `vllm/model_executor/models/internvl.py` modified +12/-39 (51 lines); hunks: -23,7 +23,6; -582,14 +581,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _init_vision_model, _init_mlp1, _parse_and_validate_multimodal_inputs, touching `__init__, _init_vision_model, _init_mlp1`; `vllm/model_executor/models/skyworkr1v.py` modified +12/-32 (44 lines); hunks: -22,7 +22,6; -178,14 +177,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _init_vision_model, _init_mlp1, _process_image_input, touching `__init__, _init_vision_model, _init_mlp1`; `vllm/model_executor/models/nvlm_d.py` modified +15/-20 (35 lines); hunks: -177,27 +177,22 @@ def _init_vision_model(; symbols: _init_vision_model, touching `_init_vision_model`.
- Code diff details:
  - `vllm/model_executor/models/internlm2_ve.py` removed +0/-139 (139 lines); hunks: -1,139 +0,0; symbols: InternLM2VEDecoderLayer, __init__, forward, InternLM2VEModel
  - `vllm/model_executor/models/internvl.py` modified +12/-39 (51 lines); hunks: -23,7 +23,6; -582,14 +581,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _init_vision_model, _init_mlp1, _parse_and_validate_multimodal_inputs
  - `vllm/model_executor/models/skyworkr1v.py` modified +12/-32 (44 lines); hunks: -22,7 +22,6; -178,14 +177,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _init_vision_model, _init_mlp1, _process_image_input
  - `vllm/model_executor/models/nvlm_d.py` modified +15/-20 (35 lines); hunks: -177,27 +177,22 @@ def _init_vision_model(; symbols: _init_vision_model
  - `vllm/model_executor/models/h2ovl.py` modified +12/-17 (29 lines); hunks: -157,27 +157,22 @@ def _init_vision_model(; symbols: _init_vision_model, get_num_mm_encoder_tokens
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/internlm2_ve.py
@@ -1,139 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from itertools import islice
-import torch
-from torch import nn
-from transformers import PretrainedConfig
diff -- vllm/model_executor/models/internvl.py
@@ -23,7 +23,6 @@
-    InternVisionPatchModel,
@@ -582,14 +581,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        llm_arch_name = config.text_config.architectures[0]
-        self.is_mono = llm_arch_name == "InternLM2VEForCausalLM"
-                is_mono=self.is_mono,
@@ -604,7 +599,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
diff -- vllm/model_executor/models/skyworkr1v.py
@@ -22,7 +22,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/internlm2_ve.py` removed +0/-139; `vllm/model_executor/models/internvl.py` modified +12/-39; `vllm/model_executor/models/skyworkr1v.py` modified +12/-32; `vllm/model_executor/models/nvlm_d.py` modified +15/-20; `vllm/model_executor/models/h2ovl.py` modified +12/-17
  - tests: `tests/models/registry.py` modified +0/-11; `tests/models/multimodal/generation/test_common.py` modified +0/-2
  - docs: `docs/models/supported_models.md` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
