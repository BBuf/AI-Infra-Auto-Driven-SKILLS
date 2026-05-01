# vllm Intern-S1 Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/tool_chat_template_internlm2_tool.jinja` | no direct PR-number commit |
| `tests/tool_parsers/test_internlm2_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/models/internlm2.py` | no direct PR-number commit |
| `vllm/model_executor/models/internlm2_ve.py` | no direct PR-number commit |
| `vllm/model_executor/models/interns1.py` | [#21628](https://github.com/vllm-project/vllm/pull/21628), [#21671](https://github.com/vllm-project/vllm/pull/21671), [#22417](https://github.com/vllm-project/vllm/pull/22417), [#23510](https://github.com/vllm-project/vllm/pull/23510), [#25644](https://github.com/vllm-project/vllm/pull/25644) |
| `vllm/model_executor/models/interns1_pro.py` | [#33636](https://github.com/vllm-project/vllm/pull/33636), [#33793](https://github.com/vllm-project/vllm/pull/33793) |
| `vllm/model_executor/models/interns1_vit.py` | [#21628](https://github.com/vllm-project/vllm/pull/21628), [#27480](https://github.com/vllm-project/vllm/pull/27480) |
| `vllm/tool_parsers/internlm2_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 8
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 8
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
| 2026-02-04 | [#33793](https://github.com/vllm-project/vllm/pull/33793) | merged | [Bugfix] Fix interns1-pro initialization and PP | `vllm/model_executor/models/interns1_pro.py` |

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
- Motivation: Title: "[Models] Intern-S1-Pro"; model line: Intern-S1; category: model support/runtime entry; main diff: `vllm/model_executor/models/interns1_pro.py`; technical summary: Covers "[Models] Intern-S1-Pro"; the main implementation surface is `vllm/model_executor/models/interns1_pro.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
