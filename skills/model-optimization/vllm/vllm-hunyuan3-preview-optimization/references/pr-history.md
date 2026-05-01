# vllm Hunyuan3 Preview PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/reasoning/test_hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `tests/tool_parsers/test_hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3_mtp.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/reasoning/hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/tool_parsers/hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/transformers_utils/configs/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-23 | [#21368](https://github.com/vllm-project/vllm/pull/21368) | merged | [Model] add Hunyuan V1 Dense Model support. | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py` |
| 2025-11-25 | [#29327](https://github.com/vllm-project/vllm/pull/29327) | merged | [Model] Add HunyuanOCR support | `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py` |
| 2026-01-27 | [#33035](https://github.com/vllm-project/vllm/pull/33035) | merged | feature: support eagle3 for HunyuanVL & Hunyuan | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/hunyuan_vision.py`, `vllm/v1/spec_decode/eagle.py` |
| 2026-04-23 | [#40681](https://github.com/vllm-project/vllm/pull/40681) | merged | [Model] Support Hy3 preview | `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py` |

## Per-PR Diff Audit Cards

### PR #21368 - [Model] add Hunyuan V1 Dense Model support.

- Link: https://github.com/vllm-project/vllm/pull/21368
- Status/date: merged / 2025-07-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +57/-19, 136 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] add Hunyuan V1 Dense Model support."; model line: Hunyuan3 Preview; category: model support/runtime entry; main diff: `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py`; technical summary: Covers "[Model] add Hunyuan V1 Dense Model support."; the main implementation surface is `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18 (70 lines); hunks: -61,6 +61,19; -140,8 +153,8 @@ def __init__(; symbols: _is_moe, _get_cla_factor, __init__, _split_qkv_weight, touching `_is_moe, _get_cla_factor, __init__`; `vllm/model_executor/models/registry.py` modified +2/-1 (3 lines); hunks: -79,7 +79,8; `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -199,6 +199,8 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -363,6 +363,7 @@ th {.
- Code diff details:
  - `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18 (70 lines); hunks: -61,6 +61,19; -140,8 +153,8 @@ def __init__(; symbols: _is_moe, _get_cla_factor, __init__, _split_qkv_weight
  - `vllm/model_executor/models/registry.py` modified +2/-1 (3 lines); hunks: -79,7 +79,8
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -199,6 +199,8 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -363,6 +363,7 @@ th {
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -61,6 +61,19 @@
+def _is_moe(config: PretrainedConfig) -> bool:
+    num_experts = getattr(config, "num_experts", None)
+    if isinstance(num_experts, int):
+        return num_experts > 1
+    if isinstance(num_experts, list) and num_experts:
+        # Ensure all elements are integers before calling max.
diff -- vllm/model_executor/models/registry.py
@@ -79,7 +79,8 @@
-    "HunYuanMoEV1ForCausalLM": ("hunyuan_v1_moe", "HunYuanMoEV1ForCausalLM"),
+    "HunYuanMoEV1ForCausalLM": ("hunyuan_v1", "HunYuanMoEV1ForCausalLM"),
+    "HunYuanDenseV1ForCausalLM": ("hunyuan_v1", "HunYuanDenseV1ForCausalLM"),
diff -- tests/models/registry.py
@@ -199,6 +199,8 @@ def check_available_online(
+    "HunYuanDenseV1ForCausalLM":_HfExamplesInfo("tencent/Hunyuan-7B-Instruct-0124",
+                                               trust_remote_code=True),
diff -- docs/models/supported_models.md
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18; `vllm/model_executor/models/registry.py` modified +2/-1
  - tests: `tests/models/registry.py` modified +2/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29327 - [Model] Add HunyuanOCR support

- Link: https://github.com/vllm-project/vllm/pull/29327
- Status/date: merged / 2025-11-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +2415/-4, 2653 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add HunyuanOCR support"; model line: Hunyuan3 Preview; category: model support/runtime entry; main diff: `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py`; technical summary: Covers "[Model] Add HunyuanOCR support"; the main implementation surface is `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0 (1028 lines); hunks: -0,0 +1,1028; symbols: HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP, __init__, touching `HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP`; `vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0 (477 lines); hunks: -0,0 +1,477; symbols: for, smart_resize, HunYuanVLImageProcessor, __init__, touching `for, smart_resize, HunYuanVLImageProcessor`; `vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig, to, touching `HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig`; `vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0 (233 lines); hunks: -0,0 +1,233; symbols: HunYuanVLProcessor, __init__, __call__, batch_decode, touching `HunYuanVLProcessor, __init__, __call__`.
- Code diff details:
  - `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0 (1028 lines); hunks: -0,0 +1,1028; symbols: HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP, __init__
  - `vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0 (477 lines); hunks: -0,0 +1,477; symbols: for, smart_resize, HunYuanVLImageProcessor, __init__
  - `vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig, to
  - `vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0 (233 lines); hunks: -0,0 +1,233; symbols: HunYuanVLProcessor, __init__, __call__, batch_decode
  - `vllm/model_executor/layers/rotary_embedding/xdrope.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: XDRotaryEmbedding, __init__, forward, get_next_input_positions
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/hunyuan_vision.py
@@ -0,0 +1,1028 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# coding=utf-8
+# Copyright 2025 The HunYuan team.
+# Copyright 2025 The vLLM team.
+# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/transformers_utils/processors/hunyuan_vl_image.py
@@ -0,0 +1,477 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://github.com/ManaEstras/transformers/blob/v4.57.1.hyvl/src/transformers/models/hunyuan_vl/image_processing_hunyuan_vl.py
+"""Image processor class for HunYuanVL."""
+# isort conflicts with ruff for transformers imports
+# isort: skip_file
diff -- vllm/transformers_utils/configs/hunyuan_vl.py
@@ -0,0 +1,322 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0; `vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0; `vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0; `vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0; `vllm/model_executor/layers/rotary_embedding/xdrope.py` added +102/-0; `vllm/model_executor/models/interfaces.py` modified +50/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33035 - feature: support eagle3 for HunyuanVL & Hunyuan

- Link: https://github.com/vllm-project/vllm/pull/33035
- Status/date: merged / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +49/-3, 165 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feature: support eagle3 for HunyuanVL & Hunyuan"; model line: Hunyuan3 Preview; category: docs/tests/CI; main diff: `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/hunyuan_vision.py`, `vllm/v1/spec_decode/eagle.py`; technical summary: Covers "feature: support eagle3 for HunyuanVL & Hunyuan"; the main implementation surface is `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/hunyuan_vision.py`, `vllm/v1/spec_decode/eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2 (19 lines); hunks: -66,7 +66,7; -630,6 +630,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, forward, _split_qkv_weight, touching `__init__, embed_input_ids, forward`; `vllm/model_executor/models/hunyuan_vision.py` modified +9/-0 (9 lines); hunks: -83,6 +83,7; -780,6 +781,7 @@ class HunYuanVLForConditionalGeneration(; symbols: HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers, touching `HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers`; `vllm/v1/spec_decode/eagle.py` modified +15/-0 (15 lines); hunks: -115,6 +115,8 @@ def __init__(; -129,6 +131,12 @@ def __init__(; symbols: __init__, _get_positions, _set_positions, touching `__init__, _get_positions, _set_positions`; `vllm/config/speculative.py` modified +8/-1 (9 lines); hunks: -675,7 +675,14 @@ def _verify_args(self) -> Self:; symbols: _verify_args, touching `_verify_args`.
- Code diff details:
  - `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2 (19 lines); hunks: -66,7 +66,7; -630,6 +630,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, forward, _split_qkv_weight
  - `vllm/model_executor/models/hunyuan_vision.py` modified +9/-0 (9 lines); hunks: -83,6 +83,7; -780,6 +781,7 @@ class HunYuanVLForConditionalGeneration(; symbols: HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers
  - `vllm/v1/spec_decode/eagle.py` modified +15/-0 (15 lines); hunks: -115,6 +115,8 @@ def __init__(; -129,6 +131,12 @@ def __init__(; symbols: __init__, _get_positions, _set_positions
  - `vllm/config/speculative.py` modified +8/-1 (9 lines); hunks: -675,7 +675,14 @@ def _verify_args(self) -> Self:; symbols: _verify_args
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -66,7 +66,7 @@
-from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
+from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
@@ -630,6 +630,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.aux_hidden_state_layers = tuple[int, ...]()
@@ -654,9 +655,13 @@ def forward(
+        aux_hidden_states = []
diff -- vllm/model_executor/models/hunyuan_vision.py
@@ -83,6 +83,7 @@
+    SupportsEagle3,
@@ -780,6 +781,7 @@ class HunYuanVLForConditionalGeneration(
+    SupportsEagle3,
@@ -966,6 +968,13 @@ def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
+    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
+        self.language_model.model.aux_hidden_state_layers = layers
diff -- vllm/v1/spec_decode/eagle.py
@@ -115,6 +115,8 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2; `vllm/model_executor/models/hunyuan_vision.py` modified +9/-0; `vllm/v1/spec_decode/eagle.py` modified +15/-0; `vllm/config/speculative.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/config/speculative.py`, `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/hunyuan_vision.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40681 - [Model] Support Hy3 preview

- Link: https://github.com/vllm-project/vllm/pull/40681
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3.py`, `vllm/model_executor/models/hy_v3_mtp.py`, `vllm/reasoning/hy_v3_reasoning_parser.py` and 7 files; associated commits `d0009ddb0b96`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +2696/-0, 2801 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Hy3 preview"; model line: Hunyuan3 Preview; category: docs/tests/CI; main diff: `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py`; technical summary: Covers "[Model] Support Hy3 preview"; the main implementation surface is `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused, touching `HYV3FeedForward, __init__, forward`; `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options, touching `HYV3ToolParser, _normalize_type, _get_arg_schema`; `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__, touching `_is_moe, _get_cla_factor, HYV3SharedHead`; `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls, touching `hy_v3_tokenizer, hy_v3_tool_parser, mock_request`.
- Code diff details:
  - `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused
  - `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options
  - `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__
  - `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls
  - `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: hy_v3_tokenizer, test_reasoning, test_is_reasoning_end_full_prompt
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/hy_v3.py
@@ -0,0 +1,707 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# coding=utf-8
+# Copyright 2026 The HY team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -0,0 +1,645 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import ast
+import json
+from collections.abc import Sequence
+from typing import Any
diff -- vllm/model_executor/models/hy_v3_mtp.py
@@ -0,0 +1,470 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/hy_v3.py` added +707/-0; `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0; `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0; `vllm/transformers_utils/configs/hy_v3.py` added +185/-0; `vllm/reasoning/hy_v3_reasoning_parser.py` added +137/-0
  - tests: `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0; `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
