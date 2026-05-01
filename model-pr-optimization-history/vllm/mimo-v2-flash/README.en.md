# vllm MiMo V2 Flash Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `vllm/model_executor/models/mimo.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433) |
| `vllm/model_executor/models/mimo_audio.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/model_executor/models/mimo_mtp.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433), [#25136](https://github.com/vllm-project/vllm/pull/25136) |
| `vllm/model_executor/models/mimo_v2.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967), [#41029](https://github.com/vllm-project/vllm/pull/41029) |
| `vllm/model_executor/models/mimo_v2_mtp.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/model_executor/models/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/transformers_utils/configs/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/transformers_utils/processors/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 3
- Total PRs in this document: 7
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-05-12 | [#17433](https://github.com/vllm-project/vllm/pull/17433) | merged | [Model] Support MiMo-7B inference with MTP | `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py` |
| 2025-09-18 | [#25136](https://github.com/vllm-project/vllm/pull/25136) | merged | [spec decode] Fix MTP inference path for MiMo-7B model | `vllm/model_executor/models/mimo_mtp.py` |
| 2025-12-19 | [#30836](https://github.com/vllm-project/vllm/pull/30836) | merged | [Model] Add MiMo-V2-Flash support | `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py` |
| 2026-01-05 | [#31175](https://github.com/vllm-project/vllm/pull/31175) | merged | [Bugfix] Properly apply v_scale for mimo_v2_flash | `vllm/model_executor/models/mimo_v2_flash.py` |
| 2026-04-24 | [#40045](https://github.com/vllm-project/vllm/pull/40045) | merged | [Attention] use diff kv backend for mimo v2 flash | `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/attention/attention.py`, `tools/pre_commit/generate_attention_backend_docs.py` |
| 2026-04-27 | [#40967](https://github.com/vllm-project/vllm/pull/40967) | merged | [Model] Add MiMo-V2.5 support | `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py` |
| 2026-04-28 | [#41029](https://github.com/vllm-project/vllm/pull/41029) | merged | [Model] update for mimo v25 | `vllm/model_executor/models/mimo_v2.py` |

## Per-PR Diff Audit Cards

### PR #17433 - [Model] Support MiMo-7B inference with MTP

- Link: https://github.com/vllm-project/vllm/pull/17433
- Status/date: merged / 2025-05-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo.py`, `vllm/model_executor/models/mimo_mtp.py`; associated commits `acee8f48aa9c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +507/-4, 576 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support MiMo-7B inference with MTP"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`; technical summary: Covers "[Model] Support MiMo-7B inference with MTP"; the main implementation surface is `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_mtp.py` added +283/-0 (283 lines); hunks: -0,0 +1,283; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMultiTokenPredictor, touching `MiMoMultiTokenPredictorLayer, __init__, forward`; `vllm/model_executor/models/mimo.py` added +190/-0 (190 lines); hunks: -0,0 +1,190; symbols: MiMoModel, forward, load_weights, MiMoForCausalLM, touching `MiMoModel, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mimo_mtp.py` added +283/-0 (283 lines); hunks: -0,0 +1,283; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMultiTokenPredictor
  - `vllm/model_executor/models/mimo.py` added +190/-0 (190 lines); hunks: -0,0 +1,190; symbols: MiMoModel, forward, load_weights, MiMoForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_mtp.py
@@ -0,0 +1,283 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from
+# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py
+# Copyright 2025 Xiaomi Corporation.
+# Copyright 2023 The vLLM team.
+# Copyright 2024 DeepSeek-AI team.
diff -- vllm/model_executor/models/mimo.py
@@ -0,0 +1,190 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
+# Copyright 2025 Xiaomi Corporation.
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_mtp.py` added +283/-0; `vllm/model_executor/models/mimo.py` added +190/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25136 - [spec decode] Fix MTP inference path for MiMo-7B model

- Link: https://github.com/vllm-project/vllm/pull/25136
- Status/date: merged / 2025-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo_mtp.py`; associated commits `c4cb0af98a8e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-6, 61 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[spec decode] Fix MTP inference path for MiMo-7B model"; model line: MiMo V2 Flash; category: bug fix; main diff: `vllm/model_executor/models/mimo_mtp.py`; technical summary: Covers "[spec decode] Fix MTP inference path for MiMo-7B model"; the main implementation surface is `vllm/model_executor/models/mimo_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_mtp.py` modified +14/-4 (18 lines); hunks: -241,17 +241,27 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name, touching `load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name`.
- Code diff details:
  - `vllm/model_executor/models/mimo_mtp.py` modified +14/-4 (18 lines); hunks: -241,17 +241,27 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_mtp.py
@@ -241,17 +241,27 @@ def load_weights(self, weights: Iterable[tuple[str,
+        # append mtp_start_layer_idx
+        pattern = r"(model\.mtp_layers\.)(\d+)(\.)"
+        match = re.match(pattern, name)
+        if match:
+            original_num = int(match.group(2))
+            new_num = original_num + self.config.num_hidden_layers
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_mtp.py` modified +14/-4
- Risk and verification: Runtime changes concentrate in `vllm/config/speculative.py`, `vllm/model_executor/models/mimo_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30836 - [Model] Add MiMo-V2-Flash support

- Link: https://github.com/vllm-project/vllm/pull/30836
- Status/date: merged / 2025-12-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +789/-13, 946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add MiMo-V2-Flash support"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py`; technical summary: Covers "[Model] Add MiMo-V2-Flash support"; the main implementation surface is `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE, touching `MiMoV2MLP, __init__, forward`; `vllm/model_executor/layers/linear.py` modified +49/-13 (62 lines); hunks: -277,6 +277,7 @@ def __init__(; -475,6 +476,7 @@ def __init__(; symbols: __init__, _maybe_allow_fp8_block_shape_mismatch, weight_loader, touching `__init__, _maybe_allow_fp8_block_shape_mismatch, weight_loader`; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0 (8 lines); hunks: -1252,6 +1252,14 @@ def validate_fp8_block_shape(; symbols: validate_fp8_block_shape, touching `validate_fp8_block_shape`; `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -459,6 +459,9 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE
  - `vllm/model_executor/layers/linear.py` modified +49/-13 (62 lines); hunks: -277,6 +277,7 @@ def __init__(; -475,6 +476,7 @@ def __init__(; symbols: __init__, _maybe_allow_fp8_block_shape_mismatch, weight_loader
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0 (8 lines); hunks: -1252,6 +1252,14 @@ def validate_fp8_block_shape(; symbols: validate_fp8_block_shape
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -459,6 +459,9 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -415,6 +415,7 @@ th {
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_v2_flash.py
@@ -0,0 +1,720 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from itertools import islice
+import torch
+from torch import nn
diff -- vllm/model_executor/layers/linear.py
@@ -277,6 +277,7 @@ def __init__(
+        self.allow_fp8_block_shape_mismatch = False
@@ -475,6 +476,7 @@ def __init__(
+        self._maybe_allow_fp8_block_shape_mismatch()
@@ -509,6 +511,33 @@ def __init__(
+    def _maybe_allow_fp8_block_shape_mismatch(self) -> None:
+        quant_config = getattr(self, "quant_config", None)
diff -- vllm/model_executor/layers/quantization/utils/fp8_utils.py
@@ -1252,6 +1252,14 @@ def validate_fp8_block_shape(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0; `vllm/model_executor/layers/linear.py` modified +49/-13; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/config/model.py` modified +5/-0; `vllm/config/__init__.py` modified +2/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31175 - [Bugfix] Properly apply v_scale for mimo_v2_flash

- Link: https://github.com/vllm-project/vllm/pull/31175
- Status/date: merged / 2026-01-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-13, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Properly apply v_scale for mimo_v2_flash"; model line: MiMo V2 Flash; category: bug fix; main diff: `vllm/model_executor/models/mimo_v2_flash.py`; technical summary: Covers "[Bugfix] Properly apply v_scale for mimo_v2_flash"; the main implementation surface is `vllm/model_executor/models/mimo_v2_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_v2_flash.py` modified +10/-13 (23 lines); hunks: -211,6 +211,7 @@ def __init__(; -241,6 +242,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_flash.py` modified +10/-13 (23 lines); hunks: -211,6 +211,7 @@ def __init__(; -241,6 +242,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_v2_flash.py
@@ -211,6 +211,7 @@ def __init__(
+        v_scale: float | None = None,
@@ -241,6 +242,7 @@ def __init__(
+        self.v_scale = v_scale
@@ -304,6 +306,10 @@ def forward(
+        # Apply v_scale before attention
+        if self.v_scale is not None:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` modified +10/-13
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mimo_v2_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40045 - [Attention] use diff kv backend for mimo v2 flash

- Link: https://github.com/vllm-project/vllm/pull/40045
- Status/date: merged / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +112/-24, 270 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] use diff kv backend for mimo v2 flash"; model line: MiMo V2 Flash; category: performance/backend optimization; main diff: `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/attention/attention.py`, `tools/pre_commit/generate_attention_backend_docs.py`; technical summary: Covers "[Attention] use diff kv backend for mimo v2 flash"; the main implementation surface is `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/attention/attention.py`, `tools/pre_commit/generate_attention_backend_docs.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/layers/attention/attention.py` modified +1/-0 (1 lines); hunks: -597,6 +597,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec, touching `get_kv_cache_spec`; `tools/pre_commit/generate_attention_backend_docs.py` modified +41/-8 (49 lines); hunks: -634,9 +634,10 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; -656,17 +657,49 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; symbols: parse_flash_attn_features, touching `parse_flash_attn_features`; `vllm/v1/attention/backends/fa_utils.py` modified +22/-3 (25 lines); hunks: -54,7 +54,10 @@ def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None...; -112,6 +115,23 @@ def get_flash_attn_version(; symbols: get_scheduler_metadata, get_flash_attn_version, flash_attn_supports_quant_query_input, flash_attn_supports_sinks, touching `get_scheduler_metadata, get_flash_attn_version, flash_attn_supports_quant_query_input`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/layers/attention/attention.py` modified +1/-0 (1 lines); hunks: -597,6 +597,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec
  - `tools/pre_commit/generate_attention_backend_docs.py` modified +41/-8 (49 lines); hunks: -634,9 +634,10 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; -656,17 +657,49 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; symbols: parse_flash_attn_features
  - `vllm/v1/attention/backends/fa_utils.py` modified +22/-3 (25 lines); hunks: -54,7 +54,10 @@ def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None...; -112,6 +115,23 @@ def get_flash_attn_version(; symbols: get_scheduler_metadata, get_flash_attn_version, flash_attn_supports_quant_query_input, flash_attn_supports_sinks
  - `vllm/v1/attention/backends/flash_attn_diffkv.py` modified +18/-4 (22 lines); hunks: -6,14 +6,16; -23,8 +25,6; symbols: FlashAttentionDiffKVBackend, get_kv_cache_stride_order, FlashAttentionDiffKVImpl, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_v2_flash.py
@@ -46,6 +46,9 @@
+from vllm.v1.attention.backends.flash_attn_diffkv import (
+    FlashAttentionDiffKVBackend,
+)
@@ -287,6 +290,15 @@ def __init__(
+        # Use DiffKV backend when V has a different head dim than K
+        if self.v_head_dim != self.head_dim:
diff -- vllm/model_executor/layers/attention/attention.py
@@ -597,6 +597,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
+                head_size_v=self.head_size_v,
diff -- tools/pre_commit/generate_attention_backend_docs.py
@@ -634,9 +634,10 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:
-    # Analyze the functions to determine FA3-specific features
+    # Analyze the functions to determine FA3/FA4-specific features
+    fa4_supports_sinks = False
@@ -656,17 +657,49 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:
-        # Check flash_attn_supports_sinks - looks for `get_flash_attn_version() == 3`
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8; `vllm/model_executor/layers/attention/attention.py` modified +1/-0; `vllm/v1/attention/backends/fa_utils.py` modified +22/-3; `vllm/v1/attention/backends/flash_attn_diffkv.py` modified +18/-4; `vllm/v1/kv_cache_interface.py` modified +14/-0; `vllm/vllm_flash_attn/flash_attn_interface.py` modified +1/-0
  - other: `tools/pre_commit/generate_attention_backend_docs.py` modified +41/-8
  - docs: `docs/design/attention_backends.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/v1/attention/backends/fa_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40967 - [Model] Add MiMo-V2.5 support

- Link: https://github.com/vllm-project/vllm/pull/40967
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo_audio.py`, `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py`, `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/transformers_utils/configs/mimo_v2_omni.py` and 6 files; associated commits `c245d35ff467`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +4737/-5, 4920 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add MiMo-V2.5 support"; model line: MiMo V2 Flash; category: model support/runtime entry; main diff: `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py`; technical summary: Covers "[Model] Add MiMo-V2.5 support"; the main implementation surface is `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0 (1488 lines); hunks: -0,0 +1,1488; symbols: MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger, __init__, touching `MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger`; `vllm/model_executor/models/mimo_audio.py` added +1389/-0 (1389 lines); hunks: -0,0 +1,1389; symbols: _vq_default, _ema_inplace, _laplace_smoothing, _uniform_init, touching `_vq_default, _ema_inplace, _laplace_smoothing`; `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0 (1285 lines); hunks: -0,0 +1,1285; symbols: ImageInput, VideoInput, AudioInput, VideoAudioInput, touching `ImageInput, VideoInput, AudioInput`; `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0 (373 lines); hunks: -0,0 +1,373; symbols: MiMoV2MTPLayer, __init__, forward, _MiMoV2MTPLayers, touching `MiMoV2MTPLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0 (1488 lines); hunks: -0,0 +1,1488; symbols: MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger, __init__
  - `vllm/model_executor/models/mimo_audio.py` added +1389/-0 (1389 lines); hunks: -0,0 +1,1389; symbols: _vq_default, _ema_inplace, _laplace_smoothing, _uniform_init
  - `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0 (1285 lines); hunks: -0,0 +1,1285; symbols: ImageInput, VideoInput, AudioInput, VideoAudioInput
  - `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0 (373 lines); hunks: -0,0 +1,373; symbols: MiMoV2MTPLayer, __init__, forward, _MiMoV2MTPLayers
  - `vllm/transformers_utils/configs/mimo_v2_omni.py` added +65/-0 (65 lines); hunks: -0,0 +1,65; symbols: Mimo_VLVisionConfig, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_v2_omni.py
@@ -0,0 +1,1488 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import math
+from collections.abc import Callable, Iterable, Mapping, Sequence
+from functools import partial
+from typing import Any
diff -- vllm/model_executor/models/mimo_audio.py
@@ -0,0 +1,1389 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MiMo audio: tokenizer, encoding utilities, and audio encoder.
+Ported from SGLang's mimo_audio.py.
+Audio tokenizer adapted from https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git
+"""
diff -- vllm/transformers_utils/processors/mimo_v2_omni.py
@@ -0,0 +1,1285 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0; `vllm/model_executor/models/mimo_audio.py` added +1389/-0; `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0; `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0; `vllm/transformers_utils/configs/mimo_v2_omni.py` added +65/-0; `vllm/model_executor/models/mimo_v2.py` renamed +22/-2
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41029 - [Model] update for mimo v25

- Link: https://github.com/vllm-project/vllm/pull/41029
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo_v2.py`; associated commits `7a1eb8ac2ec4`
- Diff scope read: GitHub Pull Request files API returned 6 files, +10/-8, 74 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] update for mimo v25"; model line: MiMo V2 Flash; category: model implementation change; main diff: `vllm/model_executor/models/mimo_v2.py`; technical summary: Covers "[Model] update for mimo v25"; the main implementation surface is `vllm/model_executor/models/mimo_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_v2.py` modified +1/-1 (2 lines); hunks: -733,7 +733,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, MiMoV2ProForCausalLM, MiMoV2ForCausalLM, touching `load_weights, MiMoV2ProForCausalLM, MiMoV2ForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2.py` modified +1/-1 (2 lines); hunks: -733,7 +733,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, MiMoV2ProForCausalLM, MiMoV2ForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_v2.py
@@ -733,7 +733,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-class MiMoV2ProForCausalLM(MiMoV2FlashForCausalLM):
+class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
