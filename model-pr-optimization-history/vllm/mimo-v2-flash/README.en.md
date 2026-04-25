# vllm MiMo V2 Flash Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `vllm/model_executor/models/mimo.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433) |
| `vllm/model_executor/models/mimo_mtp.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433), [#25136](https://github.com/vllm-project/vllm/pull/25136) |
| `vllm/model_executor/models/mimo_v2_flash.py` | [#30836](https://github.com/vllm-project/vllm/pull/30836), [#31175](https://github.com/vllm-project/vllm/pull/31175), [#40045](https://github.com/vllm-project/vllm/pull/40045) |

## PR Coverage Summary

- Git-traced PRs: 5
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 5
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-05-12 | [#17433](https://github.com/vllm-project/vllm/pull/17433) | merged | [Model] Support MiMo-7B inference with MTP | `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py` |
| 2025-09-18 | [#25136](https://github.com/vllm-project/vllm/pull/25136) | merged | [spec decode] Fix MTP inference path for MiMo-7B model | `vllm/model_executor/models/mimo_mtp.py` |
| 2025-12-19 | [#30836](https://github.com/vllm-project/vllm/pull/30836) | merged | [Model] Add MiMo-V2-Flash support | `vllm/model_executor/models/mimo_v2_flash.py` |
| 2026-01-05 | [#31175](https://github.com/vllm-project/vllm/pull/31175) | merged | [Bugfix] Properly apply v_scale for mimo_v2_flash | `vllm/model_executor/models/mimo_v2_flash.py` |
| 2026-04-24 | [#40045](https://github.com/vllm-project/vllm/pull/40045) | merged | [Attention] use diff kv backend for mimo v2 flash | `vllm/model_executor/models/mimo_v2_flash.py` |

## Per-PR Diff Audit Cards

### PR #17433 - [Model] Support MiMo-7B inference with MTP

- Link: https://github.com/vllm-project/vllm/pull/17433
- Status/date: merged / 2025-05-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo.py`, `vllm/model_executor/models/mimo_mtp.py`; associated commits `acee8f48aa9c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +507/-4, 576 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "[Model] Support MiMo-7B inference with MTP". The diff centers on `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`. PR body context: MiMo-7B is a decoder-only Transformer LM with MTP layers that exhibits extraordinary reasoning potential. The model and technical report are open-sourced in https://github.com/X...
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
- Motivation: For MiMo V2 Flash, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[spec decode] Fix MTP inference path for MiMo-7B model". The diff centers on `vllm/model_executor/models/mimo_mtp.py`. PR body context: ## Purpose Fix MiMo-7B MTP inference path ## Test Plan Acceptance rate test + lm_eval ## Test Result - acceptance rate - lm_eval baseline - lm_eval with mtp --- Essential Elemen...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo_v2_flash.py`; associated commits `969bbc7c6166`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +789/-13, 946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR adds or enables a model support/runtime surface. Title: "[Model] Add MiMo-V2-Flash support". The diff centers on `vllm/model_executor/models/mimo_v2_flash.py`. PR body context: ## Purpose Add support for MiMo-V2-Flash. ## Examples ### Example 1 ### Example 2 ## Accuracy GSM8K
- Key implementation: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE, touching `MiMoV2MLP, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE
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
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31175 - [Bugfix] Properly apply v_scale for mimo_v2_flash

- Link: https://github.com/vllm-project/vllm/pull/31175
- Status/date: merged / 2026-01-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo_v2_flash.py`; associated commits `951302989814`
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-13, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Properly apply v_scale for mimo_v2_flash". The diff centers on `vllm/model_executor/models/mimo_v2_flash.py`. PR body context: ## Purpose Noticed this when comparing with the Transformers implementation Before: After: ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Chec...
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mimo_v2_flash.py`; associated commits `e8ee2a78dbc0`
- Diff scope read: GitHub Pull Request files API returned 8 files, +112/-24, 270 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For MiMo V2 Flash, this PR optimizes an inference path or backend selection. Title: "[Attention] use diff kv backend for mimo v2 flash". The diff centers on `vllm/model_executor/models/mimo_v2_flash.py`. PR body context: ## Purpose ### Diff kv A key characteristic of mimo v2 flash architecture is that the attention layer uses different head dimensions for keys and values (v_head_dim != head_dim)...
- Key implementation: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward
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
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/v1/attention/backends/fa_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
