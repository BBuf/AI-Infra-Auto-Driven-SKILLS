# vLLM EXAONE 4/4.5/K-EXAONE Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `vllm/model_executor/models/exaone4.py` | [#21060](https://github.com/vllm-project/vllm/pull/21060), [#23918](https://github.com/vllm-project/vllm/pull/23918), [#31621](https://github.com/vllm-project/vllm/pull/31621), [#39388](https://github.com/vllm-project/vllm/pull/39388), [#50524](https://github.com/vllm-project/vllm/pull/50524) |
| `vllm/model_executor/models/exaone4_5.py` | [#39388](https://github.com/vllm-project/vllm/pull/39388), [#42246](https://github.com/vllm-project/vllm/pull/42246), [#45073](https://github.com/vllm-project/vllm/pull/45073) |
| `vllm/model_executor/models/exaone4_5_mtp.py` | [#39388](https://github.com/vllm-project/vllm/pull/39388), [#39526](https://github.com/vllm-project/vllm/pull/39526), [#42246](https://github.com/vllm-project/vllm/pull/42246) |
| `vllm/model_executor/models/exaone_moe.py` | [#31621](https://github.com/vllm-project/vllm/pull/31621), [#32196](https://github.com/vllm-project/vllm/pull/32196), [#50524](https://github.com/vllm-project/vllm/pull/50524) |
| `vllm/model_executor/models/exaone_moe_mtp.py` | [#31621](https://github.com/vllm-project/vllm/pull/31621), [#39388](https://github.com/vllm-project/vllm/pull/39388), [#50524](https://github.com/vllm-project/vllm/pull/50524) |

## PR Coverage Summary

- Git-traced PRs: 9
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 9
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-19 | [#21060](https://github.com/vllm-project/vllm/pull/21060) | merged | [Model] EXAONE 4.0 model support | `vllm/model_executor/models/exaone4.py`, `vllm/transformers_utils/configs/exaone4.py` |
| 2025-09-02 | [#23918](https://github.com/vllm-project/vllm/pull/23918) | merged | [BugFix] Fix EXAONE4 rotary embeddings | `vllm/model_executor/models/exaone4.py` |
| 2026-01-12 | [#31621](https://github.com/vllm-project/vllm/pull/31621) | merged | Add K-EXAONE-236B-A23B | `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone_moe_mtp.py`, `vllm/model_executor/models/exaone4.py` |
| 2026-01-12 | [#32196](https://github.com/vllm-project/vllm/pull/32196) | merged | [BugFix] fix FusedMoE.make_expert_params_mapping in EXAONE-MoE | `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-10 | [#39388](https://github.com/vllm-project/vllm/pull/39388) | merged | Add EXAONE-4.5 | `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`, `vllm/model_executor/models/exaone_moe_mtp.py` |
| 2026-04-11 | [#39526](https://github.com/vllm-project/vllm/pull/39526) | merged | [Bugfix] add SupportsMultiModal to Exaone4_5_MTP | `vllm/model_executor/models/exaone4_5_mtp.py` |
| 2026-05-11 | [#42246](https://github.com/vllm-project/vllm/pull/42246) | merged | Fix EXAONE-4.5 to align with Transformers update | `vllm/model_executor/models/exaone4_5_mtp.py`, `vllm/model_executor/models/exaone4_5.py` |
| 2026-06-10 | [#45073](https://github.com/vllm-project/vllm/pull/45073) | merged | [Bugfix] Fix missing sequence_lengths in EXAONE-4.5 vision encoder | `vllm/model_executor/models/exaone4_5.py` |
| 2026-08-03 | [#50524](https://github.com/vllm-project/vllm/pull/50524) | merged | [Model] Add K-EXAONE-2.0-750B-A37B | `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone_moe_mtp.py` |

## Per-PR Diff Audit Cards

### PR #21060 - [Model] EXAONE 4.0 model support

- Link: https://github.com/vllm-project/vllm/pull/21060
- Status/date: merged / 2025-07-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4.py`; associated commits `3e04107d97ae`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +809/-3, 863 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone4.py` added +547/-0 (547 lines); hunks: -0,0 +1,547; symbols: Exaone4GatedMLP, __init__, forward, Exaone4Attention, touching `Exaone4GatedMLP, __init__, forward`; `vllm/transformers_utils/configs/exaone4.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: check_is_sliding, Exaone4Config, to, __init__, touching `check_is_sliding, Exaone4Config, to`.
- Code diff details:
  - `vllm/model_executor/models/exaone4.py` added +547/-0 (547 lines); hunks: -0,0 +1,547; symbols: Exaone4GatedMLP, __init__, forward, Exaone4Attention
  - `vllm/transformers_utils/configs/exaone4.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: check_is_sliding, Exaone4Config, to, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone4.py
@@ -0,0 +1,547 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Adapted from
+# https://github.com/lgai-exaone/transformers/blob/add-exaone4/src/transformers/models/exaone4/modeling_exaone4.py
+# Copyright 2025 The LG CNS Gen AI Solution Delivery Team.
diff -- vllm/transformers_utils/configs/exaone4.py
@@ -0,0 +1,252 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Copied from
+# https://github.com/lgai-exaone/transformers/blob/add-exaone4/src/transformers/models/exaone4/configuration_exaone4.py
+# Copyright 2025 The LG CNS Gen AI Solution Delivery Team.
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone4.py` added +547/-0; `vllm/transformers_utils/configs/exaone4.py` added +252/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23918 - [BugFix] Fix EXAONE4 rotary embeddings

- Link: https://github.com/vllm-project/vllm/pull/23918
- Status/date: merged / 2025-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4.py`; associated commits `38ba061f6f44`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +3/-3, 20 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone4.py` modified +3/-3 (6 lines); hunks: -164,8 +164,8 @@ def __init__(; -201,7 +201,7 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/exaone4.py` modified +3/-3 (6 lines); hunks: -164,8 +164,8 @@ def __init__(; -201,7 +201,7 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone4.py
@@ -164,8 +164,8 @@ def __init__(
-        # apply rotary embeddings to every layer
-        self.apply_all_layers = not is_sliding
+        # apply rotary embeddings to every layer in full attention models
+        self.apply_rope_all_layers = "sliding_attention" not in config.layer_types
@@ -201,7 +201,7 @@ def forward(
-        if self.sliding_window or self.apply_all_layers:
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone4.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/exaone4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31621 - Add K-EXAONE-236B-A23B

- Link: https://github.com/vllm-project/vllm/pull/31621
- Status/date: merged / 2026-01-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone_moe_mtp.py`; associated commits `63ed2409e8cf`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +856/-0, 921 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone_moe.py` added +578/-0 (578 lines); hunks: -0,0 +1,578; symbols: ExaoneMoe, __init__, forward, ExaoneMoeDecoderLayer, touching `ExaoneMoe, __init__, forward`; `vllm/model_executor/models/exaone_moe_mtp.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: ExaoneMoeMultiTokenPredictor, __init__, get_input_embeddings, forward, touching `ExaoneMoeMultiTokenPredictor, __init__, get_input_embeddings`; `vllm/model_executor/models/exaone4.py` modified +2/-0 (2 lines); hunks: -72,6 +72,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/exaone_moe.py` added +578/-0 (578 lines); hunks: -0,0 +1,578; symbols: ExaoneMoe, __init__, forward, ExaoneMoeDecoderLayer
  - `vllm/model_executor/models/exaone_moe_mtp.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: ExaoneMoeMultiTokenPredictor, __init__, get_input_embeddings, forward
  - `vllm/model_executor/models/exaone4.py` modified +2/-0 (2 lines); hunks: -72,6 +72,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone_moe.py
@@ -0,0 +1,578 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- vllm/model_executor/models/exaone_moe_mtp.py
@@ -0,0 +1,255 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only ExaoneMoe MTP model."""
+from collections.abc import Iterable
+import torch
+from torch import nn
diff -- vllm/model_executor/models/exaone4.py
@@ -72,6 +72,7 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone_moe.py` added +578/-0; `vllm/model_executor/models/exaone_moe_mtp.py` added +255/-0; `vllm/model_executor/models/exaone4.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32196 - [BugFix] fix FusedMoE.make_expert_params_mapping in EXAONE-MoE

- Link: https://github.com/vllm-project/vllm/pull/32196
- Status/date: merged / 2026-01-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone_moe.py`; associated commits `3d962d72ab01`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone_moe.py` modified +1/-0 (1 lines); hunks: -338,6 +338,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping, touching `get_expert_mapping`.
- Code diff details:
  - `vllm/model_executor/models/exaone_moe.py` modified +1/-0 (1 lines); hunks: -338,6 +338,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone_moe.py
@@ -338,6 +338,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
+            self,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/exaone_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39388 - Add EXAONE-4.5

- Link: https://github.com/vllm-project/vllm/pull/39388
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`, `vllm/model_executor/models/exaone_moe_mtp.py`; associated commits `e7a1387e7380`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +600/-10, 727 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone4_5.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: EXAONE4_5_VisionAttention, __init__, split_qkv, forward, touching `EXAONE4_5_VisionAttention, __init__, split_qkv`; `vllm/model_executor/models/exaone4_5_mtp.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: Exaone4_5MultiTokenPredictor, __init__, Exaone4_5_MTP, load_weights, touching `Exaone4_5MultiTokenPredictor, __init__, Exaone4_5_MTP`; `vllm/model_executor/models/exaone_moe_mtp.py` modified +0/-5 (5 lines); hunks: -184,11 +184,6 @@ class ExaoneMoeMTP(nn.Module):; symbols: ExaoneMoeMTP, __init__, touching `ExaoneMoeMTP, __init__`; `vllm/model_executor/models/exaone4.py` modified +3/-0 (3 lines); hunks: -75,6 +75,7 @@ def __init__(; -83,6 +84,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/exaone4_5.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: EXAONE4_5_VisionAttention, __init__, split_qkv, forward
  - `vllm/model_executor/models/exaone4_5_mtp.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: Exaone4_5MultiTokenPredictor, __init__, Exaone4_5_MTP, load_weights
  - `vllm/model_executor/models/exaone_moe_mtp.py` modified +0/-5 (5 lines); hunks: -184,11 +184,6 @@ class ExaoneMoeMTP(nn.Module):; symbols: ExaoneMoeMTP, __init__
  - `vllm/model_executor/models/exaone4.py` modified +3/-0 (3 lines); hunks: -75,6 +75,7 @@ def __init__(; -83,6 +84,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone4_5.py
@@ -0,0 +1,366 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- vllm/model_executor/models/exaone4_5_mtp.py
@@ -0,0 +1,129 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only EXAONE-4_5 MTP model."""
+from collections.abc import Iterable
+import torch
+from torch import nn
diff -- vllm/model_executor/models/exaone_moe_mtp.py
@@ -184,11 +184,6 @@ class ExaoneMoeMTP(nn.Module):
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone4_5.py` added +366/-0; `vllm/model_executor/models/exaone4_5_mtp.py` added +129/-0; `vllm/model_executor/models/exaone_moe_mtp.py` modified +0/-5; `vllm/model_executor/models/exaone4.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39526 - [Bugfix] add SupportsMultiModal to Exaone4_5_MTP

- Link: https://github.com/vllm-project/vllm/pull/39526
- Status/date: merged / 2026-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4_5_mtp.py`; associated commits `da72daced2a8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +36/-1, 62 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone4_5_mtp.py` modified +36/-1 (37 lines); hunks: -23,8 +23,14; -85,9 +91,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, Exaone4_5_MTP, touching `__init__, embed_input_ids, Exaone4_5_MTP`.
- Code diff details:
  - `vllm/model_executor/models/exaone4_5_mtp.py` modified +36/-1 (37 lines); hunks: -23,8 +23,14; -85,9 +91,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, Exaone4_5_MTP
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone4_5_mtp.py
@@ -23,8 +23,14 @@
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsMultiModal,
+    _require_is_multimodal,
+)
+    _merge_multimodal_embeddings,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone4_5_mtp.py` modified +36/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/exaone4_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42246 - Fix EXAONE-4.5 to align with Transformers update

- Link: https://github.com/vllm-project/vllm/pull/42246
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`; associated commits `27ae67636479`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +53/-13, 147 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone4_5_mtp.py` modified +53/-9 (62 lines); hunks: -9,6 +9,7; -22,6 +23,7; symbols: __init__, embed_input_ids, forward, touching `__init__, embed_input_ids, forward`; `vllm/model_executor/models/exaone4_5.py` modified +0/-4 (4 lines); hunks: -23,7 +23,6; -304,9 +303,6 @@ def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Pr...; symbols: get_hf_processor, get_image_processor, touching `get_hf_processor, get_image_processor`.
- Code diff details:
  - `vllm/model_executor/models/exaone4_5_mtp.py` modified +53/-9 (62 lines); hunks: -9,6 +9,7; -22,6 +23,7; symbols: __init__, embed_input_ids, forward
  - `vllm/model_executor/models/exaone4_5.py` modified +0/-4 (4 lines); hunks: -23,7 +23,6; -304,9 +303,6 @@ def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Pr...; symbols: get_hf_processor, get_image_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone4_5_mtp.py
@@ -9,6 +9,7 @@
+from vllm.distributed.parallel_state import get_pp_group
@@ -22,6 +23,7 @@
+from vllm.sequence import IntermediateTensors
@@ -48,6 +50,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        text_config = config.text_config
@@ -58,18 +61,18 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
diff -- vllm/model_executor/models/exaone4_5.py
@@ -23,7 +23,6 @@
-    Exaone4_5_ImageProcessor,
@@ -304,9 +303,6 @@ def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Processor:
-    def get_image_processor(self, **kwargs: object) -> Exaone4_5_ImageProcessor:
-        return Exaone4_5_ImageProcessor(**kwargs)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone4_5_mtp.py` modified +53/-9; `vllm/model_executor/models/exaone4_5.py` modified +0/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45073 - [Bugfix] Fix missing sequence_lengths in EXAONE-4.5 vision encoder

- Link: https://github.com/vllm-project/vllm/pull/45073
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4_5.py`; associated commits `ccc05de03888`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +7/-0, 42 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone4_5.py` modified +7/-0 (7 lines); hunks: -152,6 +152,8 @@ def forward(; -176,6 +178,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/exaone4_5.py` modified +7/-0 (7 lines); hunks: -152,6 +152,8 @@ def forward(; -176,6 +178,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone4_5.py
@@ -152,6 +152,8 @@ def forward(
+        sequence_lengths: torch.Tensor
+        | None = None,  # Only used for FlashInfer CuDNN backend
@@ -176,6 +178,7 @@ def forward(
+            sequence_lengths=sequence_lengths,
@@ -190,6 +193,7 @@ def forward(
+        "sequence_lengths": 0,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone4_5.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/exaone4_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50524 - [Model] Add K-EXAONE-2.0-750B-A37B

- Link: https://github.com/vllm-project/vllm/pull/50524
- Status/date: merged / 2026-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone_moe_mtp.py`; associated commits `9ae11a6b895d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +161/-11, 289 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/exaone_moe.py` modified +152/-5 (157 lines); hunks: -29,19 +29,25; -59,6 +65,7 @@ class ExaoneMoe(nn.Module):; symbols: ExaoneMoe, __init__, forward, touching `ExaoneMoe, __init__, forward`; `vllm/model_executor/models/exaone4.py` modified +5/-2 (7 lines); hunks: -31,7 +31,7; -70,6 +70,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/exaone_moe_mtp.py` modified +1/-1 (2 lines); hunks: -81,7 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/exaone_moe.py` modified +152/-5 (157 lines); hunks: -29,19 +29,25; -59,6 +65,7 @@ class ExaoneMoe(nn.Module):; symbols: ExaoneMoe, __init__, forward
  - `vllm/model_executor/models/exaone4.py` modified +5/-2 (7 lines); hunks: -31,7 +31,7; -70,6 +70,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/exaone_moe_mtp.py` modified +1/-1 (2 lines); hunks: -81,7 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/exaone_moe.py
@@ -29,19 +29,25 @@
+from vllm.model_executor.layers.attention import Attention
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import (
+    QKVParallelLinear,
+    ReplicatedLinear,
+    RowParallelLinear,
diff -- vllm/model_executor/models/exaone4.py
@@ -31,7 +31,7 @@
-from vllm.model_executor.layers.activation import SiluAndMul
+from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
@@ -70,6 +70,7 @@ def __init__(
+        swiglu_limit: float | None = None,
@@ -95,7 +96,9 @@ def __init__(
-        self.act_fn = SiluAndMul()
diff -- vllm/model_executor/models/exaone_moe_mtp.py
@@ -81,7 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/exaone_moe.py` modified +152/-5; `vllm/model_executor/models/exaone4.py` modified +5/-2; `vllm/model_executor/models/exaone_moe_mtp.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
