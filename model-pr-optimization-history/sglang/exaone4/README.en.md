# SGLang EXAONE 4/4.5/K-EXAONE Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/models/exaone4.py` | [#8205](https://github.com/sgl-project/sglang/pull/8205), [#42303](https://github.com/sgl-project/sglang/pull/42303) |
| `python/sglang/srt/models/exaone_moe.py` | [#16294](https://github.com/sgl-project/sglang/pull/16294), [#42303](https://github.com/sgl-project/sglang/pull/42303), [#42304](https://github.com/sgl-project/sglang/pull/42304), [#42305](https://github.com/sgl-project/sglang/pull/42305) |
| `python/sglang/srt/models/exaone_moe_mtp.py` | [#16294](https://github.com/sgl-project/sglang/pull/16294) |

## PR Coverage Summary

- Git-traced PRs: 5
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 5
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-01-27 | [#8205](https://github.com/sgl-project/sglang/pull/8205) | merged | [Model] Add support for EXAONE-4.0 Model | `python/sglang/srt/models/exaone4.py` |
| 2026-01-30 | [#16294](https://github.com/sgl-project/sglang/pull/16294) | merged | [Model] Add K-EXAONE model support | `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone_moe_mtp.py` |
| 2026-10-03 | [#42303](https://github.com/sgl-project/sglang/pull/42303) | merged | [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues | `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone4.py` |
| 2026-10-03 | [#42304](https://github.com/sgl-project/sglang/pull/42304) | merged | [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries | `python/sglang/srt/models/exaone_moe.py` |
| 2026-10-03 | [#42305](https://github.com/sgl-project/sglang/pull/42305) | merged | [Fix] EXAONE MoE under DP attention and DeepEP | `python/sglang/srt/models/exaone_moe.py` |

## Per-PR Diff Audit Cards

### PR #8205 - [Model] Add support for EXAONE-4.0 Model

- Link: https://github.com/sgl-project/sglang/pull/8205
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/exaone4.py`; associated commits `81c0f5c5adf0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +728/-0, 736 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/exaone4.py` added +719/-0 (719 lines); hunks: -0,0 +1,719; symbols: get_attention_sliding_window_size, Exaone4GatedMLP, __init__, forward, touching `get_attention_sliding_window_size, Exaone4GatedMLP, __init__`.
- Code diff details:
  - `python/sglang/srt/models/exaone4.py` added +719/-0 (719 lines); hunks: -0,0 +1,719; symbols: get_attention_sliding_window_size, Exaone4GatedMLP, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/exaone4.py
@@ -0,0 +1,719 @@
+from collections.abc import Iterable
+from typing import Any, List, Optional, Tuple, Union
+import torch
+from torch import nn
+from transformers import Exaone4Config
+from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/exaone4.py` added +719/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/exaone4.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16294 - [Model] Add K-EXAONE model support

- Link: https://github.com/sgl-project/sglang/pull/16294
- Status/date: merged / 2026-01-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone_moe_mtp.py`; associated commits `c04efe030acc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +1000/-7, 1025 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/exaone_moe.py` added +881/-0 (881 lines); hunks: -0,0 +1,881; symbols: ExaoneMoEMLP, __init__, forward, ExaoneMoESparseMoEBlock, touching `ExaoneMoEMLP, __init__, forward`; `python/sglang/srt/models/exaone_moe_mtp.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: ExaoneMoEForCausalLMMTP, __init__, forward, load_weights, touching `ExaoneMoEForCausalLMMTP, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/exaone_moe.py` added +881/-0 (881 lines); hunks: -0,0 +1,881; symbols: ExaoneMoEMLP, __init__, forward, ExaoneMoESparseMoEBlock
  - `python/sglang/srt/models/exaone_moe_mtp.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: ExaoneMoEForCausalLMMTP, __init__, forward, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -0,0 +1,881 @@
+# Copyright 2025 The LG AI Research Team
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/exaone_moe_mtp.py
@@ -0,0 +1,106 @@
+# Copyright 2025 The LG AI Research Team
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/exaone_moe.py` added +881/-0; `python/sglang/srt/models/exaone_moe_mtp.py` added +106/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone_moe_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42303 - [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues

- Link: https://github.com/sgl-project/sglang/pull/42303
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/exaone4.py`, `python/sglang/srt/models/exaone_moe.py`; associated commits `f89ead192f3c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +65/-10, 168 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/exaone_moe.py` modified +12/-1 (13 lines); hunks: -786,6 +786,7 @@ def load_weights(; -807,7 +808,11 @@ def load_weights(; symbols: load_weights, set_eagle3_layers_to_capture, ExaoneMoeForCausalLM, touching `load_weights, set_eagle3_layers_to_capture, ExaoneMoeForCausalLM`; `python/sglang/srt/models/exaone4.py` modified +10/-1 (11 lines); hunks: -435,7 +435,7 @@ def __init__(; -575,6 +575,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/exaone_moe.py` modified +12/-1 (13 lines); hunks: -786,6 +786,7 @@ def load_weights(; -807,7 +808,11 @@ def load_weights(; symbols: load_weights, set_eagle3_layers_to_capture, ExaoneMoeForCausalLM
  - `python/sglang/srt/models/exaone4.py` modified +10/-1 (11 lines); hunks: -435,7 +435,7 @@ def __init__(; -575,6 +575,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -786,6 +786,7 @@ def load_weights(
+            is_expert_weight = False
@@ -807,7 +808,11 @@ def load_weights(
+                    is_expert_weight = True
+                    if name not in params_dict:
+                        # The expert's layer lives on another pipeline rank.
+                        continue
diff -- python/sglang/srt/models/exaone4.py
@@ -435,7 +435,7 @@ def __init__(
-        if config.tie_word_embeddings:
+        if config.tie_word_embeddings and self.pp_group.world_size == 1:
@@ -575,6 +575,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if (
+                name == "model.embed_tokens.weight"
+                and self.config.tie_word_embeddings
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/exaone_moe.py` modified +12/-1; `python/sglang/srt/models/exaone4.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/model_hook.py`, `python/sglang/srt/arg_groups/model_overrides/__init__.py`, `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42304 - [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries

- Link: https://github.com/sgl-project/sglang/pull/42304
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/exaone_moe.py`; associated commits `2fa17a64671b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +126/-100, 403 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/exaone_moe.py` modified +60/-42 (102 lines); hunks: -28,9 +28,16; -39,10 +46,7; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/models/exaone_moe.py` modified +60/-42 (102 lines); hunks: -28,9 +28,16; -39,10 +46,7; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -28,9 +28,16 @@
+from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
+from sglang.srt.layers.layer_boundary import (
+    declare_attn,
+    declare_ffn,
+    make_stages,
+)
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/exaone_moe.py` modified +60/-42
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/models/exaone_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42305 - [Fix] EXAONE MoE under DP attention and DeepEP

- Link: https://github.com/sgl-project/sglang/pull/42305
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/exaone_moe.py`; associated commits `b7fc51698601`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +3/-1, 18 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/exaone_moe.py` modified +3/-1 (4 lines); hunks: -406,6 +406,8 @@ def forward(; -532,7 +534,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/exaone_moe.py` modified +3/-1 (4 lines); hunks: -406,6 +406,8 @@ def forward(; -532,7 +534,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -406,6 +406,8 @@ def forward(
+        if hidden_states.shape[0] == 0:
+            return hidden_states
@@ -532,7 +534,7 @@ def forward(
-            hidden_states = self.mlp(hidden_states)
+            hidden_states = self.mlp(hidden_states, forward_batch)
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/exaone_moe.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/exaone_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
