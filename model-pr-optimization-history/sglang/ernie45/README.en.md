# SGLang ERNIE 4.5 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/cookbook/autoregressive/Ernie/Ernie4.5-VL.mdx` | no direct PR-number commit |
| `docs/cookbook/autoregressive/Ernie/Ernie4.5.mdx` | no direct PR-number commit |
| `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` | [#42303](https://github.com/sgl-project/sglang/pull/42303) |
| `python/sglang/srt/models/ernie4.py` | [#7657](https://github.com/sgl-project/sglang/pull/7657), [#26038](https://github.com/sgl-project/sglang/pull/26038), [#35222](https://github.com/sgl-project/sglang/pull/35222) |
| `python/sglang/srt/models/ernie45_moe_vl.py` | [#15679](https://github.com/sgl-project/sglang/pull/15679), [#42303](https://github.com/sgl-project/sglang/pull/42303), [#42304](https://github.com/sgl-project/sglang/pull/42304) |
| `python/sglang/srt/models/ernie45_vl.py` | [#15679](https://github.com/sgl-project/sglang/pull/15679), [#19743](https://github.com/sgl-project/sglang/pull/19743) |
| `python/sglang/srt/models/ernie4_eagle.py` | [#7657](https://github.com/sgl-project/sglang/pull/7657) |
| `python/sglang/srt/multimodal/processors/ernie45_vl.py` | [#15679](https://github.com/sgl-project/sglang/pull/15679) |

## PR Coverage Summary

- Git-traced PRs: 7
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 7
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-08 | [#7657](https://github.com/sgl-project/sglang/pull/7657) | merged | Add ernie4.py for ERNIE-4.5 | `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/ernie4_eagle.py` |
| 2026-01-26 | [#15679](https://github.com/sgl-project/sglang/pull/15679) | merged | [Model] Add Ernie4.5 VL model support | `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` |
| 2026-03-04 | [#19743](https://github.com/sgl-project/sglang/pull/19743) | merged | [VLM] Support cos sin cache for Ernie4.5-VL | `python/sglang/srt/models/ernie45_vl.py` |
| 2026-05-28 | [#26038](https://github.com/sgl-project/sglang/pull/26038) | merged | [NPU] fix model ERNIE-4.5-21B-A3B-PT bias need 1D error | `python/sglang/srt/models/ernie4.py` |
| 2026-08-27 | [#35222](https://github.com/sgl-project/sglang/pull/35222) | merged | [CPU] Enable ERNIE models on CPU | `python/sglang/srt/models/ernie4.py` |
| 2026-10-03 | [#42303](https://github.com/sgl-project/sglang/pull/42303) | merged | [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues | `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` |
| 2026-10-03 | [#42304](https://github.com/sgl-project/sglang/pull/42304) | merged | [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries | `python/sglang/srt/models/ernie45_moe_vl.py` |

## Per-PR Diff Audit Cards

### PR #7657 - Add ernie4.py for ERNIE-4.5

- Link: https://github.com/sgl-project/sglang/pull/7657
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/ernie4_eagle.py`; associated commits `113254749659`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +635/-0, 651 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/ernie4.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: MoEGate, __init__, forward, Ernie4Moe, touching `MoEGate, __init__, forward`; `python/sglang/srt/models/ernie4_eagle.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: Ernie4ModelMTP, __init__, forward, Ernie4_5_MoeForCausalLMMTP, touching `Ernie4ModelMTP, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/ernie4.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: MoEGate, __init__, forward, Ernie4Moe
  - `python/sglang/srt/models/ernie4_eagle.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: Ernie4ModelMTP, __init__, forward, Ernie4_5_MoeForCausalLMMTP
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ernie4.py
@@ -0,0 +1,426 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/ernie4_eagle.py
@@ -0,0 +1,203 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/ernie4.py` added +426/-0; `python/sglang/srt/models/ernie4_eagle.py` added +203/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/ernie4_eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15679 - [Model] Add Ernie4.5 VL model support

- Link: https://github.com/sgl-project/sglang/pull/15679
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`; associated commits `1a19b3987dca`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +2072/-0, 2103 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock, touching `Ernie4_5_VisionMLP, __init__, forward`; `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE, touching `Ernie4_5_VLMoeAttention, __init__, forward`; `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor, touching `smart_resize, resize_image, round_by_factor`.
- Code diff details:
  - `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock
  - `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE
  - `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ernie45_vl.py
@@ -0,0 +1,845 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/ernie45_moe_vl.py
@@ -0,0 +1,552 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/ernie45_vl.py
@@ -0,0 +1,417 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/ernie45_vl.py` added +845/-0; `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0; `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/ernie45_moe_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19743 - [VLM] Support cos sin cache for Ernie4.5-VL

- Link: https://github.com/sgl-project/sglang/pull/19743
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/ernie45_vl.py`; associated commits `82e7139c06a3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +34/-12, 102 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: -30,6 +30,7; -120,14 +121,16 @@ def forward(; symbols: forward, __init__, dtype, device, touching `forward, __init__, dtype`.
- Code diff details:
  - `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: -30,6 +30,7; -120,14 +121,16 @@ def forward(; symbols: forward, __init__, dtype, device
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ernie45_vl.py
@@ -30,6 +30,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -120,14 +121,16 @@ def forward(
-        position_embeddings: torch.Tensor,
+        rotary_pos_emb_cos: torch.Tensor,
+        rotary_pos_emb_sin: torch.Tensor,
-            position_embeddings=position_embeddings,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/ernie45_vl.py` modified +34/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/ernie45_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26038 - [NPU] fix model ERNIE-4.5-21B-A3B-PT bias need 1D error

- Link: https://github.com/sgl-project/sglang/pull/26038
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/ernie4.py`; associated commits `a245cae3d1e7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +9/-2, 39 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/ernie4.py` modified +8/-2 (10 lines); hunks: -42,9 +42,11; -86,12 +88,16 @@ def __init__(; symbols: MoEGate, __init__, touching `MoEGate, __init__`.
- Code diff details:
  - `python/sglang/srt/models/ernie4.py` modified +8/-2 (10 lines); hunks: -42,9 +42,11; -86,12 +88,16 @@ def __init__(; symbols: MoEGate, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ernie4.py
@@ -42,9 +42,11 @@
-from sglang.srt.utils import add_prefix, make_layers
+from sglang.srt.utils import add_prefix, is_npu, make_layers
+_is_npu = is_npu()
@@ -86,12 +88,16 @@ def __init__(
+        correction_bias = self.gate.e_score_correction_bias
+        # npu only supports 1D, but current correction_bias is 2D
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/ernie4.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/llama.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35222 - [CPU] Enable ERNIE models on CPU

- Link: https://github.com/sgl-project/sglang/pull/35222
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/ernie4.py`; associated commits `5adc2880f996`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +6/-3, 32 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/ernie4.py` modified +4/-3 (7 lines); hunks: -42,9 +42,10; -89,8 +90,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/ernie4.py` modified +4/-3 (7 lines); hunks: -42,9 +42,10; -89,8 +90,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ernie4.py
@@ -42,9 +42,10 @@
-from sglang.srt.utils import add_prefix, is_npu, make_layers
+from sglang.srt.utils import add_prefix, is_cpu, is_npu, make_layers
+_is_cpu = is_cpu()
@@ -89,8 +90,8 @@ def __init__(
-        # npu only supports 1D, but current correction_bias is 2D
-        if _is_npu:
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/ernie4.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/ernie4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42303 - [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues

- Link: https://github.com/sgl-project/sglang/pull/42303
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`; associated commits `f89ead192f3c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +65/-10, 168 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/ernie45_moe_vl.py` modified +0/-2 (2 lines); hunks: -25,7 +25,6; -479,7 +478,6 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` added +19/-0 (19 lines); hunks: -0,0 +1,19; symbols: _ernie45_vl_overrides, touching `_ernie45_vl_overrides`.
- Code diff details:
  - `python/sglang/srt/models/ernie45_moe_vl.py` modified +0/-2 (2 lines); hunks: -25,7 +25,6; -479,7 +478,6 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` added +19/-0 (19 lines); hunks: -0,0 +1,19; symbols: _ernie45_vl_overrides
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ernie45_moe_vl.py
@@ -25,7 +25,6 @@
-from sglang.srt.layers.dp_attention import is_dp_attention_enabled
@@ -479,7 +478,6 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
diff -- python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py
@@ -0,0 +1,19 @@
+"""Config-time override declarations for ernie45_vl."""
+from typing import Any
+from sglang.srt.arg_groups.model_override_base import (
+    _register_for,
+    resolving_view,
+)
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/ernie45_moe_vl.py` modified +0/-2; `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` added +19/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/model_hook.py`, `python/sglang/srt/arg_groups/model_overrides/__init__.py`, `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42304 - [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries

- Link: https://github.com/sgl-project/sglang/pull/42304
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/ernie45_moe_vl.py`; associated commits `2fa17a64671b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +126/-100, 403 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/ernie45_moe_vl.py` modified +66/-58 (124 lines); hunks: -16,15 +16,18; -107,6 +110,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer, Ernie4_5_VLMoeDecoderLayer, touching `__init__, forward, _is_moe_layer`.
- Code diff details:
  - `python/sglang/srt/models/ernie45_moe_vl.py` modified +66/-58 (124 lines); hunks: -16,15 +16,18; -107,6 +110,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer, Ernie4_5_VLMoeDecoderLayer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/ernie45_moe_vl.py
@@ -16,15 +16,18 @@
-from typing import Any, Dict, Optional, Tuple, Union
+from typing import Any, Dict, Optional, Union
-from sglang.srt.distributed import (
-    tensor_model_parallel_all_reduce,
+from sglang.srt.layers.layer_boundary import (
+    declare_attn,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/ernie45_moe_vl.py` modified +66/-58
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/models/exaone_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
