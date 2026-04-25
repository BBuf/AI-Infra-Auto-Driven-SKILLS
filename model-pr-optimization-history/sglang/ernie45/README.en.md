# sglang ERNIE 4.5 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/models/ernie45_moe_vl.py` | no direct PR-number commit |
| `python/sglang/srt/models/ernie45_vl.py` | no direct PR-number commit |
| `python/sglang/srt/multimodal/processors/ernie45_vl.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 0
- Extra PRs preserved from existing docs: 3
- Total PRs in this document: 3
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-01-26 | [#15679](https://github.com/sgl-project/sglang/pull/15679) | merged | [Model] Add Ernie4.5 VL model support | `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` |
| 2026-02-16 | [#18856](https://github.com/sgl-project/sglang/pull/18856) | merged | [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel | `python/sglang/srt/layers/rotary_embedding.py` |
| 2026-03-04 | [#19743](https://github.com/sgl-project/sglang/pull/19743) | merged | [VLM] Support cos sin cache for Ernie4.5-VL | `python/sglang/srt/models/ernie45_vl.py` |

## Per-PR Diff Audit Cards

### PR #15679 - [Model] Add Ernie4.5 VL model support

- Link: https://github.com/sgl-project/sglang/pull/15679
- Status/date: merged / 2026-01-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +2072/-0, 2103 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For ERNIE 4.5, this PR adds or enables a model support/runtime surface. Title: "[Model] Add Ernie4.5 VL model support". The diff centers on `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`. PR body context: ## Motivation Add Baidu Ernie4.5 VL model support ## Modifications `ernie45_moe_vl.py` the text backbone `ernie45_vl.py` the vit `processors/ernie45_vl.py` the processor `rotary...
- Key implementation: `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock, touching `Ernie4_5_VisionMLP, __init__, forward`; `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE, touching `Ernie4_5_VLMoeAttention, __init__, forward`; `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor, touching `smart_resize, resize_image, round_by_factor`; `python/sglang/srt/layers/rotary_embedding.py` modified +256/-0 (256 lines); hunks: -2284,6 +2284,177 @@ def get_rope_index_glm4v(; -2323,6 +2494,91 @@ def _get_llm_pos_ids_for_vision(; symbols: get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths, _get_llm_pos_ids_for_vision, touching `get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths`.
- Code diff details:
  - `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock
  - `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE
  - `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor
  - `python/sglang/srt/layers/rotary_embedding.py` modified +256/-0 (256 lines); hunks: -2284,6 +2284,177 @@ def get_rope_index_glm4v(; -2323,6 +2494,91 @@ def _get_llm_pos_ids_for_vision(; symbols: get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths, _get_llm_pos_ids_for_vision
  - `docs/supported_models/multimodal_language_models.md` modified +1/-0 (1 lines); hunks: -46,6 +46,7 @@ in the GitHub search bar.
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

- Reviewed files:
  - runtime: `python/sglang/srt/models/ernie45_vl.py` added +845/-0; `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0; `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0; `python/sglang/srt/layers/rotary_embedding.py` modified +256/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0
  - docs: `docs/supported_models/multimodal_language_models.md` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/ernie45_moe_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18856 - [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel

- Link: https://github.com/sgl-project/sglang/pull/18856
- Status/date: merged / 2026-02-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +268/-3, 308 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For ERNIE 4.5, this PR adds or enables a model support/runtime surface. Title: "[VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel". The diff centers on `python/sglang/srt/layers/rotary_embedding.py`. PR body context: ## Motivation Current Ernie4.5VL MRoPE occupies a major portion in inference time. It has many small ops which introduces quite a lot of GPU bubbles. This PR is to introduce a f...
- Key implementation: `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3 (271 lines); hunks: -2787,9 +2787,238 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -2834,14 +3063,16 @@ def forward_native( # type: ignore[override]; symbols: _compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace, Ernie4_5_VLRotaryEmbedding, touching `_compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace`.
- Code diff details:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3 (271 lines); hunks: -2787,9 +2787,238 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -2834,14 +3063,16 @@ def forward_native( # type: ignore[override]; symbols: _compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace, Ernie4_5_VLRotaryEmbedding
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -2787,9 +2787,238 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
+@triton.jit
+def _triton_ernie45_rope_qk_fused(
+    q_ptr,
+    k_ptr,
+    cos_sin_cache_ptr,
+    positions_ptr,  # [3, num_tokens]  (t/h/w)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19743 - [VLM] Support cos sin cache for Ernie4.5-VL

- Link: https://github.com/sgl-project/sglang/pull/19743
- Status/date: merged / 2026-03-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-12, 102 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For ERNIE 4.5, this PR adds or enables a model support/runtime surface. Title: "[VLM] Support cos sin cache for Ernie4.5-VL". The diff centers on `python/sglang/srt/models/ernie45_vl.py`. PR body context: ## Motivation This PR refactors the rotary positional embedding implementation to expose an explicit cos/sin cache interface and reuse it in the 2D vision RoPE code path. Instea...
- Key implementation: `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: -30,6 +30,7; -120,14 +121,16 @@ def forward(; symbols: forward, __init__, dtype, device, touching `forward, __init__, dtype`.
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

- Reviewed files:
  - runtime: `python/sglang/srt/models/ernie45_vl.py` modified +34/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/ernie45_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
