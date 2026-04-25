# SGLang Ernie4.5 / Ernie4.5-VL PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Ernie4.5-VL landing and rotary hot-path optimization

## Landed PRs

### PR #15679 - Add Ernie4.5 VL model support

- Link: https://github.com/sgl-project/sglang/pull/15679
- State: merged
- Diff coverage: full diff reviewed, `6` files, `2072` additions
- Motivation:
  - SGLang needed native Ernie4.5-VL and Ernie4.5-MoE-VL support rather than
    routing the family through an adjacent vision-language model.
- Key implementation:
  - Registers `Ernie4_5_VLMoeForConditionalGeneration` as multimodal.
  - Adds `ernie45_vl.py`, `ernie45_moe_vl.py`, and the matching multimodal
    processor.
  - Builds a dedicated vision stack with `VisionAttention`,
    variable-resolution resampling, and Ernie4.5-specific multimodal embedding
    plumbing.
- Key code excerpts:

```diff
+    "Ernie4_5_VLMoeForConditionalGeneration",
```

```diff
+from sglang.srt.models.ernie45_moe_vl import Ernie4_5_VLMoeModel
+from sglang.srt.utils.hf_transformers_utils import get_processor
...
+class Ernie4_5_VisionBlock(nn.Module):
+    self.attn = VisionAttention(... use_qkv_parallel=True, flatten_batch=True, ...)
```

- Reviewed files:
  - config: `python/sglang/srt/configs/model_config.py`
  - runtime: `python/sglang/srt/models/ernie45_vl.py`,
    `python/sglang/srt/models/ernie45_moe_vl.py`
  - processor:
    `python/sglang/srt/multimodal/processors/ernie45_vl.py`
- Validation implications:
  - Ernie4.5-VL bugs can come from the processor or the vision resampler, not
    only from the text model.

### PR #18856 - Optimize Ernie4.5-VL rotary embedding with fused triton kernel

- Link: https://github.com/sgl-project/sglang/pull/18856
- State: merged
- Diff coverage: full diff reviewed, `1` file, `268` additions, `3` deletions
- Motivation:
  - The original Ernie4.5-VL rotary path was expensive for the vision tower and
    repeatedly transformed Q/K in a generic way instead of fusing the layout.
- Key implementation:
  - Adds `_triton_ernie45_rope_qk_fused` and
    `triton_ernie45_rope_fused_inplace(...)`.
  - Encodes Ernie4.5's `(h, w, t)` layout selection directly in the kernel.
  - Extends `Ernie4_5_VLRotaryEmbedding` around the fused path.
- Key code excerpts:

```diff
+@triton.jit
+def _triton_ernie45_rope_qk_fused(
+    q_ptr,
+    k_ptr,
+    cos_sin_cache_ptr,
+    positions_ptr,  # [3, num_tokens]  (t/h/w)
+    ...
+):
```

```diff
+    section_h, section_w, section_t = mrope_section
+    assert section_h == section_w, "Ernie4.5 layout assumes section_h == section_w"
```

- Reviewed files:
  - runtime/kernel wrapper: `python/sglang/srt/layers/rotary_embedding.py`
- Validation implications:
  - If Ernie4.5-VL outputs drift only after rotary or Triton changes, inspect
    the fused rope path before changing higher-level model code.

### PR #19743 - Support cos sin cache for Ernie4.5-VL

- Link: https://github.com/sgl-project/sglang/pull/19743
- State: merged
- Diff coverage: full diff reviewed, `1` file, `34` additions, `12` deletions
- Motivation:
  - After the fused rotary kernel landed, the remaining waste was repeated
    rotary cache computation in the Ernie4.5-VL vision tower.
- Key implementation:
  - Replaces the custom rotary object with `get_rope(...)`.
  - Fetches `cos` and `sin` via `get_cos_sin(max_grid_size)`.
  - Threads `rotary_pos_emb_cos` and `rotary_pos_emb_sin` through the vision
    blocks instead of passing a prebuilt combined embedding tensor.
- Key code excerpts:

```diff
+from sglang.srt.layers.rotary_embedding import get_rope
...
+        self.rotary_pos_emb = get_rope(
+            head_size=head_dim,
+            rotary_dim=head_dim // 2,
+            max_position=8192,
+            base=10000.0,
+            is_neox_style=True,
+        )
```

```diff
+        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
+        cos_combined = cos[pos_ids].flatten(1)
+        sin_combined = sin[pos_ids].flatten(1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/ernie45_vl.py`
- Validation implications:
  - If Ernie4.5-VL regresses after rope cache refactors, verify the
    `grid_thw -> pos_ids -> cos/sin` mapping, not only the attention call site.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG ERNIE 4.5 / ERNIE 4.5 VL PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-23 | [#15679](https://github.com/sgl-project/sglang/pull/15679) | merged | [Model] Add Ernie4.5 VL model support | model wrapper, MoE/router, multimodal/processor, docs/config | `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` |
| 2026-02-15 | [#18856](https://github.com/sgl-project/sglang/pull/18856) | merged | [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel | misc | `python/sglang/srt/layers/rotary_embedding.py` |
| 2026-03-03 | [#19743](https://github.com/sgl-project/sglang/pull/19743) | merged | [VLM] Support cos sin cache for Ernie4.5-VL | model wrapper | `python/sglang/srt/models/ernie45_vl.py` |

## Diff Cards

### PR #15679 - [Model] Add Ernie4.5 VL model support

- Link: https://github.com/sgl-project/sglang/pull/15679
- Status/date: `merged`, created 2025-12-23, merged 2026-01-26; author `CSWYF3634076`.
- Diff scope read: `6` files, `+2072/-0`; areas: model wrapper, MoE/router, multimodal/processor, docs/config; keywords: config, vision, kv, moe, spec, attention, cache, cuda, expert, processor.
- Code diff details:
  - `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: +# Copyright 2023-2025 SGLang Team; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock
  - `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: +# Copyright 2023-2025 SGLang Team; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE
  - `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: +import math; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor
  - `python/sglang/srt/layers/rotary_embedding.py` modified +256/-0 (256 lines); hunks: def get_rope_index_glm4v(; def _get_llm_pos_ids_for_vision(; symbols: get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths, _get_llm_pos_ids_for_vision
  - `docs/supported_models/multimodal_language_models.md` modified +1/-0 (1 lines); hunks: in the GitHub search bar.
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`; keywords observed in patches: config, vision, kv, moe, spec, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18856 - [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel

- Link: https://github.com/sgl-project/sglang/pull/18856
- Status/date: `merged`, created 2026-02-15, merged 2026-02-16; author `yuan-luo`.
- Diff scope read: `1` files, `+268/-3`; areas: misc; keywords: cache, cuda, kv, triton.
- Code diff details:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3 (271 lines); hunks: def _compute_cos_sin_cache(self) -> torch.Tensor:; def forward_native( # type: ignore[override]; symbols: _compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace, Ernie4_5_VLRotaryEmbedding
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: cache, cuda, kv, triton. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19743 - [VLM] Support cos sin cache for Ernie4.5-VL

- Link: https://github.com/sgl-project/sglang/pull/19743
- Status/date: `merged`, created 2026-03-03, merged 2026-03-04; author `yuan-luo`.
- Diff scope read: `1` files, `+34/-12`; areas: model wrapper; keywords: cache, config, moe, processor, quant, triton, vision.
- Code diff details:
  - `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: from sglang.srt.layers.logits_processor import LogitsProcessor; def forward(; symbols: forward, __init__, dtype, device
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/ernie45_vl.py`; keywords observed in patches: cache, config, moe, processor, quant, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/ernie45_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
