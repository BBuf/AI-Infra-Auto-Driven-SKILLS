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
