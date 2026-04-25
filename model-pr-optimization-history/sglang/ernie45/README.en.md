# SGLang Ernie4.5 / Ernie4.5-VL Support and PR History

This note tracks the Ernie4.5 multimodal path in SGLang at commit
`c122d343adb969cd9bbd1af2ca86727a11be3845`.

- Status: supported on current mainline

## Key Conclusions

- Ernie4.5-VL support landed as a dedicated runtime and processor, not as a
  thin alias.
- The most important optimization work afterward is all in the rotary path:
  first a fused Triton kernel, then cos/sin cache reuse.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/ernie45_vl.py`
- `sglang/python/sglang/srt/models/ernie45_moe_vl.py`
- `sglang/python/sglang/srt/multimodal/processors/ernie45_vl.py`
- `sglang/python/sglang/srt/layers/rotary_embedding.py`

## Landed PRs

- [#15679](https://github.com/sgl-project/sglang/pull/15679)
  `Add Ernie4.5 VL model support`
  Diff reviewed: `6` files, `2072` additions.
  Adds the native Ernie4.5-VL runtime, MoE-VL runtime, processor, and
  multimodal registration.
- [#18856](https://github.com/sgl-project/sglang/pull/18856)
  `Optimize Ernie4.5-VL rotary embedding with fused triton kernel`
  Diff reviewed: `1` file, `268` additions, `3` deletions.
  Adds the fused Triton Q/K rotary kernel for Ernie4.5's `(h, w, t)` layout.
- [#19743](https://github.com/sgl-project/sglang/pull/19743)
  `Support cos sin cache for Ernie4.5-VL`
  Diff reviewed: `1` file, `34` additions, `12` deletions.
  Rewrites the vision tower to reuse `get_rope(...).get_cos_sin(...)`.

## Current Contract

If Ernie4.5-VL regresses, check which rotary path is active before modifying the
full model. Many correctness and performance changes in this family are local to
vision rotary handling.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `ERNIE 4.5 / ERNIE 4.5 VL` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-23 | [#15679](https://github.com/sgl-project/sglang/pull/15679) | merged | [Model] Add Ernie4.5 VL model support | model wrapper, MoE/router, multimodal/processor, docs/config | `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` |
| 2026-02-15 | [#18856](https://github.com/sgl-project/sglang/pull/18856) | merged | [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel | misc | `python/sglang/srt/layers/rotary_embedding.py` |
| 2026-03-03 | [#19743](https://github.com/sgl-project/sglang/pull/19743) | merged | [VLM] Support cos sin cache for Ernie4.5-VL | model wrapper | `python/sglang/srt/models/ernie45_vl.py` |

### File-level PR diff reading notes

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


### Gap and optimization follow-up

- Covered PRs: 3; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
