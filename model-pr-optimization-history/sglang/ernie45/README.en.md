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
