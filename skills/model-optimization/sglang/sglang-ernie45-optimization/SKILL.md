---
name: sglang-ernie45-optimization
description: PR-backed optimization manual for Ernie4.5 / Ernie4.5-VL in SGLang. Use when Codex needs to audit, debug, extend, or document the SGLang Ernie4.5 multimodal runtime, especially the initial VL landing, fused Triton rotary path, and later cos/sin cache rewrite for Ernie4.5-VL.
---

# SGLang Ernie4.5 / Ernie4.5-VL Optimization

## Overview

Ernie4.5 is already supported on the checked SGLang mainline. The high-signal
changes for this family are the initial VL bring-up, then two successive rotary
embedding optimizations for the vision tower.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors:
  `model-pr-optimization-history/sglang/ernie45/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the bar.
For Ernie4.5-VL, name the exact rotary path being changed:
original rotary embedding, fused Triton rope, or cached cos/sin route.

## Runtime Surfaces

- `sglang/python/sglang/srt/configs/model_config.py`
- `sglang/python/sglang/srt/models/ernie45_vl.py`
- `sglang/python/sglang/srt/models/ernie45_moe_vl.py`
- `sglang/python/sglang/srt/multimodal/processors/ernie45_vl.py`
- `sglang/python/sglang/srt/layers/rotary_embedding.py`

## Current Main Summary

- `#15679` lands the Ernie4.5-VL and Ernie4.5-MoE-VL runtime plus the multimodal
  processor and model-config registration.
- `#18856` adds a fused Triton Q/K rope kernel tailored to Ernie4.5-VL's
  `(h, w, t)` layout.
- `#19743` rewrites the vision tower to use `get_rope(...).get_cos_sin(...)`
  instead of recomputing rotary embeddings for every call.

## Key Landed PRs

- [#15679](https://github.com/sgl-project/sglang/pull/15679) `Add Ernie4.5 VL model support`
- [#18856](https://github.com/sgl-project/sglang/pull/18856) `Optimize Ernie4.5-VL rotary embedding with fused triton kernel`
- [#19743](https://github.com/sgl-project/sglang/pull/19743) `Support cos sin cache for Ernie4.5-VL`

## Validation Lanes

- Ernie4.5-VL startup and processor/token expansion.
- Vision-tower rotary correctness under both the fused kernel and cached cos/sin
  path.
- Long-input and multi-image / video cases that stress `grid_thw` and rotary
  indexing.

## References

- `references/pr-history.md`: diff-reviewed Ernie4.5 cards.
