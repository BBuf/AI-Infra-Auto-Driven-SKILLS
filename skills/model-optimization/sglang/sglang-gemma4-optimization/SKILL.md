---
name: sglang-gemma4-optimization
description: PR-backed optimization manual for Gemma 4 in SGLang. Use when an engineer needs to audit, debug, extend, or document Gemma 4 text, MoE, multimodal, reasoning, tool use, and quantized MoE serving.
---

# SGLang Gemma 4 Optimization

## Overview

This skill covers Gemma 4 text, MoE, multimodal, reasoning, tool use, and quantized MoE serving.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/gemma4/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/gemma4_causal.py`
- `sglang/python/sglang/srt/models/gemma4_mm.py`
- `sglang/python/sglang/srt/models/gemma4_vision.py`
- `sglang/python/sglang/srt/models/gemma4_audio.py`

## Current Main Summary

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## Key Landed PRs

- [#21952](https://github.com/sgl-project/sglang/pull/21952) `New Model: Gemma 4`: Initial Gemma 4 support in SGLang.
- [#22079](https://github.com/sgl-project/sglang/pull/22079) `Gemma4 nvfp4 fix`: Fixed the NVFP4 launch path.
- [#22408](https://github.com/sgl-project/sglang/pull/22408) `Adding Gemma 4 to Nightly CI`: Added model-family regression coverage.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
