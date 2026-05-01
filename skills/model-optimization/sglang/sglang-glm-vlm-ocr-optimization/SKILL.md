---
name: sglang-glm-vlm-ocr-optimization
description: PR-backed and current-main optimization manual for GLM-4V, GLM-4.1V, GLM-4.5V, GLM-4.6V, GLM-Glyph, and GLM-OCR in SGLang. Use when an engineer needs to recover, extend, or audit GLM vision processors, GLM4V MoE, vision encoder DP/PP, GLM-OCR/NextN loading, transformers compatibility, Conv3D-to-linear projection, AMD/NPU validation, or VLM/OCR cookbook recipes.
---

# SGLang GLM VLM/OCR Optimization

## Overview

This skill covers GLM visual and OCR models: GLM-4V, GLM-4.1V, GLM-4.5V, GLM-4.6V, GLM-Glyph, and GLM-OCR. It is separate from GLM text because the processor, vision encoder, image token placement, transformers compatibility, and OCR-specific NextN behavior dominate debugging.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Detailed PR diff dossier: `references/pr-history.md`
- Runtime files: `glm4v.py`, `glm4v_moe.py`, `glm_ocr.py`, `glm_ocr_nextn.py`
- Processor files: `multimodal/processors/glm4v.py`, `glmasr.py` when audio-adjacent
- Docs/snippets: GLM-4.5V, GLM-4.6V, GLM-Glyph, GLM-OCR cookbook pages

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- model: GLM-4V/4.1V/4.5V/4.6V/Glyph/OCR
- input type: image, OCR page, multi-image, or video-like visual sequence
- encoder DP/PP enabled or disabled
- quantization: BF16, FP8, compressed-tensors, or vendor format
- transformers version and whether config fields are optional/missing
- NPU/AMD/CPU backend
- whether OCR NextN is active

## Core Principle

Start at the multimodal boundary.

- If text-only GLM passes, inspect processor registration, image feature projection, position IDs, and vision model DP before text MoE.
- GLM-OCR often fails because of processor/config field drift or NextN omission, not only because of kernels.
- GLM-4.5V/4.6V share text MoE work but need separate vision validation.
- Treat `#20463` and `#20740` as a pair: `#20463` documents a real Conv3D-to-linear loader regression mechanism, while `#20740` defines the current-main state after revert.

## Main Runtime Surfaces

- `python/sglang/srt/models/glm4v.py`
- `python/sglang/srt/models/glm4v_moe.py`
- `python/sglang/srt/models/glm_ocr.py`
- `python/sglang/srt/models/glm_ocr_nextn.py`
- `python/sglang/srt/multimodal/processors/glm4v.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-Glyph.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx`

## Optimization Order

1. Validate processor registration and image feature tensor shapes.
2. Validate text-only fallback if the model supports it.
3. Validate encoder DP/PP and TP assumptions.
4. Validate quantized startup.
5. Validate OCR/Glyph examples.
6. Update cookbook and registered tests.

## Open PRs to Track

- [#9349](https://github.com/sgl-project/sglang/pull/9349): GLM-4.5V FP8.
- [#14662](https://github.com/sgl-project/sglang/pull/14662): GLM4.6V ktransformers.
- [#19728](https://github.com/sgl-project/sglang/pull/19728): ROCm GLM-4.5V-FP8 startup.
- [#22961](https://github.com/sgl-project/sglang/pull/22961): NPU GLM-4.5V.

## PR Dossier Rule

When producing or updating model optimization history, do not summarize a PR as one sentence. Read the source diff and write:

- motivation/root cause
- key implementation idea
- a short code fragment that proves the implementation surface
- validation evidence
- current-main status and risk, especially for reverted or open PRs

## Validation Lanes

- GLM-4.5V/4.6V image prompts.
- GLM-OCR page recognition.
- GLM-Glyph examples.
- Vision encoder DP/PP.
- Transformers compatibility.
- AMD/NPU startup and accuracy.
