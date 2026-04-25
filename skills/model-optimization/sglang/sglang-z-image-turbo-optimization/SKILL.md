---
name: sglang-z-image-turbo-optimization
description: PR-backed optimization manual for Z-Image and Z-Image-Turbo in SGLang Diffusion. Use when Codex needs to audit or extend Z-Image registry entries, Turbo/base sampling defaults, CFG normalization, sequence-parallel latent sharding, Cache-DiT/TeaCache behavior, LoRA/FP8 coverage, or AMD nightly validation.
---

# SGLang Z-Image-Turbo Optimization

## Overview

Z-Image and Z-Image-Turbo are native SGLang Diffusion model families, not generic Diffusers-only fallbacks. Current main registers both `Tongyi-MAI/Z-Image-Turbo` and `Tongyi-MAI/Z-Image`, uses `ZImagePipelineConfig`, and serves through `ZImagePipeline` or `ComfyUIZImagePipeline`.

Current evidence snapshot:

- SGLang `origin/main`: `bca3dd958` on `2026-04-24`
- Core runtime: `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py`
- DiT runtime: `python/sglang/multimodal_gen/runtime/models/dits/zimage.py`
- Registry: `python/sglang/multimodal_gen/registry.py`
- Cookbook: `docs_new/cookbook/diffusion/Z-Image/Z-Image-Turbo.mdx`
- Diff-reviewed PRs: #17822, #19733, #23455

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- model id: `Tongyi-MAI/Z-Image-Turbo` or `Tongyi-MAI/Z-Image`
- sampling class: `ZImageTurboSamplingParams` or `ZImageSamplingParams`
- `cfg_normalization`, `guidance_scale`, and `guidance_rescale`
- sequence parallelism: `--sp-degree`, `--ulysses-degree`, `--ring-degree`
- Cache-DiT/TeaCache env settings
- LoRA/FP8 transformer path, if used
- AMD CI coverage status for `nightly-amd-1-gpu-zimage-turbo`

## Core Principles

- Keep Turbo and base Z-Image sampling defaults separate. Turbo defaults to few-step distilled behavior, while base Z-Image keeps CFG enabled.
- Do not route `z-image` detectors through the Turbo sampling params unless the model id contains `z-image-turbo`.
- Preserve Z-Image's native H/W sequence-parallel sharding plan; it shards latent spatial axes before patchification-sensitive gather.
- `cfg_normalization` is a quality-control knob, not a generic sampling parameter to silently drop.
- AMD CI coverage is part of the support surface because #23455 restored a test that had been removed for missing `__main__`.

## Validation Lanes

- Registry resolves `Tongyi-MAI/Z-Image-Turbo` to `ZImageTurboSamplingParams` and `ZImagePipelineConfig`.
- Registry resolves `Tongyi-MAI/Z-Image` to `ZImageSamplingParams` and excludes Turbo.
- `test/registered/amd/test_zimage_turbo.py` runs via `python test/registered/amd/test_zimage_turbo.py -v`.
- Diffusion server image generation returns bytes and passes the CLIP threshold when CLIP dependencies are available.
- Multi-GPU SP tests preserve non-square latent shapes and gather/crop correctly.

## PR Dossier Rule

Before adding Z-Image evidence, open the PR diff/source and update `references/pr-history.md` with motivation, key implementation, short code excerpts, reviewed files, and validation implications.

## References

- `references/pr-history.md`: diff-reviewed Z-Image PR cards.
