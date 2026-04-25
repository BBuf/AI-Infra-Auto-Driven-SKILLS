---
name: sglang-ltx23-hq-optimization
description: PR-backed and current-main optimization manual for LTX-2.3 High Quality pipeline in SGLang Diffusion. Use when Codex needs to audit or extend LTX2TwoStageHQPipeline, LTX-2.3 two-stage LoRA switching, HQ sigma/timestep semantics, res2s RK2 refinement, audio/video denoising, Gemma prompt trimming, low-VRAM device snapshots, or LTX-2.3 HQ sampling defaults.
---

# SGLang LTX-2.3 HQ Optimization

## Overview

LTX-2.3 HQ is a SGLang Diffusion two-stage video+audio pipeline lane. It differs from legacy LTX-2 and non-HQ LTX-2.3 because stage 1 runs at half resolution, HQ sigma shift is resolution-aware, distilled LoRA strength is stage-specific, and stage 2 uses HQ timestep semantics plus res2s refinement behavior.

Current evidence snapshot:

- SGLang `origin/main`: `bca3dd958` on `2026-04-24`
- Diff-reviewed PRs: #23366 and #23624
- Main runtime files: `ltx_2_pipeline.py`, `ltx_2_denoising.py`, `configs/sample/ltx_2.py`, `configs/pipeline_configs/ltx_2.py`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- pipeline class: `LTX2Pipeline`, `LTX2TwoStagePipeline`, or `LTX2TwoStageHQPipeline`
- whether `ltx_variant == "ltx_2_3"`
- stage 1 and stage 2 distilled LoRA strengths
- sigma schedule source and number of latent tokens
- video/audio CFG, STG, rescale, modality scale, and skip-step settings
- low-VRAM snapshot mode vs resident two-transformer mode
- whether batch height/width have already been halved by `LTX2HalveResolutionStage`

## Core Principle

Do not apply HQ semantics to every LTX-2.3 path.

- HQ uses resolution-aware sigma shift because the official HQ path passes a half-resolution latent to the scheduler.
- Non-HQ one-stage and two-stage LTX-2.3 use the constant anchor schedule.
- HQ stage-2 defaults differ from non-HQ: distilled LoRA strength, STG, rescale, and sampler choices are part of the quality contract.

## PR Dossier Rule

Before adding LTX-2.3 HQ evidence, open the PR diff/source and update `references/pr-history.md` with motivation, implementation, code excerpts, reviewed files, and validation implications.

## Validation Lanes

- LTX-2 legacy one-stage unchanged.
- LTX-2.3 one-stage unchanged.
- LTX-2.3 non-HQ two-stage unchanged.
- LTX-2.3 HQ 1088x1920 default with stage-1 halving.
- HQ stage-1/stage-2 distilled LoRA strengths from request extras.
- Audio+video denoise with fixed seed and regression artifacts.
- Low-VRAM and resident two-transformer modes.

## References

- `references/pr-history.md`: diff-reviewed LTX-2.3 HQ PR cards.
