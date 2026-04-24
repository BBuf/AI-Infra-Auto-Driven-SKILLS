# SGLang LTX-2.3 HQ Support and Optimization Timeline

Scope: LTX-2.3 High Quality two-stage video+audio pipeline, stage-specific LoRA, HQ sigma/timestep semantics, res2s refinement, sampling defaults, and regression gates.

Evidence snapshot: SGLang `origin/main` `bca3dd958` (`2026-04-24`). Full dossier: `skills/model-optimization/sglang/sglang-ltx23-hq-optimization/references/pr-history.md`.

## Diff-Reviewed PRs

- #23366 added `LTX2TwoStageHQPipeline`, `LTX23HQSamplingParams`, resolution-aware sigma shift for HQ, stage-specific distilled LoRA strengths, and HQ denoising semantics. Full diff reviewed: `5411` lines, `19` files.
- #23624 tightened comments and gates so HQ sigma semantics apply only to `LTX2TwoStageHQPipeline`, not all native LTX-2.3 paths. Full diff reviewed: `505` lines, `3` files.

## Current Contract

HQ stage 1 uses half-resolution latent token count to build sigmas. Non-HQ one-stage and two-stage LTX-2.3 should keep constant-anchor sigmas. The default HQ request is 1088x1920, 15 steps, stage-1 LoRA strength 0.25, and stage-2 LoRA strength 0.5.
