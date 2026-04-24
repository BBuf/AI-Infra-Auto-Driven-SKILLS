# SGLang Z-Image/Z-Image-Turbo Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `bca3dd958` on `2026-04-24`.

Scope: Z-Image, Z-Image-Turbo, Turbo/base sampling defaults, CFG normalization, native SP latent sharding, AMD nightly validation, and direct-file CI execution.

## Summary

Z-Image-Turbo is a native SGLang Diffusion path backed by `ZImagePipelineConfig`, `ZImageTransformer2DModel`, and explicit registry entries. The important contract is that Turbo and base Z-Image use different sampling classes, while AMD support is guarded by the restored `nightly-amd-1-gpu-zimage-turbo` test.

## Diff-Reviewed PR Cards

### #17822 - Sync with upstream Z-Image

- Link: https://github.com/sgl-project/sglang/pull/17822
- State: merged at `2026-01-29T13:10:11Z`
- Diff coverage: `209` lines, `5` files.
- Adds separate `ZImageTurboSamplingParams` and `ZImageSamplingParams`, distinct registry detectors, `prepare_neg_cond_kwargs`, and `cfg_normalization` in denoising.
- Validation: verify Turbo does not inherit base-Z-Image CFG defaults.

### #19733 - Add AMD Z-Image-Turbo nightly test

- Link: https://github.com/sgl-project/sglang/pull/19733
- State: merged at `2026-03-05T16:36:18Z`
- Diff coverage: `308` lines, `4` files.
- Adds AMD nightly generation coverage for `Tongyi-MAI/Z-Image-Turbo`, with image artifacts and CLIP-score validation.
- Validation: monitor both image bytes and CLIP score, not just server startup.

### #23455 - Restore Z-Image-Turbo executable test

- Link: https://github.com/sgl-project/sglang/pull/23455
- State: merged at `2026-04-23T05:28:21Z`
- Diff coverage: `291` lines, `2` files.
- Restores `test/registered/amd/test_zimage_turbo.py` after a CI hygiene removal and adds a direct `pytest.main` entry.
- Validation: direct-file execution must keep working.

Full PR dossier: `skills/model-optimization/sglang/sglang-z-image-turbo-optimization/references/pr-history.md`.
