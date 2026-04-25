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

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Z-Image-Turbo` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-01-27 | [#17822](https://github.com/sgl-project/sglang/pull/17822) | merged | [wip] sync with upstream zImage | multimodal/processor, docs/config | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/registry.py`, `python/sglang/multimodal_gen/configs/sample/sampling_params.py` |
| 2026-03-03 | [#19733](https://github.com/sgl-project/sglang/pull/19733) | merged | [AMD] [Z-Image-Turbo Day 0] Add Z-Image-Turbo nightly test for AMD GPUs | multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml` |
| 2026-04-22 | [#23455](https://github.com/sgl-project/sglang/pull/23455) | merged | [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry | MoE/router, quantization, multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py` |

### File-level PR diff reading notes

### PR #17822 - [wip] sync with upstream zImage

- Link: https://github.com/sgl-project/sglang/pull/17822
- Status/date: `merged`, created 2026-01-27, merged 2026-01-29; author `yhyang201`.
- Diff scope read: `5` files, `+79/-4`; areas: multimodal/processor, docs/config; keywords: config, cache.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` modified +26/-0 (26 lines); hunks: def _predict_noise_with_cfg(; def _predict_noise_with_cfg(; symbols: _predict_noise_with_cfg, _predict_noise_with_cfg
  - `python/sglang/multimodal_gen/registry.py` modified +16/-3 (19 lines); hunks: WanT2V_1_3B_SamplingParams,; def _register_configs():; symbols: _register_configs
  - `python/sglang/multimodal_gen/configs/sample/sampling_params.py` modified +13/-0 (13 lines); hunks: class SamplingParams:; def _finite_non_negative_float(; symbols: SamplingParams:, _finite_non_negative_float, add_cli_args
  - `python/sglang/multimodal_gen/configs/sample/zimage.py` modified +12/-1 (13 lines); hunks: @dataclass; class ZImageSamplingParams(SamplingParams):; symbols: ZImageSamplingParams, ZImageTurboSamplingParams, ZImageSamplingParams, ZImageSamplingParams
  - `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py` modified +12/-0 (12 lines); hunks: def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):; symbols: prepare_pos_cond_kwargs, prepare_neg_cond_kwargs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/registry.py`, `python/sglang/multimodal_gen/configs/sample/sampling_params.py`; keywords observed in patches: config, cache. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/registry.py`, `python/sglang/multimodal_gen/configs/sample/sampling_params.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19733 - [AMD] [Z-Image-Turbo Day 0] Add Z-Image-Turbo nightly test for AMD GPUs

- Link: https://github.com/sgl-project/sglang/pull/19733
- Status/date: `merged`, created 2026-03-03, merged 2026-03-05; author `michaelzhang-ai`.
- Diff scope read: `4` files, `+235/-0`; areas: multimodal/processor, tests/benchmarks; keywords: test, config, doc, processor.
- Code diff details:
  - `test/registered/amd/test_zimage_turbo.py` added +150/-0 (150 lines); hunks: +"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""; symbols: _save_image_and_write_summary, _compute_clip_score, TestZImageTurboAMD, teardown_class
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +42/-0 (42 lines); hunks: jobs:; jobs:
  - `.github/workflows/nightly-test-amd.yml` modified +42/-0 (42 lines); hunks: jobs:; jobs:
  - `test/run_suite.py` modified +1/-0 (1 lines); hunks: "nightly-amd",
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/test_zimage_turbo.py`, `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`; keywords observed in patches: test, config, doc, processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/test_zimage_turbo.py`, `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23455 - [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry

- Link: https://github.com/sgl-project/sglang/pull/23455
- Status/date: `merged`, created 2026-04-22, merged 2026-04-23; author `bingxche`.
- Diff scope read: `2` files, `+220/-0`; areas: MoE/router, quantization, multimodal/processor, tests/benchmarks; keywords: test, attention, config, fp8, moe, processor, quant, triton.
- Code diff details:
  - `test/registered/amd/test_zimage_turbo.py` added +156/-0 (156 lines); hunks: +"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""; symbols: _save_image_and_write_summary, _compute_clip_score, TestZImageTurboAMD, teardown_class
  - `test/registered/quant/test_int4fp8_moe.py` added +64/-0 (64 lines); hunks: +from types import SimpleNamespace; symbols: TestMixtralAccuracy, setUpClass, tearDownClass, test_gsm8k
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py`; keywords observed in patches: test, attention, config, fp8, moe, processor. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 3; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
