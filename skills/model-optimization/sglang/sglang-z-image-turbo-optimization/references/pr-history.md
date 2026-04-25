# SGLang Z-Image/Z-Image-Turbo PR History

Evidence snapshot: SGLang `origin/main` `bca3dd958` on `2026-04-24`.

Scope: native Z-Image and Z-Image-Turbo support in SGLang Diffusion, including registry selection, sampling defaults, CFG normalization, sequence-parallel latent sharding, docs, and AMD validation.

## Current-Main Code Surfaces

- `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py`
- `python/sglang/multimodal_gen/configs/sample/zimage.py`
- `python/sglang/multimodal_gen/configs/sample/sampling_params.py`
- `python/sglang/multimodal_gen/registry.py`
- `python/sglang/multimodal_gen/runtime/models/dits/zimage.py`
- `python/sglang/multimodal_gen/runtime/pipelines/zimage_pipeline.py`
- `docs_new/cookbook/diffusion/Z-Image/Z-Image-Turbo.mdx`
- `test/registered/amd/test_zimage_turbo.py`

## PR #17822 - Sync with upstream Z-Image

- Link: https://github.com/sgl-project/sglang/pull/17822
- State: merged at `2026-01-29T13:10:11Z`
- Diff coverage: full diff fetched, `209` lines, `5` files.
- Motivation: SGLang already had a Z-Image path, but upstream Z-Image changes required separate Turbo/base sampling defaults, explicit negative-conditioning kwargs, and CFG normalization for quality control.
- Key implementation: split `ZImageTurboSamplingParams` from `ZImageSamplingParams`, register Turbo and base model ids separately, add `cfg_normalization` to generic sampling params, and renormalize CFG predictions in denoising.

```python
class ZImageTurboSamplingParams(SamplingParams):
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    cfg_normalization: float | bool = False

class ZImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50
    negative_prompt: str = " "
    guidance_scale: float = 5.0
    cfg_normalization: float | bool = True
```

```python
register_configs(
    sampling_param_cls=ZImageTurboSamplingParams,
    pipeline_config_cls=ZImagePipelineConfig,
    hf_model_paths=["Tongyi-MAI/Z-Image-Turbo"],
    model_detectors=[lambda hf_id: "z-image-turbo" in hf_id.lower()],
)
```

- Validation implications: confirm `z-image-turbo` does not accidentally pick base CFG defaults; test non-Turbo `z-image` detection separately.

## PR #19733 - Add AMD Z-Image-Turbo nightly test

- Link: https://github.com/sgl-project/sglang/pull/19733
- State: merged at `2026-03-05T16:36:18Z`
- Diff coverage: full diff fetched, `308` lines, `4` files.
- Motivation: Z-Image-Turbo was supported in SGLang Diffusion but lacked AMD nightly coverage.
- Key implementation: add `test/registered/amd/test_zimage_turbo.py`, register `nightly-amd-1-gpu-zimage-turbo`, launch `Tongyi-MAI/Z-Image-Turbo`, and validate generated image bytes with a CLIP score threshold.

```python
register_amd_ci(est_time=1800, suite="nightly-amd-1-gpu-zimage-turbo", nightly=True)

DiffusionServerArgs(model_path="Tongyi-MAI/Z-Image-Turbo", modality="image")
```

- Validation implications: AMD regressions should inspect both perf records and generated-image summaries, not just server boot.

## PR #23455 - Restore Z-Image-Turbo and int4fp8 MoE tests with executable entries

- Link: https://github.com/sgl-project/sglang/pull/23455
- State: merged at `2026-04-23T05:28:21Z`
- Diff coverage: full diff fetched, `291` lines, `2` files.
- Motivation: #23305 removed tests missing executable `__main__` entries. Z-Image-Turbo AMD nightly coverage had to be restored with a proper Python entrypoint.
- Key implementation: re-add `test/registered/amd/test_zimage_turbo.py` and end it with `sys.exit(pytest.main([__file__, "-v"]))`.

```python
if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
```

- Validation implications: CI discovery alone is insufficient. The registered test must also run under the direct-file execution mode used by the suite.

## Current Gaps

- No model-code change landed in the 2026-04-24 incremental range for Z-Image itself. The active gap was documentation/skill coverage, now added here.
- If future PRs change Cache-DiT, TeaCache, LoRA, FP8 transformer paths, or SP gather for Z-Image, they should be added as full PR cards rather than title-only rows.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Z-Image-Turbo PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-01-27 | [#17822](https://github.com/sgl-project/sglang/pull/17822) | merged | [wip] sync with upstream zImage | multimodal/processor, docs/config | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/registry.py`, `python/sglang/multimodal_gen/configs/sample/sampling_params.py` |
| 2026-03-03 | [#19733](https://github.com/sgl-project/sglang/pull/19733) | merged | [AMD] [Z-Image-Turbo Day 0] Add Z-Image-Turbo nightly test for AMD GPUs | multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml` |
| 2026-04-22 | [#23455](https://github.com/sgl-project/sglang/pull/23455) | merged | [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry | MoE/router, quantization, multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py` |

## Diff Cards

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


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
