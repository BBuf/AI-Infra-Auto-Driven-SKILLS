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
