# LTX-2.3 HQ PR History

Evidence sweep:

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
- Manual diff review date: `2026-04-24`
- Searched paths: LTX-2 pipeline config, sampling params, two-stage pipeline, LTX-2 denoising stage, registry, server tests, compatibility docs.
- Searched PR terms: `LTX2.3`, `LTX-2.3`, `LTX2TwoStageHQPipeline`, `ltx_2_3`, `HQ`, `res2s`.

## Runtime Surfaces

- `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`
- `python/sglang/multimodal_gen/configs/sample/ltx_2.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2.py`
- `python/sglang/multimodal_gen/registry.py`
- `python/sglang/multimodal_gen/test/server/gpu_cases.py`

## Diff-Reviewed PR Cards

### PR #23366 - Support LTX-2.3 high quality pipeline

- Link: https://github.com/sgl-project/sglang/pull/23366
- State: merged at `2026-04-24T06:18:21Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `5411` lines, `19` files; current-main source rechecked at `bca3dd958`.
- Motivation: LTX-2.3 HQ is not just a larger-resolution setting. It follows the official two-stage HQ pipeline with half-resolution stage 1, resolution-aware sigma shift, stage-specific distilled LoRA strengths, HQ timestep semantics, audio/video guidance differences, and dedicated defaults.
- Key implementation:
  - adds `LTX23HQSamplingParams` with `1088x1920`, `15` steps, stage-1 LoRA strength `0.25`, stage-2 LoRA strength `0.5`, and HQ guidance defaults;
  - trims Gemma prompt whitespace in `LTX2PipelineConfig.tokenize_prompt`;
  - adds `number_of_tokens` to `build_official_ltx2_sigmas` and makes HQ stage 1 compute it from half-resolution latents;
  - threads sampler names and stage-specific LoRA strengths through `LTX2TwoStagePipeline`;
  - adds `LTX2TwoStageHQPipeline` with `LTX23HQSamplingParams`;
  - expands `ltx_2_denoising.py` for HQ timestep semantics and res2s refinement.
- Key code excerpts:

```python
class LTX23HQSamplingParams(LTX23SamplingParams):
    height: int = 1088
    width: int = 1920
    num_inference_steps: int = 15
    distilled_lora_strength_stage_1: float = 0.25
    distilled_lora_strength_stage_2: float = 0.5
```

```python
batch.sigmas = build_official_ltx2_sigmas(
    int(batch.num_inference_steps),
    number_of_tokens=latent_num_frames * latent_height * latent_width,
)
```

```python
class LTX2TwoStageHQPipeline(LTX2TwoStagePipeline):
    pipeline_name = "LTX2TwoStageHQPipeline"
    pipeline_config_cls = LTX2PipelineConfig
    sampling_params_cls = LTX23HQSamplingParams
```

- Reviewed files:
  - runtime: `ltx_2_pipeline.py`, `ltx_2_denoising.py`, `denoising_av.py`, `ltx_2.py` pipeline config
  - sampling/docs/tests: `configs/sample/ltx_2.py`, `registry.py`, `gpu_cases.py`, compatibility docs
- Validation implications: validate legacy LTX-2, non-HQ LTX-2.3, and HQ LTX-2.3 separately. HQ changes should be checked with fixed seeds, stage-1 half-resolution latents, and audio+video quality artifacts.

### PR #23624 - Unify LTX-2.3 HQ codepath gates

- Link: https://github.com/sgl-project/sglang/pull/23624
- State: merged at `2026-04-24T09:44:08Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `505` lines, `3` files.
- Motivation: after #23366, the key risk was accidentally applying HQ's resolution-aware sigma shift to every native LTX-2.3 variant. The follow-up clarifies that official HQ, non-HQ two-stage, and one-stage entry points call the scheduler differently.
- Key implementation: leaves the gate on `server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"` and adds source-of-truth comments explaining why only HQ computes token-count-aware sigmas from the half-resolution stage-1 latent.
- Key code excerpt:

```python
if server_args.pipeline_class_name == "LTX2TwoStageHQPipeline":
    # batch.height/width have already been halved by
    # LTX2HalveResolutionStage, so these latents are the
    # half-resolution stage-1 shape.
    batch.sigmas = build_official_ltx2_sigmas(
        int(batch.num_inference_steps),
        number_of_tokens=latent_num_frames * latent_height * latent_width,
    )
```

- Reviewed files: `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` plus the PR patch for related gate comments.
- Validation implications: regression tests should assert that one-stage and non-HQ two-stage LTX-2.3 keep constant-anchor sigmas while HQ uses resolution-aware sigmas.

## Validation Notes

- Any future LTX-2.3 patch must name which pipeline class it affects.
- If visual quality drifts, inspect sigma/timestep semantics and LoRA strength before changing model weights.
