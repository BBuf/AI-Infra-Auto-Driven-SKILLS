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

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG LTX-Video 2.3 HQ PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-21 | [#23366](https://github.com/sgl-project/sglang/pull/23366) | merged | [diffusion] model: support LTX2.3 high quality pipeline | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` |
| 2026-04-24 | [#23624](https://github.com/sgl-project/sglang/pull/23624) | merged | [diffusion] fix: unify LTX-2.3 HQ codepath gates for all LTX-2.3 variants | multimodal/processor, tests/benchmarks | `python/sglang/multimodal_gen/test/server/perf_baselines.json`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` |

## Diff Cards

### PR #23366 - [diffusion] model: support LTX2.3 high quality pipeline

- Link: https://github.com/sgl-project/sglang/pull/23366
- Status/date: `merged`, created 2026-04-21, merged 2026-04-24; author `mickqian`.
- Diff scope read: `19` files, `+1501/-419`; areas: model wrapper, multimodal/processor, tests/benchmarks, docs/config; keywords: config, lora, scheduler, cache, spec, attention, cuda, test.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py` modified +974/-313 (1287 lines); hunks: import copy; is_ltx2_two_stage_pipeline_name,; symbols: LTX2DenoisingContext, LTX2DenoisingContext, LTX2DenoisingStage, __init__
  - `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` modified +129/-26 (155 lines); hunks: from diffusers import FlowMatchEulerDiscreteScheduler; def build_official_ltx2_sigmas(; symbols: build_official_ltx2_sigmas, build_official_ltx2_sigmas, LTX2SigmaPreparationStage, forward
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` modified +129/-23 (152 lines); hunks: def __init__(; def _should_reset_stage2_generators(server_args: ServerArgs) -> bool:; symbols: __init__, _should_reset_stage2_generators, _build_stage2_renoise_generator, _ltx2_renoise_like
  - `python/sglang/multimodal_gen/runtime/layers/lora/linear.py` modified +45/-10 (55 lines); hunks: torch._dynamo.config.recompile_limit = 64; def _merge_lora_into_data(; symbols: BaseLayerWithLoRA, __init__, _merge_lora_into_data, merge_lora_weights
  - `python/sglang/multimodal_gen/test/server/perf_baselines.json` modified +41/-0 (41 lines); hunks: "expected_avg_denoise_ms": 890.17,
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`; keywords observed in patches: config, lora, scheduler, cache, spec, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23624 - [diffusion] fix: unify LTX-2.3 HQ codepath gates for all LTX-2.3 variants

- Link: https://github.com/sgl-project/sglang/pull/23624
- Status/date: `merged`, created 2026-04-24, merged 2026-04-24; author `mickqian`.
- Diff scope read: `3` files, `+65/-61`; areas: multimodal/processor, tests/benchmarks; keywords: config, scheduler, lora, test.
- Code diff details:
  - `python/sglang/multimodal_gen/test/server/perf_baselines.json` modified +36/-35 (71 lines); hunks: },
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` modified +16/-22 (38 lines); hunks: def forward(self, batch: Req, server_args: ServerArgs) -> Req:; def forward(self, batch: Req, server_args: ServerArgs) -> Req:; symbols: forward, forward, forward, forward
  - `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` modified +13/-4 (17 lines); hunks: class LTX2SigmaPreparationStage(PipelineStage):; symbols: LTX2SigmaPreparationStage, forward, to
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/test/server/perf_baselines.json`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`; keywords observed in patches: config, scheduler, lora, test. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/test/server/perf_baselines.json`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
