# SGLang LTX-2.3 HQ Support and Optimization Timeline

Scope: LTX-2.3 High Quality two-stage video+audio pipeline, stage-specific LoRA, HQ sigma/timestep semantics, res2s refinement, sampling defaults, and regression gates.

Evidence snapshot: SGLang `origin/main` `bca3dd958` (`2026-04-24`). Full dossier: `skills/model-optimization/sglang/sglang-ltx23-hq-optimization/references/pr-history.md`.

## Diff-Reviewed PRs

- #23366 added `LTX2TwoStageHQPipeline`, `LTX23HQSamplingParams`, resolution-aware sigma shift for HQ, stage-specific distilled LoRA strengths, and HQ denoising semantics. Full diff reviewed: `5411` lines, `19` files.
- #23624 tightened comments and gates so HQ sigma semantics apply only to `LTX2TwoStageHQPipeline`, not all native LTX-2.3 paths. Full diff reviewed: `505` lines, `3` files.

## Current Contract

HQ stage 1 uses half-resolution latent token count to build sigmas. Non-HQ one-stage and two-stage LTX-2.3 should keep constant-anchor sigmas. The default HQ request is 1088x1920, 15 steps, stage-1 LoRA strength 0.25, and stage-2 LoRA strength 0.5.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `LTX-Video 2.3 HQ` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-21 | [#23366](https://github.com/sgl-project/sglang/pull/23366) | merged | [diffusion] model: support LTX2.3 high quality pipeline | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` |
| 2026-04-24 | [#23624](https://github.com/sgl-project/sglang/pull/23624) | merged | [diffusion] fix: unify LTX-2.3 HQ codepath gates for all LTX-2.3 variants | multimodal/processor, tests/benchmarks | `python/sglang/multimodal_gen/test/server/perf_baselines.json`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` |

### File-level PR diff reading notes

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


### Gap and optimization follow-up

- Covered PRs: 2; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
