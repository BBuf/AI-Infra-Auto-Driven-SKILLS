# SGLang LTX-2.3 HQ 支持与优化时间线

范围：LTX-2.3 High Quality 两阶段 video+audio pipeline、分阶段 LoRA、HQ sigma/timestep 语义、res2s refinement、采样默认值和回归 gates。

证据快照：SGLang `origin/main` `bca3dd958`（`2026-04-24`）。完整卡片见：`skills/model-optimization/sglang/sglang-ltx23-hq-optimization/references/pr-history.md`。

## 已阅读 Diff 的 PR

- #23366 新增 `LTX2TwoStageHQPipeline`、`LTX23HQSamplingParams`、HQ resolution-aware sigma shift、分阶段 distilled LoRA strength 和 HQ denoising 语义。已阅读完整 diff：`5411` 行、`19` 个文件。
- #23624 明确 gate，确保 HQ sigma 语义只用于 `LTX2TwoStageHQPipeline`，不会误套到所有 native LTX-2.3 路径。已阅读完整 diff：`505` 行、`3` 个文件。

## 当前契约

HQ stage 1 使用半分辨率 latent token 数生成 sigmas。非 HQ one-stage / two-stage LTX-2.3 仍使用 constant-anchor sigmas。默认 HQ 请求是 1088x1920、15 steps、stage-1 LoRA strength 0.25、stage-2 LoRA strength 0.5。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `LTX-Video 2.3 HQ`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-21 | [#23366](https://github.com/sgl-project/sglang/pull/23366) | merged | [diffusion] model: support LTX2.3 high quality pipeline | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` |
| 2026-04-24 | [#23624](https://github.com/sgl-project/sglang/pull/23624) | merged | [diffusion] fix: unify LTX-2.3 HQ codepath gates for all LTX-2.3 variants | multimodal/processor, tests/benchmarks | `python/sglang/multimodal_gen/test/server/perf_baselines.json`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` |

### 逐 PR 代码 diff 阅读记录

### PR #23366 - [diffusion] model: support LTX2.3 high quality pipeline

- 链接：https://github.com/sgl-project/sglang/pull/23366
- 状态/时间：`merged`，created 2026-04-21, merged 2026-04-24；作者 `mickqian`。
- 代码 diff 已读范围：`19` 个文件，`+1501/-419`；代码面：model wrapper, multimodal/processor, tests/benchmarks, docs/config；关键词：config, lora, scheduler, cache, spec, attention, cuda, test。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py` modified +974/-313 (1287 lines); hunk: import copy; is_ltx2_two_stage_pipeline_name,; 符号: LTX2DenoisingContext, LTX2DenoisingContext, LTX2DenoisingStage, __init__
  - `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` modified +129/-26 (155 lines); hunk: from diffusers import FlowMatchEulerDiscreteScheduler; def build_official_ltx2_sigmas(; 符号: build_official_ltx2_sigmas, build_official_ltx2_sigmas, LTX2SigmaPreparationStage, forward
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` modified +129/-23 (152 lines); hunk: def __init__(; def _should_reset_stage2_generators(server_args: ServerArgs) -> bool:; 符号: __init__, _should_reset_stage2_generators, _build_stage2_renoise_generator, _ltx2_renoise_like
  - `python/sglang/multimodal_gen/runtime/layers/lora/linear.py` modified +45/-10 (55 lines); hunk: torch._dynamo.config.recompile_limit = 64; def _merge_lora_into_data(; 符号: BaseLayerWithLoRA, __init__, _merge_lora_into_data, merge_lora_weights
  - `python/sglang/multimodal_gen/test/server/perf_baselines.json` modified +41/-0 (41 lines); hunk: "expected_avg_denoise_ms": 890.17,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`；patch 关键词为 config, lora, scheduler, cache, spec, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23624 - [diffusion] fix: unify LTX-2.3 HQ codepath gates for all LTX-2.3 variants

- 链接：https://github.com/sgl-project/sglang/pull/23624
- 状态/时间：`merged`，created 2026-04-24, merged 2026-04-24；作者 `mickqian`。
- 代码 diff 已读范围：`3` 个文件，`+65/-61`；代码面：multimodal/processor, tests/benchmarks；关键词：config, scheduler, lora, test。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/test/server/perf_baselines.json` modified +36/-35 (71 lines); hunk: },
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py` modified +16/-22 (38 lines); hunk: def forward(self, batch: Req, server_args: ServerArgs) -> Req:; def forward(self, batch: Req, server_args: ServerArgs) -> Req:; 符号: forward, forward, forward, forward
  - `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` modified +13/-4 (17 lines); hunk: class LTX2SigmaPreparationStage(PipelineStage):; 符号: LTX2SigmaPreparationStage, forward, to
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/test/server/perf_baselines.json`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`；patch 关键词为 config, scheduler, lora, test。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/test/server/perf_baselines.json`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py`, `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：2；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
