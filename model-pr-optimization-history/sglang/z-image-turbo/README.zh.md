# SGLang Z-Image/Z-Image-Turbo 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `bca3dd958`（2026-04-24）整理。

范围：Z-Image、Z-Image-Turbo、Turbo/base 采样默认值、CFG normalization、原生 SP latent sharding、AMD nightly 验证和 direct-file CI 执行。

## 结论

Z-Image-Turbo 是 SGLang Diffusion 的原生路径，核心由 `ZImagePipelineConfig`、`ZImageTransformer2DModel` 和 registry 条目承载。关键契约是 Turbo 与 base Z-Image 必须走不同采样类；AMD 支持面由已恢复的 `nightly-amd-1-gpu-zimage-turbo` 测试兜住。

## 已阅读 diff 的 PR 卡片

### #17822 - 同步上游 Z-Image

- 链接：https://github.com/sgl-project/sglang/pull/17822
- 状态：已合入，`2026-01-29T13:10:11Z`
- Diff 覆盖：`209` 行，`5` 个文件。
- 新增独立的 `ZImageTurboSamplingParams` 和 `ZImageSamplingParams`，拆分 registry detector，补 `prepare_neg_cond_kwargs`，并在 denoising 中加入 `cfg_normalization`。
- 验证影响：确认 Turbo 不会误用 base Z-Image 的 CFG 默认值。

### #19733 - 增加 AMD Z-Image-Turbo nightly 测试

- 链接：https://github.com/sgl-project/sglang/pull/19733
- 状态：已合入，`2026-03-05T16:36:18Z`
- Diff 覆盖：`308` 行，`4` 个文件。
- 为 `Tongyi-MAI/Z-Image-Turbo` 增加 AMD nightly generation 覆盖，包含生成图像 artifact 和 CLIP score 校验。
- 验证影响：AMD 回归不能只看 server boot，还要看图像字节和 CLIP 分数。

### #23455 - 恢复 Z-Image-Turbo 可直接执行测试

- 链接：https://github.com/sgl-project/sglang/pull/23455
- 状态：已合入，`2026-04-23T05:28:21Z`
- Diff 覆盖：`291` 行，`2` 个文件。
- 在 CI hygiene 删除后恢复 `test/registered/amd/test_zimage_turbo.py`，并补上 direct `pytest.main` 入口。
- 验证影响：direct-file execution 模式必须继续可运行。

完整 PR dossier：`skills/model-optimization/sglang/sglang-z-image-turbo-optimization/references/pr-history.md`。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Z-Image-Turbo`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-01-27 | [#17822](https://github.com/sgl-project/sglang/pull/17822) | merged | [wip] sync with upstream zImage | multimodal/processor, docs/config | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/registry.py`, `python/sglang/multimodal_gen/configs/sample/sampling_params.py` |
| 2026-03-03 | [#19733](https://github.com/sgl-project/sglang/pull/19733) | merged | [AMD] [Z-Image-Turbo Day 0] Add Z-Image-Turbo nightly test for AMD GPUs | multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml` |
| 2026-04-22 | [#23455](https://github.com/sgl-project/sglang/pull/23455) | merged | [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry | MoE/router, quantization, multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py` |

### 逐 PR 代码 diff 阅读记录

### PR #17822 - [wip] sync with upstream zImage

- 链接：https://github.com/sgl-project/sglang/pull/17822
- 状态/时间：`merged`，created 2026-01-27, merged 2026-01-29；作者 `yhyang201`。
- 代码 diff 已读范围：`5` 个文件，`+79/-4`；代码面：multimodal/processor, docs/config；关键词：config, cache。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` modified +26/-0 (26 lines); hunk: def _predict_noise_with_cfg(; def _predict_noise_with_cfg(; 符号: _predict_noise_with_cfg, _predict_noise_with_cfg
  - `python/sglang/multimodal_gen/registry.py` modified +16/-3 (19 lines); hunk: WanT2V_1_3B_SamplingParams,; def _register_configs():; 符号: _register_configs
  - `python/sglang/multimodal_gen/configs/sample/sampling_params.py` modified +13/-0 (13 lines); hunk: class SamplingParams:; def _finite_non_negative_float(; 符号: SamplingParams:, _finite_non_negative_float, add_cli_args
  - `python/sglang/multimodal_gen/configs/sample/zimage.py` modified +12/-1 (13 lines); hunk: @dataclass; class ZImageSamplingParams(SamplingParams):; 符号: ZImageSamplingParams, ZImageTurboSamplingParams, ZImageSamplingParams, ZImageSamplingParams
  - `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py` modified +12/-0 (12 lines); hunk: def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):; 符号: prepare_pos_cond_kwargs, prepare_neg_cond_kwargs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/registry.py`, `python/sglang/multimodal_gen/configs/sample/sampling_params.py`；patch 关键词为 config, cache。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/registry.py`, `python/sglang/multimodal_gen/configs/sample/sampling_params.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19733 - [AMD] [Z-Image-Turbo Day 0] Add Z-Image-Turbo nightly test for AMD GPUs

- 链接：https://github.com/sgl-project/sglang/pull/19733
- 状态/时间：`merged`，created 2026-03-03, merged 2026-03-05；作者 `michaelzhang-ai`。
- 代码 diff 已读范围：`4` 个文件，`+235/-0`；代码面：multimodal/processor, tests/benchmarks；关键词：test, config, doc, processor。
- 代码 diff 细节：
  - `test/registered/amd/test_zimage_turbo.py` added +150/-0 (150 lines); hunk: +"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""; 符号: _save_image_and_write_summary, _compute_clip_score, TestZImageTurboAMD, teardown_class
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +42/-0 (42 lines); hunk: jobs:; jobs:
  - `.github/workflows/nightly-test-amd.yml` modified +42/-0 (42 lines); hunk: jobs:; jobs:
  - `test/run_suite.py` modified +1/-0 (1 lines); hunk: "nightly-amd",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/test_zimage_turbo.py`, `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`；patch 关键词为 test, config, doc, processor。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/test_zimage_turbo.py`, `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23455 - [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry

- 链接：https://github.com/sgl-project/sglang/pull/23455
- 状态/时间：`merged`，created 2026-04-22, merged 2026-04-23；作者 `bingxche`。
- 代码 diff 已读范围：`2` 个文件，`+220/-0`；代码面：MoE/router, quantization, multimodal/processor, tests/benchmarks；关键词：test, attention, config, fp8, moe, processor, quant, triton。
- 代码 diff 细节：
  - `test/registered/amd/test_zimage_turbo.py` added +156/-0 (156 lines); hunk: +"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""; 符号: _save_image_and_write_summary, _compute_clip_score, TestZImageTurboAMD, teardown_class
  - `test/registered/quant/test_int4fp8_moe.py` added +64/-0 (64 lines); hunk: +from types import SimpleNamespace; 符号: TestMixtralAccuracy, setUpClass, tearDownClass, test_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py`；patch 关键词为 test, attention, config, fp8, moe, processor。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
