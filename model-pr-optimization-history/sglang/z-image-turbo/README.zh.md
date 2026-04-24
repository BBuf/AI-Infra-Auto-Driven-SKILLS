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
