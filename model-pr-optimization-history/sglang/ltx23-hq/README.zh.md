# SGLang LTX-2.3 HQ 支持与优化时间线

范围：LTX-2.3 High Quality 两阶段 video+audio pipeline、分阶段 LoRA、HQ sigma/timestep 语义、res2s refinement、采样默认值和回归 gates。

证据快照：SGLang `origin/main` `bca3dd958`（`2026-04-24`）。完整卡片见：`skills/model-optimization/sglang/sglang-ltx23-hq-optimization/references/pr-history.md`。

## 已阅读 Diff 的 PR

- #23366 新增 `LTX2TwoStageHQPipeline`、`LTX23HQSamplingParams`、HQ resolution-aware sigma shift、分阶段 distilled LoRA strength 和 HQ denoising 语义。已阅读完整 diff：`5411` 行、`19` 个文件。
- #23624 明确 gate，确保 HQ sigma 语义只用于 `LTX2TwoStageHQPipeline`，不会误套到所有 native LTX-2.3 路径。已阅读完整 diff：`505` 行、`3` 个文件。

## 当前契约

HQ stage 1 使用半分辨率 latent token 数生成 sigmas。非 HQ one-stage / two-stage LTX-2.3 仍使用 constant-anchor sigmas。默认 HQ 请求是 1088x1920、15 steps、stage-1 LoRA strength 0.25、stage-2 LoRA strength 0.5。
