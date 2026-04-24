# SGLang DeepSeek-V4 支持与优化时间线

范围：DeepSeek-V4 Flash/Pro cookbook、H200 FP8 checkpoint 选择、Blackwell FP4 recipe、DeepEP dispatch-token budget、MTP/EAGLE、CP、PD-disaggregation 命令生成。

证据快照：SGLang `origin/main` `bca3dd958`（`2026-04-24`）。完整卡片见：`skills/model-optimization/sglang/sglang-deepseek-v4-optimization/references/pr-history.md`。

## 已阅读 Diff 的 PR

- #23605 新增 DeepSeek-V4 cookbook 和命令生成器。已阅读 `2113` 行 diff、`4` 个文件，核心是 Flash/Pro、B200/GB300/H200 矩阵和未验证 recipe 注释行为。
- #23617 把 H200 Flash 从 placeholder 改成 `sgl-project/DeepSeek-V4-Flash-FP8`。已阅读 `33` 行 diff。
- #23628 在 cookbook 加 H200 必须使用 `sgl-project` checkpoint 的提示。已阅读 `24` 行 diff。
- #23622 增加 Docker 启动骨架、扩展 verified recipe，并清理 CP `--max-running-requests`。已阅读 `292` 行 diff。
- #23634 把 H200 Pro 改成 `sgl-project/DeepSeek-V4-Pro-FP8`。已阅读 `26` 行 diff。

## 当前契约

H200 命令必须使用 SGLang FP8 checkpoint；Blackwell 命令使用 DeepSeek Flash/Pro repo。未验证的命令生成器 cell 会被整体注释，避免用户误复制。parser 分别是 `--reasoning-parser deepseek-v4` 和 `--tool-call-parser deepseekv4`。
