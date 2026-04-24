# SGLang Hunyuan 3 Preview 支持与优化时间线

范围：Tencent Hunyuan 3 Preview BF16 cookbook、硬件 TP 配置、reasoning/tool parser、MTP/EAGLE flags、Blackwell attention backend、`--trust-remote-code` 启动建议。

证据快照：SGLang `origin/main` `bca3dd958`（`2026-04-24`）。完整卡片见：`skills/model-optimization/sglang/sglang-hunyuan3-preview-optimization/references/pr-history.md`。

## 已阅读 Diff 的 PR

#23532 新增 Hunyuan 3 Preview cookbook 和命令生成器。已阅读完整 diff：`1309` 行、`3` 个文件。命令生成器将 H200/B200 映射到 TP=8，B300/GB300 映射到 TP=4，并加入 `hunyuan` parser toggle、MTP/EAGLE flags、`--trust-remote-code`、Blackwell `--attention-backend trtllm_mha`。

当前注意：这是 docs/recipe 支持；该 PR 没有新增 Hunyuan3 专用的 SGLang model 实现文件。
