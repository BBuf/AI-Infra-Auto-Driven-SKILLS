# SGLang Hunyuan 3 Preview 支持与优化时间线

范围：Tencent Hunyuan 3 Preview BF16 cookbook、硬件 TP 配置、reasoning/tool parser、MTP/EAGLE flags、Blackwell attention backend、`--trust-remote-code` 启动建议。

证据快照：SGLang `origin/main` `bca3dd958`（`2026-04-24`）。完整卡片见：`skills/model-optimization/sglang/sglang-hunyuan3-preview-optimization/references/pr-history.md`。

## 已阅读 Diff 的 PR

#23532 新增 Hunyuan 3 Preview cookbook 和命令生成器。已阅读完整 diff：`1309` 行、`3` 个文件。命令生成器将 H200/B200 映射到 TP=8，B300/GB300 映射到 TP=4，并加入 `hunyuan` parser toggle、MTP/EAGLE flags、`--trust-remote-code`、Blackwell `--attention-backend trtllm_mha`。

当前注意：这是 docs/recipe 支持；该 PR 没有新增 Hunyuan3 专用的 SGLang model 实现文件。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Hunyuan3 Preview`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-23 | [#23532](https://github.com/sgl-project/sglang/pull/23532) | merged | docs: add Hunyuan 3 Preview cookbook | docs/config | `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`, `docs_new/docs.json` |

### 逐 PR 代码 diff 阅读记录

### PR #23532 - docs: add Hunyuan 3 Preview cookbook

- 链接：https://github.com/sgl-project/sglang/pull/23532
- 状态/时间：`merged`，created 2026-04-23, merged 2026-04-23；作者 `JustinTong0323`。
- 代码 diff 已读范围：`3` 个文件，`+707/-0`；代码面：docs/config；关键词：doc, attention, config, eagle, moe, spec, topk, benchmark, expert, flash。
- 代码 diff 细节：
  - `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0 (527 lines); hunk: +---; 符号: GPUs
  - `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0 (174 lines); hunk: +export const Hunyuan3PreviewDeployment = () => {; 符号: GPUs
  - `docs_new/docs.json` modified +6/-0 (6 lines); hunk: "pages": [
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`, `docs_new/docs.json`；patch 关键词为 doc, attention, config, eagle, moe, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`, `docs_new/docs.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：1；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
