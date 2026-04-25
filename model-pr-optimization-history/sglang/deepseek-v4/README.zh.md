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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V4`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx` |
| 2026-04-24 | [#23617](https://github.com/sgl-project/sglang/pull/23617) | merged | Further update Deepseek V4 docs | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-24 | [#23622](https://github.com/sgl-project/sglang/pull/23622) | merged | Again update DeepSeek V4 cookbook | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23628](https://github.com/sgl-project/sglang/pull/23628) | merged | [codex] docs: note H200 DeepSeek-V4 checkpoint | docs/config | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23634](https://github.com/sgl-project/sglang/pull/23634) | merged | Update pro fp8 checkpoint in DeepSeek V4 cookbook | docs/config | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |

### 逐 PR 代码 diff 阅读记录

### PR #23605 - Add DeepSeek V4 cookbook

- 链接：https://github.com/sgl-project/sglang/pull/23605
- 状态/时间：`merged`，created 2026-04-24, merged 2026-04-24；作者 `wisclmy0611`。
- 代码 diff 已读范围：`4` 个文件，`+1024/-1`；代码面：docs/config；关键词：doc, attention, config, cuda, deepep, eagle, expert, flash, fp4, fp8。
- 代码 diff 细节：
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunk: +export const DeepSeekV4Deployment = () => {; 符号: uses
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunk: +---
  - `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunk: metatags:
  - `docs_new/docs.json` modified +1/-0 (1 lines); hunk: {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`；patch 关键词为 doc, attention, config, cuda, deepep, eagle。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23617 - Further update Deepseek V4 docs

- 链接：https://github.com/sgl-project/sglang/pull/23617
- 状态/时间：`merged`，created 2026-04-24, merged 2026-04-24；作者 `fzyzcjy`。
- 代码 diff 已读范围：`1` 个文件，`+5/-6`；代码面：docs/config；关键词：doc, flash, fp4, fp8, kv, spec。
- 代码 diff 细节：
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunk: export const DeepSeekV4Deployment = () => {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；patch 关键词为 doc, flash, fp4, fp8, kv, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23622 - Again update DeepSeek V4 cookbook

- 链接：https://github.com/sgl-project/sglang/pull/23622
- 状态/时间：`merged`，created 2026-04-24, merged 2026-04-24；作者 `fzyzcjy`。
- 代码 diff 已读范围：`2` 个文件，`+32/-9`；代码面：docs/config；关键词：doc, cache, cuda, deepep, kv, router, spec, test。
- 代码 diff 细节：
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunk: export const DeepSeekV4Deployment = () => {; export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunk: Please refer to the [official SGLang installation guide](../../../docs/get-start
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；patch 关键词为 doc, cache, cuda, deepep, kv, router。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23628 - [codex] docs: note H200 DeepSeek-V4 checkpoint

- 链接：https://github.com/sgl-project/sglang/pull/23628
- 状态/时间：`merged`，created 2026-04-24, merged 2026-04-24；作者 `zijiexia`。
- 代码 diff 已读范围：`1` 个文件，`+4/-0`；代码面：docs/config；关键词：config, doc, spec。
- 代码 diff 细节：
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunk: Please refer to the [official SGLang installation guide](../../../docs/get-start
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；patch 关键词为 config, doc, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23634 - Update pro fp8 checkpoint in DeepSeek V4 cookbook

- 链接：https://github.com/sgl-project/sglang/pull/23634
- 状态/时间：`merged`，created 2026-04-24, merged 2026-04-24；作者 `fzyzcjy`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：docs/config；关键词：doc, flash, fp4, fp8, kv, spec。
- 代码 diff 细节：
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunk: export const DeepSeekV4Deployment = () => {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；patch 关键词为 doc, flash, fp4, fp8, kv, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
