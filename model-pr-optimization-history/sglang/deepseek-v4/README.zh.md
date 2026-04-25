# sglang DeepSeek V4 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23628](https://github.com/sgl-project/sglang/pull/23628), [#23684](https://github.com/sgl-project/sglang/pull/23684), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697) |
| `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23617](https://github.com/sgl-project/sglang/pull/23617), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23634](https://github.com/sgl-project/sglang/pull/23634), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23690](https://github.com/sgl-project/sglang/pull/23690), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23698](https://github.com/sgl-project/sglang/pull/23698) |

## PR 覆盖总览

- git 追溯 PR 数: 11
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 11
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23617](https://github.com/sgl-project/sglang/pull/23617) | merged | Further update Deepseek V4 docs | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-24 | [#23628](https://github.com/sgl-project/sglang/pull/23628) | merged | [codex] docs: note H200 DeepSeek-V4 checkpoint | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23622](https://github.com/sgl-project/sglang/pull/23622) | merged | Again update DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23634](https://github.com/sgl-project/sglang/pull/23634) | merged | Update pro fp8 checkpoint in DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23684](https://github.com/sgl-project/sglang/pull/23684) | merged | docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23689](https://github.com/sgl-project/sglang/pull/23689) | merged | docs(DeepSeek-V4): mark b200\|small\|pd-disagg + h200\|small\|{cp,pd-disagg} verified | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23691](https://github.com/sgl-project/sglang/pull/23691) | merged | docs(DeepSeek-V4): mark gb300\|{small,big}\|{cp,pd-disagg} verified + GB300-specific fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23690](https://github.com/sgl-project/sglang/pull/23690) | merged | Small udpate gb300 recipe for deepseek v4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23697](https://github.com/sgl-project/sglang/pull/23697) | merged | update: b300 container for dsv4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23698](https://github.com/sgl-project/sglang/pull/23698) | merged | docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |

## 逐 PR diff 审计卡

### PR #23605 - Add DeepSeek V4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23605
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `492883c8ca66`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1024/-1，可读 patch 1041 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 cookbook」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -0,0 +1,569 @@
+export const DeepSeekV4Deployment = () => {
+  // DeepSeek-V4 deployment matrix (small / real checkpoint):
+  //   Hardware × Recipe → concrete launch command.
+  //
+  //   Hardware (quantization determined by GPU generation):
+  //     B200  → FP4 weights, Flash TP=4 / Pro TP=8 single-node
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -0,0 +1,453 @@
+---
+title: DeepSeek-V4
+metatags:
+    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek. Blackwell deployments use the FP4 checkpoint; Hopper deployments use the FP8 checkpoi
+tag: NEW
+---
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23617 - Further update Deepseek V4 docs

- 链接: https://github.com/sgl-project/sglang/pull/23617
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `734e1e2965cb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-6，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Further update Deepseek V4 docs」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {
-    // H200 needs a separate FP8-only Instruct ckpt (Flash / Pro public repos
-    // ship FP4-mixed weights). That ckpt is still being uploaded, so we emit a
-    // placeholder that fails loudly on copy-paste instead of silently pulling
-    // the wrong weights. Replace with the real slug once Hopper ckpts are public.
-    "h200|small":  { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Flash-hopper>", tp: 4,  multinode: false },
-    "h200|big":    { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-hopper>",   tp: 16, multinode: true, nnodes: 2 },
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23628 - [codex] docs: note H200 DeepSeek-V4 checkpoint

- 链接: https://github.com/sgl-project/sglang/pull/23628
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；关联提交 `1a37e57fb1ae`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[codex] docs: note H200 DeepSeek-V4 checkpoint」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；PR 正文摘要: - Add a highlighted note in the DeepSeek-V4 deployment section for H200 GPU users. - Clarify that H200 deployments should use the SGLang checkpoint under `sgl-project` instead o...。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../....。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../....
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+<Note>
+For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
+</Note>
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23622 - Again update DeepSeek V4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23622
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `3a620cb761ff`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+32/-9，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Again update DeepSeek V4 cookbook」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../....。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../....
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {
-        { id: "low-latency",    label: "Low-Latency",      default: true,  subtitle: "MTP 3/4" },
-        { id: "balanced",       label: "Balanced",         default: false, subtitle: "MTP 1/2 + DeepEP" },
-        { id: "max-throughput", label: "Max-Throughput",   default: false, subtitle: "DP + DeepEP" },
-        { id: "cp",             label: "Context-Parallel", default: false, subtitle: "long prompts" },
-        { id: "pd-disagg",      label: "PD-Disagg",        default: false, subtitle: "1P + 1D + router" },
+        { id: "low-latency",    label: "Low-Latency",      default: true  },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+For how to actually launch one of these images, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). A minimal example (substitute the
+'''bash Command
+docker run --gpus all \
+    --shm-size 32g \
+    -p 30000:30000 \
+    -v ~/.cache/huggingface:/root/.cache/huggingface \
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23634 - Update pro fp8 checkpoint in DeepSeek V4 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23634
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `92bb5c6bbee9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update pro fp8 checkpoint in DeepSeek V4 cookbook」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
-    // repackagings; Flash is public, Pro is still being uploaded.
+    // repackagings for both variants.
-    "h200|big":    { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-FP8>",     tp: 16, multinode: true, nnodes: 2 },
+    "h200|big":    { slug: "sgl-project/DeepSeek-V4-Pro-FP8",          tp: 16, multinode: true, nnodes: 2 },
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23684 - docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models

- 链接: https://github.com/sgl-project/sglang/pull/23684
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；关联提交 `fd401c2fb451`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -147,6 +147,10 @@ The generator currently picks values on the **conservative** side (mirroring an
+**Base model usage**
+In order to use base models, please enable `SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1` and use latest code, before the next round of testing matrix is finished.
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23689 - docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified

- 链接: https://github.com/sgl-project/sglang/pull/23689
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `d2c61acf2597`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+22/-1，可读 patch 59 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {
+    "b200|small|pd-disagg",
+    "h200|small|cp",
+    "h200|small|pd-disagg",
+    // h200|big|pd-disagg: pending verification (needs 4-node H200 cluster with
+    //   shared IB fabric: 2-node prefill + 2-node decode).
+  // Recipes whose command is intentionally not yet provided (e.g. blocked by an
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -145,7 +145,14 @@ The generator currently picks values on the **conservative** side (mirroring an
-The H200 image and checkpoint are currently being uploaded — public path coming shortly.
+H200 image (`lmsysorg/sglang:deepseek-v4-hopper`) and FP8 checkpoints
+(`sgl-project/DeepSeek-V4-Flash-FP8`, `sgl-project/DeepSeek-V4-Pro-FP8`) are
+publicly available.
+PD-Disagg recipes on H200 may require `docker run --privileged --ulimit memlock=-1`
+(or `--device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK`) so mooncake
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23691 - docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes

- 链接: https://github.com/sgl-project/sglang/pull/23691
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `8a395994edcf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+56/-5，可读 patch 113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|small|cp",
+    "gb300|big|cp",
+    "gb300|small|pd-disagg",
+    "gb300|big|pd-disagg",
@@ -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {
-      flags.push("  --mem-fraction-static 0.78");
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpoints.
+**GB300 PD-Disagg cross-pod MNNVL**
+On some GB300 clusters with cross-pod KV transfer over NVLink, mooncake may
+fail with `nvlink_transport.cpp:497 Requested address ... not found!`. If
+this happens, prepend `MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`
+to both prefill and decode `sglang serve` commands.
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23690 - Small udpate gb300 recipe for deepseek v4

- 链接: https://github.com/sgl-project/sglang/pull/23690
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `69485a176c87`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Small udpate gb300 recipe for deepseek v4」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|small|low-latency",
+    "gb300|small|balanced",
+    "gb300|small|max-throughput",
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23697 - update: b300 container for dsv4

- 链接: https://github.com/sgl-project/sglang/pull/23697
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `0d224e505333`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-2，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update: b300 container for dsv4」；模型线: DeepSeek V4；类别: 模型实现调整；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../....。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../....
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "b300",  label: "B300 (FP4)",  default: false  },
@@ -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {
-    const { hardware, modelSize, recipe, reasoningParser, toolcall } = values;
+    const { hardware: rawHardware, modelSize, recipe, reasoningParser, toolcall } = values;
+    // B300 usage is identical to B200 — alias so we don't duplicate every spec entry.
+    const hardware = rawHardware === "b300" ? "b200" : rawHardware;
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA B300</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>lmsysorg/sglang:deepseek-v4-b300</code></td>
+    </tr>
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23698 - docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9

- 链接: https://github.com/sgl-project/sglang/pull/23698
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；关联提交 `880599cd430f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-3，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9」；模型线: DeepSeek V4；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {
-        // OOM during CG capture. Verified working on 2026-04-25 (journal
-        // 2026-04-25-001 Cell D, Δ10).
+        // OOM during CG capture. mem-frac sweep at 0.83 / 0.87 / 0.89 / 0.91
+        // all pass static smoke; 0.9 picked as the default — leaves
+        // ~14 GB / GPU post-CG headroom for mooncake transfer + activation
+        // peaks while giving ~1M-token KV pool.
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
