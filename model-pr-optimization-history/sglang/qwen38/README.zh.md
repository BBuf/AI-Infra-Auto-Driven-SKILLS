# sglang Qwen3.8 模型 PR 优化历史

2026-08-23 的 SGLang head 没有独立的 `python/sglang/srt/models/qwen3_8*.py`。
公开的 `Qwen/Qwen3.8-27B` 使用 `model_type=qwen3_5` 的混合 GDN/GQA，loader
和 kernel 历史应查 `qwen35`。本 slug 只跟踪 Qwen3.8 cookbook / 部署面。
2.4T 的 `Qwen3.8-2.4T-A95B` 是另一条多机 MoE recipe，不要抄进单卡 27B
serving 方案。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` | [#34860](https://github.com/sgl-project/sglang/pull/34860), [#34863](https://github.com/sgl-project/sglang/pull/34863), [#35064](https://github.com/sgl-project/sglang/pull/35064), [#35065](https://github.com/sgl-project/sglang/pull/35065), [#35121](https://github.com/sgl-project/sglang/pull/35121), [#35663](https://github.com/sgl-project/sglang/pull/35663), [#35753](https://github.com/sgl-project/sglang/pull/35753), [#35767](https://github.com/sgl-project/sglang/pull/35767), [#35786](https://github.com/sgl-project/sglang/pull/35786), [#35825](https://github.com/sgl-project/sglang/pull/35825) |
| `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` | [#34587](https://github.com/sgl-project/sglang/pull/34587), [#34590](https://github.com/sgl-project/sglang/pull/34590), [#34601](https://github.com/sgl-project/sglang/pull/34601), [#34860](https://github.com/sgl-project/sglang/pull/34860) |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx` | [#34836](https://github.com/sgl-project/sglang/pull/34836) |
| `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` | [#34860](https://github.com/sgl-project/sglang/pull/34860), [#35064](https://github.com/sgl-project/sglang/pull/35064) |
| `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` | [#34863](https://github.com/sgl-project/sglang/pull/34863), [#35065](https://github.com/sgl-project/sglang/pull/35065) |
| `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` | [#34860](https://github.com/sgl-project/sglang/pull/34860), [#34863](https://github.com/sgl-project/sglang/pull/34863), [#35065](https://github.com/sgl-project/sglang/pull/35065), [#35121](https://github.com/sgl-project/sglang/pull/35121), [#35663](https://github.com/sgl-project/sglang/pull/35663), [#35753](https://github.com/sgl-project/sglang/pull/35753), [#35786](https://github.com/sgl-project/sglang/pull/35786), [#35825](https://github.com/sgl-project/sglang/pull/35825) |
| `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx` | [#34587](https://github.com/sgl-project/sglang/pull/34587) |
| `docs/src/snippets/configs/Qwen/qwen3.8.jsx` | [#34587](https://github.com/sgl-project/sglang/pull/34587), [#34590](https://github.com/sgl-project/sglang/pull/34590), [#34601](https://github.com/sgl-project/sglang/pull/34601) |

## PR 覆盖总览

- git 追溯 PR 数: 14
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 14
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-12 | [#34587](https://github.com/sgl-project/sglang/pull/34587) | merged | [Docs] Add Qwen3.8 cookbook | `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` |
| 2026-08-12 | [#34590](https://github.com/sgl-project/sglang/pull/34590) | merged | [Docs] Rename Qwen3.8-Max-DSpark to Qwen3.8-2.4T-A95B-DSpark | `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` |
| 2026-08-12 | [#34601](https://github.com/sgl-project/sglang/pull/34601) | merged | docs: update Qwen3.8 disaggregated serving configs | `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` |
| 2026-08-14 | [#34836](https://github.com/sgl-project/sglang/pull/34836) | merged | [NPU] [DOC] Add Qwen3.8-Max deployment tutorial on Ascend NPUs | `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx` |
| 2026-08-14 | [#34860](https://github.com/sgl-project/sglang/pull/34860) | merged | [Docs] Add Qwen3.8-27B cookbook page | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` |
| 2026-08-14 | [#34863](https://github.com/sgl-project/sglang/pull/34863) | merged | [Docs] Add GB300 cells and benchmarks for Qwen3.8-27B | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-17 | [#35064](https://github.com/sgl-project/sglang/pull/35064) | merged | docs: fix Qwen3.8-27B mamba ratio calculator for speculative decoding | `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-17 | [#35065](https://github.com/sgl-project/sglang/pull/35065) | merged | docs(cookbook): Qwen3.8-27B deployment grid rework | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-17 | [#35121](https://github.com/sgl-project/sglang/pull/35121) | merged | docs(cookbook): add Qwen3.8-27B DGX Spark configs | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-20 | [#35663](https://github.com/sgl-project/sglang/pull/35663) | merged | [docs] Add DFlash2 speculative cells to the Qwen3.8-27B cookbook | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-20 | [#35753](https://github.com/sgl-project/sglang/pull/35753) | merged | [docs] Tell Qwen3.8-27B DFLASH2 users to build from main | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-20 | [#35767](https://github.com/sgl-project/sglang/pull/35767) | merged | [docs] Point the Qwen3.8-27B DFLASH2 note back at the rolling dev image tag | `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-21 | [#35786](https://github.com/sgl-project/sglang/pull/35786) | merged | [docs] Retune the Qwen3.8-27B RTX 5090 DFLASH2 cells against 1cf2b8c | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-22 | [#35825](https://github.com/sgl-project/sglang/pull/35825) | merged | [docs] Re-measure the Qwen3.8-27B RTX 5090, RTX PRO 6000 and DGX Spark grids on 1cf2b8c | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |

## 逐 PR diff 审计卡

### PR #34587 - [Docs] Add Qwen3.8 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/34587
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8.jsx`；关联提交 `8e7c07fae734`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1305/-6，可读 patch 1368 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add Qwen3.8 cookbook」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`；技术摘要: 覆盖「[Docs] Add Qwen3.8 cookbook」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8.jsx` added +939/-0 (939 lines); hunks: -0,0 +1,939；`docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx` added +23/-0 (23 lines); hunks: -0,0 +1,23；`docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` added +315/-0 (315 lines); hunks: -0,0 +1,315。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8.jsx` added +939/-0 (939 lines); hunks: -0,0 +1,939
  - `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx` added +23/-0 (23 lines); hunks: -0,0 +1,23
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` added +315/-0 (315 lines); hunks: -0,0 +1,315
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8.jsx
@@ -0,0 +1,939 @@
+// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
+// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
+//
+// Qwen3.8-2.4T-A95B: 92 layers as 23 repeats of (3 x Gated DeltaNet -> MoE, then
+// 1 x Gated Attention -> MoE), so 69 linear-attention layers to 23 full-attention
+// ones; MoE with 512 experts, 10 routed + 1 shared active; 2.4T total / 95B
diff -- docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx
@@ -0,0 +1,23 @@
+// One entry per cell `match` tuple (same 5 keys as config cells). Every entry is
+// a bare match with no numbers, so the card shows "pending".
+export const benchmarks = [
+  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-4" } },
+  { match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" } },
+  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "multi-2" } },
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx
@@ -0,0 +1,315 @@
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8.jsx` added +939/-0; `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx` added +23/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` added +315/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/cookbook/autoregressive/intro.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34590 - [Docs] Rename Qwen3.8-Max-DSpark to Qwen3.8-2.4T-A95B-DSpark

- 链接: https://github.com/sgl-project/sglang/pull/34590
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8.jsx`；关联提交 `d21eefc94ff8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-7，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Rename Qwen3.8-Max-DSpark to Qwen3.8-2.4T-A95B-DSpark」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`；技术摘要: 覆盖「[Docs] Rename Qwen3.8-Max-DSpark to Qwen3.8-2.4T-A95B-DSpark」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8.jsx` modified +5/-5 (10 lines); hunks: -215,7 +215,7 @@ export const config = {; -825,7 +825,7 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +2/-2 (4 lines); hunks: -95,7 +95,7 @@ import { Playground } from "/src/snippets/_playground.jsx";; -295,7 +295,7 @@ We trained a **DSpark** draft model for Qwen3.8 with SpecFor...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8.jsx` modified +5/-5 (10 lines); hunks: -215,7 +215,7 @@ export const config = {; -825,7 +825,7 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +2/-2 (4 lines); hunks: -95,7 +95,7 @@ import { Playground } from "/src/snippets/_playground.jsx";; -295,7 +295,7 @@ We trained a **DSpark** draft model for Qwen3.8 with SpecFor...
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8.jsx
@@ -215,7 +215,7 @@ export const config = {
-                  "--speculative-draft-model-path RadixArk/Qwen3.8-Max-DSpark"],
+                  "--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark"],
@@ -825,7 +825,7 @@ export const config = {
-        "--speculative-draft-model-path RadixArk/Qwen3.8-Max-DSpark",
+        "--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark",
@@ -858,7 +858,7 @@ export const config = {
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx
@@ -95,7 +95,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
-**Resources:** Each precision is its own repo — [BF16](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) · [FP8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) · [NVFP4, NVIDIA B
+**Resources:** Each precision is its own repo — [BF16](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) · [FP8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) · [NVFP4, NVIDIA B
@@ -295,7 +295,7 @@ We trained a **DSpark** draft model for Qwen3.8 with SpecForge. Turn it on with
+--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8.jsx` modified +5/-5; `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34601 - docs: update Qwen3.8 disaggregated serving configs

- 链接: https://github.com/sgl-project/sglang/pull/34601
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8.jsx`；关联提交 `e6250c7c70b0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-0，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: update Qwen3.8 disaggregated serving configs」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`；技术摘要: 覆盖「docs: update Qwen3.8 disaggregated serving configs」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8.jsx` modified +3/-0 (3 lines); hunks: -272,13 +272,16 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +12/-0 (12 lines); hunks: -179,6 +179,18 @@ Balanced and High Throughput share one shape and differ onl...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8.jsx` modified +3/-0 (3 lines); hunks: -272,13 +272,16 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +12/-0 (12 lines); hunks: -179,6 +179,18 @@ Balanced and High Throughput share one shape and differ onl...
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8.jsx
@@ -272,13 +272,16 @@ export const config = {
+      // In PD mode, --policy is the prefill fallback; keep decode explicit.
+  --policy round_robin \\
+  --decode-policy round_robin \\
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx
@@ -179,6 +179,18 @@ Balanced and High Throughput share one shape and differ only in capacity. Low La
+### GB300 PD disaggregation layouts
+The PD role selector adds the role and transfer flags to the selected base recipe. It does not resize the prefill and decode workers. Use separate workers with the layouts below f
+| Checkpoint and operating point | Prefill workers | Decode worker | Capacity setting |
+|---|---|---|---|
+| FP8, high throughput | 2 × TP1 / PP16 | DP4-attention / TP4 / EP16, DeepEP v2 + EPLB | Keep frontend concurrency above the decode MRR so prefill remains queued |
+| NVFP4, high throughput | 2 × TP1 / PP6 | DP2-attention / TP4 / EP8, FlashInfer one-sided A2A | Prefill MRR 128 per worker; decode MRR 512; frontend concurrency 1536 |
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8.jsx` modified +3/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +12/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34836 - [NPU] [DOC] Add Qwen3.8-Max deployment tutorial on Ascend NPUs

- 链接: https://github.com/sgl-project/sglang/pull/34836
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx`；关联提交 `fe0c18effd21`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+288/-1，可读 patch 297 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] [DOC] Add Qwen3.8-Max deployment tutorial on Ascend NPUs」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx`；技术摘要: 覆盖「[NPU] [DOC] Add Qwen3.8-Max deployment tutorial on Ascend NPUs」；主要实现面是 `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx` added +286/-0 (286 lines); hunks: -0,0 +1,286。
- 代码 diff 细节:
  - `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx` added +286/-0 (286 lines); hunks: -0,0 +1,286
- 关键代码摘录:

```diff
diff -- docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx
@@ -0,0 +1,286 @@
+---
+title: "Qwen3.8-Max"
+metatags:
+  description: "Deploy Qwen3.8-Max model with SGLang on Ascend NPUs, including multi-node deployment mode."
+---
+## Introduction
```

- 已读文件:
  - docs: `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx` added +286/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/docs.json`, `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34860 - [Docs] Add Qwen3.8-27B cookbook page

- 链接: https://github.com/sgl-project/sglang/pull/34860
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `29c6be15a4ef`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+1203/-1，可读 patch 1228 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add Qwen3.8-27B cookbook page」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`；技术摘要: 覆盖「[Docs] Add Qwen3.8-27B cookbook page」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` added +435/-0 (435 lines); hunks: -0,0 +1,435；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` added +391/-0 (391 lines); hunks: -0,0 +1,391; symbols: GPUs，涉及 `GPUs`；`docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` added +359/-0 (359 lines); hunks: -0,0 +1,359；`docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` added +435/-0 (435 lines); hunks: -0,0 +1,435
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` added +391/-0 (391 lines); hunks: -0,0 +1,391; symbols: GPUs
  - `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` added +359/-0 (359 lines); hunks: -0,0 +1,359
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -0,0 +1,435 @@
+// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
+// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
+//
+// Qwen3.8-27B: DENSE hybrid Gated Delta Networks VISION-LANGUAGE model — a 27B
+// causal LM plus a vision encoder, served through SGLang's Qwen3-VL path
+// (Qwen3_5ForConditionalGeneration extends Qwen3VLForConditionalGeneration and
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -0,0 +1,391 @@
+---
+title: Qwen3.8-27B
+description: "Deploy Qwen3.8-27B with SGLang — dense hybrid GDN vision-language model with BF16/FP8/NVFP4 W4A4 checkpoints and in-checkpoint MTP, single-GPU on H200, RTX PRO 6000,
+tag: NEW
+---
+## Deployment
diff -- docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx
@@ -0,0 +1,359 @@
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` added +435/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` added +391/-0; `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` added +359/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` modified +0/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx`, `docs/docs.json`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34863 - [Docs] Add GB300 cells and benchmarks for Qwen3.8-27B

- 链接: https://github.com/sgl-project/sglang/pull/34863
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `70e291b70f5a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+227/-6，可读 patch 295 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add GB300 cells and benchmarks for Qwen3.8-27B」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「[Docs] Add GB300 cells and benchmarks for Qwen3.8-27B」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +118/-3 (121 lines); hunks: -37,7 +37,7; -69,6 +69,7 @@ export const config = {；`docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` added +106/-0 (106 lines); hunks: -0,0 +1,106；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +3/-3 (6 lines); hunks: -161,9 +161,9 @@ checkpoint's calibration scales automatically.。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +118/-3 (121 lines); hunks: -37,7 +37,7; -69,6 +69,7 @@ export const config = {
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` added +106/-0 (106 lines); hunks: -0,0 +1,106
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +3/-3 (6 lines); hunks: -161,9 +161,9 @@ checkpoint's calibration scales automatically.
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -37,7 +37,7 @@
-  supportedHardware: ["h200", "rtx6000", "rtx5090", "dgx-spark"],
+  supportedHardware: ["h200", "rtx6000", "rtx5090", "dgx-spark", "gb300"],
@@ -69,6 +69,7 @@ export const config = {
+    { id: "high-throughput", label: "High-Throughput" },
@@ -130,6 +131,7 @@ export const config = {
+    gb300:   "lmsysorg/sglang:dev",
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx
@@ -0,0 +1,106 @@
+// Qwen3.8-27B per-cell benchmark numbers, keyed by the same `match` tuple as
+// qwen3.8-27b.jsx cells. See _deployment.jsx for the speed/accuracy schema.
+//
+// All six rows are ONE-BATCH measurements (sglang.bench_serving --flush-cache,
+// random dataset, ISL=1024 / OSL=1024, --random-range-ratio 1, request-rate inf,
+// max_concurrency 1/16/64, n=64/64/256 respectively) on a single GB300 GPU
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -161,9 +161,9 @@ checkpoint's calibration scales automatically.
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +118/-3; `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` added +106/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +3/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35064 - docs: fix Qwen3.8-27B mamba ratio calculator for speculative decoding

- 链接: https://github.com/sgl-project/sglang/pull/35064
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`；关联提交 `07a28ec5cf3a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+62/-21，可读 patch 137 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: fix Qwen3.8-27B mamba ratio calculator for speculative decoding」；模型线: Qwen3.8；类别: 缺陷修复；主要 diff: `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「docs: fix Qwen3.8-27B mamba ratio calculator for speculative decoding」；主要实现面是 `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +46/-15 (61 lines); hunks: -73,7 +73,7 @@ export const Qwen38MambaRatioCalculator = () => {; -115,21 +115,42 @@ export const Qwen38MambaRatioCalculator = () => {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +16/-6 (22 lines); hunks: -73,20 +73,30 @@ ratio = (S + D) x state_bytes / (L x kv_bytes_per_token)。
- 代码 diff 细节:
  - `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +46/-15 (61 lines); hunks: -73,7 +73,7 @@ export const Qwen38MambaRatioCalculator = () => {; -115,21 +115,42 @@ export const Qwen38MambaRatioCalculator = () => {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +16/-6 (22 lines); hunks: -73,20 +73,30 @@ ratio = (S + D) x state_bytes / (L x kv_bytes_per_token)
- 关键代码摘录:

```diff
diff -- docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx
@@ -73,7 +73,7 @@ export const Qwen38MambaRatioCalculator = () => {
-  const derive = (flags) => {
+  const derive = (flags, env) => {
@@ -115,21 +115,42 @@ export const Qwen38MambaRatioCalculator = () => {
-    // S mirrors kv_cache_configurator._calculate_mamba_ratio (single GPU,
-    // overlap scheduler on): extra_buffer=5, extra_buffer_lazy=4,
-    // no_buffer=3, radix cache disabled=1.
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -73,20 +73,30 @@ ratio = (S + D) x state_bytes / (L x kv_bytes_per_token)
-  `extra_buffer_lazy=4`, `no_buffer=3`, disabled radix cache `=1`.
+  `extra_buffer_lazy=4`, `no_buffer=3`, disabled radix cache `=1`. For the two
+  `extra_buffer` strategies, `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` frees one
+  slot, and `extra_buffer` frees one more with the overlap scheduler off; the
+  calculator reads both knobs.
-  `--speculative-num-draft-tokens` (4 at the recommended EAGLE 3/1/4), 0 otherwise.
```

- 已读文件:
  - docs: `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +46/-15; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +16/-6
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35065 - docs(cookbook): Qwen3.8-27B deployment grid rework

- 链接: https://github.com/sgl-project/sglang/pull/35065
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `e03c53fc13ab`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+263/-174，可读 patch 672 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): Qwen3.8-27B deployment grid rework」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「docs(cookbook): Qwen3.8-27B deployment grid rework」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +212/-171 (383 lines); hunks: -1,38 +1,16; -49,32 +27,141 @@ export const config = {；`docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` modified +12/-2 (14 lines); hunks: -1,5 +1,15；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +39/-1 (40 lines); hunks: -56,6 +56,14 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qw...; -188,7 +196,19 @@ checkpoint's calibration scales automatically.; symbols: GPUs，涉及 `GPUs`。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +212/-171 (383 lines); hunks: -1,38 +1,16; -49,32 +27,141 @@ export const config = {
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` modified +12/-2 (14 lines); hunks: -1,5 +1,15
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +39/-1 (40 lines); hunks: -56,6 +56,14 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qw...; -188,7 +196,19 @@ checkpoint's calibration scales automatically.; symbols: GPUs
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -1,38 +1,16 @@
-// Qwen3.8-27B: DENSE hybrid Gated Delta Networks VISION-LANGUAGE model — a 27B
-// causal LM plus a vision encoder, served through SGLang's Qwen3-VL path
-// (Qwen3_5ForConditionalGeneration extends Qwen3VLForConditionalGeneration and
-// is registered in the multimodal arch lists). 64 layers as 16 repeats of
-// 3 x (Gated DeltaNet -> FFN) then 1 x (Gated Attention -> FFN): 48
-// linear-attention layers to 16 full-attention. GDN runs 48 value heads and 16
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx
@@ -1,5 +1,15 @@
-// Qwen3.8-27B per-cell benchmark numbers, keyed by the same `match` tuple as
-// qwen3.8-27b.jsx cells. See _deployment.jsx for the speed/accuracy schema.
+// Qwen3.8-27B per-cell benchmark numbers. See _deployment.jsx for the
+// speed/accuracy schema.
+//
+// STRUCTURALLY UNMATCHED since the strategy->overlay migration: every entry
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -56,6 +56,14 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qwen38_mamba_ratio_ca
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +212/-171; `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` modified +12/-2; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +39/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35121 - docs(cookbook): add Qwen3.8-27B DGX Spark configs

- 链接: https://github.com/sgl-project/sglang/pull/35121
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `b956e916ae33`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+44/-21，可读 patch 134 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): add Qwen3.8-27B DGX Spark configs」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「docs(cookbook): add Qwen3.8-27B DGX Spark configs」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +28/-15 (43 lines); hunks: -338,8 +338,9 @@ export const config = {; -480,21 +481,30 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +16/-6 (22 lines); hunks: -59,7 +59,10 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qw...; -179,9 +182,17 @@ checkpoint's calibration scales automatically.。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +28/-15 (43 lines); hunks: -338,8 +338,9 @@ export const config = {; -480,21 +481,30 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +16/-6 (22 lines); hunks: -59,7 +59,10 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qw...; -179,9 +182,17 @@ checkpoint's calibration scales automatically.
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -338,8 +338,9 @@ export const config = {
-  // non-default overlay picks there are valid but unmeasured. DGX Spark stays
-  // unverified (SM121 / aarch64 unvalidated).
+  // non-default overlay picks there are valid but unmeasured. DGX Spark was
+  // measured across its whole overlay envelope too, but to a weaker standard
+  // (boot-and-serve only — see the cell block comment below).
@@ -480,21 +481,30 @@ export const config = {
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -59,7 +59,10 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qwen38_mamba_ratio_ca
-  ISL 8192 / OSL 1024, concurrency 1. The other platforms' recipes carry their
+  ISL 8192 / OSL 1024, concurrency 1. The DGX Spark cells cover that same full
+  combination set, but to a weaker standard: each was confirmed to **boot and
+  serve** at ISL 8192 / OSL 1024, concurrency 1, with no throughput or
+  acceptance-length numbers taken. The remaining platforms' recipes carry their
@@ -179,9 +182,17 @@ checkpoint's calibration scales automatically.
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +28/-15; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +16/-6
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35663 - [docs] Add DFlash2 speculative cells to the Qwen3.8-27B cookbook

- 链接: https://github.com/sgl-project/sglang/pull/35663
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `d9f6861359ce`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+79/-3，可读 patch 117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Add DFlash2 speculative cells to the Qwen3.8-27B cookbook」；模型线: Qwen3.8；类别: 性能/后端优化；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「[docs] Add DFlash2 speculative cells to the Qwen3.8-27B cookbook」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +48/-0 (48 lines); hunks: -116,6 +116,49 @@ export const config = {; -260,6 +303,11 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +31/-3 (34 lines); hunks: -59,8 +59,12 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qw...; -89,7 +93,8 @@ ratio = (S + D) x state_bytes / (L x kv_bytes_per_token); symbols: GPUs，涉及 `GPUs`。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +48/-0 (48 lines); hunks: -116,6 +116,49 @@ export const config = {; -260,6 +303,11 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +31/-3 (34 lines); hunks: -59,8 +59,12 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qw...; -89,7 +93,8 @@ ratio = (S + D) x state_bytes / (L x kv_bytes_per_token); symbols: GPUs
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -116,6 +116,49 @@ export const config = {
+        {
+          id: "dflash", label: "DFLASH2",
+          // Trained block-diffusion draft, a separate checkpoint. The
+          // selector projects through the target lm_head — including
+          // quantized heads — so it runs on the NVFP4 checkpoint too.
+          // Validated on the SM120 pair (NVFP4 measured end to end; the
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -59,8 +59,12 @@ import { Qwen38MambaRatioCalculator } from "/src/snippets/_qwen38_mamba_ratio_ca
-  ISL 8192 / OSL 1024, concurrency 1. The DGX Spark cells cover that same full
-  combination set, but to a weaker standard: each was confirmed to **boot and
+  ISL 8192 / OSL 1024, concurrency 1 — for DFLASH2, to that full standard on
+  NVFP4, and to boot-and-serve on the RTX PRO 6000 BF16/FP8 cells. On the
+  remaining platforms the DFLASH2 pick is offered but not yet exercised, and
+  the composed command carries a `# DFLASH2 on this platform: final
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +48/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +31/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35753 - [docs] Tell Qwen3.8-27B DFLASH2 users to build from main

- 链接: https://github.com/sgl-project/sglang/pull/35753
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `1a138e13b936`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+122/-25，可读 patch 300 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Tell Qwen3.8-27B DFLASH2 users to build from main」；模型线: Qwen3.8；类别: 性能/后端优化；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「[docs] Tell Qwen3.8-27B DFLASH2 users to build from main」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +35/-8 (43 lines); hunks: -122,17 +122,12 @@ export const config = {; -405,6 +400,10 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +45/-12 (57 lines); hunks: -20,6 +20,9 @@ For all methods and hardware platforms, see the [official SGLa...; -30,6 +33,10 @@ Then run the **Python** output of the command panel below in...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +35/-8 (43 lines); hunks: -122,17 +122,12 @@ export const config = {; -405,6 +400,10 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +45/-12 (57 lines); hunks: -20,6 +20,9 @@ For all methods and hardware platforms, see the [official SGLa...; -30,6 +33,10 @@ Then run the **Python** output of the command panel below in...
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -122,17 +122,12 @@ export const config = {
-          // RTX PRO 6000 BF16/FP8 cells boot-and-serve). On the other
-          // platforms the pick is offered with the in-progress hint below
-          // — the recipe composes from validated cells but has not been
-          // exercised there yet.
+          // RTX PRO 6000 BF16/FP8 cells boot-and-serve). The platforms where
+          // it has not been exercised carry verificationStatus "in-progress"
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -20,6 +20,9 @@ For all methods and hardware platforms, see the [official SGLang installation gu
+# For the DFLASH2 cells only — DFlash2 selector support is newer than the
+# latest release, so install from source (Method 2) instead of the line above.
@@ -30,6 +33,10 @@ Then run the **Python** output of the command panel below in that environment.
+# For the DFLASH2 cells only — that tag predates DFlash2 selector support,
+# so pull a nightly built from main instead:
+# docker pull lmsysorg/sglang:dev-nightly-0820
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +35/-8; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +45/-12
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/scripts/check_cookbook_configs.mjs`, `docs/src/snippets/_deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35767 - [docs] Point the Qwen3.8-27B DFLASH2 note back at the rolling dev image tag

- 链接: https://github.com/sgl-project/sglang/pull/35767
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；关联提交 `14795dcb1afb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Point the Qwen3.8-27B DFLASH2 note back at the rolling dev image tag」；模型线: Qwen3.8；类别: 性能/后端优化；主要 diff: `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「[docs] Point the Qwen3.8-27B DFLASH2 note back at the rolling dev image tag」；主要实现面是 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +1/-1 (2 lines); hunks: -36,7 +36,7 @@ docker pull lmsysorg/sglang:qwen38-27b。
- 代码 diff 细节:
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +1/-1 (2 lines); hunks: -36,7 +36,7 @@ docker pull lmsysorg/sglang:qwen38-27b
- 关键代码摘录:

```diff
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -36,7 +36,7 @@ docker pull lmsysorg/sglang:qwen38-27b
-# docker pull lmsysorg/sglang:dev-nightly-0820
+# docker pull lmsysorg/sglang:dev
```

- 已读文件:
  - docs: `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35786 - [docs] Retune the Qwen3.8-27B RTX 5090 DFLASH2 cells against 1cf2b8c

- 链接: https://github.com/sgl-project/sglang/pull/35786
- 状态/时间: merged / 2026-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `3efa0574496b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+71/-37，可读 patch 160 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Retune the Qwen3.8-27B RTX 5090 DFLASH2 cells against 1cf2b8c」；模型线: Qwen3.8；类别: 性能/后端优化；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「[docs] Retune the Qwen3.8-27B RTX 5090 DFLASH2 cells against 1cf2b8c」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +39/-13 (52 lines); hunks: -128,10 +128,8 @@ export const config = {; -142,15 +140,19 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +32/-24 (56 lines); hunks: -22,7 +22,11 @@ pip install uv; -34,9 +38,11 @@ Then run the **Python** output of the command panel below in...; symbols: GPUs，涉及 `GPUs`。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +39/-13 (52 lines); hunks: -128,10 +128,8 @@ export const config = {; -142,15 +140,19 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +32/-24 (56 lines); hunks: -22,7 +22,11 @@ pip install uv; -34,9 +38,11 @@ Then run the **Python** output of the command panel below in...; symbols: GPUs
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -128,10 +128,8 @@ export const config = {
-          // 5090: mem-fraction re-pins like DSPARK's, and fp32 additionally
-          // re-pins the ratio — the balanced L=9216 value leaves the state
-          // pool one slot short at every serviceable mem-fraction (see the
-          // DFlash2 bullet in Configuration Tips).
+          // fp32 is the one case that needs the balanced ratio overridden, so
+          // that family is stripped too and re-emitted below.
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -22,7 +22,11 @@ pip install uv
-# latest release, so install from source (Method 2) instead of the line above.
+# latest release, so build from the commit those cells were validated on
+# instead of the line above:
+# git clone https://github.com/sgl-project/sglang.git && cd sglang
+# git checkout 1cf2b8c54d81802abc15dcf23a29b9cc687bc01e   # PR #35496
+# uv pip install -e "python[all]"
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +39/-13; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +32/-24
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35825 - [docs] Re-measure the Qwen3.8-27B RTX 5090, RTX PRO 6000 and DGX Spark grids on 1cf2b8c

- 链接: https://github.com/sgl-project/sglang/pull/35825
- 状态/时间: merged / 2026-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `4cb5aebfe08f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+119/-150，可读 patch 408 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Re-measure the Qwen3.8-27B RTX 5090, RTX PRO 6000 and DGX Spark grids on 1cf2b8c」；模型线: Qwen3.8；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`；技术摘要: 覆盖「[docs] Re-measure the Qwen3.8-27B RTX 5090, RTX PRO 6000 and DGX Spark grids on 1cf2b8c」；主要实现面是 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +73/-73 (146 lines); hunks: -83,12 +83,14 @@ export const config = {; -107,13 +109,13 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +46/-77 (123 lines); hunks: -19,14 +19,10 @@ For all methods and hardware platforms, see the [official SG...; -36,13 +32,7 @@ Then run the **Python** output of the command panel below in...; symbols: GPUs，涉及 `GPUs`。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +73/-73 (146 lines); hunks: -83,12 +83,14 @@ export const config = {; -107,13 +109,13 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +46/-77 (123 lines); hunks: -19,14 +19,10 @@ For all methods and hardware platforms, see the [official SG...; -36,13 +32,7 @@ Then run the **Python** output of the command panel below in...; symbols: GPUs
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -83,12 +83,14 @@ export const config = {
-            // Measured on the 5090: bf16 state serves at 0.92, fp32 needs
-            // 0.94 (an fp32 slot is 146.81 MiB vs bf16's 74.81).
+            // Measured on the 5090 at commit 1cf2b8c: fp32 serves at 0.94,
+            // bf16 at 0.93. bf16 moved UP from 0.92 with the dense-lm_head
+            // checkpoint -- the heavier weights need a larger static budget
+            // before the state pool fits.
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -19,14 +19,10 @@ For all methods and hardware platforms, see the [official SGLang installation gu
-uv pip install --prerelease=allow sglang
-# For the DFLASH2 cells only — DFlash2 selector support is newer than the
-# latest release, so build from the commit those cells were validated on
-# instead of the line above:
-# git clone https://github.com/sgl-project/sglang.git && cd sglang
-# git checkout 1cf2b8c54d81802abc15dcf23a29b9cc687bc01e   # PR #35496
```

- 已读文件:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +73/-73; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +46/-77
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
