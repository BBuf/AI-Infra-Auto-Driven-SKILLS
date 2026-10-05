# SGLang Qwen3.8 模型 PR 优化历史

2026-08-23 的 SGLang head 没有独立的 `python/sglang/srt/models/qwen3_8*.py`。
公开的 `Qwen/Qwen3.8-27B` 使用 `model_type=qwen3_5` 的混合 GDN/GQA，loader
和 kernel 历史应查 `qwen35`。本 slug 只跟踪 Qwen3.8 cookbook / 部署面。
2.4T 的 `Qwen3.8-2.4T-A95B` 是另一条多机 MoE recipe，不要抄进单卡 27B
serving 方案。

2026-10-05 补充：Qwen3.8-Flash-Next 的实现文件是
`python/sglang/srt/models/qwen4_exp.py`（#37500），模型与 kernel PR 见
`qwen4-exp` slug；它的 cookbook 页面仍在本页跟踪。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` | [#34860](https://github.com/sgl-project/sglang/pull/34860), [#34863](https://github.com/sgl-project/sglang/pull/34863), [#35064](https://github.com/sgl-project/sglang/pull/35064), [#35065](https://github.com/sgl-project/sglang/pull/35065), [#35121](https://github.com/sgl-project/sglang/pull/35121), [#35663](https://github.com/sgl-project/sglang/pull/35663), [#35753](https://github.com/sgl-project/sglang/pull/35753), [#35767](https://github.com/sgl-project/sglang/pull/35767), [#35786](https://github.com/sgl-project/sglang/pull/35786), [#35825](https://github.com/sgl-project/sglang/pull/35825), [#36020](https://github.com/sgl-project/sglang/pull/36020), [#36496](https://github.com/sgl-project/sglang/pull/36496), ... (13 total) |
| `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` | [#36496](https://github.com/sgl-project/sglang/pull/36496), [#36497](https://github.com/sgl-project/sglang/pull/36497), [#36499](https://github.com/sgl-project/sglang/pull/36499), [#37995](https://github.com/sgl-project/sglang/pull/37995), [#41046](https://github.com/sgl-project/sglang/pull/41046) |
| `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` | [#34587](https://github.com/sgl-project/sglang/pull/34587), [#34590](https://github.com/sgl-project/sglang/pull/34590), [#34601](https://github.com/sgl-project/sglang/pull/34601), [#34860](https://github.com/sgl-project/sglang/pull/34860) |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx` | [#34836](https://github.com/sgl-project/sglang/pull/34836) |
| `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` | [#34860](https://github.com/sgl-project/sglang/pull/34860), [#35064](https://github.com/sgl-project/sglang/pull/35064), [#36020](https://github.com/sgl-project/sglang/pull/36020), [#38611](https://github.com/sgl-project/sglang/pull/38611) |
| `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` | [#34863](https://github.com/sgl-project/sglang/pull/34863), [#35065](https://github.com/sgl-project/sglang/pull/35065), [#36020](https://github.com/sgl-project/sglang/pull/36020) |
| `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` | [#34860](https://github.com/sgl-project/sglang/pull/34860), [#34863](https://github.com/sgl-project/sglang/pull/34863), [#35065](https://github.com/sgl-project/sglang/pull/35065), [#35121](https://github.com/sgl-project/sglang/pull/35121), [#35663](https://github.com/sgl-project/sglang/pull/35663), [#35753](https://github.com/sgl-project/sglang/pull/35753), [#35786](https://github.com/sgl-project/sglang/pull/35786), [#35825](https://github.com/sgl-project/sglang/pull/35825), [#36020](https://github.com/sgl-project/sglang/pull/36020), [#38611](https://github.com/sgl-project/sglang/pull/38611) |
| `docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx` | [#34587](https://github.com/sgl-project/sglang/pull/34587) |
| `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx` | [#36496](https://github.com/sgl-project/sglang/pull/36496), [#37995](https://github.com/sgl-project/sglang/pull/37995) |
| `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` | [#36496](https://github.com/sgl-project/sglang/pull/36496), [#36611](https://github.com/sgl-project/sglang/pull/36611), [#37995](https://github.com/sgl-project/sglang/pull/37995), [#41046](https://github.com/sgl-project/sglang/pull/41046) |
| `docs/src/snippets/configs/Qwen/qwen3.8.jsx` | [#34587](https://github.com/sgl-project/sglang/pull/34587), [#34590](https://github.com/sgl-project/sglang/pull/34590), [#34601](https://github.com/sgl-project/sglang/pull/34601) |
| `python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/README.md` | 无直接 PR 号提交 |
| `python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/__init__.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/kernel.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py` | [#35383](https://github.com/sgl-project/sglang/pull/35383) |
| `test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py` | [#36901](https://github.com/sgl-project/sglang/pull/36901) |

## PR 覆盖总览

- git 追溯 PR 数: 24
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 24
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
| 2026-08-24 | [#35383](https://github.com/sgl-project/sglang/pull/35383) | merged | [AMD][CI] Add the Qwen3.8 MXFP4 MI35x nightly | `test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py` |
| 2026-08-24 | [#36020](https://github.com/sgl-project/sglang/pull/36020) | merged | [docs] Split the Qwen3.8-27B NVFP4 cells by lm_head precision | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` |
| 2026-08-26 | [#36496](https://github.com/sgl-project/sglang/pull/36496) | merged | Add Qwen3.8-Flash-Next cookbook | `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` |
| 2026-08-26 | [#36499](https://github.com/sgl-project/sglang/pull/36499) | merged | docs: point the Qwen3.8-Flash-Next cookbook at model support PR #36497 | `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` |
| 2026-08-27 | [#36611](https://github.com/sgl-project/sglang/pull/36611) | merged | docs(cookbook): fix Qwen3.8 Flash Next H200 MTP verify with BF16 SSM state | `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` |
| 2026-09-07 | [#37995](https://github.com/sgl-project/sglang/pull/37995) | merged | docs(cookbook): Qwen3.8-Flash-Next NVFP4 recipes for DGX Spark (1x, 2x) and RTX PRO 6000 | `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` |
| 2026-09-08 | [#36497](https://github.com/sgl-project/sglang/pull/36497) | closed | Introduce Qwen 3.8 Flash Next | `python/sglang/srt/models/qwen4_exp.py`, `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`, `python/sglang/srt/layers/attention/qsa/qsa_indexer.py` |
| 2026-09-11 | [#38611](https://github.com/sgl-project/sglang/pull/38611) | merged | [docs] Add the NVIDIA NVFP4 export to the Qwen3.8-27B cookbook | `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` |
| 2026-09-24 | [#41046](https://github.com/sgl-project/sglang/pull/41046) | merged | [Docs] Enable Qwen3.8 Flash Next NVIDIA NVFP4 on B200/B300/GB300 | `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` |
| 2026-09-30 | [#36901](https://github.com/sgl-project/sglang/pull/36901) | merged | [AMD] Add Qwen3.8-Flash-Next-FP8 nightly validation | `test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py` |

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

### PR #35383 - [AMD][CI] Add the Qwen3.8 MXFP4 MI35x nightly

- 链接: https://github.com/sgl-project/sglang/pull/35383
- 状态/时间: merged / 2026-08-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py`；关联提交 `20064623ab46`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+368/-78，可读 patch 583 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: TestQwen38Mxfp4MI35x, setUpClass, test_a_gsm8k_accuracy, test_b_serving_perf，涉及 `TestQwen38Mxfp4MI35x, setUpClass, test_a_gsm8k_accuracy`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: TestQwen38Mxfp4MI35x, setUpClass, test_a_gsm8k_accuracy, test_b_serving_perf
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py
@@ -0,0 +1,214 @@
+"""MI35x Qwen3.8-2.4T-A95B MXFP4 GSM8K accuracy + serving-perf test (8-GPU)
+Tests amd/Qwen3.8-2.4T-A95B-Quark-MXFP4, AMD's day-0 Quark quantization of
+Qwen/Qwen3.8-2.4T-A95B-FP8, on a single 8-GPU MI35x node.
+Qwen3.8 is a 2.4T-parameter / 95B-active hybrid MoE: 23 repeats of 3 x Gated
+DeltaNet -> MoE then 1 x Gated Attention -> MoE, 512 experts with 10 routed + 1
+shared active. It reuses the Qwen3.5 architecture -- the checkpoint reports
```

- 提取文件（未人工审阅）:
  - tests: `test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py` added +214/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py`, `test/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36020 - [docs] Split the Qwen3.8-27B NVFP4 cells by lm_head precision

- 链接: https://github.com/sgl-project/sglang/pull/36020
- 状态/时间: merged / 2026-08-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `5030637c65ba`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+188/-30，可读 patch 359 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +162/-17 (179 lines); hunks: -44,7 +44,13 @@ export const config = {; -62,7 +68,7 @@ export const config = {；`docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` modified +2/-2 (4 lines); hunks: -30,7 +30,7; -44,7 +44,7 @@ export const benchmarks = [；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +22/-9 (31 lines); hunks: -168,14 +168,25 @@ context from earlier messages.; -251,12 +262,14 @@ checkpoint's calibration scales automatically.; symbols: GPUs，涉及 `GPUs`；`docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +2/-2 (4 lines); hunks: -56,7 +56,7 @@ export const Qwen38MambaRatioCalculator = () => {; -99,7 +99,7 @@ export const Qwen38MambaRatioCalculator = () => {。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +162/-17 (179 lines); hunks: -44,7 +44,13 @@ export const config = {; -62,7 +68,7 @@ export const config = {
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` modified +2/-2 (4 lines); hunks: -30,7 +30,7; -44,7 +44,7 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +22/-9 (31 lines); hunks: -168,14 +168,25 @@ context from earlier messages.; -251,12 +262,14 @@ checkpoint's calibration scales automatically.; symbols: GPUs
  - `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +2/-2 (4 lines); hunks: -56,7 +56,7 @@ export const Qwen38MambaRatioCalculator = () => {; -99,7 +99,7 @@ export const Qwen38MambaRatioCalculator = () => {
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -44,7 +44,13 @@ export const config = {
-      { id: "nvfp4", label: "NVFP4" },
+      // Two NVFP4 exports ship separately, differing only in the lm_head:
+      // one keeps it dense bf16, the other packs it to FP4. The bf16 head is
+      // ~1.7GB larger on disk (~3.2GB at runtime), so it is strictly the
+      // harder of the two to fit -- which is why the FP4-head cells reuse the
+      // BF16-head recipes verbatim.
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx
@@ -30,7 +30,7 @@
-    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
+    match: { hw: "gb300", variant: "default", quant: "nvfp4-fp4-head", strategy: "balanced", nodes: "single" },
@@ -44,7 +44,7 @@ export const benchmarks = [
-    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
+    match: { hw: "gb300", variant: "default", quant: "nvfp4-fp4-head", strategy: "high-throughput", nodes: "single" },
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -168,14 +168,25 @@ context from earlier messages.
-      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Qwen3.8-27B-NVFP4</td>
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +162/-17; `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx` modified +2/-2; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +22/-9; `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b-benchmarks.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36496 - Add Qwen3.8-Flash-Next cookbook

- 链接: https://github.com/sgl-project/sglang/pull/36496
- 状态/时间: merged / 2026-08-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`；关联提交 `c7b5e76fa925`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+1027/-10，可读 patch 1074 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` added +723/-0 (723 lines); hunks: -0,0 +1,723；`docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx` added +72/-0 (72 lines); hunks: -0,0 +1,72；`docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` added +222/-0 (222 lines); hunks: -0,0 +1,222；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` added +723/-0 (723 lines); hunks: -0,0 +1,723
  - `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx` added +72/-0 (72 lines); hunks: -0,0 +1,72
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` added +222/-0 (222 lines); hunks: -0,0 +1,222
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx
@@ -0,0 +1,723 @@
+// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
+// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
+//
+// Qwen3.8-Flash-Next: 176B total params (51B of that is the N-gram embedding
+// table) with 6B active per token. Hybrid linear/full attention — 3 of every 4
+// layers are Gated DeltaNet, the 4th is global attention running Qwen Sparse
diff -- docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx
@@ -0,0 +1,72 @@
+export const benchmarks = [
+  {
+    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
+    sglang_version: "qwen4-main @ e17062a1d",
+    accuracy: { gsm8k_pct: 97.73, aime26_pct: 97.92 },
+  },
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx
@@ -0,0 +1,222 @@
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` added +723/-0; `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx` added +72/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` added +222/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +0/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`, `docs/cookbook/autoregressive/intro.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36499 - docs: point the Qwen3.8-Flash-Next cookbook at model support PR #36497

- 链接: https://github.com/sgl-project/sglang/pull/36499
- 状态/时间: merged / 2026-08-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`；关联提交 `8eaffdf382e0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 19 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +3/-3 (6 lines); hunks: -23,15 +23,15 @@ pip install -U uv。
- 代码 diff 细节:
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +3/-3 (6 lines); hunks: -23,15 +23,15 @@ pip install -U uv
- 关键代码摘录:

```diff
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx
@@ -23,15 +23,15 @@ pip install -U uv
-# https://github.com/sgl-project/sglang/pull/<PR-NUMBER>
+# https://github.com/sgl-project/sglang/pull/36497
-git fetch origin pull/<PR-NUMBER>/head && git checkout FETCH_HEAD
+git fetch origin pull/36497/head && git checkout FETCH_HEAD
-`<PR-NUMBER>` is a placeholder — substitute the number of the SGLang PR that adds Qwen3.8-Flash-Next model support. Once that PR is in a release, `uv pip install sglang` is enough
+Model support lands in [#36497](https://github.com/sgl-project/sglang/pull/36497). Once it is in a release, `uv pip install sglang` is enough and this whole step goes away.
```

- 提取文件（未人工审阅）:
  - docs: `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +3/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36611 - docs(cookbook): fix Qwen3.8 Flash Next H200 MTP verify with BF16 SSM state

- 链接: https://github.com/sgl-project/sglang/pull/36611
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`；关联提交 `536f570e6692`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 16 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +2/-0 (2 lines); hunks: -202,6 +202,7 @@ export const config = {; -443,6 +444,7 @@ export const config = {。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +2/-0 (2 lines); hunks: -202,6 +202,7 @@ export const config = {; -443,6 +444,7 @@ export const config = {
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx
@@ -202,6 +202,7 @@ export const config = {
+        "--linear-attn-verify-backend triton",
@@ -443,6 +444,7 @@ export const config = {
+        "--linear-attn-verify-backend triton",
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +2/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #37995 - docs(cookbook): Qwen3.8-Flash-Next NVFP4 recipes for DGX Spark (1x, 2x) and RTX PRO 6000

- 链接: https://github.com/sgl-project/sglang/pull/37995
- 状态/时间: merged / 2026-09-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`；关联提交 `f4b75b5c36ca`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+791/-13，可读 patch 912 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +562/-9 (571 lines); hunks: -7,29 +7,49; -39,8 +59,11 @@ export const config = {；`docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx` modified +181/-0 (181 lines); hunks: -65,6 +65,187 @@ export const benchmarks = [；`docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +48/-4 (52 lines); hunks: -40,13 +40,19 @@ Then run the **Python** output of the command panel below in...; -68,6 +74,39 @@ import { benchmarks } from "/src/snippets/configs/Qwen/qwen3....。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +562/-9 (571 lines); hunks: -7,29 +7,49; -39,8 +59,11 @@ export const config = {
  - `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx` modified +181/-0 (181 lines); hunks: -65,6 +65,187 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +48/-4 (52 lines); hunks: -40,13 +40,19 @@ Then run the **Python** output of the command panel below in...; -68,6 +74,39 @@ import { benchmarks } from "/src/snippets/configs/Qwen/qwen3....
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx
@@ -7,29 +7,49 @@
-// Every recipe on this page is single-node: BF16 and FP8 run TP4 (so four GPUs
-// of an 8-GPU H200/B200/B300 host, or a whole 4-GPU GB300 node), NVFP4 runs on
-// a single GPU, and the AMD cells run TP8. That fits because 6B active params
-// keeps compute small and the N-gram table is the only large weight block.
+// Every datacenter recipe on this page is single-node: BF16 and FP8 run TP4 (so
+// four GPUs of an 8-GPU H200/B200/B300 host, or a whole 4-GPU GB300 node), NVFP4
diff -- docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx
@@ -65,6 +65,187 @@ export const benchmarks = [
+  // 2x DGX Spark, TP=2, lmsysorg/sglang:qwen38flashnext (SGLang 593134d17a),
+  // 2026-09-04. GSM8K is the full 1,319-question set via the chat API (thinking
+  // off, greedy, 8192 max tokens) on lmsysorg/sglang:dev-qwen38-next-local
+  // (qwen4-main-squashed 4ccff141db). AIME26 and MMMU-Pro not run.
+  {
+    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx
@@ -40,13 +40,19 @@ Then run the **Python** output of the command panel below in that environment.
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +562/-9; `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx` modified +181/-0; `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +48/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36497 - Introduce Qwen 3.8 Flash Next

- 链接: https://github.com/sgl-project/sglang/pull/36497
- 状态/时间: closed / 2026-09-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`；关联提交 `8eaffdf382e0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 115 个文件，+20609/-116，可读 patch 22023 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/qwen4_exp.py` added +2182/-0 (2182 lines); hunks: -0,0 +1,2182; symbols: _ple_table_is_fp8, _get_ple_forward_mode, _get_processed_token_count, _PLEBatch，涉及 `_ple_table_is_fp8, _get_ple_forward_mode, _get_processed_token_count`；`python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` added +1883/-0 (1883 lines); hunks: -0,0 +1,1883; symbols: _resolve_trtllm_sparse_decode, _resolve_flash_attn_varlen_func, flash_attn_varlen_func, QwenSparseAttnMetadata，涉及 `_resolve_trtllm_sparse_decode, _resolve_flash_attn_varlen_func, flash_attn_varlen_func`；`python/sglang/srt/layers/attention/qsa/qsa_indexer.py` added +652/-0 (652 lines); hunks: -0,0 +1,652; symbols: _qsa_prefill_row_chunk_size, QSAIndexer, __init__, _validate_config，涉及 `_qsa_prefill_row_chunk_size, QSAIndexer, __init__`；`python/sglang/srt/models/qwen4_exp_ple_table.py` added +483/-0 (483 lines); hunks: -0,0 +1,483; symbols: PleFilePrefetcher, __init__, pages_for_rows, _advise，涉及 `PleFilePrefetcher, __init__, pages_for_rows`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen4_exp.py` added +2182/-0 (2182 lines); hunks: -0,0 +1,2182; symbols: _ple_table_is_fp8, _get_ple_forward_mode, _get_processed_token_count, _PLEBatch
  - `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` added +1883/-0 (1883 lines); hunks: -0,0 +1,1883; symbols: _resolve_trtllm_sparse_decode, _resolve_flash_attn_varlen_func, flash_attn_varlen_func, QwenSparseAttnMetadata
  - `python/sglang/srt/layers/attention/qsa/qsa_indexer.py` added +652/-0 (652 lines); hunks: -0,0 +1,652; symbols: _qsa_prefill_row_chunk_size, QSAIndexer, __init__, _validate_config
  - `python/sglang/srt/models/qwen4_exp_ple_table.py` added +483/-0 (483 lines); hunks: -0,0 +1,483; symbols: PleFilePrefetcher, __init__, pages_for_rows, _advise
  - `python/sglang/srt/layers/attention/qsa/sparse_attn.py` added +456/-0 (456 lines); hunks: -0,0 +1,456; symbols: _get_best_config, _sparse_gqa_prefill, sparse_gqa_fwd_interface_triton, _sparse_gqa_chunk_prefill
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -0,0 +1,2182 @@
+"""Inference-only Qwen4-Exp (text + VL) on the Qwen3.5 backbone."""
+import math
+from contextlib import nullcontext
+from typing import Any, Iterable, Optional, Set, Tuple
+import msgspec
+import sympy
diff -- python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
@@ -0,0 +1,1883 @@
+"""Sparse-attention backend for Qwen4-Exp models with an indexer.
+The backend is installed as the full-attention side of Qwen4-Exp's hybrid backend;
+linear-attention layers continue to use GDN.
+"""
+from __future__ import annotations
+import logging
diff -- python/sglang/srt/layers/attention/qsa/qsa_indexer.py
@@ -0,0 +1,652 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/qwen4_exp.py` added +2182/-0; `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` added +1883/-0; `python/sglang/srt/layers/attention/qsa/qsa_indexer.py` added +652/-0; `python/sglang/srt/models/qwen4_exp_ple_table.py` added +483/-0; `python/sglang/srt/layers/attention/qsa/sparse_attn.py` added +456/-0; `python/sglang/srt/layers/attention/qsa/mqa.py` added +422/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_aime25.py`, `python/sglang/test/simple_eval_aime26.py`, `python/sglang/test/simple_eval_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38611 - [docs] Add the NVIDIA NVFP4 export to the Qwen3.8-27B cookbook

- 链接: https://github.com/sgl-project/sglang/pull/38611
- 状态/时间: merged / 2026-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；关联提交 `df6424967ab8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+283/-96，可读 patch 601 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +203/-47 (250 lines); hunks: -28,10 +28,11 @@ export const config = {; -51,6 +52,15 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +73/-44 (117 lines); hunks: -19,9 +19,7 @@ For all methods and hardware platforms, see the [official SGLa...; -31,7 +29,7 @@ Then run the **Python** output of the command panel below in t...；`docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +7/-5 (12 lines); hunks: -89,17 +89,19 @@ export const Qwen38MambaRatioCalculator = () => {。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +203/-47 (250 lines); hunks: -28,10 +28,11 @@ export const config = {; -51,6 +52,15 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +73/-44 (117 lines); hunks: -19,9 +19,7 @@ For all methods and hardware platforms, see the [official SGLa...; -31,7 +29,7 @@ Then run the **Python** output of the command panel below in t...
  - `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +7/-5 (12 lines); hunks: -89,17 +89,19 @@ export const Qwen38MambaRatioCalculator = () => {
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
@@ -28,10 +28,11 @@ export const config = {
-  // (sign-off recorded in the PR description). NVFP4: a no-op made visible
-  // (the checkpoint's `kv_cache_quant_algo: FP8` already resolved `auto` to
-  // fp8_e4m3). BF16/FP8: a real quality/capacity trade — halves
-  // kv_bytes_per_token but those checkpoints carry no fp8 KV calibration.
+  // (sign-off recorded in the PR description). The two RadixArk NVFP4 exports:
+  // a no-op made visible (their `kv_cache_quant_algo: FP8` already resolved
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
@@ -19,9 +19,7 @@ For all methods and hardware platforms, see the [official SGLang installation gu
-git clone https://github.com/sgl-project/sglang.git && cd sglang
-git checkout 1cf2b8c54d81802abc15dcf23a29b9cc687bc01e
-uv pip install --prerelease=allow -e "python[all]"
+uv pip install --prerelease=allow sglang
@@ -31,7 +29,7 @@ Then run the **Python** output of the command panel below in that environment.
-docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2
diff -- docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx
@@ -89,17 +89,19 @@ export const Qwen38MambaRatioCalculator = () => {
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` modified +203/-47; `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` modified +73/-44; `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx` modified +7/-5
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx`, `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #41046 - [Docs] Enable Qwen3.8 Flash Next NVIDIA NVFP4 on B200/B300/GB300

- 链接: https://github.com/sgl-project/sglang/pull/41046
- 状态/时间: merged / 2026-09-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`；关联提交 `a1eb691ab51b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+120/-52，可读 patch 213 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +117/-3 (120 lines); hunks: -8,8 +8,9; -44,7 +45,7 @@ export const config = {；`docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +3/-49 (52 lines); hunks: -10,59 +10,13 @@ tag: NEW。
- 代码 diff 细节:
  - `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +117/-3 (120 lines); hunks: -8,8 +8,9; -44,7 +45,7 @@ export const config = {
  - `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +3/-49 (52 lines); hunks: -10,59 +10,13 @@ tag: NEW
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx
@@ -8,8 +8,9 @@
-// four GPUs of an 8-GPU H200/B200/B300 host, or a whole 4-GPU GB300 node), NVFP4
-// runs on a single GPU, and the AMD cells run TP8. That fits because 6B active
+// four GPUs of an 8-GPU H200/B200/B300 host, or a whole 4-GPU GB300 node).
+// Both RadixArk and NVIDIA NVFP4 recipes use TP1. The AMD cells run TP8.
+// That fits because 6B active
@@ -44,7 +45,7 @@ export const config = {
diff -- docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx
@@ -10,59 +10,13 @@ tag: NEW
-For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle i
-<Tabs>
-<Tab title="Python (pip / uv)">
-Qwen3.8-Flash-Next support is not in a tagged release yet, so build the model-support PR rather than installing from PyPI:
+Use an SGLang build that includes Qwen3.8-Flash-Next support (v0.5.20 or later).
-pip install -U uv
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx` modified +117/-3; `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx` modified +3/-49
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`, `docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36901 - [AMD] Add Qwen3.8-Flash-Next-FP8 nightly validation

- 链接: https://github.com/sgl-project/sglang/pull/36901
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py`；关联提交 `b87a241a6f97`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+331/-0，可读 patch 353 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: TestQwen38FlashNextFP8AMD, _assert_multimodal_generation, test_gsm8k_accuracy，涉及 `TestQwen38FlashNextFP8AMD, _assert_multimodal_generation, test_gsm8k_accuracy`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: TestQwen38FlashNextFP8AMD, _assert_multimodal_generation, test_gsm8k_accuracy
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py
@@ -0,0 +1,229 @@
+"""AMD Qwen3.8-Flash-Next-FP8 graph-mode GSM8K accuracy test.
+The released FP8 checkpoint uses TP1 on gfx950 and TP2+EP2 on gfx942, where
+the 192 GiB device capacity cannot hold the target, MTP draft, and graph/KV
+state on one GPU. These nightly tests pin the checkpoint revision and exercise
+the same AITER-attention decode-graph path with EAGLE speculation on both
+architectures. SGLANG_USE_AITER=0 keeps direct AITER paged QSA disabled so the
```

- 提取文件（未人工审阅）:
  - tests: `test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py` added +229/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/test_qwen38_flash_next_fp8_eval.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
