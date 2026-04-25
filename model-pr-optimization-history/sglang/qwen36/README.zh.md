# sglang Qwen3.6 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` | [#23486](https://github.com/sgl-project/sglang/pull/23486) |
| `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` | [#23486](https://github.com/sgl-project/sglang/pull/23486) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-22 | [#23486](https://github.com/sgl-project/sglang/pull/23486) | merged | docs(cookbook): add Qwen3.6-27B dense variant | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |

## 逐 PR diff 审计卡

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- 链接: https://github.com/sgl-project/sglang/pull/23034
- 状态/时间: merged / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 73 个文件，+2214/-215，可读 patch 3198 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs」；模型线: Qwen3.6；类别: 缺陷修复；主要 diff: `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`；PR 正文摘要: - **Add Qwen3.6 documentation**: New full deployment guide for Qwen3.6-35B-A3B (hybrid GDN + sparse MoE architecture) with JSX deployment snippet, covering MTP, tool calling (`q...。
- 实现要点: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ====")；`docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that，涉及 `NewModelDetector, that`；`docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509；`docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471。
- 代码 diff 细节:
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ====")
  - `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunks: -0,0 +1,299; symbols: per_token_group_quant_8bit, add
- 关键代码摘录:

```diff
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -207,7 +207,7 @@ print_highlight("==== Text ====")
-The reasoning separation is enable by default when specify .
+The reasoning separation is enable by default when specify .
@@ -226,7 +226,7 @@ print_highlight("==== Original Output ====")
-### SGLang Native API
+### SGLang Native API
@@ -315,4 +315,3 @@ llm.shutdown()
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -718,7 +718,7 @@ for tool_call in tool_calls:
-> **Note:**
+> **Note:**
@@ -738,4 +738,3 @@ terminate_process(server_process)
diff -- docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx
@@ -0,0 +1,509 @@
+---
+title: "DP, DPA and SGLang DP Router"
+metatags:
```

- 已读文件:
  - docs: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3; `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2; `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0; `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0; `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/.gitignore`, `docs_new/cards/logos/google.png`, `docs_new/cards/logos/mova.png`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- 链接: https://github.com/sgl-project/sglang/pull/23474
- 状态/时间: open / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+284/-8，可读 patch 330 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models」；模型线: Qwen3.6；类别: 缺陷修复；主要 diff: `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`；PR 正文摘要: Fixes #23150. `--cpu-offload-gb > 0` was broken on hybrid linear-attention models (Qwen3-Next, Qwen3.5, Kimi-Linear): the first `/v1/chat/completions` request raised While fixin...。
- 实现要点: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent，涉及 `_TiedChild, __init__, forward`；`python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward，涉及 `maybe_offload_to_cpu, forward`。
- 代码 diff 细节:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward
- 关键代码摘录:

```diff
diff -- test/registered/unit/utils/test_offloader_tied_params.py
@@ -0,0 +1,199 @@
+"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).
+Two failure modes caused the Qwen3-Next / Qwen3.5 CPU-offload regression:
+1. **Tied parameters**: a single nn.Parameter is registered under both a parent
+   and a child module (Qwen3GatedDeltaNet + RadixLinearAttention share
+   ``A_log`` / ``dt_bias``). state_dict() then lists the same tensor under
+   multiple keys, and functional_call(..., tie_weights=True) rejects it when
diff -- python/sglang/srt/utils/offloader.py
@@ -1,7 +1,7 @@
-from typing import Callable, Generator, List, Optional
+from typing import Callable, Dict, Generator, List, Optional
@@ -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
+        # Record tensor views that alias each parameter's *original* storage
+        # BEFORE we rebind .data to pinned CPU memory. Some hybrid linear-attn
+        # models (e.g. Qwen3-Next) cache such views, which would otherwise point
```

- 已读文件:
  - tests: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0
  - runtime: `python/sglang/srt/utils/offloader.py` modified +85/-8
- 验证与风险: diff 自带测试面 `test/registered/unit/utils/test_offloader_tied_params.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- 链接: https://github.com/sgl-project/sglang/pull/23467
- 状态/时间: merged / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+31/-4，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert」；模型线: Qwen3.6；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/utils.py`；PR 正文摘要: - `is_layer_skipped` uses naive substring match (`ignored in prefix`) on `modules_to_not_convert` entries, which silently fires when an entry is a prefix-substring of a fused li...。
- 实现要点: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped，涉及 `__getattr__, _module_path_match, is_layer_skipped`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/utils.py
@@ -43,6 +43,28 @@ def __getattr__(self, name):
+def _module_path_match(ignored: str, prefix: str) -> bool:
+    # Match on dotted module-path boundaries so that `mlp.gate` does NOT
+    # match `mlp.gate_up_proj`. Needed for quant configs (e.g. Qwen3.6-FP8)
+    # whose `modules_to_not_convert` lists MoE-template names like `mlp.gate`
+    # that collide with fused dense MLP names by plain substring.
+    if ignored == prefix:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23486 - docs(cookbook): add Qwen3.6-27B dense variant

- 链接: https://github.com/sgl-project/sglang/pull/23486
- 状态/时间: merged / 2026-04-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；关联提交 `de962f327432`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+55/-17，可读 patch 170 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): add Qwen3.6-27B dense variant」；模型线: Qwen3.6；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；PR 正文摘要: Qwen3.6 ships a 27B dense variant (`Qwen3.6-27B` / `Qwen3.6-27B-FP8`) alongside the existing 35B-A3B MoE. Update the cookbook page and deployment snippet to cover both. - Rewrit...。
- 实现要点: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunks: -1,26 +1,29; -29,30 +32,43 @@ Qwen3.6 features a Gated Delta Networks combined with sparse...；`docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunks: -10,6 +10,14 @@ export const Qwen36Deployment = () => {; -66,9 +74,18 @@ export const Qwen36Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunks: -1,26 +1,29; -29,30 +32,43 @@ Qwen3.6 features a Gated Delta Networks combined with sparse...
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunks: -10,6 +10,14 @@ export const Qwen36Deployment = () => {; -66,9 +74,18 @@ export const Qwen36Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx
@@ -1,26 +1,29 @@
-    description: "Deploy Qwen3.6 with SGLang - open-weight 35B MoE multimodal model with 3B active parameters, thinking preservation, tool calling, MTP, and long-context support."
+    description: "Deploy Qwen3.6 with SGLang - open-weight multimodal series with a 35B MoE (3B active) variant and a 27B dense variant, hybrid reasoning, tool calling, MTP, and l
-[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is the first open-weight variant of the Qwen3.6 series developed by Alibaba. Built on direct feedback from the commu
+The Qwen3.6 series is developed by Alibaba. Built on direct feedback from the community, Qwen3.6 prioritizes stability and real-world utility, delivering substantial upgrades in a
-Qwen3.6 features a Gated Delta Networks combined with sparse Mixture-of-Experts architecture (35B total parameters, 3B activated), supporting multimodal inputs (text, image, video
+- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — **Sparse MoE** (35B total, 3B active) on a Gated Delta Networks backbone.
diff -- docs_new/src/snippets/autoregressive/qwen36-deployment.jsx
@@ -10,6 +10,14 @@ export const Qwen36Deployment = () => {
+    modelSize: {
+      name: 'modelSize',
+      title: 'Model Size',
+      items: [
+        { id: '35b-a3b', label: '35B-A3B (MoE)', default: true },
+        { id: '27b', label: '27B (Dense)', default: false },
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10; `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
