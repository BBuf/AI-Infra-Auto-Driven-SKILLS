# sglang Hunyuan3 Preview 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/basic_usage/hy3_preview.md` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |
| `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` | [#23532](https://github.com/sgl-project/sglang/pull/23532) |
| `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` | [#23532](https://github.com/sgl-project/sglang/pull/23532) |
| `python/sglang/srt/function_call/hunyuan_detector.py` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |
| `test/registered/unit/function_call/test_hunyuan_detector.py` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 2
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-23 | [#23532](https://github.com/sgl-project/sglang/pull/23532) | merged | docs: add Hunyuan 3 Preview cookbook | `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` |
| 2026-04-24 | [#23533](https://github.com/sgl-project/sglang/pull/23533) | merged | support Hy3 preview | `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py`, `docs/basic_usage/hy3_preview.md` |

## 逐 PR diff 审计卡

### PR #23532 - docs: add Hunyuan 3 Preview cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23532
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`；关联提交 `4868e367f851`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+707/-0，可读 patch 716 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add Hunyuan 3 Preview cookbook」；模型线: Hunyuan3 Preview；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`；PR 正文摘要: Adds a cookbook entry for Tencent **Hunyuan 3 Preview** (`Hy3-preview` / `Hy3-preview-FP8` / `Hy3-preview-Base`) under `docs_new/cookbook/autoregressive/Tencent/`. - **Doc** (`c...。
- 实现要点: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0 (527 lines); hunks: -0,0 +1,527; symbols: GPUs，涉及 `GPUs`；`docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: GPUs，涉及 `GPUs`。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0 (527 lines); hunks: -0,0 +1,527; symbols: GPUs
  - `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: GPUs
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx
@@ -0,0 +1,527 @@
+---
+title: Hunyuan 3 Preview
+metatags:
+    description: "Deploy Tencent Hunyuan 3 Preview BF16 (~276B / ~20B active MoE) on NVIDIA GPUs with SGLang — hybrid thinking, native tool calling, 256K context, and built-in MTP
+tag: NEW
+---
diff -- docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx
@@ -0,0 +1,174 @@
+export const Hunyuan3PreviewDeployment = () => {
+  // Hunyuan 3 Preview (~276B total / ~20B active MoE) — BF16 only.
+  // ~552GB weights; 80GB-class GPUs (A100/H100) cannot fit single-node.
+  //   H200 (141GB): tp=8
+  //   B200 (180GB): tp=8
+  //   B300 (275GB): tp=4
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0; `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23533 - support Hy3 preview

- 链接: https://github.com/sgl-project/sglang/pull/23533
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/hy3_preview.md`, `python/sglang/srt/function_call/hunyuan_detector.py`, `test/registered/unit/function_call/test_hunyuan_detector.py`；关联提交 `6d038614760f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 25 个文件，+4095/-3，可读 patch 4205 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support Hy3 preview」；模型线: Hunyuan3 Preview；类别: 模型支持/运行时入口；主要 diff: `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py`, `docs/basic_usage/hy3_preview.md`；PR 正文摘要: Add support for Tencent Hunyuan V3 (Hy3-preview) models in sglang. Components - **Model**: `python/sglang/srt/models/hunyuan_v3.py` (+ MTP variant `hunyuan_v3_nextn.py`) - **Too...。
- 实现要点: `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: _make_tools, TestHunyuanDetectorHasToolCall, setUp, test_has_tool_call_true，涉及 `_make_tools, TestHunyuanDetectorHasToolCall, setUp`；`python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: HunyuanDetector, __init__, _normalize_type, _get_arg_schema，涉及 `HunyuanDetector, __init__, _normalize_type`；`docs/basic_usage/hy3_preview.md` added +191/-0 (191 lines); hunks: -0,0 +1,191。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: _make_tools, TestHunyuanDetectorHasToolCall, setUp, test_has_tool_call_true
  - `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: HunyuanDetector, __init__, _normalize_type, _get_arg_schema
  - `docs/basic_usage/hy3_preview.md` added +191/-0 (191 lines); hunks: -0,0 +1,191
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_hunyuan_detector.py
@@ -0,0 +1,733 @@
+"""Unit tests for HunyuanDetector - no server, no model loading."""
+import json
+import unittest
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/function_call/hunyuan_detector.py
@@ -0,0 +1,476 @@
+import json
+import logging
+import re
+from typing import Any, Dict, List, Optional, Set
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.environ import envs
diff -- docs/basic_usage/hy3_preview.md
@@ -0,0 +1,191 @@
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0
  - runtime: `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0
  - docs: `docs/basic_usage/hy3_preview.md` added +191/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `test/registered/unit/function_call/test_hunyuan_detector.py`, `test/registered/unit/parser/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
