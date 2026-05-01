# sglang Hunyuan3 Preview PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 0
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/hy3_preview.md` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |
| `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` | [#23532](https://github.com/sgl-project/sglang/pull/23532) |
| `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` | [#23532](https://github.com/sgl-project/sglang/pull/23532) |
| `python/sglang/srt/function_call/hunyuan_detector.py` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |
| `test/registered/unit/function_call/test_hunyuan_detector.py` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-23 | [#23532](https://github.com/sgl-project/sglang/pull/23532) | merged | docs: add Hunyuan 3 Preview cookbook | `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` |
| 2026-04-24 | [#23533](https://github.com/sgl-project/sglang/pull/23533) | merged | support Hy3 preview | `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py`, `docs/basic_usage/hy3_preview.md` |

## Per-PR Diff Audit Cards

### PR #23532 - docs: add Hunyuan 3 Preview cookbook

- Link: https://github.com/sgl-project/sglang/pull/23532
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`; associated commits `4868e367f851`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +707/-0, 716 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: add Hunyuan 3 Preview cookbook"; model line: Hunyuan3 Preview; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`; technical summary: Covers "docs: add Hunyuan 3 Preview cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0 (527 lines); hunks: -0,0 +1,527; symbols: GPUs, touching `GPUs`; `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: GPUs, touching `GPUs`.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0 (527 lines); hunks: -0,0 +1,527; symbols: GPUs
  - `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0 (174 lines); hunks: -0,0 +1,174; symbols: GPUs
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` added +527/-0; `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` added +174/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23533 - support Hy3 preview

- Link: https://github.com/sgl-project/sglang/pull/23533
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/hy3_preview.md`, `python/sglang/srt/function_call/hunyuan_detector.py`, `test/registered/unit/function_call/test_hunyuan_detector.py`; associated commits `6d038614760f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 25 files, +4095/-3, 4205 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support Hy3 preview"; model line: Hunyuan3 Preview; category: model support/runtime entry; main diff: `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py`, `docs/basic_usage/hy3_preview.md`; technical summary: Covers "support Hy3 preview"; the main implementation surface is `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py`, `docs/basic_usage/hy3_preview.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: _make_tools, TestHunyuanDetectorHasToolCall, setUp, test_has_tool_call_true, touching `_make_tools, TestHunyuanDetectorHasToolCall, setUp`; `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: HunyuanDetector, __init__, _normalize_type, _get_arg_schema, touching `HunyuanDetector, __init__, _normalize_type`; `docs/basic_usage/hy3_preview.md` added +191/-0 (191 lines); hunks: -0,0 +1,191.
- Code diff details:
  - `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: _make_tools, TestHunyuanDetectorHasToolCall, setUp, test_has_tool_call_true
  - `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: HunyuanDetector, __init__, _normalize_type, _get_arg_schema
  - `docs/basic_usage/hy3_preview.md` added +191/-0 (191 lines); hunks: -0,0 +1,191
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0
  - runtime: `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0
  - docs: `docs/basic_usage/hy3_preview.md` added +191/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `test/registered/unit/function_call/test_hunyuan_detector.py`, `test/registered/unit/parser/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
