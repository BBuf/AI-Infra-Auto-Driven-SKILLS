# sglang Hunyuan3 Preview Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` | [#23532](https://github.com/sgl-project/sglang/pull/23532), [#30201](https://github.com/sgl-project/sglang/pull/30201) |
| `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx` | [#30201](https://github.com/sgl-project/sglang/pull/30201) |
| `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` | [#23532](https://github.com/sgl-project/sglang/pull/23532) |
| `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx` | [#30201](https://github.com/sgl-project/sglang/pull/30201) |
| `docs_new/src/snippets/configs/tencent/hy3.jsx` | [#30201](https://github.com/sgl-project/sglang/pull/30201) |
| `python/sglang/srt/function_call/hunyuan_detector.py` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |
| `test/registered/unit/function_call/test_hunyuan_detector.py` | [#23533](https://github.com/sgl-project/sglang/pull/23533) |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 3
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-23 | [#23532](https://github.com/sgl-project/sglang/pull/23532) | merged | docs: add Hunyuan 3 Preview cookbook | `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` |
| 2026-04-24 | [#23533](https://github.com/sgl-project/sglang/pull/23533) | merged | support Hy3 preview | `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py` |
| 2026-07-06 | [#30201](https://github.com/sgl-project/sglang/pull/30201) | merged | cookbook: add Hunyuan 3 (Hy3) Day-0 page | `docs_new/src/snippets/configs/tencent/hy3.jsx`, `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx`, `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx` |

## Per-PR Diff Audit Cards

### PR #23532 - docs: add Hunyuan 3 Preview cookbook

- Link: https://github.com/sgl-project/sglang/pull/23532
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`; associated commits `4868e367f851`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +707/-0, 716 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: add Hunyuan 3 Preview cookbook"; model line: Hunyuan3 Preview; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`; technical summary: Covers "docs: add Hunyuan 3 Preview cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/hunyuan_detector.py`, `test/registered/unit/function_call/test_hunyuan_detector.py`; associated commits `6d038614760f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 25 files, +4095/-3, 4205 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support Hy3 preview"; model line: Hunyuan3 Preview; category: docs/tests/CI; main diff: `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py`; technical summary: Covers "support Hy3 preview"; the main implementation surface is `test/registered/unit/function_call/test_hunyuan_detector.py`, `python/sglang/srt/function_call/hunyuan_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: _make_tools, TestHunyuanDetectorHasToolCall, setUp, test_has_tool_call_true, touching `_make_tools, TestHunyuanDetectorHasToolCall, setUp`; `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: HunyuanDetector, __init__, _normalize_type, _get_arg_schema, touching `HunyuanDetector, __init__, _normalize_type`.
- Code diff details:
  - `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: _make_tools, TestHunyuanDetectorHasToolCall, setUp, test_has_tool_call_true
  - `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: HunyuanDetector, __init__, _normalize_type, _get_arg_schema
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
```

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_hunyuan_detector.py` added +733/-0
  - runtime: `python/sglang/srt/function_call/hunyuan_detector.py` added +476/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `test/registered/unit/function_call/test_hunyuan_detector.py`, `test/registered/unit/parser/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30201 - cookbook: add Hunyuan 3 (Hy3) Day-0 page

- Link: https://github.com/sgl-project/sglang/pull/30201
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx`, `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx`, `docs_new/src/snippets/configs/tencent/hy3.jsx`; associated commits `6f22790943a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +944/-2, 970 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "cookbook: add Hunyuan 3 (Hy3) Day-0 page"; model line: Hunyuan3 Preview; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/tencent/hy3.jsx`, `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx`, `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx`; technical summary: Covers "cookbook: add Hunyuan 3 (Hy3) Day-0 page"; the main implementation surface is `docs_new/src/snippets/configs/tencent/hy3.jsx`, `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx`, `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/tencent/hy3.jsx` added +546/-0 (546 lines); hunks: -0,0 +1,546; `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx` added +26/-0 (26 lines); hunks: -0,0 +1,26; `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx` added +370/-0 (370 lines); hunks: -0,0 +1,370; `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6.
- Code diff details:
  - `docs_new/src/snippets/configs/tencent/hy3.jsx` added +546/-0 (546 lines); hunks: -0,0 +1,546
  - `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx` added +26/-0 (26 lines); hunks: -0,0 +1,26
  - `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx` added +370/-0 (370 lines); hunks: -0,0 +1,370
  - `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/tencent/hy3.jsx
@@ -0,0 +1,546 @@
+// Hy3 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
+// see _deployment.jsx header for the field contract.
+//
+// The shipping Hy3 tokenizer appends a shared suffix to every special token
+// (e.g. <tool_calls:TAG>); SGLang's `hunyuan` reasoning/tool-call parsers
+// resolve the real token strings from the vocab at runtime (PR #29920), so the
diff -- docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx
@@ -0,0 +1,26 @@
+// Hy3 per-cell benchmark numbers, keyed by the same `match` tuple as hy3.jsx cells.
+// See _deployment.jsx for the speed/accuracy schema.
+// H200 BF16 low-latency + balanced verified on 8×H200 (sgl-eval, single-shot, temp=0).
+// FP8 cells not yet verified.
+export const benchmarks = [
+  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" }, gsm8k_pct: 95.75 },
diff -- docs_new/cookbook/autoregressive/Tencent/Hy3.mdx
@@ -0,0 +1,370 @@
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/tencent/hy3.jsx` added +546/-0; `docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx` added +26/-0; `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx` added +370/-0; `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx` modified +0/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/cookbook/autoregressive/Tencent/Hy3.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
