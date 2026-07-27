# vllm Hunyuan3 Preview Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/reasoning/test_hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `tests/tool_parsers/test_hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3_mtp.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/reasoning/hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681), [#47192](https://github.com/vllm-project/vllm/pull/47192) |
| `vllm/tool_parsers/hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681), [#47192](https://github.com/vllm-project/vllm/pull/47192) |
| `vllm/transformers_utils/configs/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 2
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-23 | [#40681](https://github.com/vllm-project/vllm/pull/40681) | merged | [Model] Support Hy3 preview | `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py` |
| 2026-07-01 | [#47192](https://github.com/vllm-project/vllm/pull/47192) | merged | [Model] Support Hy3 token suffix and JSON Schema array types | `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/reasoning/hy_v3_reasoning_parser.py` |

## Per-PR Diff Audit Cards

### PR #40681 - [Model] Support Hy3 preview

- Link: https://github.com/vllm-project/vllm/pull/40681
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3.py`, `vllm/model_executor/models/hy_v3_mtp.py`, `vllm/reasoning/hy_v3_reasoning_parser.py` and 7 files; associated commits `d0009ddb0b96`
- Diff scope read: GitHub Pull Request files API returned 16 files, +2696/-0, 2801 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Hy3 preview"; model line: Hunyuan3 Preview; category: model support/runtime entry; main diff: `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py`; technical summary: Covers "[Model] Support Hy3 preview"; the main implementation surface is `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused, touching `HYV3FeedForward, __init__, forward`; `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options, touching `HYV3ToolParser, _normalize_type, _get_arg_schema`; `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__, touching `_is_moe, _get_cla_factor, HYV3SharedHead`; `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls, touching `hy_v3_tokenizer, hy_v3_tool_parser, mock_request`.
- Code diff details:
  - `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused
  - `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options
  - `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__
  - `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls
  - `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: hy_v3_tokenizer, test_reasoning, test_is_reasoning_end_full_prompt
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/hy_v3.py
@@ -0,0 +1,707 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# coding=utf-8
+# Copyright 2026 The HY team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -0,0 +1,645 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import ast
+import json
+from collections.abc import Sequence
+from typing import Any
diff -- vllm/model_executor/models/hy_v3_mtp.py
@@ -0,0 +1,470 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/hy_v3.py` added +707/-0; `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0; `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0; `vllm/transformers_utils/configs/hy_v3.py` added +185/-0; `vllm/reasoning/hy_v3_reasoning_parser.py` added +137/-0
  - tests: `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0; `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #47192 - [Model] Support Hy3 token suffix and JSON Schema array types

- Link: https://github.com/vllm-project/vllm/pull/47192
- Status/date: merged / 2026-07-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/hy_v3_reasoning_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`; associated commits `cc56379e28a0`
- Diff scope read: GitHub Pull Request files API returned 2 files, +24/-11, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Hy3 token suffix and JSON Schema array types"; model line: Hunyuan3 Preview; category: bug fix; main diff: `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/reasoning/hy_v3_reasoning_parser.py`; technical summary: Covers "[Model] Support Hy3 token suffix and JSON Schema array types"; the main implementation surface is `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/reasoning/hy_v3_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/hy_v3_tool_parser.py` modified +20/-9 (29 lines); hunks: -108,6 +108,14 @@ def _get_schema_options(arg_schema: dict) -> list[dict]:; -261,19 +269,22 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: _get_schema_options, __init__, touching `_get_schema_options, __init__`; `vllm/reasoning/hy_v3_reasoning_parser.py` modified +4/-2 (6 lines); hunks: -26,6 +26,8 @@ class HYV3ReasoningParser(BaseThinkingReasoningParser):; -52,12 +54,12 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; symbols: HYV3ReasoningParser, __init__, start_token, end_token, touching `HYV3ReasoningParser, __init__, start_token`.
- Code diff details:
  - `vllm/tool_parsers/hy_v3_tool_parser.py` modified +20/-9 (29 lines); hunks: -108,6 +108,14 @@ def _get_schema_options(arg_schema: dict) -> list[dict]:; -261,19 +269,22 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: _get_schema_options, __init__
  - `vllm/reasoning/hy_v3_reasoning_parser.py` modified +4/-2 (6 lines); hunks: -26,6 +26,8 @@ class HYV3ReasoningParser(BaseThinkingReasoningParser):; -52,12 +54,12 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; symbols: HYV3ReasoningParser, __init__, start_token, end_token
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -108,6 +108,14 @@ def _get_schema_options(arg_schema: dict) -> list[dict]:
+            type_val = arg_schema["type"]
+            # JSON Schema allows "type" to be an array to represent union types,
+            # e.g. "type": ["string", "object"].
+            # Expand it into an anyOf-equivalent format:
+            #   [{"type": "string"}, {"type": "object"}]
+            # so that _get_types / _parse_value can handle it uniformly later.
diff -- vllm/reasoning/hy_v3_reasoning_parser.py
@@ -26,6 +26,8 @@ class HYV3ReasoningParser(BaseThinkingReasoningParser):
+        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
+        self.suffix: str = init_kwargs.get("token_suffix") or ""
@@ -52,12 +54,12 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
-        return "<think>"
+        return f"<think{self.suffix}>"
-        return "</think>"
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/hy_v3_tool_parser.py` modified +20/-9; `vllm/reasoning/hy_v3_reasoning_parser.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/hy_v3_reasoning_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
