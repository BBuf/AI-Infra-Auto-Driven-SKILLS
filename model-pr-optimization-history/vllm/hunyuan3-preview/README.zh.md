# vllm Hunyuan3 Preview 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/reasoning/test_hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `tests/tool_parsers/test_hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3_mtp.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/reasoning/hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681), [#47192](https://github.com/vllm-project/vllm/pull/47192) |
| `vllm/tool_parsers/hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681), [#47192](https://github.com/vllm-project/vllm/pull/47192) |
| `vllm/transformers_utils/configs/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 2
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-23 | [#40681](https://github.com/vllm-project/vllm/pull/40681) | merged | [Model] Support Hy3 preview | `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py` |
| 2026-07-01 | [#47192](https://github.com/vllm-project/vllm/pull/47192) | merged | [Model] Support Hy3 token suffix and JSON Schema array types | `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/reasoning/hy_v3_reasoning_parser.py` |

## 逐 PR diff 审计卡

### PR #40681 - [Model] Support Hy3 preview

- 链接: https://github.com/vllm-project/vllm/pull/40681
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3.py`, `vllm/model_executor/models/hy_v3_mtp.py`, `vllm/reasoning/hy_v3_reasoning_parser.py` 等 7 个文件；关联提交 `d0009ddb0b96`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+2696/-0，可读 patch 2801 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Hy3 preview」；模型线: Hunyuan3 Preview；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py`；技术摘要: 覆盖「[Model] Support Hy3 preview」；主要实现面是 `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused，涉及 `HYV3FeedForward, __init__, forward`；`vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options，涉及 `HYV3ToolParser, _normalize_type, _get_arg_schema`；`vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__，涉及 `_is_moe, _get_cla_factor, HYV3SharedHead`；`tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls，涉及 `hy_v3_tokenizer, hy_v3_tool_parser, mock_request`。
- 代码 diff 细节:
  - `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused
  - `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options
  - `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__
  - `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls
  - `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: hy_v3_tokenizer, test_reasoning, test_is_reasoning_end_full_prompt
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/hy_v3.py` added +707/-0; `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0; `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0; `vllm/transformers_utils/configs/hy_v3.py` added +185/-0; `vllm/reasoning/hy_v3_reasoning_parser.py` added +137/-0
  - tests: `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0; `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #47192 - [Model] Support Hy3 token suffix and JSON Schema array types

- 链接: https://github.com/vllm-project/vllm/pull/47192
- 状态/时间: merged / 2026-07-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/reasoning/hy_v3_reasoning_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`；关联提交 `cc56379e28a0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+24/-11，可读 patch 71 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Hy3 token suffix and JSON Schema array types」；模型线: Hunyuan3 Preview；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/reasoning/hy_v3_reasoning_parser.py`；技术摘要: 覆盖「[Model] Support Hy3 token suffix and JSON Schema array types」；主要实现面是 `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/reasoning/hy_v3_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/hy_v3_tool_parser.py` modified +20/-9 (29 lines); hunks: -108,6 +108,14 @@ def _get_schema_options(arg_schema: dict) -> list[dict]:; -261,19 +269,22 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: _get_schema_options, __init__，涉及 `_get_schema_options, __init__`；`vllm/reasoning/hy_v3_reasoning_parser.py` modified +4/-2 (6 lines); hunks: -26,6 +26,8 @@ class HYV3ReasoningParser(BaseThinkingReasoningParser):; -52,12 +54,12 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; symbols: HYV3ReasoningParser, __init__, start_token, end_token，涉及 `HYV3ReasoningParser, __init__, start_token`。
- 代码 diff 细节:
  - `vllm/tool_parsers/hy_v3_tool_parser.py` modified +20/-9 (29 lines); hunks: -108,6 +108,14 @@ def _get_schema_options(arg_schema: dict) -> list[dict]:; -261,19 +269,22 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: _get_schema_options, __init__
  - `vllm/reasoning/hy_v3_reasoning_parser.py` modified +4/-2 (6 lines); hunks: -26,6 +26,8 @@ class HYV3ReasoningParser(BaseThinkingReasoningParser):; -52,12 +54,12 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; symbols: HYV3ReasoningParser, __init__, start_token, end_token
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/hy_v3_tool_parser.py` modified +20/-9; `vllm/reasoning/hy_v3_reasoning_parser.py` modified +4/-2
- 验证与风险: runtime 路径改动集中在 `vllm/reasoning/hy_v3_reasoning_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
