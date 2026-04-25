# vllm Qwen3 Coder 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/pooling/test_colqwen3.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/colqwen3.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/qwen3.py` | 无直接 PR 号提交 |
| `vllm/transformers_utils/configs/colqwen3.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 4
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-22 | [#21396](https://github.com/vllm-project/vllm/pull/21396) | merged | [Model] Add Qwen3CoderToolParser | `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2026-03-05 | [#36032](https://github.com/vllm-project/vllm/pull/36032) | merged | qwen3coder tool parser fix anyOf double encoded parameters | `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-04-01 | [#37831](https://github.com/vllm-project/vllm/pull/37831) | merged | [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-04-08 | [#38848](https://github.com/vllm-project/vllm/pull/38848) | merged | [Bugfix] Fix Qwen3 tool parser for Responses API tools | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py` |

## 逐 PR diff 审计卡

### PR #21396 - [Model] Add Qwen3CoderToolParser

- 链接: https://github.com/vllm-project/vllm/pull/21396
- 状态/时间: merged / 2025-07-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+1289/-0，可读 patch 1303 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen3 Coder 补齐模型支持入口或运行时能力，标题为「[Model] Add Qwen3CoderToolParser」，变更集中在 `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`。PR 描述补充为：Edit from @simon-mo Tested locally for both tool use example and unit test.
- 实现要点: `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0 (669 lines); hunks: -0,0 +1,669; symbols: Qwen3CoderToolParser, __init__, _generate_tool_call_id, _reset_streaming_state，涉及 `Qwen3CoderToolParser, __init__, _generate_tool_call_id`；`tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: qwen3_tokenizer, qwen3_tool_parser, sample_tools, assert_tool_calls，涉及 `qwen3_tokenizer, qwen3_tool_parser, sample_tools`；`vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunks: -17,6 +17,7; -38,4 +39,5。
- 代码 diff 细节:
  - `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0 (669 lines); hunks: -0,0 +1,669; symbols: Qwen3CoderToolParser, __init__, _generate_tool_call_id, _reset_streaming_state
  - `tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: qwen3_tokenizer, qwen3_tool_parser, sample_tools, assert_tool_calls
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunks: -17,6 +17,7; -38,4 +39,5
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py
@@ -0,0 +1,669 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import uuid
+from collections.abc import Sequence
+from typing import Any, Optional, Union
diff -- tests/tool_use/test_qwen3coder_tool_parser.py
@@ -0,0 +1,618 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+from collections.abc import Generator
+from typing import Optional
+import pytest
diff -- vllm/entrypoints/openai/tool_parsers/__init__.py
@@ -17,6 +17,7 @@
```

- 已读文件:
  - runtime: `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0
  - tests: `tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0
- 验证与风险: diff 自带测试面 `tests/tool_use/test_qwen3coder_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36032 - qwen3coder tool parser fix anyOf double encoded parameters

- 链接: https://github.com/vllm-project/vllm/pull/36032
- 状态/时间: merged / 2026-03-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-0，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen3 Coder 修复已暴露的启动、加载、解析或数值问题，标题为「qwen3coder tool parser fix anyOf double encoded parameters」，变更集中在 `vllm/tool_parsers/qwen3coder_tool_parser.py`。PR 描述补充为：## Problem When a tool parameter uses anyOf instead of an explicit type, _convert_param_value falls back to param_type = "string" because anyOf schemas have no top-level "type"...
- 实现要点: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0 (6 lines); hunks: -157,6 +157,12 @@ def _convert_param_value(; symbols: _convert_param_value，涉及 `_convert_param_value`。
- 代码 diff 细节:
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0 (6 lines); hunks: -157,6 +157,12 @@ def _convert_param_value(; symbols: _convert_param_value
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/qwen3coder_tool_parser.py
@@ -157,6 +157,12 @@ def _convert_param_value(
+        elif (
+            isinstance(param_config[param_name], dict)
+            and "anyOf" in param_config[param_name]
+        ):
+            # anyOf has no top-level "type"; treat as object to trigger json.loads.
+            param_type = "object"
```

- 已读文件:
  - runtime: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/qwen3coder_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37831 - [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params

- 链接: https://github.com/vllm-project/vllm/pull/37831
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+254/-14，可读 patch 293 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen3 Coder 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params」，变更集中在 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`。PR 描述补充为：## Purpose Fix incorrect type resolution for `anyOf`/`oneOf` schemas, type-as-array patterns, and `$ref` schemas in `Qwen3CoderToolParser._convert_param_value`. The previous fix...
- 实现要点: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0 (202 lines); hunks: -430,6 +430,208 @@ def test_extract_tool_calls_type_conversion(qwen3_tool_par...; symbols: test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming，涉及 `test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming`；`vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14 (66 lines); hunks: -133,11 +133,58 @@ def _get_arguments_config(; -152,19 +199,10 @@ def _convert_param_value(; symbols: _get_arguments_config, _first_non_null_type, _resolve_param_type, _convert_param_value，涉及 `_get_arguments_config, _first_non_null_type, _resolve_param_type`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0 (202 lines); hunks: -430,6 +430,208 @@ def test_extract_tool_calls_type_conversion(qwen3_tool_par...; symbols: test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14 (66 lines); hunks: -133,11 +133,58 @@ def _get_arguments_config(; -152,19 +199,10 @@ def _convert_param_value(; symbols: _get_arguments_config, _first_non_null_type, _resolve_param_type, _convert_param_value
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_qwen3coder_tool_parser.py
@@ -430,6 +430,208 @@ def test_extract_tool_calls_type_conversion(qwen3_tool_parser_parametrized):
+def test_extract_tool_calls_anyof_type_conversion(qwen3_tool_parser):
+    """Test type conversion for anyOf/oneOf nullable schemas (Pydantic v2).
+    Pydantic v2 emits anyOf for Optional[T] fields, e.g.:
+        Optional[int] -> {"anyOf": [{"type": "integer"}, {"type": "null"}]}
+    The parser must extract the non-null type and apply the correct
+    conversion (int(), float(), etc.) instead of returning a raw string.
diff -- vllm/tool_parsers/qwen3coder_tool_parser.py
@@ -133,11 +133,58 @@ def _get_arguments_config(
+    @staticmethod
+    def _first_non_null_type(type_value: Any) -> str | None:
+        """Extract the first non-null type from a type value.
+        Handles both scalar types ("integer") and type-as-array
+        (["integer", "null"]) per JSON Schema spec.
+        """
```

- 已读文件:
  - tests: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0
  - runtime: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_qwen3coder_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38848 - [Bugfix] Fix Qwen3 tool parser for Responses API tools

- 链接: https://github.com/vllm-project/vllm/pull/38848
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+99/-113，可读 patch 425 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen3 Coder 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Qwen3 tool parser for Responses API tools」，变更集中在 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py`。PR 描述补充为：## Purpose - Both `Qwen3CoderToolParser` and `Qwen3XMLToolParser` assumed all tools have a `.function.name` / `.function.parameters` structure (`ChatCompletionToolsParam`). Resp...
- 实现要点: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55 (128 lines); hunks: -5,6 +5,7; -49,41 +50,62 @@ def qwen3_tool_parser_parametrized(qwen3_tool_parser, qwen3_...; symbols: qwen3_tool_parser_parametrized, sample_tools, assert_tool_calls, test_extract_tool_calls_no_tools，涉及 `qwen3_tool_parser_parametrized, sample_tools, assert_tool_calls`；`vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31 (37 lines); hunks: -25,6 +25,7; -109,28 +110,6 @@ def _reset_streaming_state(self):; symbols: _reset_streaming_state, _get_arguments_config, _convert_param_value, _parse_xml_function_call，涉及 `_reset_streaming_state, _get_arguments_config, _convert_param_value`；`vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27 (33 lines); hunks: -26,6 +26,7; -1000,33 +1001,11 @@ def _get_param_type(self, param_name: str) -> str:; symbols: _get_param_type, repair_param_type，涉及 `_get_param_type, repair_param_type`；`vllm/tool_parsers/utils.py` modified +14/-0 (14 lines); hunks: -142,6 +142,20 @@ def _extract_tool_info(; symbols: _extract_tool_info, find_tool_properties, _get_tool_schema_from_tool，涉及 `_extract_tool_info, find_tool_properties, _get_tool_schema_from_tool`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55 (128 lines); hunks: -5,6 +5,7; -49,41 +50,62 @@ def qwen3_tool_parser_parametrized(qwen3_tool_parser, qwen3_...; symbols: qwen3_tool_parser_parametrized, sample_tools, assert_tool_calls, test_extract_tool_calls_no_tools
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31 (37 lines); hunks: -25,6 +25,7; -109,28 +110,6 @@ def _reset_streaming_state(self):; symbols: _reset_streaming_state, _get_arguments_config, _convert_param_value, _parse_xml_function_call
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27 (33 lines); hunks: -26,6 +26,7; -1000,33 +1001,11 @@ def _get_param_type(self, param_name: str) -> str:; symbols: _get_param_type, repair_param_type
  - `vllm/tool_parsers/utils.py` modified +14/-0 (14 lines); hunks: -142,6 +142,20 @@ def _extract_tool_info(; symbols: _extract_tool_info, find_tool_properties, _get_tool_schema_from_tool
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_qwen3coder_tool_parser.py
@@ -5,6 +5,7 @@
+from openai.types.responses.function_tool import FunctionTool
@@ -49,41 +50,62 @@ def qwen3_tool_parser_parametrized(qwen3_tool_parser, qwen3_xml_tool_parser, req
-@pytest.fixture
-def sample_tools():
-    return [
-        ChatCompletionToolsParam(
diff -- vllm/tool_parsers/qwen3coder_tool_parser.py
@@ -25,6 +25,7 @@
+from vllm.tool_parsers.utils import find_tool_properties
@@ -109,28 +110,6 @@ def _reset_streaming_state(self):
-    def _get_arguments_config(self, func_name: str, tools: list[Tool] | None) -> dict:
-        """Extract argument configuration for a function."""
-        if tools is None:
-            return {}
diff -- vllm/tool_parsers/qwen3xml_tool_parser.py
@@ -26,6 +26,7 @@
```

- 已读文件:
  - tests: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55
  - runtime: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27; `vllm/tool_parsers/utils.py` modified +14/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_qwen3coder_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
