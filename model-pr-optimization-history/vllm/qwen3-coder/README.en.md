# vllm Qwen3 Coder Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/models/multimodal/pooling/test_colqwen3.py` | no direct PR-number commit |
| `vllm/model_executor/models/colqwen3.py` | no direct PR-number commit |
| `vllm/model_executor/models/qwen3.py` | no direct PR-number commit |
| `vllm/transformers_utils/configs/colqwen3.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 0
- Extra PRs preserved from existing docs: 4
- Total PRs in this document: 4
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-07-22 | [#21396](https://github.com/vllm-project/vllm/pull/21396) | merged | [Model] Add Qwen3CoderToolParser | `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2026-03-05 | [#36032](https://github.com/vllm-project/vllm/pull/36032) | merged | qwen3coder tool parser fix anyOf double encoded parameters | `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-04-01 | [#37831](https://github.com/vllm-project/vllm/pull/37831) | merged | [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-04-08 | [#38848](https://github.com/vllm-project/vllm/pull/38848) | merged | [Bugfix] Fix Qwen3 tool parser for Responses API tools | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py` |

## Per-PR Diff Audit Cards

### PR #21396 - [Model] Add Qwen3CoderToolParser

- Link: https://github.com/vllm-project/vllm/pull/21396
- Status/date: merged / 2025-07-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +1289/-0, 1303 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Qwen3CoderToolParser"; model line: Qwen3 Coder; category: docs/tests/CI; main diff: `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`; technical summary: Covers "[Model] Add Qwen3CoderToolParser"; the main implementation surface is `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0 (669 lines); hunks: -0,0 +1,669; symbols: Qwen3CoderToolParser, __init__, _generate_tool_call_id, _reset_streaming_state, touching `Qwen3CoderToolParser, __init__, _generate_tool_call_id`; `tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: qwen3_tokenizer, qwen3_tool_parser, sample_tools, assert_tool_calls, touching `qwen3_tokenizer, qwen3_tool_parser, sample_tools`; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunks: -17,6 +17,7; -38,4 +39,5.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0 (669 lines); hunks: -0,0 +1,669; symbols: Qwen3CoderToolParser, __init__, _generate_tool_call_id, _reset_streaming_state
  - `tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: qwen3_tokenizer, qwen3_tool_parser, sample_tools, assert_tool_calls
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunks: -17,6 +17,7; -38,4 +39,5
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0
  - tests: `tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_qwen3coder_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36032 - qwen3coder tool parser fix anyOf double encoded parameters

- Link: https://github.com/vllm-project/vllm/pull/36032
- Status/date: merged / 2026-03-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-0, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "qwen3coder tool parser fix anyOf double encoded parameters"; model line: Qwen3 Coder; category: bug fix; main diff: `vllm/tool_parsers/qwen3coder_tool_parser.py`; technical summary: Covers "qwen3coder tool parser fix anyOf double encoded parameters"; the main implementation surface is `vllm/tool_parsers/qwen3coder_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0 (6 lines); hunks: -157,6 +157,12 @@ def _convert_param_value(; symbols: _convert_param_value, touching `_convert_param_value`.
- Code diff details:
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0 (6 lines); hunks: -157,6 +157,12 @@ def _convert_param_value(; symbols: _convert_param_value
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/qwen3coder_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37831 - [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params

- Link: https://github.com/vllm-project/vllm/pull/37831
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +254/-14, 293 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params"; model line: Qwen3 Coder; category: bug fix; main diff: `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params"; the main implementation surface is `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0 (202 lines); hunks: -430,6 +430,208 @@ def test_extract_tool_calls_type_conversion(qwen3_tool_par...; symbols: test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming, touching `test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming`; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14 (66 lines); hunks: -133,11 +133,58 @@ def _get_arguments_config(; -152,19 +199,10 @@ def _convert_param_value(; symbols: _get_arguments_config, _first_non_null_type, _resolve_param_type, _convert_param_value, touching `_get_arguments_config, _first_non_null_type, _resolve_param_type`.
- Code diff details:
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0 (202 lines); hunks: -430,6 +430,208 @@ def test_extract_tool_calls_type_conversion(qwen3_tool_par...; symbols: test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14 (66 lines); hunks: -133,11 +133,58 @@ def _get_arguments_config(; -152,19 +199,10 @@ def _convert_param_value(; symbols: _get_arguments_config, _first_non_null_type, _resolve_param_type, _convert_param_value
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0
  - runtime: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_qwen3coder_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38848 - [Bugfix] Fix Qwen3 tool parser for Responses API tools

- Link: https://github.com/vllm-project/vllm/pull/38848
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +99/-113, 425 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3 tool parser for Responses API tools"; model line: Qwen3 Coder; category: bug fix; main diff: `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Qwen3 tool parser for Responses API tools"; the main implementation surface is `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55 (128 lines); hunks: -5,6 +5,7; -49,41 +50,62 @@ def qwen3_tool_parser_parametrized(qwen3_tool_parser, qwen3_...; symbols: qwen3_tool_parser_parametrized, sample_tools, assert_tool_calls, test_extract_tool_calls_no_tools, touching `qwen3_tool_parser_parametrized, sample_tools, assert_tool_calls`; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31 (37 lines); hunks: -25,6 +25,7; -109,28 +110,6 @@ def _reset_streaming_state(self):; symbols: _reset_streaming_state, _get_arguments_config, _convert_param_value, _parse_xml_function_call, touching `_reset_streaming_state, _get_arguments_config, _convert_param_value`; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27 (33 lines); hunks: -26,6 +26,7; -1000,33 +1001,11 @@ def _get_param_type(self, param_name: str) -> str:; symbols: _get_param_type, repair_param_type, touching `_get_param_type, repair_param_type`; `vllm/tool_parsers/utils.py` modified +14/-0 (14 lines); hunks: -142,6 +142,20 @@ def _extract_tool_info(; symbols: _extract_tool_info, find_tool_properties, _get_tool_schema_from_tool, touching `_extract_tool_info, find_tool_properties, _get_tool_schema_from_tool`.
- Code diff details:
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55 (128 lines); hunks: -5,6 +5,7; -49,41 +50,62 @@ def qwen3_tool_parser_parametrized(qwen3_tool_parser, qwen3_...; symbols: qwen3_tool_parser_parametrized, sample_tools, assert_tool_calls, test_extract_tool_calls_no_tools
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31 (37 lines); hunks: -25,6 +25,7; -109,28 +110,6 @@ def _reset_streaming_state(self):; symbols: _reset_streaming_state, _get_arguments_config, _convert_param_value, _parse_xml_function_call
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27 (33 lines); hunks: -26,6 +26,7; -1000,33 +1001,11 @@ def _get_param_type(self, param_name: str) -> str:; symbols: _get_param_type, repair_param_type
  - `vllm/tool_parsers/utils.py` modified +14/-0 (14 lines); hunks: -142,6 +142,20 @@ def _extract_tool_info(; symbols: _extract_tool_info, find_tool_properties, _get_tool_schema_from_tool
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55
  - runtime: `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27; `vllm/tool_parsers/utils.py` modified +14/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_qwen3coder_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
