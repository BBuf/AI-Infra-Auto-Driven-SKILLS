# vLLM Qwen3 Coder 支持与 PR 历史

本文记录 vLLM 中与 Qwen3 Coder 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持
- 该家族继承 `qwen3-core` 的基础 runtime，这里只记录增量 PR。

## 核心结论

- Qwen3 Coder inherits the base Qwen3 runtime and adds coder-specific tool parsing.
- The main regressions are in JSON-schema edge cases, anyOf / oneOf handling, and Responses API tools.

## 主要代码面

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/entrypoints/openai/tool_parsers/`

## 已合入 PR

- [#21396](https://github.com/vllm-project/vllm/pull/21396) `Add Qwen3CoderToolParser`：Created the dedicated coder-tool parser instead of reusing a generic Qwen parser.
- [#36032](https://github.com/vllm-project/vllm/pull/36032) `Fix anyOf double encoded parameters`：Fixed a concrete schema serialization bug in coder tool calls.
- [#37831](https://github.com/vllm-project/vllm/pull/37831) `Fix anyOf/oneOf type resolution for nullable params`：Improved nullable parameter handling in complex schemas.
- [#38848](https://github.com/vllm-project/vllm/pull/38848) `Fix Qwen3 tool parser for Responses API tools`：Aligned the tool parser with the Responses API tool surface.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen3-coder-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-coder-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3 Coder`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-22 | [#21396](https://github.com/vllm-project/vllm/pull/21396) | merged | [Model] Add Qwen3CoderToolParser | tests/benchmarks | `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2026-03-04 | [#36032](https://github.com/vllm-project/vllm/pull/36032) | merged | qwen3coder tool parser fix anyOf double encoded parameters | misc | `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-03-23 | [#37831](https://github.com/vllm-project/vllm/pull/37831) | merged | [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params | tests/benchmarks | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-04-02 | [#38848](https://github.com/vllm-project/vllm/pull/38848) | merged | [Bugfix] Fix Qwen3 tool parser for Responses API tools | tests/benchmarks | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py` |

### 逐 PR 代码 diff 阅读记录

### PR #21396 - [Model] Add Qwen3CoderToolParser

- 链接：https://github.com/vllm-project/vllm/pull/21396
- 状态/时间：`merged`，created 2025-07-22, merged 2025-07-22；作者 `ranpox`。
- 代码 diff 已读范围：`3` 个文件，`+1289/-0`；代码面：tests/benchmarks；关键词：config, fp8, moe, spec, test。
- 代码 diff 细节：
  - `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0 (669 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Qwen3CoderToolParser, __init__, _generate_tool_call_id, _reset_streaming_state
  - `tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0 (618 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: qwen3_tokenizer, qwen3_tool_parser, sample_tools, assert_tool_calls
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunk: from .mistral_tool_parser import MistralToolParser; "KimiK2ToolParser",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`；patch 关键词为 config, fp8, moe, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #36032 - qwen3coder tool parser fix anyOf double encoded parameters

- 链接：https://github.com/vllm-project/vllm/pull/36032
- 状态/时间：`merged`，created 2026-03-04, merged 2026-03-05；作者 `cmunley1`。
- 代码 diff 已读范围：`1` 个文件，`+6/-0`；代码面：misc；关键词：config。
- 代码 diff 细节：
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0 (6 lines); hunk: def _convert_param_value(; 符号: _convert_param_value
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/tool_parsers/qwen3coder_tool_parser.py`；patch 关键词为 config。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `vllm/tool_parsers/qwen3coder_tool_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37831 - [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params

- 链接：https://github.com/vllm-project/vllm/pull/37831
- 状态/时间：`merged`，created 2026-03-23, merged 2026-04-01；作者 `AAISSJ`。
- 代码 diff 已读范围：`2` 个文件，`+254/-14`；代码面：tests/benchmarks；关键词：config, spec, test。
- 代码 diff 细节：
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0 (202 lines); hunk: def test_extract_tool_calls_type_conversion(qwen3_tool_parser_parametrized):; 符号: test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14 (66 lines); hunk: def _get_arguments_config(; def _convert_param_value(; 符号: _get_arguments_config, _first_non_null_type, _resolve_param_type, _convert_param_value
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`；patch 关键词为 config, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #38848 - [Bugfix] Fix Qwen3 tool parser for Responses API tools

- 链接：https://github.com/vllm-project/vllm/pull/38848
- 状态/时间：`merged`，created 2026-04-02, merged 2026-04-08；作者 `sfeng33`。
- 代码 diff 已读范围：`4` 个文件，`+99/-113`；代码面：tests/benchmarks；关键词：config, spec, test。
- 代码 diff 细节：
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55 (128 lines); hunk: from collections.abc import Generator; def qwen3_tool_parser_parametrized(qwen3_tool_parser, qwen3_xml_tool_parser, req; 符号: qwen3_tool_parser_parametrized, sample_tools, sample_tools, assert_tool_calls
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31 (37 lines); hunk: Tool,; def _reset_streaming_state(self):; 符号: _reset_streaming_state, _get_arguments_config, _convert_param_value, _convert_param_value
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27 (33 lines); hunk: Tool,; def _get_param_type(self, param_name: str) -> str:; 符号: _get_param_type, repair_param_type
  - `vllm/tool_parsers/utils.py` modified +14/-0 (14 lines); hunk: def _extract_tool_info(; 符号: _extract_tool_info, find_tool_properties, _get_tool_schema_from_tool
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py`；patch 关键词为 config, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：4；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
