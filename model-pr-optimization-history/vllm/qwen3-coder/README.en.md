# vLLM Qwen3 Coder Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen3 Coder.

- Status: supported on current mainline
- This family inherits the base runtime from `qwen3-core` and only records the delta here.

## Key Conclusions

- Qwen3 Coder inherits the base Qwen3 runtime and adds coder-specific tool parsing.
- The main regressions are in JSON-schema edge cases, anyOf / oneOf handling, and Responses API tools.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/entrypoints/openai/tool_parsers/`

## Landed PRs

- [#21396](https://github.com/vllm-project/vllm/pull/21396) `Add Qwen3CoderToolParser`: Created the dedicated coder-tool parser instead of reusing a generic Qwen parser.
- [#36032](https://github.com/vllm-project/vllm/pull/36032) `Fix anyOf double encoded parameters`: Fixed a concrete schema serialization bug in coder tool calls.
- [#37831](https://github.com/vllm-project/vllm/pull/37831) `Fix anyOf/oneOf type resolution for nullable params`: Improved nullable parameter handling in complex schemas.
- [#38848](https://github.com/vllm-project/vllm/pull/38848) `Fix Qwen3 tool parser for Responses API tools`: Aligned the tool parser with the Responses API tool surface.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen3-coder-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-coder-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3 Coder` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-22 | [#21396](https://github.com/vllm-project/vllm/pull/21396) | merged | [Model] Add Qwen3CoderToolParser | tests/benchmarks | `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2026-03-04 | [#36032](https://github.com/vllm-project/vllm/pull/36032) | merged | qwen3coder tool parser fix anyOf double encoded parameters | misc | `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-03-23 | [#37831](https://github.com/vllm-project/vllm/pull/37831) | merged | [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params | tests/benchmarks | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py` |
| 2026-04-02 | [#38848](https://github.com/vllm-project/vllm/pull/38848) | merged | [Bugfix] Fix Qwen3 tool parser for Responses API tools | tests/benchmarks | `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py` |

### File-level PR diff reading notes

### PR #21396 - [Model] Add Qwen3CoderToolParser

- Link: https://github.com/vllm-project/vllm/pull/21396
- Status/date: `merged`, created 2025-07-22, merged 2025-07-22; author `ranpox`.
- Diff scope read: `3` files, `+1289/-0`; areas: tests/benchmarks; keywords: config, fp8, moe, spec, test.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py` added +669/-0 (669 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Qwen3CoderToolParser, __init__, _generate_tool_call_id, _reset_streaming_state
  - `tests/tool_use/test_qwen3coder_tool_parser.py` added +618/-0 (618 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: qwen3_tokenizer, qwen3_tool_parser, sample_tools, assert_tool_calls
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunks: from .mistral_tool_parser import MistralToolParser; "KimiK2ToolParser",
- Optimization/support interpretation: The concrete diff surface is `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`; keywords observed in patches: config, fp8, moe, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`, `tests/tool_use/test_qwen3coder_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #36032 - qwen3coder tool parser fix anyOf double encoded parameters

- Link: https://github.com/vllm-project/vllm/pull/36032
- Status/date: `merged`, created 2026-03-04, merged 2026-03-05; author `cmunley1`.
- Diff scope read: `1` files, `+6/-0`; areas: misc; keywords: config.
- Code diff details:
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-0 (6 lines); hunks: def _convert_param_value(; symbols: _convert_param_value
- Optimization/support interpretation: The concrete diff surface is `vllm/tool_parsers/qwen3coder_tool_parser.py`; keywords observed in patches: config. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `vllm/tool_parsers/qwen3coder_tool_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37831 - [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params

- Link: https://github.com/vllm-project/vllm/pull/37831
- Status/date: `merged`, created 2026-03-23, merged 2026-04-01; author `AAISSJ`.
- Diff scope read: `2` files, `+254/-14`; areas: tests/benchmarks; keywords: config, spec, test.
- Code diff details:
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +202/-0 (202 lines); hunks: def test_extract_tool_calls_type_conversion(qwen3_tool_parser_parametrized):; symbols: test_extract_tool_calls_type_conversion, test_extract_tool_calls_anyof_type_conversion, test_extract_tool_calls_anyof_type_conversion_streaming
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +52/-14 (66 lines); hunks: def _get_arguments_config(; def _convert_param_value(; symbols: _get_arguments_config, _first_non_null_type, _resolve_param_type, _convert_param_value
- Optimization/support interpretation: The concrete diff surface is `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`; keywords observed in patches: config, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #38848 - [Bugfix] Fix Qwen3 tool parser for Responses API tools

- Link: https://github.com/vllm-project/vllm/pull/38848
- Status/date: `merged`, created 2026-04-02, merged 2026-04-08; author `sfeng33`.
- Diff scope read: `4` files, `+99/-113`; areas: tests/benchmarks; keywords: config, spec, test.
- Code diff details:
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +73/-55 (128 lines); hunks: from collections.abc import Generator; def qwen3_tool_parser_parametrized(qwen3_tool_parser, qwen3_xml_tool_parser, req; symbols: qwen3_tool_parser_parametrized, sample_tools, sample_tools, assert_tool_calls
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +6/-31 (37 lines); hunks: Tool,; def _reset_streaming_state(self):; symbols: _reset_streaming_state, _get_arguments_config, _convert_param_value, _convert_param_value
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +6/-27 (33 lines); hunks: Tool,; def _get_param_type(self, param_name: str) -> str:; symbols: _get_param_type, repair_param_type
  - `vllm/tool_parsers/utils.py` modified +14/-0 (14 lines); hunks: def _extract_tool_info(; symbols: _extract_tool_info, find_tool_properties, _get_tool_schema_from_tool
- Optimization/support interpretation: The concrete diff surface is `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py`; keywords observed in patches: config, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/tool_parsers/test_qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 4; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
