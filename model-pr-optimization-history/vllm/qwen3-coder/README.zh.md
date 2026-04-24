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
