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
