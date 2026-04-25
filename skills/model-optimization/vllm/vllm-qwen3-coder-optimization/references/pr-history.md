# vLLM Qwen3 Coder PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Qwen3 Coder tool parser, structured tool arguments, and coding-oriented parser behavior layered on top of the base Qwen3 runtime.

This family inherits the base runtime context from `qwen3-core`; this file records only the delta that is specific to `qwen3-coder`.

## Landed PRs

### PR #21396 - Add Qwen3CoderToolParser

- Link: https://github.com/vllm-project/vllm/pull/21396
- Why it mattered: Created the dedicated coder-tool parser instead of reusing a generic Qwen parser.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/entrypoints/openai/tool_parsers/
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36032 - Fix anyOf double encoded parameters

- Link: https://github.com/vllm-project/vllm/pull/36032
- Why it mattered: Fixed a concrete schema serialization bug in coder tool calls.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/entrypoints/openai/tool_parsers/
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37831 - Fix anyOf/oneOf type resolution for nullable params

- Link: https://github.com/vllm-project/vllm/pull/37831
- Why it mattered: Improved nullable parameter handling in complex schemas.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/entrypoints/openai/tool_parsers/
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #38848 - Fix Qwen3 tool parser for Responses API tools

- Link: https://github.com/vllm-project/vllm/pull/38848
- Why it mattered: Aligned the tool parser with the Responses API tool surface.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/entrypoints/openai/tool_parsers/
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
