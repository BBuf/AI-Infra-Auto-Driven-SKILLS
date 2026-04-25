---
name: vllm-qwen3-coder-optimization
description: PR-backed optimization manual for Qwen3 Coder in vLLM. Use when Codex needs to audit, debug, extend, or document Qwen3 Coder tool parser, structured tool arguments, and coding-oriented parser behavior layered on top of the base Qwen3 runtime.
---

# vLLM Qwen3 Coder Optimization

## Overview

This skill covers Qwen3 Coder tool parser, structured tool arguments, and coding-oriented parser behavior layered on top of the base Qwen3 runtime.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Family baseline: read `qwen3-core` first; this dossier only records the delta for `qwen3-coder`.
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/qwen3-coder/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/entrypoints/openai/tool_parsers/`

## Current Main Summary

- Qwen3 Coder inherits the base Qwen3 runtime and adds coder-specific tool parsing.
- The main regressions are in JSON-schema edge cases, anyOf / oneOf handling, and Responses API tools.

## Key Landed PRs

- [#21396](https://github.com/vllm-project/vllm/pull/21396) `Add Qwen3CoderToolParser`: Created the dedicated coder-tool parser instead of reusing a generic Qwen parser.
- [#36032](https://github.com/vllm-project/vllm/pull/36032) `Fix anyOf double encoded parameters`: Fixed a concrete schema serialization bug in coder tool calls.
- [#37831](https://github.com/vllm-project/vllm/pull/37831) `Fix anyOf/oneOf type resolution for nullable params`: Improved nullable parameter handling in complex schemas.
- [#38848](https://github.com/vllm-project/vllm/pull/38848) `Fix Qwen3 tool parser for Responses API tools`: Aligned the tool parser with the Responses API tool surface.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Complex tool schema extraction with anyOf / oneOf / nullable parameters.
- Responses API tool execution on coder checkpoints.
- Streaming tool-call integrity under speculative decode.
