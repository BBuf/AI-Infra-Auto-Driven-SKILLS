---
name: vllm-deepseek-v31-optimization
description: PR-backed optimization manual for DeepSeek V3.1 in vLLM. Use when an engineer needs to audit, debug, extend, or document DeepSeek V3.1 parser, scale-format, DeepGEMM, and reasoning-tooling deltas layered on top of the base DeepSeek V3 runtime.
---

# vLLM DeepSeek V3.1 Optimization

## Overview

This skill covers DeepSeek V3.1 parser, scale-format, DeepGEMM, and reasoning-tooling deltas layered on top of the base DeepSeek V3 runtime.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Family baseline: read `deepseek-v3-r1` first; this dossier only records the delta for `deepseek-v31`.
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/deepseek-v31/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`

## Current Main Summary

- V3.1 mostly reuses the base V3 runtime and adds parser plus scale-format correctness work.
- The practical blast radius is in tool calling, DeepGEMM scale handling, and reasoning-parser behavior.

## Key Landed PRs

- [#23454](https://github.com/vllm-project/vllm/pull/23454) `Support DeepSeek-V3.1 tool call`: Added the first V3.1-specific tool-call parser surface to vLLM.
- [#23666](https://github.com/vllm-project/vllm/pull/23666) `Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt`: Tuned the scale-format path used by DeepGEMM-based DeepSeek V3.1 kernels.
- [#25589](https://github.com/vllm-project/vllm/pull/25589) `Add DeepSeek-V3.1 reasoning parser`: Separated V3.1 reasoning output handling from generic DeepSeek parsing.
- [#32361](https://github.com/vllm-project/vllm/pull/32361) `Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes`: Patched a concrete shape mismatch between newer checkpoints and DeepGEMM assumptions.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Tool-call extraction on non-streaming and streaming outputs.
- DeepGEMM-backed launch with V3.1 scale-format checkpoints.
- Reasoning parser output separation under long responses.
