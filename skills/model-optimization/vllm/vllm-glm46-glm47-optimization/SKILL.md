---
name: vllm-glm46-glm47-optimization
description: PR-backed optimization manual for GLM-4.6 / 4.7 in vLLM. Use when an engineer needs to audit, debug, extend, or document GLM-4.6, GLM-4.6V, GLM-4.7, GLM-4.7-Flash, GLM-Lite, and the parser / quant / fused-MoE deltas after the 4.5 generation.
---

# vLLM GLM-4.6 / 4.7 Optimization

## Overview

This skill covers GLM-4.6, GLM-4.6V, GLM-4.7, GLM-4.7-Flash, GLM-Lite, and the parser / quant / fused-MoE deltas after the 4.5 generation.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/glm46-glm47/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## Current Main Summary

- The 4.6/4.7 generation mainly extends the 4.5 base with new tuning tables, parser behavior, and Lite variants.
- AWQ / Marlin compatibility and content-normalization in tool parsing are the recurring pitfalls.

## Key Landed PRs

- [#26818](https://github.com/vllm-project/vllm/pull/26818) `Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on B200`: Added fused-MoE tuning configs for the new Blackwell deployment lane.
- [#30210](https://github.com/vllm-project/vllm/pull/30210) `Fix glm46 awq marlin moe compatibility`: Closed an incompatibility between GLM-4.6 AWQ checkpoints and Marlin MoE assumptions.
- [#30876](https://github.com/vllm-project/vllm/pull/30876) `GLM-4.7 Tool Parser and Doc Update`: Brought parser behavior and docs up to date for 4.7 / 4.7-Flash.
- [#31386](https://github.com/vllm-project/vllm/pull/31386) `GLM Model support for GLM-Lite`: Extended the same runtime family to the Lite checkpoint line.
- [#37386](https://github.com/vllm-project/vllm/pull/37386) `Improve tool call parsing and content normalization for glm47`: Fixed concrete parsing errors that surfaced in newer GLM-4.7 outputs.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- GLM-4.6 text and vision startup.
- GLM-4.7 tool-call extraction and streaming normalization.
- AWQ / Marlin launch for GLM-4.6 MoE.
