---
name: vllm-glm45-optimization
description: PR-backed optimization manual for GLM-4.5 / 4.5V in vLLM. Use when an engineer needs to audit, debug, extend, or document GLM-4.5 text, GLM-4.5V, GLM-4.5-Air, shared MoE routing, and tool/reasoning parser behavior in vLLM.
---

# vLLM GLM-4.5 / 4.5V Optimization

## Overview

This skill covers GLM-4.5 text, GLM-4.5V, GLM-4.5-Air, shared MoE routing, and tool/reasoning parser behavior in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/glm45/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## Current Main Summary

- The GLM-4.5 lane is where vLLM reorganized the GLM family around text, MoE, and vision variants.
- Most regressions are in MoE gate behavior, tie-word-embedding policy, and vendor-specific fused MoE tuning.

## Key Landed PRs

- [#22171](https://github.com/vllm-project/vllm/pull/22171) `Modify the organization of GLM series`: Reworked the family layout so 4.5-era models reused a cleaner GLM structure.
- [#22460](https://github.com/vllm-project/vllm/pull/22460) `not tie_word_embeddings for glm-4.5 and glm-4.5v`: Aligned the loader with the real 4.5 checkpoint contract instead of forcing tied embeddings.
- [#22832](https://github.com/vllm-project/vllm/pull/22832) `Modify the gate implementation of glm4_moe`: Changed the GLM4.5 MoE gating path used by text and VL variants.
- [#23695](https://github.com/vllm-project/vllm/pull/23695) `Add triton fused moe config for GLM-4.5-Air-FP8 on B200`: Added a production kernel-tuning lane for the 4.5 Air FP8 deployment path.
- [#24589](https://github.com/vllm-project/vllm/pull/24589) `Add documentation for GLM-4.5 series tool-calling and reasoning parser`: Codified the parser choices needed to serve 4.5 reasoning / tool checkpoints correctly.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- GLM-4.5 and GLM-4.5-Air text generation.
- GLM-4.5V startup with image inputs.
- Tool calling and reasoning parser smoke tests.
- FP8 fused-MoE launch on Blackwell where relevant.
