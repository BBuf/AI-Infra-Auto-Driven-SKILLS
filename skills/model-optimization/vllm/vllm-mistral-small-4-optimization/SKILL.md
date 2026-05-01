---
name: vllm-mistral-small-4-optimization
description: PR-backed optimization manual for Mistral Small 4 in vLLM. Use when an engineer needs to audit, debug, extend, or document Mistral Small 4, Leanstral, and closely related Mistral Large 3 / Ministral serving behavior, including multimodal and MoE execution.
---

# vLLM Mistral Small 4 Optimization

## Overview

This skill covers Mistral Small 4, Leanstral, and closely related Mistral Large 3 / Ministral serving behavior, including multimodal and MoE execution.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/mistral-small-4/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/mistral_large_3.py`
- `vllm/vllm/model_executor/models/mistral_large_3_eagle.py`
- `vllm/vllm/model_executor/models/mistral3.py`

## Current Main Summary

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- MoE execution and multimodal projector behavior are the main risk areas.

## Key Landed PRs

- [#29757](https://github.com/vllm-project/vllm/pull/29757) `Add Mistral Large 3 and Ministral 3`: Landed the runtime family that Mistral Small 4 deployments build on in vLLM.
- [#33174](https://github.com/vllm-project/vllm/pull/33174) `Add support for Mistral Large 3 inference with Flashinfer MoE`: Improved the practical MoE serving path for the same family.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
