---
name: vllm-ernie45-optimization
description: PR-backed optimization manual for Ernie4.5 / Ernie4.5-VL in vLLM. Use when Codex needs to audit, debug, extend, or document Baidu Ernie4.5 text/VL/MoE runtime, vision rotary, and long-input stability.
---

# vLLM Ernie4.5 / Ernie4.5-VL Optimization

## Overview

This skill covers Baidu Ernie4.5 text/VL/MoE runtime, vision rotary, and long-input stability.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/ernie45/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/ernie45.py`
- `vllm/vllm/model_executor/models/ernie45_moe.py`
- `vllm/vllm/model_executor/models/ernie45_vl.py`

## Current Main Summary

- Ernie4.5 spans dense, MoE, and VL paths in vLLM.
- The highest-risk work items are shared-expert behavior, VL rotary/timestamp logic, and long-input stability.

## Key Landed PRs

- [#20220](https://github.com/vllm-project/vllm/pull/20220) `Add Ernie4.5 and Ernie4.5MoE Model Support`: Landed text and MoE support.
- [#21717](https://github.com/vllm-project/vllm/pull/21717) `Fix Ernie4_5_MoeForCausalLM shared experts`: Fixed shared-expert correctness.
- [#22514](https://github.com/vllm-project/vllm/pull/22514) `Add Ernie4.5 VL Model Support`: Added the multimodal Ernie4.5-VL lane.
- [#24074](https://github.com/vllm-project/vllm/pull/24074) `Fix Ernie4.5-VL hanging on long inputs`: Closed a production long-input stall.
- [#31274](https://github.com/vllm-project/vllm/pull/31274) `Support video metadata for timestamp rendering`: Improved VL video output fidelity.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
