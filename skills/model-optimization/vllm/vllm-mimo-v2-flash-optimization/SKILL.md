---
name: vllm-mimo-v2-flash-optimization
description: PR-backed optimization manual for MiMo-V2-Flash in vLLM. Use when Codex needs to audit, debug, extend, or document MiMo-V2-Flash inference-centric MoE runtime, MTP behavior, and the transition from older MiMo checkpoints in vLLM.
---

# vLLM MiMo-V2-Flash Optimization

## Overview

This skill covers MiMo-V2-Flash inference-centric MoE runtime, MTP behavior, and the transition from older MiMo checkpoints in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/mimo-v2-flash/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/mimo_v2_flash.py`
- `vllm/vllm/model_executor/models/mimo.py`
- `vllm/vllm/model_executor/models/mimo_mtp.py`

## Current Main Summary

- MiMo-V2-Flash is a throughput-oriented MoE serving family in vLLM.
- MTP correctness and the split between older MiMo checkpoints and V2-Flash are the key maintenance points.

## Key Landed PRs

- [#17433](https://github.com/vllm-project/vllm/pull/17433) `Support MiMo-7B inference with MTP`: Historical base for the MiMo family.
- [#25136](https://github.com/vllm-project/vllm/pull/25136) `Fix MTP inference path for MiMo-7B model`: Closed a concrete draft-path bug.
- [#30836](https://github.com/vllm-project/vllm/pull/30836) `Add MiMo-V2-Flash support`: Landed the dedicated V2-Flash runtime.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
