---
name: vllm-mimo-v2-flash-optimization
description: PR-backed optimization manual for MiMo-V2 / MiMo-V2-Flash / MiMo-V2.5 in vLLM. Use when an engineer needs to audit, debug, extend, or document MiMo-V2 inference-centric MoE runtime, MTP behavior, MiMo-V2.5 Pro/Omni support, and the transition from older MiMo checkpoints in vLLM.
---

# vLLM MiMo-V2 / MiMo-V2.5 Optimization

## Overview

This skill covers the MiMo-V2 family in vLLM: historical MiMo-V2-Flash,
current `mimo_v2.py` runtime, MTP behavior, MiMo-V2.5-Pro, MiMo-V2.5-Omni, and
the transition from older MiMo checkpoints.

Evidence snapshot:

- vLLM `origin/main`: `fd74c90d9` on `2026-04-27`
- Latest family update: `#40967` added MiMo-V2.5 Pro and Omni support, moving
  the active runtime surface to `mimo_v2.py`, `mimo_v2_mtp.py`, and
  `mimo_v2_omni.py` while retaining the historical Flash architecture alias.
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/mimo-v2-flash/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- Current runtime: `vllm/vllm/model_executor/models/mimo_v2.py`
- Current MTP runtime: `vllm/vllm/model_executor/models/mimo_v2_mtp.py`
- Current Omni runtime: `vllm/vllm/model_executor/models/mimo_v2_omni.py`
- Historical Flash runtime: `vllm/vllm/model_executor/models/mimo_v2_flash.py`
- `vllm/vllm/model_executor/models/mimo.py`
- `vllm/vllm/model_executor/models/mimo_mtp.py`

## Current Main Summary

- MiMo-V2-Flash is the historical throughput-oriented MoE landing in vLLM.
- Current mainline maps `MiMoV2FlashForCausalLM` and `MiMoV2ProForCausalLM` to
  `mimo_v2`, with MiMo-V2.5 Omni handled through `mimo_v2_omni`.
- MTP correctness, registry aliases, multimodal processor coverage, and the
  split between older MiMo checkpoints and V2.5 checkpoints are the key
  maintenance points.

## Key Landed PRs

- [#17433](https://github.com/vllm-project/vllm/pull/17433) `Support MiMo-7B inference with MTP`: Historical base for the MiMo family.
- [#25136](https://github.com/vllm-project/vllm/pull/25136) `Fix MTP inference path for MiMo-7B model`: Closed a concrete draft-path bug.
- [#30836](https://github.com/vllm-project/vllm/pull/30836) `Add MiMo-V2-Flash support`: Landed the dedicated V2-Flash runtime.
- [#40045](https://github.com/vllm-project/vllm/pull/40045) `use diff kv backend for mimo v2 flash`: Added the DiffKV attention backend path.
- [#40967](https://github.com/vllm-project/vllm/pull/40967) `Add MiMo-V2.5 support`: Added Pro, Omni, MTP, registry, and tests/docs coverage for the current family.

## Validation Lanes

- Startup on current MiMo-V2-Flash, MiMo-V2.5-Pro, and MiMo-V2.5-Omni checkpoints where access is available.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.

## References

- `references/pr-history.md`: diff-reviewed MiMo-V2 family PR cards; it includes historical Flash-file cards and current MiMo-V2.5 updates.
