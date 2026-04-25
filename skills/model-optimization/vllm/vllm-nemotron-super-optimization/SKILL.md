---
name: vllm-nemotron-super-optimization
description: PR-backed optimization manual for Nemotron Super / Nano Hybrid in vLLM. Use when Codex needs to audit, debug, extend, or document NemotronH, Nemotron 3 Super, Nemotron Nano hybrid Mamba+Attention+MoE, MTP, NVFP4, and VL adjacencies.
---

# vLLM Nemotron Super / Nano Hybrid Optimization

## Overview

This skill covers NemotronH, Nemotron 3 Super, Nemotron Nano hybrid Mamba+Attention+MoE, MTP, NVFP4, and VL adjacencies.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/nemotron-super/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/nemotron_h.py`
- `vllm/vllm/model_executor/models/nemotron_h_mtp.py`
- `vllm/vllm/model_executor/models/nano_nemotron_vl.py`
- `vllm/vllm/model_executor/models/nemotron_vl.py`

## Current Main Summary

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## Key Landed PRs

- [#18863](https://github.com/vllm-project/vllm/pull/18863) `NemotronH support`: Initial NemotronH landing in vLLM.
- [#25863](https://github.com/vllm-project/vllm/pull/25863) `Add MoE support for NemotronH`: Extended the hybrid family to routed experts.
- [#33726](https://github.com/vllm-project/vllm/pull/33726) `Nemotron-H MTP and Mamba Speculative Decoding Support`: Opened the MTP / spec-decode path.
- [#36803](https://github.com/vllm-project/vllm/pull/36803) `E2E Nemotron-3-Super tests`: Added direct Super-family regression coverage.
- [#37803](https://github.com/vllm-project/vllm/pull/37803) `Enable NemotronHPuzzle + NemotronHMTP`: Expanded hybrid and MTP coverage for the family.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
