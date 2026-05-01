---
name: vllm-deepseek-v3-r1-optimization
description: PR-backed optimization manual for DeepSeek V3 / R1 in vLLM. Use when an engineer needs to audit, debug, extend, or document DeepSeek V3 and DeepSeek R1 MLA, MoE, packed-module loading, LoRA, MTP/Eagle, and quantized ROCm/CUDA validation paths.
---

# vLLM DeepSeek V3 / R1 Optimization

## Overview

This skill covers DeepSeek V3 and DeepSeek R1 MLA, MoE, packed-module loading, LoRA, MTP/Eagle, and quantized ROCm/CUDA validation paths.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/deepseek-v3-r1/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/deepseek_eagle.py`
- `vllm/vllm/model_executor/models/deepseek_eagle3.py`
- `vllm/vllm/model_executor/models/deepseek_mtp.py`

## Current Main Summary

- `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` remain the shared runtime for V3 and R1.
- The highest-risk regressions cluster around packed module mapping, quantized MLA/MoE weight loading, LoRA, and MTP draft paths.
- R1 validation should split BF16, FP8/ModelOpt, and compressed-tensors or ROCm lanes.

## Key Landed PRs

- [#22352](https://github.com/vllm-project/vllm/pull/22352) `Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM``: Fixed quantized and packed-weight loading for DeepSeek V2/V3/R1 style checkpoints.
- [#23971](https://github.com/vllm-project/vllm/pull/23971) `Add LoRA support for DeepSeek models (V2, V3, R1-0528)`: Enabled adapter injection on the DeepSeek family rather than only base dense models.
- [#29545](https://github.com/vllm-project/vllm/pull/29545) `Fix DeepSeek R1 MTP weight loading`: Hardened R1 NextN / MTP draft loading after launch failures on draft weights.
- [#36247](https://github.com/vllm-project/vllm/pull/36247) `Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x`: Closed a production ROCm gap for compressed-tensors DeepSeek-R1 deployment.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- BF16 DeepSeek-V3 / R1 serve plus generation sanity.
- FP8 or compressed-tensors startup on Hopper/Blackwell and MI300X.
- MTP / Eagle draft-model acceptance tests.
- LoRA adapter load and unload on the same checkpoint.
