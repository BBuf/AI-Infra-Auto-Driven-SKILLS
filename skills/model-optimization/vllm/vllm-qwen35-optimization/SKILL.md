---
name: vllm-qwen35-optimization
description: PR-backed optimization manual for Qwen3.5 in vLLM. Use when Codex needs to audit, debug, extend, or document Qwen3.5 dense / MoE / GDN runtime, MTP, FP8 and NVFP4 quantization, LoRA, and Eagle3 in vLLM.
---

# vLLM Qwen3.5 Optimization

## Overview

This skill covers Qwen3.5 dense / MoE / GDN runtime, MTP, FP8 and NVFP4 quantization, LoRA, and Eagle3 in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/qwen35/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3_5.py`
- `vllm/vllm/model_executor/models/qwen3_5_mtp.py`

## Current Main Summary

- Qwen3.5 builds on the Qwen3-Next era work but has its own model registration and quantization details.
- The hot spots are GDN fusion, FP8/NVFP4 loading, LoRA target naming, and MoE EP precision.

## Key Landed PRs

- [#34110](https://github.com/vllm-project/vllm/pull/34110) `Adding Support for Qwen3.5 Models`: Landed the Qwen3.5 runtime family.
- [#34697](https://github.com/vllm-project/vllm/pull/34697) `Redo Qwen3.5/Qwen3-Next GDN projector fusion`: Reworked an earlier fusion that had to be reverted.
- [#35289](https://github.com/vllm-project/vllm/pull/35289) `Fix Qwen3.5 FP8 quantization tuple shard_id weight loading`: Closed a concrete FP8 weight-loading failure.
- [#36658](https://github.com/vllm-project/vllm/pull/36658) `Add Eagle3 support for Qwen3.5`: Enabled the draft-model fast path.
- [#37975](https://github.com/vllm-project/vllm/pull/37975) `Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5`: Reduced duplicated GDN logic across related families.
- [#39181](https://github.com/vllm-project/vllm/pull/39181) `Fix EP precision for Qwen3.5, Qwen3-Next`: Patched a serving-precision bug under expert parallelism.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- BF16 and FP8 Qwen3.5 serve.
- NVFP4 or MXFP4 quantized startup.
- LoRA and MTP paths.
- Expert-parallel precision regression checks.
