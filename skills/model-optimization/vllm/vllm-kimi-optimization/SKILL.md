---
name: vllm-kimi-optimization
description: PR-backed optimization manual for Kimi K2 / K2.5 / Linear / Audio / VL in vLLM. Use when an engineer needs to audit, debug, extend, or document Kimi-VL, Kimi-Linear, Kimi-K2.5, Kimi-Audio, parser aliases, and quantized MLA behavior in vLLM.
---

# vLLM Kimi K2 / K2.5 / Linear / Audio / VL Optimization

## Overview

This skill covers Kimi-VL, Kimi-Linear, Kimi-K2.5, Kimi-Audio, parser aliases, and quantized MLA behavior in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/kimi/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/kimi_vl.py`
- `vllm/vllm/model_executor/models/kimi_linear.py`
- `vllm/vllm/model_executor/models/kimi_k25.py`
- `vllm/vllm/model_executor/models/kimi_audio.py`

## Current Main Summary

- The Kimi family in vLLM spans vision, linear-attention, K2.5, and audio checkpoints.
- The most fragile areas are MLA plus FP8/NVFP4 loading, processor evolution, and parser alias compatibility between K2 and K2.5.

## Key Landed PRs

- [#16387](https://github.com/vllm-project/vllm/pull/16387) `Add Kimi-VL model support`: Landed the original Kimi-VL multimodal runtime.
- [#27809](https://github.com/vllm-project/vllm/pull/27809) `Introduce Kimi Linear to vLLM`: Added the linear-attention Kimi family instead of only the VL path.
- [#33131](https://github.com/vllm-project/vllm/pull/33131) `Kimi-K2.5`: Brought the K2.5 generation into mainline.
- [#33876](https://github.com/vllm-project/vllm/pull/33876) `Fix Kimi-K2.5 NVFP4 checkpoints weight loading`: Closed a concrete launch blocker for quantized K2.5 checkpoints.
- [#36127](https://github.com/vllm-project/vllm/pull/36127) `Add support for moonshotai/Kimi-Audio-7B-Instruct`: Extended the family to audio-conditioned serving.
- [#37438](https://github.com/vllm-project/vllm/pull/37438) `Add Kimi-K2.5 reasoning/tool parser aliases`: Aligned parser aliases and tool-call IDs with the newer model outputs.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- K2.5 text and VL launch.
- Kimi-Linear long-context inference.
- Kimi-Audio processor and transcription path.
- NVFP4 or FP8 MLA startup on K2.5.
