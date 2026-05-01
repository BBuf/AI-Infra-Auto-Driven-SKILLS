---
name: vllm-gpt-oss-optimization
description: PR-backed optimization manual for GPT-OSS in vLLM. Use when an engineer needs to audit, debug, extend, or document OpenAI GPT-OSS MoE, MXFP4/FP8 quantization, DP/EP, reasoning parser, tool calling, and Eagle/spec decode.
---

# vLLM GPT-OSS Optimization

## Overview

This skill covers OpenAI GPT-OSS MoE, MXFP4/FP8 quantization, DP/EP, reasoning parser, tool calling, and Eagle/spec decode.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/gpt-oss/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/gpt_oss.py`

## Current Main Summary

- GPT-OSS is a flagship MoE family in vLLM.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## Key Landed PRs

- [#22327](https://github.com/vllm-project/vllm/pull/22327) `Add GPT-OSS model code and config`: Initial GPT-OSS landing in vLLM.
- [#23819](https://github.com/vllm-project/vllm/pull/23819) `Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE`: Opened large-scale GPT-OSS serving topologies.
- [#25246](https://github.com/vllm-project/vllm/pull/25246) `Enable Eagle3 speculative decoding for GPT-OSS model`: Added draft-model acceleration.
- [#25515](https://github.com/vllm-project/vllm/pull/25515) `Structure_Tag support for gpt-oss tool-call in cot`: Improved tool calling in reasoning-mode outputs.
- [#30647](https://github.com/vllm-project/vllm/pull/30647) `Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE`: Targeted the hot MXFP4/MXFP8 path for throughput.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
