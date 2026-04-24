---
name: vllm-llama4-optimization
description: PR-backed optimization manual for Llama 4 in vLLM. Use when Codex needs to audit, debug, extend, or document Llama4 text and multimodal runtime, FP8/FP4 quantization, router behavior, long-context attention, and Eagle support.
---

# vLLM Llama 4 Optimization

## Overview

This skill covers Llama4 text and multimodal runtime, FP8/FP4 quantization, router behavior, long-context attention, and Eagle support.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/llama4/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/llama4.py`
- `vllm/vllm/model_executor/models/mllama4.py`
- `vllm/vllm/model_executor/models/llama4_eagle.py`

## Current Main Summary

- Llama4 is mature on the vLLM side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## Key Landed PRs

- [#16104](https://github.com/vllm-project/vllm/pull/16104) `Support Llama4 in vLLM`: Initial Llama4 landing.
- [#20419](https://github.com/vllm-project/vllm/pull/20419) `Enable ModelOpt Llama4 fp8 checkpoint deployment`: Added ModelOpt FP8 coverage.
- [#20591](https://github.com/vllm-project/vllm/pull/20591) `Llama4 EAGLE Support`: Opened speculative decoding for Llama4.
- [#22511](https://github.com/vllm-project/vllm/pull/22511) `Fix Llama4 FlashInfer FP4 MoE issues`: Stabilized the FP4 MoE path.
- [#25889](https://github.com/vllm-project/vllm/pull/25889) `Fix misplaced dtype cast in Llama4VisionRotaryEmbedding`: Patched a multimodal rotary bug.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
