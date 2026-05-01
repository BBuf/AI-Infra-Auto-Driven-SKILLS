---
name: vllm-gemma4-optimization
description: PR-backed optimization manual for Gemma 4 in vLLM. Use when an engineer needs to audit, debug, extend, or document Gemma 4 text, MoE, multimodal, reasoning, tool use, and quantized MoE serving.
---

# vLLM Gemma 4 Optimization

## Overview

This skill covers Gemma 4 text, MoE, multimodal, reasoning, tool use, and quantized MoE serving.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/gemma4/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/gemma4.py`
- `vllm/vllm/model_executor/models/gemma4_mm.py`

## Current Main Summary

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## Key Landed PRs

- [#38826](https://github.com/vllm-project/vllm/pull/38826) `Implement Google Gemma 4 architecture support`: Initial Gemma 4 text/MoE/multimodal landing.
- [#38879](https://github.com/vllm-project/vllm/pull/38879) `Enable Fast Prefill Optimization`: Added YOCO KV-sharing based fast prefill for Gemma4.
- [#39045](https://github.com/vllm-project/vllm/pull/39045) `Support quantized MoE`: Extended Gemma4 to quantized MoE checkpoints.
- [#38844](https://github.com/vllm-project/vllm/pull/38844) `Enable Gemma4ForCausalLM to load LoRA adapters correctly`: Fixed adapter naming/load behavior.
- [#39450](https://github.com/vllm-project/vllm/pull/39450) `Add Gemma4 Eagle3 support`: Enabled speculative decode for Gemma4.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
