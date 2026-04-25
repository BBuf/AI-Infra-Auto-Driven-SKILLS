---
name: sglang-llama4-optimization
description: PR-backed optimization manual for Llama 4 in SGLang. Use when Codex needs to audit, debug, extend, or document Llama4 text and multimodal runtime, FP8/FP4 quantization, router behavior, long-context attention, and Eagle support.
---

# SGLang Llama 4 Optimization

## Overview

This skill covers Llama4 text and multimodal runtime, FP8/FP4 quantization, router behavior, long-context attention, and Eagle support.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/llama4/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/llama4.py`
- `sglang/python/sglang/srt/models/mllama4.py`

## Current Main Summary

- Llama4 is mature on the SGLang side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## Key Landed PRs

- [#5092](https://github.com/sgl-project/sglang/pull/5092) `Add Llama4 support`: Initial Llama4 landing in SGLang.
- [#5194](https://github.com/sgl-project/sglang/pull/5194) `Support Llama4 fp8 inference`: Enabled the first production quantized lane.
- [#6162](https://github.com/sgl-project/sglang/pull/6162) `Fix Llama4 gibberish output with long context and CUDA graph`: Closed a major correctness bug.
- [#7129](https://github.com/sgl-project/sglang/pull/7129) `Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang`: Added the ModelOpt checkpoint path.
- [#13421](https://github.com/sgl-project/sglang/pull/13421) `Add Llama4 attention backend auto-selection`: Stabilized backend choice for real deployments.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
