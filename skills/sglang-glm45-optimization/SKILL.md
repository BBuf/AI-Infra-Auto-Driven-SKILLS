---
name: sglang-glm45-optimization
description: PR-backed and current-main optimization manual for GLM-4.5 and GLM-4.5 Air/MoE in SGLang. Use when Codex needs to recover, extend, or audit GLM-4.5 MoE loading, A2A/DeepEP, reduce-scatter behavior, NVFP4 padding, tool parser behavior, AMD/NPU/Blackwell validation, or GLM-4.5 cookbook recipes.
---

# SGLang GLM-4.5 Optimization

## Overview

GLM-4.5 is the first GLM lane because it introduced the GLM MoE support and later GLM-4.6/4.7/5.x work inherits many model, parser, quantization, and platform decisions from it.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Three-pass completeness audit: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`
- Runtime files: `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`
- Parser files: `python/sglang/srt/function_call/glm4_moe_detector.py`
- Docs/snippets: `docs/basic_usage/glm45.md`, `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx`, `glm-45-deployment.jsx`

## Before You Change Anything

Capture:

- checkpoint: GLM-4.5, GLM-4.5-Air, GLM-4.5-FP8/NVFP4, or GLM-4.5V if the issue is actually VLM
- TP/EP/PP/DP shape and A2A backend
- `--enable-deepep-moe`, reduce-scatter settings, and MoE backend
- quantization: BF16, FP8, NVFP4, compressed-tensors
- tool parser and streaming mode
- backend: NVIDIA, AMD ROCm, NPU, CPU/XPU

## Core Principle

Treat GLM-4.5 as the GLM MoE baseline.

- A2A initialization, reduce-scatter, MoE block layout, and NVFP4 padding are foundational for later GLM models.
- Tool-parser fixes in GLM-4.5 can affect GLM-4.6/4.7 parser behavior.
- VLM issues should move to `sglang-glm-vlm-ocr-optimization`.

## Main Runtime Surfaces

- `python/sglang/srt/models/glm4.py`
- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `docs/basic_usage/glm45.md`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx`
- `docs_new/src/snippets/autoregressive/glm-45-deployment.jsx`

## Optimization Order

1. Verify BF16 GLM-4.5 MoE correctness.
2. Check A2A backend initialization and reduce-scatter flags.
3. Validate quantized weight-scale padding and compressed-tensors metadata.
4. Validate tool parser with streaming chunks.
5. Tune MoE backend configs per hardware.
6. Update cookbook commands and registered tests.

## Open PRs to Track

- [#13711](https://github.com/sgl-project/sglang/pull/13711): fused-MoE TP2 RTX Pro 6000 configs for GLM-4.5-Air/4.5V.
- [#19106](https://github.com/sgl-project/sglang/pull/19106): GLM4 MoE Lite compressed-tensors support.
- [#19728](https://github.com/sgl-project/sglang/pull/19728): ROCm GLM-4.5V-FP8 startup, VLM-specific but uses adjacent GLM4 MoE code.
- [#20917](https://github.com/sgl-project/sglang/pull/20917): `enable_thinking` for Qwen3/GLM45.
- [#23067](https://github.com/sgl-project/sglang/pull/23067): GLM45 detector `continue_final_message`.

## Validation Lanes

- GLM-4.5 BF16 and quantized startup.
- MoE A2A/reduce-scatter matrix.
- Tool parser streaming.
- AMD/NPU docs and cookbook commands.
