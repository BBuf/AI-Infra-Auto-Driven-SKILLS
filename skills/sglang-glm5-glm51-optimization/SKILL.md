---
name: sglang-glm5-glm51-optimization
description: PR-backed and current-main optimization manual for GLM-5 and GLM-5.1 in SGLang. Use when Codex needs to recover, extend, or audit GLM-5 DSA/NSA/NSA indexer paths, GLM-5.1 FP8/MXFP4/NVFP4, NextN/MTP, dense-attention threshold, NSA TileLang/AITER, tool templates, EAGLE, PCG, AMD/Blackwell/NPU validation, or GLM-5 cookbook recipes.
---

# SGLang GLM-5/5.1 Optimization

## Overview

GLM-5/5.1 is a separate lane because it moves from GLM4 MoE into GLM MoE DSA/NSA-adjacent behavior, NextN/MTP, FP8/MXFP4/NVFP4, NSA indexer work, and GLM-5.1 tool-template details.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Three-pass completeness audit: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`
- Runtime files: `glm4_moe.py`, `glm4_moe_nextn.py`, `deepseek_nextn.py` when shared MTP infrastructure is touched
- Docs/snippets: `GLM-5.mdx`, `GLM-5.1.mdx`, `glm-5-deployment.jsx`, `glm-51-deployment.jsx`
- Tests: `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/gb300/test_glm5_fp8.py`, `test/registered/gb300/test_glm5_nvfp4.py`

## Before You Change Anything

Capture:

- checkpoint: GLM-5, GLM-5-FP8, GLM-5-MXFP4, GLM-5.1-FP8, GLM-5.1-MXFP4/NVFP4
- whether DSA/NSA or dense-attention fallback is active
- MTP/NextN enabled or disabled
- KV cache dtype, NSA backend, and dense-attention threshold envs
- quantization exporter: FP8, MXFP4, NVFP4, Quark/ModelOpt/compressed-tensors
- EAGLE, PCG, DP attention, PD, or CUDA graph enabled
- backend: AMD MI300/MI325/MI355, B200/GB300, NPU

## Core Principle

GLM-5/5.1 should be debugged like a GLM-specific DSA/MTP system, not like plain GLM4.

- NSA indexer and dense-attention threshold changes may be shared with DeepSeek V3.2.
- MTP/NextN fixes can touch `deepseek_nextn.py` even when the user-facing model is GLM.
- GLM-5.1 template/tool-result behavior is user-visible and must be tested separately.

## Main Runtime Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/layers/attention/nsa/`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`
- `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`

## Optimization Order

1. Verify BF16/FP8 baseline with dense-attention threshold recorded.
2. Validate NSA indexer and backend selection.
3. Validate quantized loading and KV path.
4. Enable MTP/NextN and verify draft quant config.
5. Validate tool template and tool-result normalization.
6. Tune NSA/AMD/Blackwell kernels.
7. Add PCG/EAGLE only after base DSA/MTP is stable.

## Open PRs to Track

- [#20275](https://github.com/sgl-project/sglang/pull/20275): thinking budget logit processor.
- [#21332](https://github.com/sgl-project/sglang/pull/21332): TRTLLM MHA on B200.
- [#21529](https://github.com/sgl-project/sglang/pull/21529): ROCm MXFP4/Quark W4A4.
- [#21889](https://github.com/sgl-project/sglang/pull/21889): FP4 KV NSA TileLang.
- [#22409](https://github.com/sgl-project/sglang/pull/22409): GLM-5.1-MXFP4 MI30x/MI35x accuracy/perf.
- [#22473](https://github.com/sgl-project/sglang/pull/22473): dense MLA decode fallback.
- [#22488](https://github.com/sgl-project/sglang/pull/22488): GLM-5 256-expert JIT fused gate.
- [#22638](https://github.com/sgl-project/sglang/pull/22638): AITER decode backend auto-detect.
- [#22851](https://github.com/sgl-project/sglang/pull/22851): NSA top-k backend.
- [#22977](https://github.com/sgl-project/sglang/pull/22977): GLM5.1 tool-result template.
- [#23037](https://github.com/sgl-project/sglang/pull/23037): EAGLE CUDA graph IMA PD+DP+MTP for GLM-5.1.
- [#23346](https://github.com/sgl-project/sglang/pull/23346): decode state retract-resume for GLM-5.1.
- [#23351](https://github.com/sgl-project/sglang/pull/23351): PCG with NSA.

## Validation Lanes

- GLM-5 FP8/NVFP4 GB300 tests.
- GLM-5.1 FP8 8-GPU test.
- AMD GLM-5/5.1 FP8/MXFP4 accuracy and perf.
- MTP/NextN with draft quant config.
- Tool-template and tool-result normalization.
