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
- Three-pass completeness audit: `model-pr-optimization-history/sglang/audits/qwen-glm-three-pass-audit-2026-04-23.md`
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

## PR Dossier Rule

Before adding GLM-5/5.1 PR evidence, open the PR diff/source and write a full card in `references/pr-history.md` with motivation, key implementation, code excerpt, reviewed files, and validation implications.

Do not add one-line open PR radar lists. Open PRs can be recorded only after their diffs are reviewed and they are clearly separated from merged history.

## Validation Lanes

- GLM-5 FP8/NVFP4 GB300 tests.
- GLM-5.1 FP8 8-GPU test.
- AMD GLM-5/5.1 FP8/MXFP4 accuracy and perf.
- MTP/NextN with draft quant config.
- Tool-template and tool-result normalization.
