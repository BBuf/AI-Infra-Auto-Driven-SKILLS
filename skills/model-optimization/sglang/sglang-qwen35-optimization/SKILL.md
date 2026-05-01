---
name: sglang-qwen35-optimization
description: PR-backed and current-main optimization manual for Qwen3.5 in SGLang. Use when an engineer needs to recover, extend, or audit Qwen3.5 dense/MoE, Qwen3.5 FP8/NVFP4/MXFP4, MTP, GDN projection, PP, EPLB, AMD/NPU/Blackwell deployments, FP8 KV caution paths, or Qwen3.5 cookbook recipes.
---

# SGLang Qwen3.5 Optimization

## Overview

Qwen3.5 is a separate optimization lane because SGLang carries dedicated model files, MTP files, quantized checkpoint support, PP fixes, GDN projection fusion, and platform-specific tests.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Runtime files: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`
- Config: `python/sglang/srt/configs/qwen3_5.py`
- Cookbook/docs: `docs/basic_usage/qwen3_5.md`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- checkpoint: dense, MoE, FP8, NVFP4, MXFP4, or compressed-tensors variant
- whether MTP is enabled
- TP/EP/PP/DP shape and whether EPLB is active
- KV cache dtype and any FP8 KV caution note
- all-reduce backend, FlashInfer all-reduce fusion, and MoE backend
- hardware lane: H200, B200/GB200/GB300, AMD MI300/MI325/MI355, NPU, CPU/XPU

## Core Principle

Qwen3.5 is a high-risk loader and topology model.

- Most regressions come from weight-loader mapping, PP layer splitting, tied embeddings, quantization guards, or MTP cache behavior.
- Performance work is concentrated around GDN projection fusion, shared-expert fusion, MoE configs, all-reduce, and backend-specific quant kernels.
- Cookbook recipes are active and should be kept in sync with SGLang docs when server defaults change.

## Main Runtime Surfaces

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen3_5_mtp.py`
- `python/sglang/srt/configs/qwen3_5.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`
- `test/registered/8-gpu-models/test_qwen35.py`
- `test/registered/gb300/test_qwen35_fp8.py`
- `test/registered/gb300/test_qwen35_nvfp4.py`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`

## Optimization Order

1. Prove BF16 or reference checkpoint correctness.
2. Confirm weight-loader mapping, tied embeddings, and PP split rules.
3. Validate quantized loading under the exact target exporter.
4. Add MTP and EPLB compatibility tests.
5. Tune GDN projections and shared experts.
6. Update registered tests and cookbook configs for each hardware lane.

## PR Dossier Rule

Before adding or updating any Qwen3.5 PR entry, open the PR diff/source and then write a complete card in `references/pr-history.md`:

- motivation/root cause
- key implementation idea
- concrete code snippet copied from the reviewed diff
- reviewed files
- validation implication

Do not add one-line open PR lists to this skill. If a PR is still open, keep it in a clearly marked open optimization section only after reading its diff, as done for `#23474`.

## Validation Lanes

- 4-GPU FP4/MTP and Triton tests.
- 8-GPU Qwen3.5 test.
- GB300 FP8 and NVFP4 tests.
- AMD MI300/MI325/MI355 accuracy and perf tests.
- Cookbook command generation for H200/B200/AMD.
