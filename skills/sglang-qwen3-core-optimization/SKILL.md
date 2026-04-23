---
name: sglang-qwen3-core-optimization
description: PR-backed and current-main optimization manual for Qwen3 dense and Qwen3 MoE in SGLang. Use when Codex needs to recover, extend, or audit Qwen3/Qwen3-30B/Qwen3-235B-A22B, FP8/NVFP4/MXFP4/W4A4, fused QK-norm/RoPE/KV-store paths, FlashInfer TRTLLM-GEN-MoE, context parallel, EAGLE3, LoRA, NPU/AMD/Blackwell validation, or Qwen3 reasoning/tool-parser behavior.
---

# SGLang Qwen3 Core Optimization

## Overview

This skill covers the non-hybrid Qwen3 text path: dense Qwen3, Qwen3 MoE, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Instruct/Thinking variants, and the shared Qwen3 model infrastructure that later Qwen3.5/Qwen3-Next/Qwen3.6 work builds on.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Three-pass completeness audit: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`
- Runtime files: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`
- Parser/config files: `python/sglang/srt/function_call/qwen25_detector.py`, `python/sglang/srt/configs/qwen3_5.py` when shared with later variants
- Cookbook/docs: `docs/basic_usage/qwen3.md`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`

The historical evidence lives in:

- [references/pr-history.md](references/pr-history.md)
- [references/playbook.md](references/playbook.md)

## Before You Change Anything

Record the serving shape first:

- exact checkpoint and architecture: dense Qwen3 or Qwen3 MoE
- TP/DP/EP/PP/attention-CP topology
- quantization: BF16, FP8, NVFP4, MXFP4, W4A4, GPTQ, or ModelOpt export
- attention backend: FlashInfer, TRTLLM MHA, Triton, NPU custom backend, or AMD AITER path
- MoE backend: fused MoE, TRTLLM-GEN-MoE, DeepEP, AITER, or fallback
- parser flags: `--reasoning-parser qwen3`, `--tool-call-parser qwen25` or a downstream Qwen parser
- LoRA, EAGLE3, context parallel, or radix/prefix cache enabled or not
- hardware: H100/H200, B200/GB200/GB300, AMD MI300/MI325/MI355, Ascend NPU, Intel XPU, or CPU

## Core Principle

Treat Qwen3 core as the baseline compatibility layer.

- Qwen3 dense and Qwen3 MoE carry the common RoPE, QK-norm, MLP/MoE, embedding, LM-head, reasoning parser, and many quantization pathways.
- Qwen3.5 and Qwen3-Next reuse a large amount of Qwen3-era validation: PP splitting, tied embeddings, RoPE config fallback, fused QK norm, FP8/NVFP4 loading, EAGLE3 and platform tests.
- Bugs in Qwen3 core often surface later as Qwen3.5/Qwen3.6 regressions, so read this history before duplicating a fix in a newer model file.

## Main Runtime Surfaces

Start from these files:

- `python/sglang/srt/models/qwen3.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/models/qwen2.py`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/configs/`
- `python/sglang/srt/layers/quantization/`
- `python/sglang/srt/function_call/qwen25_detector.py`
- `test/registered/models/test_qwen_models.py`
- `test/registered/4-gpu-models/test_qwen3_30b.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- `test/srt/models/test_lora_qwen3.py`

## Optimization Order

1. Confirm the model class and config detection before touching kernels.
2. Stabilize correctness under BF16 and the smallest TP shape.
3. Add quantized loading checks for FP8/NVFP4/MXFP4/W4A4.
4. Tune fused QK-norm/RoPE/KV-store only after RoPE parameters and QK-norm shapes are stable.
5. Tune MoE backend configs per GPU generation.
6. Add LoRA, EAGLE3, CP, or platform-specific logic only after baseline decode and prefill match reference output.
7. Update cookbook commands and CI tests in the same model lane.

## Open PRs to Track

Check these before declaring a Qwen3 gap fixed or unsupported:

- [#22674](https://github.com/sgl-project/sglang/pull/22674): NPU Qwen3.5-MoE and Qwen3-Next quantization, touches shared Qwen quant logic.
- [#20474](https://github.com/sgl-project/sglang/pull/20474): Intel XPU Qwen3 layernorm/MRoPE support.
- [#22529](https://github.com/sgl-project/sglang/pull/22529): sliding window attention for Qwen3.
- [#22837](https://github.com/sgl-project/sglang/pull/22837): Qwen3 reasoning detector swallowing tool calls.
- [#20127](https://github.com/sgl-project/sglang/pull/20127): tied embeddings for Qwen MoE and Qwen3-Next.
- [#9147](https://github.com/sgl-project/sglang/pull/9147): Qwen3-MoE W4AFP8.

## Validation Lanes

- Correctness: `test/registered/models/test_qwen_models.py`, Qwen3 eval tests, logprob-diff LoRA tests.
- MoE performance: benchmark fused MoE/DeepEP/AITER/TRTLLM-GEN-MoE with fixed prompt mix and TP/EP shape.
- Quantization: run one BF16 reference plus one quantized checkpoint per backend, and include long-context decode.
- Platform lanes: keep AMD, NPU, XPU, and CPU changes isolated behind backend-specific guards.
- Parser lane: stream tool calls and reasoning content together; Qwen parser bugs often only show up with streaming deltas.
