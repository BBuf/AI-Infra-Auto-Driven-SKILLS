---
name: sglang-qwen3-next-optimization
description: PR-backed and current-main optimization manual for Qwen3-Next and Qwen3-Next MTP in SGLang. Use when Codex needs to recover, extend, or audit Qwen3-Next hybrid Gated Delta Network paths, RadixLinearAttention, Mamba/radix-cache scheduling, MTP/EAGLE, FP8/NVFP4/ModelOpt checkpoints, FlashInfer all-reduce, GDN projection fusion, CPU offload, AMD/NPU validation, or cookbook deployment flags.
---

# SGLang Qwen3-Next Optimization

## Overview

Qwen3-Next is its own optimization lane because it combines attention, Gated Delta Networks, MoE, hybrid cache state, and MTP. Do not collapse it into generic Qwen3 MoE work unless the patch only touches shared loader/parser code.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Three-pass completeness audit: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`
- Runtime files: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`
- Config: `python/sglang/srt/configs/qwen3_next.py`
- Cookbook/docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Next.mdx`, `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx`

## Before You Change Anything

Capture:

- checkpoint: base Qwen3-Next, Qwen3-Coder-Next, or ModelOpt NVFP4/FP8 variant
- MTP enabled or disabled, and whether `SGLANG_ENABLE_SPEC_V2=1`
- `--mamba-scheduler-strategy`: `no_buffer` or `extra_buffer`
- page size, radix cache, prefix cache, and CPU offload flags
- TP/DP/EP/PP, and whether MoE expert parallel is active
- backend: NVIDIA/Hopper/Blackwell, AMD MI325/MI355, Ascend NPU, CPU/XPU
- quantization: BF16, FP8, NVFP4, compressed-tensors, or ModelOpt

## Core Principle

Qwen3-Next bugs are often state bugs, not just matmul bugs.

- RadixLinearAttention and Mamba/GDN state scheduling interact with cache reuse.
- MTP changes draft/target paths and can expose missing cache sharding.
- CPU offload, PP, and quantization all need weight-loader coverage.
- Projection fusion must preserve GDN numerical behavior and tensor layout.

## Main Runtime Surfaces

- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_next_mtp.py`
- `python/sglang/srt/configs/qwen3_next.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/radix_attention.py`
- `python/sglang/srt/speculative/`
- `test/registered/4-gpu-models/test_qwen3_next_models.py`
- `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py`
- `test/registered/models/test_qwen3_next_models_fp4.py`

## Optimization Order

1. Verify BF16 no-MTP correctness.
2. Add quantized loading, especially FP8/NVFP4 weight-loader rules.
3. Validate Mamba/GDN state behavior with prefix/radix cache.
4. Enable MTP and run target/draft consistency tests.
5. Tune GDN projection fusion and all-reduce only after correctness is stable.
6. Re-run AMD/NPU lanes because Qwen3-Next platform paths have distinct kernel coverage.

## Open PRs to Track

- [#10657](https://github.com/sgl-project/sglang/pull/10657): EAGLE3 for Qwen3-Next.
- [#12892](https://github.com/sgl-project/sglang/pull/12892): avoid SSM/conv state copy for speculative decoding.
- [#13964](https://github.com/sgl-project/sglang/pull/13964): Qwen3-Next kernel performance optimization.
- [#14502](https://github.com/sgl-project/sglang/pull/14502): Qwen3-Next PCG optimization.
- [#16488](https://github.com/sgl-project/sglang/pull/16488): TBO support for Qwen3-Next.
- [#17981](https://github.com/sgl-project/sglang/pull/17981): CuteDSL decode/MTP kernel on Blackwell.
- [#17983](https://github.com/sgl-project/sglang/pull/17983): prefill kernel/GDN Gluon/cumsum.
- [#19812](https://github.com/sgl-project/sglang/pull/19812): MTP + EPLB compatibility.
- [#20397](https://github.com/sgl-project/sglang/pull/20397): NPU Qwen3-Next MTP.
- [#23474](https://github.com/sgl-project/sglang/pull/23474): CPU offload on hybrid linear-attention models.

## Validation Lanes

- BF16 and FP8/NVFP4 functional tests.
- MTP target/draft tests with and without radix cache.
- Long-context prompt reuse with both Mamba scheduler strategies.
- AMD/NPU registered tests and cookbook deployment commands.
