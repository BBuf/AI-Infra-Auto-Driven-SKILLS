---
name: sglang-qwen-vlm-omni-asr-optimization
description: PR-backed and current-main optimization manual for Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, and Qwen3-ASR in SGLang. Use when Codex needs to recover, extend, or audit multimodal processor behavior, ViT DP/PP/chunked attention/cache, mRoPE, audio encoder, streaming ASR, DP encoder, EAGLE3, AMD/NPU/CPU support, or Qwen VLM cookbook recipes.
---

# SGLang Qwen VLM/Omni/ASR Optimization

## Overview

This skill covers the Qwen multimodal lane: Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, and Qwen3-ASR. It is separate from Qwen3 text because the main risk sits in processors, vision/audio encoders, mRoPE, multimodal DP/PP, and streaming ASR.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Three-pass completeness audit: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`
- Runtime files: `qwen2_5_vl.py`, `qwen3_vl.py`, `qwen3_vl_moe.py`, `qwen3_omni_moe.py`, `qwen3_asr.py`
- Processors: `qwen_vl.py`, `qwen_audio.py`, `qwen3_asr.py`
- Docs: `docs/basic_usage/qwen3_vl.md`, Qwen VLM/ASR cookbook docs/snippets

## Before You Change Anything

Capture:

- model: Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, or Qwen3-ASR
- input type: image, video, audio, mixed media, or streaming ASR
- vision/audio encoder DP or PP enabled
- `--mm-attention-backend` and vision attention backend
- mRoPE/3D mRoPE path and position embedding cache status
- quantization and backend: BF16/FP8/NVFP4, AMD/NPU/CPU
- whether EAGLE3/speculative decoding is enabled for VLM

## Core Principle

Separate language-model bugs from encoder/processor bugs.

- If text-only Qwen3 passes, inspect multimodal processors, feature transfer, and mRoPE first.
- Qwen VLM performance is often dominated by ViT attention, chunking, encoder DP, and feature transfer.
- ASR bugs often involve streaming chunk boundaries rather than model logits.

## Main Runtime Surfaces

- `python/sglang/srt/models/qwen2_5_vl.py`
- `python/sglang/srt/models/qwen3_vl.py`
- `python/sglang/srt/models/qwen3_vl_moe.py`
- `python/sglang/srt/models/qwen3_omni_moe.py`
- `python/sglang/srt/models/qwen3_asr.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `python/sglang/srt/multimodal/processors/qwen_audio.py`
- `python/sglang/srt/multimodal/processors/qwen3_asr.py`
- `test/srt/models/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`

## Optimization Order

1. Reproduce with one image or one audio chunk before multi-turn media.
2. Validate processor output shapes and feature placement.
3. Validate mRoPE/3D mRoPE and position embedding cache.
4. Enable encoder DP/PP or chunked ViT attention.
5. Tune memory transfer and chunk-aware cache.
6. Add EAGLE3 or quantization only after baseline media correctness passes.

## Open PRs to Track

- [#12662](https://github.com/sgl-project/sglang/pull/12662): CPU Qwen3-VL and Qwen3-Omni support.
- [#12261](https://github.com/sgl-project/sglang/pull/12261): Qwen multimodal open support track.
- [#13918](https://github.com/sgl-project/sglang/pull/13918): Qwen VLM follow-up.
- [#17276](https://github.com/sgl-project/sglang/pull/17276): Qwen VLM follow-up.
- [#20857](https://github.com/sgl-project/sglang/pull/20857): VLM/audio follow-up.
- [#22052](https://github.com/sgl-project/sglang/pull/22052): VLM/ASR follow-up.
- [#22848](https://github.com/sgl-project/sglang/pull/22848): Qwen multimodal follow-up.
- [#23220](https://github.com/sgl-project/sglang/pull/23220): Qwen3-VL-MoE encoder-only adaptation.
- [#23469](https://github.com/sgl-project/sglang/pull/23469): NPU Qwen3-ASR adaptation.

## Validation Lanes

- image, multi-image, and video prompts
- encoder DP and PP
- ASR chunked streaming
- LoRA Qwen3-VL logprob diff
- AMD/NPU/CPU lanes separately
