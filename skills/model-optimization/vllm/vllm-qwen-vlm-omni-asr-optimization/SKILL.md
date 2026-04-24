---
name: vllm-qwen-vlm-omni-asr-optimization
description: PR-backed optimization manual for Qwen2.5-VL / Qwen3-VL / Qwen3-Omni / Qwen3-ASR in vLLM. Use when Codex needs to audit, debug, extend, or document the multimodal Qwen runtime in vLLM, especially Qwen2.5-VL attention hot paths, Qwen3-VL video and interleaved MRoPE handling, Qwen3-Omni thinker audio-in-video logic, and Qwen3-ASR / realtime speech support.
---

# vLLM Qwen2.5-VL / Qwen3-VL / Qwen3-Omni / Qwen3-ASR Optimization

## Overview

This family is already supported on the checked mainline commit, but the
high-risk logic lives in processors and multimodal position handling more than
in the text decoder itself. The important milestones are:

- `#13155`: Qwen2.5-VL hot-path optimization
- `#24727`: Qwen3-VL and Qwen3-VL-MoE bring-up
- `#25055`: interleaved MRoPE support for Qwen3-VL
- `#25550`: Qwen3-Omni thinker path
- `#33312`: Qwen3-ASR base transcription path
- `#34613`: Qwen3-ASR realtime streaming path

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors:
  `model-pr-optimization-history/vllm/qwen-vlm-omni-asr/README.zh.md` and
  `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the bar.
When touching this family, name the exact processor/model file because regressions
usually come from placeholder expansion, timestamps, audio lengths, or MRoPE
layout, not from generic "multimodal" code.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen2_5_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl_moe.py`
- `vllm/vllm/model_executor/models/qwen3_omni_moe_thinker.py`
- `vllm/vllm/model_executor/models/qwen3_asr.py`
- `vllm/vllm/model_executor/models/qwen3_asr_realtime.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`
- `vllm/vllm/transformers_utils/processors/qwen3_asr.py`

## Current Main Summary

- Qwen2.5-VL optimized its vision attention fallback path and RMSNorm usage.
- Qwen3-VL introduces native image and video placeholders, video timestamp
  calculation, and interleaved MRoPE handling.
- Qwen3-Omni thinker reimplements prompt updates to support
  `use_audio_in_video`.
- Qwen3-ASR reuses Qwen3 text plus Omni audio encoder pieces, then adds a
  separate realtime subclass with an audio buffer and prompt expansion logic.

## Key Landed PRs

- [#13155](https://github.com/vllm-project/vllm/pull/13155) `Qwen2.5-VL Optimization`
- [#24727](https://github.com/vllm-project/vllm/pull/24727) `Support Qwen3-VL Model Series`
- [#25055](https://github.com/vllm-project/vllm/pull/25055) `Add Triton kernel for Qwen3-VL interleaved MRoPE`
- [#25550](https://github.com/vllm-project/vllm/pull/25550) `Add Qwen3-Omni moe thinker`
- [#33312](https://github.com/vllm-project/vllm/pull/33312) `Qwen3-ASR`
- [#34613](https://github.com/vllm-project/vllm/pull/34613) `Add Qwen3-ASR realtime streaming support`

## Validation Lanes

- Qwen2.5-VL image and long-video requests on the chosen attention backend.
- Qwen3-VL image plus video placeholder expansion, timestamp generation, and
  interleaved MRoPE correctness.
- Qwen3-Omni thinker runs with and without `use_audio_in_video`.
- Qwen3-ASR batch transcription and realtime streaming.

## References

- `references/pr-history.md`: diff-reviewed Qwen multimodal cards.
