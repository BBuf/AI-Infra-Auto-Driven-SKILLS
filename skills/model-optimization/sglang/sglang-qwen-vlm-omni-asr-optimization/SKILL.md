---
name: sglang-qwen-vlm-omni-asr-optimization
description: PR-diff-backed optimization manual for Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, Qwen3-ASR, and Qwen3.5 multimodal paths in SGLang. Use when Codex needs to audit, debug, extend, or document multimodal processors, ViT DP/PP/chunk/cache, mRoPE, DeepStack, EAGLE3, LoRA, audio encoder, streaming ASR, encoder disaggregation, AMD/NPU/CPU support, or Qwen VLM cookbook deployment recipes.
---

# SGLang Qwen VLM / Omni / ASR Optimization

## Non-Negotiable Evidence Rule

Before writing or changing model optimization documentation, read the relevant PR
diffs directly and record the motivation, implementation path, key code snippet,
reviewed files, and validation notes. Do not summarize a PR with only a title-level
sentence.

For this skill, start with:

- `references/pr-history.md`: manually written PR cards with source-diff evidence.
- `references/playbook.md`: symptom-to-file investigation path and validation lanes.

## Scope

This skill covers the Qwen multimodal lane:

- Qwen2.5-VL
- Qwen3-VL
- Qwen3-VL-MoE
- Qwen3-Omni
- Qwen3-ASR
- Qwen3.5 multimodal cross-family paths that share `qwen_vl.py`,
  Qwen VLM processor code, or encoder-disaggregation code.

It is separate from Qwen3 text-only optimization because most regressions here sit
in processors, vision/audio encoders, mRoPE/3D mRoPE, DeepStack, multimodal cache,
DP/PP encoder routing, and streaming ASR.

## Current Evidence Snapshot

- SGLang main evidence: around `b3e6cf60a` on 2026-04-22.
- sgl-cookbook evidence: around `816bad5` on 2026-04-21.
- Public docs/blogs checked:
  - SGLang Qwen3-VL usage docs.
  - LMSYS AMD Qwen3/Qwen3-VL latency optimization blog.
  - SGLang tracking issue `#18466`.

## Runtime Files

- `python/sglang/srt/models/qwen2_5_vl.py`
- `python/sglang/srt/models/qwen3_vl.py`
- `python/sglang/srt/models/qwen3_vl_moe.py`
- `python/sglang/srt/models/qwen3_omni_moe.py`
- `python/sglang/srt/models/qwen3_asr.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `python/sglang/srt/multimodal/processors/qwen_audio.py`
- `python/sglang/srt/multimodal/processors/qwen3_asr.py`
- `python/sglang/srt/managers/mm_utils.py`
- `python/sglang/srt/disaggregation/encode_server.py`
- `python/sglang/srt/entrypoints/openai/serving_transcription.py`

## How To Use This Skill

1. Identify the exact model family and modality: image, video, audio, streaming ASR,
   or mixed media.
2. Read the corresponding PR cards in `references/pr-history.md`; do not rely on
   the compact table in the README.
3. Inspect the current SGLang source files touched by the relevant PRs.
4. Reproduce or validate in the narrowest lane first: one image, one video, one audio
   clip, or one ASR chunk.
5. Only then add DP/PP, chunked prefill, CUDA graph, EAGLE3, LoRA, quantization, or
   hardware-specific backends.

## Optimization Map

- Processor/token placement: `qwen_vl.py`, `qwen3_asr.py`, `base_processor.py`.
- mRoPE / 3D mRoPE: `qwen3_vl.py`, `qwen3_omni_moe.py`, `mrope_rope_index.py`.
- ViT speed/memory: `qwen3_vl.py`, `attention/vision.py`, `mm_utils.py`,
  `vit_cuda_graph_runner.py`.
- Encoder DP/PP/EPD: `qwen3_vl.py`, `qwen3_vl_moe.py`, `encode_server.py`,
  `server_args.py`.
- DeepStack / EAGLE3: `qwen3_vl.py`, `qwen3_vl_moe.py`, speculative decoding helpers.
- Qwen3-Omni audio: `qwen3_omni_moe.py`, `qwen_vl.py`.
- Qwen3-ASR: `qwen3_asr.py`, `qwen3_asr.py` processor, transcription serving,
  `streaming_asr.py`.
- Hardware-specific lanes: AMD AITER/ROCm, NPU processor/audio loader, CPU SDPA/AMX.

## Validation Lanes

- Single image and multi-image requests.
- Raw video URL/path and `processor_output` video.
- Long-video chunked prefill and multimodal cache.
- Encoder DP, pipeline parallelism, and encoder-only/language-only EPD.
- Qwen3-VL-MoE LoRA logprob diff if adapter routing changes.
- Qwen3-Omni audio/video/audio-in-video when audio code changes.
- Qwen3-ASR final transcription, HTTP streaming, websocket streaming if present.
- Hardware-specific reruns on AMD, NPU, or CPU when those paths are touched.

## Open Radar

Track open/closed-unmerged PRs in `references/pr-history.md` before using them as
implementation guidance. Important radar lanes include CPU Qwen3-VL/Omni, Qwen3-Omni
DP encoder, Qwen3-VL EVS, precise embedding interpolation, ASR websocket streaming,
NPU ASR audio loading, and Qwen3-VL-MoE encoder-only guards.
## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.
