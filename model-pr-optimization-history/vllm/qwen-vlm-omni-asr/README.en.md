# vLLM Qwen2.5-VL / Qwen3-VL / Qwen3-Omni / Qwen3-ASR Support and PR History

This note tracks the multimodal Qwen family in vLLM at commit
`0f7be0f2f76814f80f9091220a5fbbb53912ad00`.

- Status: supported on current mainline

## Key Conclusions

- The family now spans Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni thinker,
  Qwen3-ASR, and realtime Qwen3-ASR.
- The main risk areas are:
  processor placeholder expansion, video timestamps, interleaved MRoPE,
  `use_audio_in_video`, audio feature-length accounting, and realtime prompt
  expansion.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen2_5_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl_moe.py`
- `vllm/vllm/model_executor/models/qwen3_omni_moe_thinker.py`
- `vllm/vllm/model_executor/models/qwen3_asr.py`
- `vllm/vllm/model_executor/models/qwen3_asr_realtime.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`

## Landed PRs

- [#13155](https://github.com/vllm-project/vllm/pull/13155)
  `Qwen2.5-VL Optimization`
  Diff reviewed: `2` files, `47` additions, `51` deletions.
  Optimizes the Qwen2.5-VL vision attention fallback and switches to shared
  `RMSNorm`.
- [#24727](https://github.com/vllm-project/vllm/pull/24727)
  `Support Qwen3-VL Model Series`
  Diff reviewed: `13` files, `2084` additions, `17` deletions.
  Lands native Qwen3-VL / Qwen3-VL-MoE plus video placeholders and processing.
- [#25055](https://github.com/vllm-project/vllm/pull/25055)
  `Add Triton kernel for Qwen3-VL interleaved MRoPE`
  Diff reviewed: `2` files, `88` additions, `46` deletions.
  Adds interleaved MRoPE support and test coverage for Qwen3-VL.
- [#25550](https://github.com/vllm-project/vllm/pull/25550)
  `Add Qwen3-Omni moe thinker`
  Diff reviewed: `6` files, `1795` additions, `36` deletions.
  Adds the thinker runtime and special `use_audio_in_video` placeholder logic.
- [#33312](https://github.com/vllm-project/vllm/pull/33312)
  `Qwen3-ASR`
  Diff reviewed: `9` files, `1269` additions.
  Adds Qwen3-ASR configs, processor, model, and transcription path.
- [#34613](https://github.com/vllm-project/vllm/pull/34613)
  `Add Qwen3-ASR realtime streaming support`
  Diff reviewed: `5` files, `256` additions, `1` deletion.
  Adds a dedicated realtime subclass, audio buffer, and prompt expansion logic.

## Current Contract

When this family breaks, look at the exact modality path first:
Qwen2.5-VL attention fallback, Qwen3-VL video prompt replacement, Qwen3-Omni
audio-in-video bookkeeping, or Qwen3-ASR prompt/audio length logic. These are
the real fault lines.
