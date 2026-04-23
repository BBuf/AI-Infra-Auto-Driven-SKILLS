# Qwen VLM/Omni/ASR Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| Image/video output wrong | Qwen VLM processors and mRoPE | `#10985`, `#14292`, `#19333` |
| Vision encoder slow/OOM | ViT attention/cache/DP | `#14907`, `#15320`, `#16366`, `#22038` |
| Encoder DP hang | DP encoder path | `#13724`, `#17624`, `#20759`, `#20788` |
| Qwen3-Omni audio slow | `qwen3_omni_moe.py`, audio encoder | `#10911`, `#18185` |
| ASR stream boundary bug | `qwen3_asr.py`, ASR processor | `#22073`, `#22089`, `#23469` |
| VLM speculative issue | EAGLE3 + VLM path | `#22230` |

## Investigation Order

1. Test one image or one audio chunk.
2. Verify processor output and media token placement.
3. Verify mRoPE indices and position embedding cache.
4. Enable encoder DP/PP.
5. Enable chunked ViT attention/cache.
6. Enable quantization or speculative decoding.

## Official Deployment Checklist

- Launch flags: record FP8/BF16 checkpoint, `--keep-mm-feature-on-device`, `--mm-max-concurrent-calls`, and backend-specific attention settings.
- Request shape: preserve separate image, video, multi-image, and audio/ASR request examples.
- Device cache: when VLM cache or feature transfer changes, record memory peak and feature-transfer time.
- ASR: use `test/manual/models/test_qwen3_asr.py` and validate chunk boundaries, partial output, and final transcript accumulation.

## Validation Checklist

- single image
- video input
- multi-image conversation
- encoder DP
- ASR chunked streaming
- VLM EAGLE3 if touched
- LoRA Qwen3-VL logprob diff if LoRA path changed

## Change Rules

- Do not debug VLM output only from language-model logits; inspect processor and media feature tensors.
- Keep audio and vision processors separate.
- Encoder DP/PP fixes need a no-DP fallback comparison.
