# Qwen VLM / Omni / ASR Playbook

Use this playbook after reading the detailed PR cards in `pr-history.md`.

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| Qwen3-VL image/video output wrong | `qwen_vl.py`, `qwen3_vl.py`, visual weight loader | `#10985`, `#14292`, `#18024`, `#19333`, `#22052` |
| Qwen3-VL ViT slow | `attention/vision.py`, `qwen3_vl.py`, `vit_cuda_graph_runner.py` | `#13736`, `#14292`, `#15205`, `#15320`, `#19003` |
| Long video or many images OOM | `qwen3_vl.py`, `mm_utils.py`, multimodal cache | `#14907`, `#16366`, `#22038`, `#20857` |
| Encoder DP hang | `qwen3_vl.py`, `mm_utils.py`, `vocab_parallel_embedding.py` | `#13724`, `#17624`, `#20788`, `#20759`, `#18721` |
| Pipeline parallel or EPD launch fails | `qwen3_vl.py`, `qwen3_vl_moe.py`, `server_args.py`, `encode_server.py` | `#12333`, `#21849`, `#23115`, `#23220` |
| Qwen3-VL-MoE LoRA issue | `qwen3_vl_moe.py`, LoRA tests | `#21469` |
| Qwen3-VL EAGLE3 issue | `qwen3_vl.py`, speculative decoding helpers | `#13918`, `#17276`, `#22230` |
| Qwen3-Omni audio slow or wrong | `qwen3_omni_moe.py`, `qwen_vl.py` | `#10911`, `#18185`, `#14886`, `#16996` |
| Qwen3-ASR transcription or stream bug | `qwen3_asr.py`, `qwen3_asr.py` processor, `serving_transcription.py`, `streaming_asr.py` | `#22073`, `#22089`, `#22848`, `#23469` |
| Hardware-specific mismatch | AMD/NPU/CPU backend files | `#12662`, `#16571`, `#19693`, `#21458`, `#22266`, `#23469` |

## Investigation Order

1. Confirm the exact architecture in `hf_config.architectures`.
2. Capture request modality and shape: image count, video frames, audio length, and
   whether data is raw URL/path or processor output.
3. Reproduce with the smallest request: one image, one short video, or one short
   audio clip.
4. Inspect processor output: token placeholders, offsets, `grid_thw`, audio feature
   lengths, timestamp metadata, and feature device.
5. Inspect mRoPE/position embedding: mRoPE position shape, 2D/3D RoPE, DeepStack
   slices, and cos/sin cache use.
6. Add execution features one by one: DP encoder, PP, EPD, chunked prefill, ViT CUDA
   graph, FlashInfer CUDNN, EAGLE3, LoRA, quantization.
7. Only then evaluate hardware-specific backends.

## Validation Checklist

- Single image request.
- Multi-image request.
- Raw video URL/path request.
- `processor_output` video request.
- Long video with chunked prefill and cache reporting.
- Qwen3-VL encoder DP and no-DP comparison.
- Qwen3-VL PP or EPD if launch/rank code changed.
- Qwen3-VL-MoE LoRA logprob diff if adapter routing changed.
- Qwen3-Omni image/audio/video prompt if Omni code changed.
- Qwen3-ASR final transcription.
- Qwen3-ASR streaming chunk boundary, whitespace, final transcript.
- Hardware-specific run when touching AMD, NPU, or CPU code.

## Deployment Notes From Public Docs

- SGLang Qwen3-VL docs recommend FP8 checkpoints for memory-efficient H100/H200
  deployments and BF16/full precision where FP8 is not used.
- `--keep-mm-feature-on-device` avoids D2H copies but changes memory pressure.
- `--mm-max-concurrent-calls` controls image/video processor concurrency and memory.
- `--mm-attention-backend fa3` and CUDA IPC transport are documented performance
  knobs.
- LMSYS AMD latency notes split Qwen3-VL work into preprocessing acceleration,
  multimodal data transfer, ViT DP, and ViT kernel fusion; keep those buckets when
  writing new optimization docs.

## Change Rules

- Do not debug Qwen VLM from language-model logits alone; first inspect processor
  tensors and media feature placement.
- Do not call an open PR current-main behavior.
- Do not merge Qwen3 text-only history into Qwen VLM unless the diff touches shared
  multimodal files or Qwen3-VL-specific config/model paths.
- For every new model optimization card, include motivation, implementation, code
  snippet, reviewed files, and validation.
