# Qwen-Image Playbook

## Symptom Map

| Symptom | First check | PR trail |
| --- | --- | --- |
| Image changes under CUDA graph | graph capture path, fixed seed output | `#19516`, `#20810` |
| FP8 image quality regression | ModelOpt export/load and BF16 comparison | `#23155` |
| Edit mode broken | edit conditioning and multi-output batch | `#21988`, `#22953` |
| Layer serving fails | layer route and weight mapping | `#22362`, `#22397` |
| AMD slow kernel | norm/RoPE/LN fusion | `#18530`, `#20429` |
| TeaCache wrong output | cache key and denoise-step reuse | `#20447` |

## Investigation Order

1. Fixed-seed BF16 text-to-image.
2. Fixed-seed BF16 edit.
3. CUDA graph on/off.
4. TeaCache on/off.
5. IMA/conditional batching.
6. ModelOpt FP8.
7. AMD kernel tuning.

## Runtime Surface Checklist

- Pipeline config: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`.
- ComfyUI executor: `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/qwen_image.py`.
- Tests: Qwen-Image and Qwen-Image-Edit pipeline tests under `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/`.
- Public scope: SGLang Diffusion docs/blogs now cover Qwen-Image, Qwen-Image-Edit, Qwen-Image-Edit-2511, Qwen-Image-2512, Qwen-Image-Layered, GLM-Image, and LoRA API.

## Validation Checklist

- prompt, negative prompt, seed, resolution, steps, guidance scale recorded
- image output saved for BF16 baseline and optimized variant
- latency measured over full denoise, not only one kernel
- memory peak recorded
- ModelOpt FP8 quality compared to BF16

## Change Rules

- Do not use autoregressive Qwen tests as proof for Qwen-Image.
- Kernel changes require e2e denoise validation.
- FP8 changes require both loading and image-quality validation.
