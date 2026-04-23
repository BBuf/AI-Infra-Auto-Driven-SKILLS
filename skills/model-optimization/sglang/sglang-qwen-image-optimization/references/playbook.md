# Qwen-Image Playbook

Use this playbook after reading `references/pr-history.md`. Qwen-Image is a diffusion runtime lane; autoregressive Qwen validation is not sufficient.

## Symptom Map

| Symptom | First check | PR trail |
| --- | --- | --- |
| AMD denoise slow in cross-attention | fused RoPE/RMSNorm and AITER envs | `#18530` |
| Qwen2.5-VL encoder slow or inaccurate on AMD | custom ViT, SDPA contiguity, RoPE cache | `#19066` |
| Image changes or graph memory grows under CUDA graph | graph bucket/cache path | `#19516`, `#20810` |
| Local model path not detected | registry model detectors | `#19521` |
| Modulation/LN kernels dominate profile | Triton fused modulate kernel | `#20429` |
| B200 latency has avoidable text/image gaps | dual-stream QKV/MLP overlap | `#20432` |
| TeaCache output quality changes | residual cache, CFG branch separation | `#20447` |
| `num_outputs_per_prompt > 1` shape mismatch | condition batch expansion | `#21988` |
| Qwen-Image-Layered serve fails without prompt or RGBA save | OpenAI image edit endpoint and output extension | `#22362` |
| Weight loading fails on `to_out` or split added QKV names | `QwenImageArchConfig.param_names_mapping` | `#22397` |
| CUDA illegal memory access in RoPE with many images | text RoPE cache length check | `#22953` |
| FP8 image is dark or blurry | ModelOpt fallback set and converter order | `#23155` |

## PR Dossier Rule

Every PR cited in Qwen-Image docs must be manually diff-reviewed. For each PR, record link, state, diff stats, motivation, implementation path, code excerpt, reviewed files, and validation implications.

## Investigation Order

1. Fixed-seed BF16 text-to-image baseline.
2. Fixed-seed BF16 edit baseline.
3. Qwen-Image-Layered / multi-image edge cases if the issue touches serving or RoPE cache.
4. Conditional batch and `num_outputs_per_prompt`.
5. CUDA graph on/off.
6. TeaCache on/off.
7. Dual stream on/off for target GPU.
8. ModelOpt FP8 with BF16 comparison.
9. AMD AITER and Triton kernel toggles.

## Runtime Surface Checklist

- Pipeline config: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- DiT model: `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- Qwen2.5-VL encoder path: `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`
- Vision attention helper: `python/sglang/multimodal_gen/runtime/models/encoders/vision.py`
- TeaCache: `python/sglang/multimodal_gen/runtime/cache/teacache.py`
- CUDA graph utils: `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`
- ModelOpt converter: `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`
- Image API: `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py`
- Quant docs: `docs/diffusion/quantization.md`

## Toggle Validation

For every performance or quality claim, capture:

- model and checkpoint path
- prompt, negative prompt, input image path when editing, seed, width, height, steps, guidance scale
- output image for BF16 baseline and optimized variant
- E2E latency and denoise latency
- peak memory
- hardware and backend envs
- profiler artifact if the claim is kernel-level

## Independent Toggles

Validate independently before combining:

- `SGLANG_ENABLE_FUSED_ROPE_RMS_2WAY`
- `_FUSE_LN_SCALE_SHIFT_SELECT01`
- `QWEN_IMAGE_DUAL_STREAM_FORWARD`
- `--enable-cuda-graph`
- `--cuda-graph-txt-lengths`
- TeaCache params and threshold
- `num_outputs_per_prompt`
- ModelOpt FP8 fallback profile

## Change Rules

- Do not use autoregressive Qwen tests as proof for Qwen-Image.
- Kernel changes require end-to-end denoise validation.
- FP8 changes require image-quality validation, not just successful loading.
- CUDA graph and torch.compile are mutually exclusive in the reland design.
- Open PRs are design references until they merge; re-check PR state before relying on them.
- Layered/RGBA serving should default to PNG unless request format says otherwise.
