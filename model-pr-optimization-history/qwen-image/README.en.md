# SGLang Qwen-Image Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen-Image, Qwen-Image-Edit, CUDA graph, TeaCache, IMA, conditional batching, ModelOpt FP8, and AMD diffusion kernels.

## Summary

Qwen-Image is a diffusion lane. Autoregressive Qwen tests are not sufficient. Most current PRs are open/active, so validation should center on fixed-seed BF16/FP8 image quality, denoise latency, and separate CUDA graph/TeaCache/IMA toggle matrices.

## Code Surfaces

- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image.mdx`
- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image-Edit.mdx`
- `docs_new/src/snippets/diffusion/qwen-image-deployment.jsx`
- `docs_new/src/snippets/diffusion/qwen-image-edit-deployment.jsx`

## Open Radar

Track `#18530`, `#19066`, `#19516`, `#19521`, `#20429`, `#20432`, `#20447`, `#20810`, `#21988`, `#22362`, `#22397`, `#22953`, and `#23155`.

## Cookbook Evidence

Track `sgl-cookbook#49`, `#55`, `#60`, `#103`, `#146`, and `#147`.

## Next Work

Build fixed prompt/seed/resolution/step BF16 baselines, validate CUDA graph, TeaCache, IMA, conditional batching, and ModelOpt FP8 independently, and require full denoise profiles for AMD kernel work.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. Runtime surfaces now include `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`, the ComfyUI SGLDiffusion executor, and Qwen-Image/Qwen-Image-Edit pipeline tests. Public blog evidence adds SGLang Diffusion launch and the two-month update, covering Qwen-Image, Qwen-Image-Edit, Qwen-Image-Edit-2511, Qwen-Image-2512, Qwen-Image-Layered, GLM-Image, and LoRA API.
