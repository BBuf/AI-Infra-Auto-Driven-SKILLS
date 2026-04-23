---
name: sglang-qwen-image-optimization
description: PR-backed and current-main optimization manual for Qwen-Image and Qwen-Image-Edit in SGLang Diffusion. Use when Codex needs to recover, extend, or audit diffusion transformer loading, layer serving, CUDA graph, TeaCache, IMA, ModelOpt FP8, AMD kernels, Qwen-Image detectors, or cookbook diffusion recipes.
---

# SGLang Qwen-Image Optimization

## Overview

Qwen-Image is the diffusion Qwen lane. It is intentionally split from autoregressive Qwen because the runtime surfaces are SGLang Diffusion pipelines, transformer layers, ModelOpt export/loading, image-edit conditioning, CUDA graph, TeaCache, and AMD diffusion kernels.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Manual PR diff review date: `2026-04-23`
- Three-pass completeness audit: `model-pr-optimization-history/sglang/qwen-glm-three-pass-audit-2026-04-23.md`
- Docs/snippets: `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image.mdx`, `Qwen-Image-Edit.mdx`, and `docs_new/src/snippets/diffusion/qwen-image*.jsx`
- Relevant skill: use the local SGLang diffusion/modelopt skills for implementation-level quantization or profiling work.

## PR Dossier Rule

When writing or updating Qwen-Image optimization docs, manually inspect every cited PR diff. Include PR link, state, diff stats, motivation, implementation path, key code excerpt, reviewed files, and validation implications. Use `model-pr-diff-dossier` for new PR-history production.

## Before You Change Anything

Capture:

- model: Qwen-Image or Qwen-Image-Edit
- mode: text-to-image, edit, multi-output, or layer serving
- quantization: BF16, FP8 ModelOpt, or backend-native path
- CUDA graph enabled or disabled
- TeaCache enabled or disabled
- IMA or conditional batching enabled
- backend: NVIDIA or AMD
- whether any external Diffusers pipeline assumptions are being imported

## Core Principle

Keep diffusion pipeline changes separate from autoregressive Qwen.

- Qwen-Image issues usually involve denoise-step scheduling, transformer/layer loading, conditional batches, or CUDA graph capture.
- ModelOpt FP8 support must validate image quality and layer-loading compatibility, not just server startup.
- AMD kernels need diffusion-specific microbench and denoise validation.

## Main Runtime Surfaces

- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image.mdx`
- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image-Edit.mdx`
- `docs_new/src/snippets/diffusion/qwen-image-deployment.jsx`
- `docs_new/src/snippets/diffusion/qwen-image-edit-deployment.jsx`
- `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`
- `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`
- `python/sglang/multimodal_gen/runtime/cache/teacache.py`
- `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`
- `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`

## Optimization Order

1. Validate BF16 image generation with fixed seed and prompt.
2. Validate edit mode separately from text-to-image.
3. Enable CUDA graph and compare image hash/metrics or visual regression outputs.
4. Enable TeaCache or IMA after baseline quality is stable.
5. Add ModelOpt FP8 and compare BF16-vs-FP8 output.
6. Run AMD kernel benchmarks only after e2e denoise is correct.

## Open PRs to Track

- [#18530](https://github.com/sgl-project/sglang/pull/18530): AMD norm/RoPE fusion.
- [#19066](https://github.com/sgl-project/sglang/pull/19066): Qwen2.5-VL ViT/text encoder optimization for diffusion-adjacent path.
- [#19516](https://github.com/sgl-project/sglang/pull/19516): CUDA graph.
- [#19521](https://github.com/sgl-project/sglang/pull/19521): detectors.
- [#20429](https://github.com/sgl-project/sglang/pull/20429): LN/scale-shift-gate kernel fusion.
- [#20432](https://github.com/sgl-project/sglang/pull/20432): dual-stream forward.
- [#20447](https://github.com/sgl-project/sglang/pull/20447): TeaCache.
- [#20810](https://github.com/sgl-project/sglang/pull/20810): reland CUDA graph.
- [#21988](https://github.com/sgl-project/sglang/pull/21988): conditional batch multi-output.
- [#22362](https://github.com/sgl-project/sglang/pull/22362): layer-serving fix.
- [#22397](https://github.com/sgl-project/sglang/pull/22397): weight-name mapping.
- [#22953](https://github.com/sgl-project/sglang/pull/22953): IMA bugfix.
- [#23155](https://github.com/sgl-project/sglang/pull/23155): ModelOpt FP8.

Read `references/pr-history.md` for the diff-reviewed card of each PR. Most entries are currently open; do not describe them as current-main behavior unless their state changes.

## Validation Lanes

- fixed-seed BF16 generation
- fixed-seed edit generation
- CUDA graph on/off comparison
- TeaCache on/off comparison
- ModelOpt FP8 image quality comparison
- AMD denoise latency and kernel profile

## References

- `references/pr-history.md`: manually diff-reviewed PR cards.
- `references/playbook.md`: symptom map, toggle matrix, and validation rules.
