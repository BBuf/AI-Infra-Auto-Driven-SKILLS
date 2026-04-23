# Qwen-Image PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: diffusion Qwen-Image docs/snippets, diffusion model/pipeline files, diffusion tests.
- Searched PR terms: `Qwen-Image`, `Qwen Image`, `Qwen-Image-Edit`, `qwen_image`, `QwenImage`.

## Runtime and Docs Surfaces

- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image.mdx`
- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image-Edit.mdx`
- `docs_new/src/snippets/diffusion/qwen-image-deployment.jsx`
- `docs_new/src/snippets/diffusion/qwen-image-edit-deployment.jsx`
- SGLang Diffusion model/pipeline files found by `rg -n "Qwen-Image|qwen_image|QwenImage" python docs_new test`

## Open PR Radar

- [#18530](https://github.com/sgl-project/sglang/pull/18530): AMD fuse norm/RoPE.
- [#19066](https://github.com/sgl-project/sglang/pull/19066): Qwen2.5-VL ViT/text encoder optimization in diffusion-adjacent path.
- [#19516](https://github.com/sgl-project/sglang/pull/19516): CUDA graph.
- [#19521](https://github.com/sgl-project/sglang/pull/19521): Qwen-Image detectors.
- [#20429](https://github.com/sgl-project/sglang/pull/20429): LN/scale-shift-gate fusion.
- [#20432](https://github.com/sgl-project/sglang/pull/20432): dual-stream forward.
- [#20447](https://github.com/sgl-project/sglang/pull/20447): TeaCache.
- [#20810](https://github.com/sgl-project/sglang/pull/20810): reland CUDA graph.
- [#21988](https://github.com/sgl-project/sglang/pull/21988): conditional batch multi-output.
- [#22362](https://github.com/sgl-project/sglang/pull/22362): Qwen-Image layer-serving fix.
- [#22397](https://github.com/sgl-project/sglang/pull/22397): weight-name mapping.
- [#22953](https://github.com/sgl-project/sglang/pull/22953): IMA bugfix.
- [#23155](https://github.com/sgl-project/sglang/pull/23155): ModelOpt FP8.

## Cookbook Evidence

- sgl-cookbook [#49](https://github.com/sgl-project/sgl-cookbook/pull/49): diffusion benchmark/model initialization groundwork.
- sgl-cookbook [#55](https://github.com/sgl-project/sgl-cookbook/pull/55): diffusion docs restructuring.
- sgl-cookbook [#60](https://github.com/sgl-project/sgl-cookbook/pull/60): diffusion command generator cleanup.
- sgl-cookbook [#103](https://github.com/sgl-project/sgl-cookbook/pull/103): diffusion index/doc update.
- sgl-cookbook [#146](https://github.com/sgl-project/sgl-cookbook/pull/146): Qwen-Image-Edit AMD MI300X/MI325X/MI355X support.
- sgl-cookbook [#147](https://github.com/sgl-project/sgl-cookbook/pull/147): Qwen-Image AMD MI300X/MI325X/MI355X support.

## Validation Notes

- Qwen-Image PRs are mostly open/active; re-check PR state before starting implementation.
- Treat CUDA graph, TeaCache, IMA, FP8, and AMD kernel work as independent toggles in validation.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Concrete runtime surfaces found by local search:

- `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/qwen_image.py`
- `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_qwen_image_pipeline.py`
- `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_qwen_image_edit_pipeline.py`
- `python/sglang/multimodal_gen/apps/webui/README.md`

Public-blog evidence: the LMSYS SGLang Diffusion launch blog documents Qwen-Image/Qwen-Image-Edit support across OpenAI-compatible API, CLI, and Python interfaces; the two-month update adds Qwen-Image-Edit-2511, Qwen-Image-2512, Qwen-Image-Layered, GLM-Image, LoRA API coverage, and continued diffusion performance work.
