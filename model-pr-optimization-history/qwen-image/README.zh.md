# SGLang Qwen-Image 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 Qwen-Image、Qwen-Image-Edit、CUDA graph、TeaCache、IMA、conditional batch、ModelOpt FP8、AMD diffusion kernels。

结论：Qwen-Image 是 diffusion lane，不能用 autoregressive Qwen 的测试替代。当前主要 PR 多数仍处于 open/active 状态，后续应以固定 seed 的 BF16/FP8 图像质量对齐、denoise latency、CUDA graph/TeaCache/IMA 开关矩阵为核心。

## 代码面

- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image.mdx`
- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image-Edit.mdx`
- `docs_new/src/snippets/diffusion/qwen-image-deployment.jsx`
- `docs_new/src/snippets/diffusion/qwen-image-edit-deployment.jsx`
- SGLang Diffusion 中可用 `rg -n "Qwen-Image|qwen_image|QwenImage" python docs_new test` 定位 pipeline/model 实现。

## Open PR 雷达

- `#18530`：AMD norm/RoPE fusion。
- `#19066`：Qwen2.5-VL ViT/text encoder diffusion-adjacent 优化。
- `#19516`、`#20810`：CUDA graph / reland。
- `#19521`：detectors。
- `#20429`：LN/scale-shift-gate fusion。
- `#20432`：dual-stream forward。
- `#20447`：TeaCache。
- `#21988`：conditional batch multi-output。
- `#22362`：layer-serving 修复。
- `#22397`：weight-name mapping。
- `#22953`：IMA bugfix。
- `#23155`：ModelOpt FP8。

## sgl-cookbook / 文档

- `sgl-cookbook#49`、`#55`、`#60`、`#103`：diffusion benchmark/docs/command generator 基础演进。
- `sgl-cookbook#146`：Qwen-Image-Edit AMD MI300X/MI325X/MI355X。
- `sgl-cookbook#147`：Qwen-Image AMD MI300X/MI325X/MI355X。

## 下一步优化建议

1. 先建立固定 prompt/seed/resolution/steps 的 BF16 基线图。
2. 分别验证 CUDA graph、TeaCache、IMA、conditional batch、ModelOpt FP8，不要一次性全开。
3. AMD kernel 优化必须跑完整 denoise profile，而不是只看单 kernel。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。代码面补充 `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`、ComfyUI SGLDiffusion executor 和 Qwen-Image/Qwen-Image-Edit pipeline tests。公开博客补充 SGLang Diffusion launch 与 two-month update，覆盖 Qwen-Image、Qwen-Image-Edit、Qwen-Image-Edit-2511、Qwen-Image-2512、Qwen-Image-Layered、GLM-Image 和 LoRA API。
