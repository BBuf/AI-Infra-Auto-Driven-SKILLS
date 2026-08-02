# SGLang Diffusion 优化 Flow

这是一组可以逐个交给 coding agent 执行的简单 Markdown flow。它们不是新的
controller 或 skill：每个模型文件只定义入口、基准 shape、共享组件和模型特有
热点；真正重复的工作只在 `components/` 中定义一次。

## Source lock

- SGLang: `sgl-project/sglang@f8e62a9224815cc9c6fc56b940eb7fde791a8870`
- Source date: 2026-08-02
- Registry: `python/sglang/multimodal_gen/registry.py`
- Native pipelines: `python/sglang/multimodal_gen/runtime/pipelines/`
- Native model components: `python/sglang/multimodal_gen/runtime/models/`
- Nightly comparison cases: `scripts/ci/utils/diffusion/comparison_configs.json`
- Local benchmark presets:
  `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py`
- Companion skills update: `sgl-project/sglang#33282` / commit `d9641da26b`。其中新增
  `minimax-h3-t2va` preset；PR 未合并时用 H3 flow 内的手工命令。

完整、逐 ID 的当前 registry 清单见
[`common/model-inventory.md`](common/model-inventory.md)。

如果执行时 SGLang 已经越过该 revision，先重新对 registry、GPU cases、nightly
config 和 benchmark preset 做一次差异检查。不要默认本目录仍然完整覆盖新模型。

## 怎么使用

1. 先读 [`common/execution-contract.md`](common/execution-contract.md)。
2. 从下表选择一个模型族文件。
3. 只展开该文件引用的共享 component flow；不要重复优化同一个 VAE。
4. 每轮只接受一个有独立正确性与性能证据的改动。
5. 不满意就回到 profile，不要靠猜测继续堆 kernel。

推荐给 agent 的启动提示：

```text
执行 sglang-diffusion-optimization-flows/models/<family>.md。
严格遵守 common/execution-contract.md；只使用 native SGLang backend；
先完成 baseline、component profile 和已有 fast-path 排查，再决定是否改代码。
每轮只提交一个通过精度和速度门槛的候选。
```

## 模型覆盖

| Flow | Registry 中覆盖的模型/checkpoint |
| --- | --- |
| **[`minimax-h3.md`](models/minimax-h3.md)** | **`MiniMaxAI/MiniMax-H3`：T2VA、FL2VA、Ref2VA；优先执行** |
| [`flux.md`](models/flux.md) | `FLUX.1-dev`, `FLUX.2-dev`, `FLUX.2-dev-NVFP4`, Klein 4B/9B, Klein Base 4B/9B |
| [`qwen-image.md`](models/qwen-image.md) | Qwen-Image, 2512, Edit, Edit-2509, Edit-2511, Layered, NVIDIA NVFP4, FireRed Edit 1.0/1.1 |
| [`z-image.md`](models/z-image.md) | Z-Image, Z-Image-Turbo |
| [`wan.md`](models/wan.md) | Wan2.1 T2V/I2V, Wan2.2 T2V/I2V/TI2V, FastWan, TurboWan, Wan-Fun, NVIDIA NVFP4 |
| [`ltx.md`](models/ltx.md) | LTX-2, LTX-2.3 one-stage/two-stage/HQ |
| [`joyai-echo.md`](models/joyai-echo.md) | JoyAI-Echo 长视频生成 |
| [`hunyuan.md`](models/hunyuan.md) | HunyuanVideo, FastHunyuan |
| [`hunyuan3d.md`](models/hunyuan3d.md) | Hunyuan3D-2 shape/mesh |
| [`cosmos3.md`](models/cosmos3.md) | Cosmos3 Nano/Super, Super T2I/I2V variants |
| [`ideogram4.md`](models/ideogram4.md) | Ideogram 4 FP8/NF4, Comfy-Org, Fast, Instant |
| [`sana.md`](models/sana.md) | SANA 0.6B/1.6B/4.8B, SANA-WM bidirectional/streaming/realtime |
| [`ernie-image.md`](models/ernie-image.md) | ERNIE-Image, ERNIE-Image-Turbo |
| [`glm-image.md`](models/glm-image.md) | GLM-Image family detector paths |
| [`stable-diffusion-3.md`](models/stable-diffusion-3.md) | SD3 Medium, SD3.5 Medium/Large and Diffusers variants |
| [`krea2.md`](models/krea2.md) | Krea-2 |
| [`mova.md`](models/mova.md) | MOVA-360p, MOVA-720p |
| [`helios.md`](models/helios.md) | Helios Base, Mid, Distilled |
| [`joyai-image-edit.md`](models/joyai-image-edit.md) | JoyAI-Image-Edit-Diffusers |
| [`lingbot-world.md`](models/lingbot-world.md) | LingBot World fast and World V2 causal fast |
| [`longlive2.md`](models/longlive2.md) | LongLive-2.0-5B official/diffusers layouts |
| [`pi05.md`](models/pi05.md) | `lerobot/pi05_base`, `lerobot/pi05_libero_base` action diffusion |

`DiffusersPipeline` 是显式 fallback，不是 native 优化目标。ComfyUI 的 FLUX、Qwen
和 Z-Image wrapper 复用对应模型 flow，不再单独定义。后处理的 Real-ESRGAN 和
frame interpolation 也不复制模型组件 flow；只有 profile 证明其成为 E2E 热点时
才单独立项。

## 去重后的关键组件

所有 VAE component 在自己的 shape/config 之外，还统一执行
[`components/vae-rewrite-candidates.md`](components/vae-rewrite-candidates.md)。该清单把
参考报告中已经证明有效的语义 rewrite 作为“先验证能否复用”的候选，不假设它们
在新模型上一定更快。

| 组件 flow | 代表实现 | 被哪些模型复用 |
| --- | --- | --- |
| [`vae-autoencoder-kl-2d.md`](components/vae-autoencoder-kl-2d.md) | `AutoencoderKL` | FLUX.1, Z-Image, SD3, GLM-Image（按 config 分别取 shape） |
| [`vae-flux2.md`](components/vae-flux2.md) | `AutoencoderKLFlux2` | FLUX.2/Klein, Ideogram 4, ERNIE-Image |
| [`vae-qwen-causal-3d.md`](components/vae-qwen-causal-3d.md) | `AutoencoderKLQwenImage` | Qwen-Image, Krea-2, FireRed |
| [`vae-wan-causal-3d.md`](components/vae-wan-causal-3d.md) | `AutoencoderKLWan` | Wan, Cosmos3, Helios, JoyAI, MOVA, LingBot/LongLive derivatives |
| [`vae-hunyuan-video.md`](components/vae-hunyuan-video.md) | `AutoencoderKLHunyuanVideo` | HunyuanVideo/FastHunyuan |
| [`vae-ltx-video.md`](components/vae-ltx-video.md) | `AutoencoderKLLTX2Video` | LTX, SANA-WM, JoyAI-Echo |
| [`vae-dc-ae.md`](components/vae-dc-ae.md) | `AutoencoderDC` | SANA and DC-AE checkpoints |
| **[`vae-minimax-h3-video.md`](components/vae-minimax-h3-video.md)** | **`MiniMaxH3VideoVAE`** | **MiniMax-H3 video** |
| [`vae-audio.md`](components/vae-audio.md) | H3/LTX audio VAE、MOVA DAC、Cosmos3 AVAE | 各实现独立子项，不跨实现复用结论 |
| [`vae-hunyuan3d-shape.md`](components/vae-hunyuan3d-shape.md) | `ShapeVAE` | Hunyuan3D-2 |
| [`dit-attention.md`](components/dit-attention.md) | DiT/attention/modulation | 所有 diffusion transformer |
| [`distributed-runtime.md`](components/distributed-runtime.md) | TP/SP/CFG/offload/cache | 多 GPU 与低显存路径 |
| [`encoders.md`](components/encoders.md) | T5/CLIP/Qwen/Gemma/Mistral 等 | 文本、图像与条件编码器 |
| [`schedulers-postprocess.md`](components/schedulers-postprocess.md) | scheduler、guardrail、mux/export | 所有模型的非 DiT 尾部组件 |

这里的“复用”按 checkpoint `vae/config.json` 的 `_class_name` 与 SGLang runtime
implementation 去重。相同实现但 config/shape
不同的模型仍需跑代表 shape；不能拿一个 checkpoint 的数值结果替代另一个。
