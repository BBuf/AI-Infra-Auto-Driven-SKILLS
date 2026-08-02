# Latest native model inventory

锁定 `sgl-project/sglang@f8e62a9224815cc9c6fc56b940eb7fde791a8870`。表内 ID 来自
`registry.py` 的精确路径；MOVA/GLM 是 detector path，采用同 revision benchmark
preset 验证的公开 ID。每个 ID 都按 [`execution-contract.md`](execution-contract.md)
下载到稳定本地目录。不要一次下载全表，只下载当前 flow 所需 checkpoint。

| Flow | Hugging Face / registry ID |
| --- | --- |
| MiniMax-H3 | `MiniMaxAI/MiniMax-H3`；ModelScope 已登记别名 `MiniMax/MiniMax-H3` |
| Pi0.5 | `lerobot/pi05_base`, `lerobot/pi05_libero_base` |
| FLUX.1 | `black-forest-labs/FLUX.1-dev` |
| FLUX.2 | `black-forest-labs/FLUX.2-dev`, `black-forest-labs/FLUX.2-dev-NVFP4` |
| FLUX.2 Klein | `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B`, `black-forest-labs/FLUX.2-klein-base-4B`, `black-forest-labs/FLUX.2-klein-base-9B` |
| Z-Image | `Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo` |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`, `nvidia/Qwen-Image-NVFP4` |
| Qwen edit/layered | `Qwen/Qwen-Image-Edit`, `Qwen/Qwen-Image-Edit-2509`, `Qwen/Qwen-Image-Edit-2511`, `Qwen/Qwen-Image-Layered` |
| FireRed edit | `FireRedTeam/FireRed-Image-Edit-1.0`, `FireRedTeam/FireRed-Image-Edit-1.1` |
| Krea-2 | `krea/Krea-2` |
| SD3/3.5 | `stabilityai/stable-diffusion-3-medium`, `stabilityai/stable-diffusion-3-medium-diffusers`, `stabilityai/stable-diffusion-3.5-medium`, `stabilityai/stable-diffusion-3.5-medium-diffusers`, `stabilityai/stable-diffusion-3.5-large`, `stabilityai/stable-diffusion-3.5-large-diffusers` |
| ERNIE | `baidu/ERNIE-Image`, `baidu/ERNIE-Image-Turbo` |
| GLM | `zai-org/GLM-Image`（detector） |
| SANA | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`, `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers`, `Efficient-Large-Model/Sana_1600M_1024px_diffusers`, `Efficient-Large-Model/Sana_600M_1024px_diffusers`, `Efficient-Large-Model/Sana_1600M_512px_diffusers`, `Efficient-Large-Model/Sana_600M_512px_diffusers` |
| SANA-WM | `Efficient-Large-Model/SANA-WM_bidirectional`, `Efficient-Large-Model/SANA-WM_streaming` |
| Ideogram 4 | `fal/ideogram-v4-fast`, `fal/ideogram-v4-instant`, `ideogram-ai/ideogram-4-fp8`, `ideogram-ai/ideogram-4-nf4`, `Comfy-Org/Ideogram-4` |
| Wan2.1 T2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers`, `IPostYellow/TurboWan2.1-T2V-14B-Diffusers`, `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers` |
| Wan2.1 I2V/Fun | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`, `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`, `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` |
| Wan2.2 | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers`, `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers` |
| FastWan/NVFP4 | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`, `FastVideo/FastWan2.2-TI2V-5B-Diffusers`, `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`, `nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4` |
| LingBot | `IPostYellow/lingbot-world-fast-diffusers`, `robbyant/lingbot-world-fast-diffusers`, `robbyant/lingbot-world-v2-14b-causal-fast-diffusers` |
| LongLive 2 | `Rabinovich/LongLive-2.0-5B-Diffusers`, `Efficient-Large-Model/LongLive-2.0-5B` |
| Helios | `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled` |
| HunyuanVideo | `hunyuanvideo-community/HunyuanVideo`, `FastVideo/FastHunyuan-diffusers` |
| Hunyuan3D | `tencent/Hunyuan3D-2` |
| LTX | `Lightricks/LTX-2`, `Lightricks/LTX-2.3` |
| JoyAI-Echo | `jdopensource/JoyAI-Echo` |
| JoyAI edit | `jdopensource/JoyAI-Image-Edit-Diffusers` |
| Cosmos3 | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super`, `nvidia/Cosmos3-Super-Text2Image`, `nvidia/Cosmos3-Super-Image2Video` |
| MOVA | `OpenMOSS-Team/MOVA-360p`, `OpenMOSS-Team/MOVA-720p`（detector） |

最小下载模板：

```bash
MODEL_ID=<表中完整 ID>
MODEL_DIR=/data/models/<稳定目录名>
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
# 国内环境可在命令前：export HF_ENDPOINT=https://hf-mirror.com
```

若没有经过 registry/source 验证的 ModelScope ID，使用 HF 镜像，不猜别名。
