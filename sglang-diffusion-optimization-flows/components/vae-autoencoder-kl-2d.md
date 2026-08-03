# AutoencoderKL 2D VAE 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载一个使用 runtime/models/vaes/autoencoder.py::AutoencoderKL 的实际 checkpoint 并把 VAE 单独跑起来。下面用 FLUX.1-dev 作为代表；gated repo 设置 HF_TOKEN，国内机器可使用 HF_ENDPOINT 或已验证的 ModelScope 镜像。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   hf download black-forest-labs/FLUX.1-dev
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-autoencoder-kl-2d
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 从对应 native pipeline 保存进入 decoder 的真实 latent，并在 ImageNet-val 或固定重建集上建立 eager reference。分别记录 untiled 和生产 tiled shape 的 output cosine、normalized MSE、PSNR、SSIM、LPIPS、颜色及 tile seam；FLUX.1、Z-Image、SD3、GLM 的 config/latent scaling 各自验收。

3. 分析 AutoencoderKL 架构，拆分 decoder conv、resblock、GroupNorm、SiLU、mid-block attention、upsample、tiling、blend 和 postprocess；记录 channels、scaling/shift、stride、padding、tile/overlap、dtype、layout 与调用次数，GLM 额外记录 latent mean/std。

4. 用保存的 latent 建 decode-only harness，20 次 warmup、100 次计时，并 profile torch.compile 后各组件与 kernel 耗时；定位 conv、norm/activation、mid attention、upsample、tile blend、layout copy、graph break 和 launch overhead，同时保留 eager trace。

5. （并行）针对真实 decoder shape 调研 SGLang、Diffusers、PyTorch/cuDNN、FlashAttention、CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断优化空间。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 model/config/dtype/channels/stride/tile/device guard、测试和 fallback 的 kernel。

6. （并行）研究 compile 后仍未融合好的数学等价操作，优先减少 global memory 读写、reshape/shuffle、tile materialize 和重复 layout；重点证明 nearest 2x upsample+3x3 Conv 到等价 ConvTranspose2d、GroupNorm+SiLU、bias/residual、attention output projection 与 tile overlap/blend 的融合条件。

7. 用未参与开发的 ImageNet-val/重建样本和固定 seed E2E 独立验收。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无颜色偏移和 tile seam；组件与 E2E 收益超过方差才接受，否则回到第 4 步继续优化。
