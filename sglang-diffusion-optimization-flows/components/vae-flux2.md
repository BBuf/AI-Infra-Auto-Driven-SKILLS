# AutoencoderKLFlux2 VAE 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 FLUX.2 checkpoint 并通过 native pipeline 把 runtime/models/vaes/autoencoder_kl_flux2.py::AutoencoderKLFlux2 单独跑起来。gated repo 设置 HF_TOKEN；国内机器可使用 HF_ENDPOINT 或已验证的 ModelScope 镜像。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   hf download black-forest-labs/FLUX.2-dev
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-flux2
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux2 --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 从 FLUX.2、Ideogram 4 和 ERNIE-Image 各保存一个真实 latent shape，在固定重建集上分别建立 eager reference。记录 output cosine、normalized MSE、PSNR、SSIM、LPIPS、颜色和 tile seam；FP8/NVFP4 checkpoint 也记录 VAE 的实际 dtype，不能跨 config 复用数值结论。

3. 分析 AutoencoderKLFlux2 架构，拆分 encode/decode、batch norm、resblock、attention、upsample、parallel tiling、tile blend 和 postprocess；记录每个 config 的 latent/patch、channels、stride、tile、dtype、layout 与调用次数。

4. 用保存的 latent 建 decode-only harness，profile torch.compile 后各组件与 kernel 耗时，定位 batch norm、conv、attention、resblock、upsample、tile blend、layout conversion、graph break 和 launch overhead；20 次 warmup、100 次计时并保留 eager trace。

5. （并行）针对真实 shape 调研 SGLang、Diffusers、PyTorch/cuDNN、FlashAttention、CUTLASS/Triton 已有 kernel，并用 ncu-report skill分析优化空间。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 model/config/dtype/channels/stride/tile/device guard、测试和 fallback 的 kernel。

6. （并行）研究 compile 后仍未融合好的数学等价路径，优先减少 global memory 读写、reshape/shuffle、layout 转换和 tile 临时张量；重点检查 batch norm/activation、residual、upsample+conv、attention output projection 与 tile overlap/blend，一次只合入一个可归因改动。

7. 使用独立重建样本和 FLUX.2、Ideogram、ERNIE 固定 seed E2E 验收。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/色偏；组件与三种 config 的 E2E 收益都超过方差才接受，否则回到第 4 步。
