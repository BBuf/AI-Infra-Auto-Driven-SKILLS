# SANA AutoencoderDC 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 SANA checkpoint，并通过 native pipeline 把 runtime/models/vaes/autoencoder_dc.py::AutoencoderDC 单独跑起来。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope checkpoint 下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-dc-ae
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model sana-1.5-1.6b --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 保存 512px 和 1024px 的真实 latent，在 ImageNet-val 或固定重建集上建立 eager reference。记录 output cosine、normalized MSE、PSNR、SSIM、LPIPS，并重点检查 32x compression 下的细线、文字、小物体、颜色和 tile seam；600M、1.6B、4.8B 的代表 config 分别验收。

3. 分析 AutoencoderDC wrapper 与 Diffusers inner model，拆分 encode/decode、conv、channel mixing、norm/activation、upsample、tiling 和 postprocess；记录真实 channels、stride、compression ratio、tile、dtype、layout 和 wrapper copy。

4. 用保存的 latent 建 decode-only harness，profile torch.compile 后 wrapper、inner model、各层和 kernel 耗时，定位 conv、channel mixing、upsample、layout conversion、tile blend、graph break 和 launch overhead；20 次 warmup、100 次计时并保留 eager trace。

5. （并行）针对真实 DC-AE shape 调研 Diffusers、PyTorch/cuDNN、SGLang、CUTLASS/Triton 已有实现，并用 ncu-report skill判断优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 config/dtype/channels/stride/tile/device guard、测试和 fallback 的 kernel。

6. （并行）研究 compile 后仍未融合好的数学等价操作，优先减少 global memory 读写、wrapper copy、reshape/shuffle、layout 转换和 tile 临时张量；重点检查 channel mixing、norm+activation、residual、upsample+conv 和 tile overlap/blend，不能假设与其他 VAE 的 tiling 语义相同。

7. 用独立 ImageNet-val/重建样本和 SANA 600M/1.6B E2E 验收精度、速度和显存。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且细节/颜色无回归；组件和 E2E 都超过方差才接受，否则回到第 4 步。
