# AutoencoderKLHunyuanVideo VAE 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 HunyuanVideo checkpoint，并通过 native pipeline 把 runtime/models/vaes/hunyuanvae.py::AutoencoderKLHunyuanVideo 单独跑起来。国内机器可使用 HF_ENDPOINT，或把验证过的 ModelScope checkpoint 下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download hunyuanvideo-community/HunyuanVideo
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-hunyuan-video
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model hunyuanvideo --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 保存生产分辨率和帧数的真实 latent，在固定视频重建集上建立 eager reference。记录 latent/output cosine、normalized MSE、逐帧 PSNR/SSIM、最差帧、首尾、seam 和 flicker；HunyuanVideo 与 FastHunyuan 分别验收，少步数质量变化不能归因给 VAE。

3. 分析 AutoencoderKLHunyuanVideo 架构，拆分 decoder conv、resblock、attention、upsampler GroupNorm+SiLU、temporal/spatial tiling、tile blend 与 decode postprocess；记录 channels、stride、padding、tile/overlap、dtype、layout、调用次数和 causal 边界。

4. 用保存的 latent 建 decode-only harness，profile torch.compile 后各组件与 kernel 耗时，先确认已有 Hunyuan/LTX upsampler fusion 是否命中，再定位 conv、resblock、attention、GroupNorm+SiLU、upsample、tile blend、layout copy 和 graph break；20 次 warmup、100 次计时。

5. （并行）针对真实视频 shape 调研 SGLang、Diffusers、PyTorch/cuDNN、FlashAttention、CUTLASS/Triton 已有 kernel，并用 ncu-report skill确认优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 dtype/channels/frames/tile/device guard、测试和 fallback 的 kernel。

6. （并行）研究 compile 后仍未融合好的数学等价操作，优先减少 global memory 读写、reshape/shuffle、temporal/spatial layout 转换和 tile 临时张量；重点检查 GroupNorm+SiLU、residual、upsample+conv、attention output projection 与 tile overlap/blend，保持边界和时间语义。

7. 用独立 latent、视频和 FastHunyuan E2E 验收精度、速度与峰值显存。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/flicker；component 与 E2E 均稳定获益才接受，否则回到第 4 步。
