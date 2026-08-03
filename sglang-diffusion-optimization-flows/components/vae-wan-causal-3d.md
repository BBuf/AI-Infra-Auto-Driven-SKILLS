# AutoencoderKLWan causal 3D VAE 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 Wan checkpoint，并通过 native pipeline 把 runtime/models/vaes/wanvae.py::AutoencoderKLWan 单独跑起来。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope checkpoint 下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-wan-causal-3d
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model wan-t2v --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 保存 Wan 720p decode、I2V encode+decode 和 Cosmos3 z_dim=48 的真实 latent，在固定视频重建集上分别建立 eager reference。记录 cosine、normalized MSE、逐帧 PSNR/SSIM、最差帧、首尾、seam、flicker 和 cache；Wan、Cosmos3、Helios、JoyAI、MOVA、LingBot、LongLive 的代表 config 分别验收。

3. 分析 AutoencoderKLWan 架构，拆分 causal Conv3d、resblock、RMS/GroupNorm、attention、upsample、temporal/spatial/patch tiling、跨 rank gather/blend 和 postprocess；记录 z_dim、channels、frames、stride、padding、dtype、layout、tile、cache 与合法 parallel decode 模式。

4. 用保存的 latent 建 encode/decode harness，profile torch.compile 后各组件与 kernel 耗时，定位 causal Conv3d、norm、resblock、attention、upsample、tile overlap/blend、layout copy、跨 rank gather 和 graph break；分别测 full/spatial/temporal/patch 合法模式，20 次 warmup、100 次计时。

5. （并行）针对真实视频 shape 调研 SGLang、Diffusers、PyTorch/cuDNN、FlashAttention、CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断空间。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 model/z_dim/causal/cache/dtype/tile/topology/device guard、测试和 fallback 的 kernel。

6. （并行）研究 compile 后仍未融合好的数学等价路径，优先减少 global memory 读写、reshape/shuffle、temporal/spatial layout 转换、跨 rank materialize 和 tile 重叠计算；仅在严格 T=1 且无 cache 时证明 Conv3d 到 Conv2d 等价，并检查 norm+activation、residual、upsample+conv 和 tile blend。

7. 用独立 Wan、Cosmos3 和其他目标 config 的 latent/E2E 验收精度、速度、显存和 parallel decode。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/flicker；所有 guard/fallback 和代表 config 都通过才接受，否则回到第 4 步。
