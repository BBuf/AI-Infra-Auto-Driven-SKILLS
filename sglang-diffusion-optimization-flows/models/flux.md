# FLUX family 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，根据目标分别下载 FLUX.1-dev、FLUX.2-dev 或 Klein checkpoint；gated repo 先配置 HF_TOKEN。国内机器可使用 HF_ENDPOINT，或用已经验证存在的 ModelScope 对应仓库下载到本地。FLUX.1、FLUX.2、Klein 必须分别跑 baseline。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   hf download black-forest-labs/FLUX.1-dev
   hf download black-forest-labs/FLUX.2-dev
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/flux
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux2 --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux2-klein --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、seed、分辨率、steps、guidance、dtype 和 GPU topology，保存 eager 图像、DiT 输出和 VAE latent。用固定照片、文字和高频纹理样本比较 PSNR、SSIM、LPIPS；FLUX.1、FLUX.2、Klein 4B/9B/Base 以及 FP8/NVFP4 路线各自建立 reference，不能混表。

3. 分析每个 checkpoint 的 native pipeline，拆分 T5/CLIP encoder、double/single transformer blocks、joint attention、modulation、RoPE、MLP、scheduler 和 VAE。FLUX.1 使用 AutoencoderKL 2D；FLUX.2/Klein 使用 AutoencoderKLFlux2，逐个记录 latent channels、scaling、tile 和真实 attention shape。

4. profile torch.compile 后各组件及 kernel 耗时，重点定位 joint attention、packed QKV、head_dim、GEMM、Nunchaku GELU MLP、modulation、RoPE、VAE norm/resblock/upsample、tile blend 和多卡通信；用相同输入保留 eager trace确认 graph break 与 fallback。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model flux2 --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）对关键 shape 调研 SGLang、Nunchaku、FlashInfer、FlashAttention、Diffusers、PyTorch、CUTLASS/Triton 的现有 fast path，再用 ncu-report skill 判断优化空间。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill 开发带完整 guard、测试和 fallback 的 attention、MLP 或 VAE kernel。

6. （并行）研究 compile 后仍存在的 fuse 机会，优先减少 global memory 读写、reshape、shuffle、permute、cat 和无效 materialize；重点检查 QKV/RoPE/norm、scale-shift-gate、residual、GELU MLP、upsample+conv、norm+activation 与 tile blend 的数学等价融合。NVFP4/FP8 属于单独质量预算。

7. 用独立 prompt 和 latent 验收每个目标 checkpoint 的精度、速度、峰值显存和多卡 scaling。component cosine 至少 0.999、normalized MSE 不超过 1e-4，图像 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002；20 次 warmup、100 次计时且组件与 E2E 都稳定获益才接受，否则回到第 4 步。
