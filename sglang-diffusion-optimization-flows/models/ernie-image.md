# ERNIE-Image 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 ERNIE-Image-Turbo 并建立 native SGLang baseline；Base checkpoint 也至少完成加载和固定 seed smoke。国内机器可使用 HF_ENDPOINT，或用已验证的 ModelScope 对应仓库下载到本地目录。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download baidu/ERNIE-Image-Turbo
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/ernie-image
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model ernie-image-turbo --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、seed、1024px 分辨率、steps、guidance、dtype 和 GPU topology，保存 eager 输出与进入 VAE decoder 的 latent。使用固定图像重建集比较 PSNR、SSIM、LPIPS、颜色和文字细节；Turbo 与 Base 分开建立质量和性能 reference。

3. 分析 ERNIE-Image native pipeline，拆分 condition encoder、DiT attention/MLP/modulation、scheduler、AutoencoderKLFlux2 encode/decode 和 postprocess；记录 latent scaling、patch/config、batch norm、tile 参数和各阶段真实 shape。

4. profile torch.compile 后的完整 E2E、denoise 和 VAE stage，按组件及 kernel 汇总耗时，定位 joint attention、GEMM、norm、RoPE、batch norm、resblock、upsample、tile blend、layout copy 与 graph break；保留相同输入的 eager trace用于对照。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model ernie-image-turbo --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对 profile 的关键 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch、CUTLASS/Triton 已有实现，并用 ncu-report skill 分析瓶颈。已有 kernel 不够快时启动 kernel design sub agent，使用 ultra 模式和 KernelWiki、ncu-report skill 开发专用 kernel，保留 dtype、channels、stride、tile 和设备 guard 以及原实现 fallback。

6. （并行）研究 compile 后仍未融合好的数学等价路径，优先消除 global memory 往返、reshape/shuffle、permute/cat 和重复标准化；重点检查 modulation、norm+activation、residual、upsample+conv、tile overlap/blend 与 latent scaling 的融合。量化、少步数和 cache 路线单独记为近似优化。

7. 使用独立 prompt 与重建样本复验 Turbo 和 Base 的精度、速度与显存。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002；20 次 warmup、100 次计时并确认 native backend 无 fallback，收益超过方差才接受，否则回到第 4 步继续优化。
