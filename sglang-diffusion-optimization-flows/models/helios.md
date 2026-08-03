# Helios 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境并下载 Helios-Base；Mid 和 Distilled 也分别下载并至少跑一轮 smoke。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope 镜像下载到本地路径。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download BestWishYsh/Helios-Base
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/helios
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model helios --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、seed、分辨率、帧数、fps、steps、dtype 和 topology，分别保存 Base/Mid/Distilled 的 eager reference。用固定视频集检查逐帧 PSNR/SSIM、最差帧、首尾、闪烁、运动连续性、cache reset 和长视频显存有界。

3. 分析 Helios native pipeline，拆分 condition encoder、专用 denoising stage、DiT attention/MLP/modulation、block cache/causal state、scheduler、Wan-style causal 3D VAE、postprocess 和多卡通信，记录各 variant 的真实 shape 与调用次数。

4. profile torch.compile 后的完整 E2E 和所有 stage，重点定位 attention、GEMM、causal cache、A2A、causal Conv3d、resblock、upsample、tile overlap/blend、CPU 同步与 graph break；保留 eager trace比较首次和 steady-state latency。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model helios --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）对关键 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 已有实现，使用 ncu-report skill 判断 kernel 或通信是否还有空间。需要新 kernel 时启动 kernel design sub agent，使用 ultra 模式和 KernelWiki、ncu-report skill实现严格 guard、cache 语义测试及 fallback。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍不理想的等价 fuse，优先减少 global memory 读写、reshape/shuffle、layout copy、cache materialize 和跨 rank gather；重点检查 modulation、norm/activation、residual、causal Conv3d、upsample+conv 与 tile blend，保持 chunk/cache 语义。

7. 使用独立视频和长时 soak 验收 Base/Mid/Distilled 的精度、速度、显存和 cache 稳定性。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/flicker；20 次 warmup、100 次计时，收益超过方差才接受，否则回到第 4 步。
