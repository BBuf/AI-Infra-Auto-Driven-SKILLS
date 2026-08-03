# Z-Image 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，分别下载 Tongyi-MAI/Z-Image 与 Z-Image-Turbo 并建立 native baseline。国内机器可使用 HF_ENDPOINT，也可使用 ModelScope 上已经确认存在的同名 checkpoint 下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Tongyi-MAI/Z-Image
   hf download Tongyi-MAI/Z-Image-Turbo
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/z-image
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model zimage --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model zimage-base --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、seed、resolution、steps、guidance、dtype 和 topology，分别保存 Base/Turbo 的 eager 图像、DiT 输出和 VAE latent。用固定 ImageNet-val 子集或重建集比较 PSNR、SSIM、LPIPS、颜色和细节；Base 与 Turbo 的步数/质量不得混表。

3. 分析 Z-Image native pipeline，拆分 condition encoder、DiT attention/MLP/modulation、bf16 Triton norm/tanh residual、scheduler、AutoencoderKL 2D conv/resblock/mid attention/upsample/tile 和 postprocess，记录真实 shape、dtype 与 fast-path dispatch。

4. profile torch.compile 后各组件及 kernel 耗时，先确认现有 bf16 Triton norm/tanh residual 是否命中，再定位 attention、GEMM、modulation、decoder conv、GroupNorm/SiLU、upsample、tile blend 和 graph break；保留 eager trace。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model zimage --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 已有 kernel，并用 ncu-report skill确认现有 Triton fast path 和其他热点的优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 base/turbo/dtype/shape/device guard、测试和 fallback 的关键 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未完成的数学等价 fuse，优先减少 global memory 读写、reshape/shuffle、condition materialize 和 tile 临时张量；重点检查 norm+tanh residual、QKV/RoPE、modulation、GroupNorm+SiLU、residual、upsample+conv 与 tile overlap/blend。

7. 用独立 prompt 和重建样本验收 Base/Turbo 的精度、速度、显存和 native dispatch。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002；20 次 warmup、100 次计时且组件和 E2E 都稳定获益才接受，否则回到第 4 步。
