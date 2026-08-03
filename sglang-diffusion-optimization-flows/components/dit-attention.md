# DiT / attention / modulation 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载一个真实 checkpoint 并跑 native DiT baseline。下面用 FLUX.1-dev 作为独立例子；gated repo 设置 HF_TOKEN，国内机器可使用 HF_ENDPOINT 或验证过的 ModelScope 镜像。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   hf download black-forest-labs/FLUX.1-dev
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/dit-attention
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 checkpoint、prompt、seed、latent、timestep、context、dtype 和 topology，保存 eager block/DiT/E2E reference。比较 block output cosine 和 normalized MSE，再用固定图像或视频输出比较 PSNR、SSIM、LPIPS；不同 model/head_dim/token shape 各自建 reference。

3. 分析 DiT 架构，拆分 self/cross/joint attention、QKV projection、QK norm、RoPE、softmax、output projection、MLP、norm、scale/shift/gate、residual、packed/varlen layout 和通信；记录每个热点的 batch、heads、tokens、head_dim、dtype、stride 与调用次数。

4. profile torch.compile 后各 block、组件和 kernel 耗时，检查 fused QK norm、QK norm+RoPE、packed QKV/KV、varlen USP、scale-shift-gate、residual-gate 是否命中，并定位 attention backend、GEMM、materialize、A2A/all-gather、graph break 和 launch overhead。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model flux --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实 shape 调研 SGLang、FlashInfer、FlashAttention、SDPA、xFormers、PyTorch、CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断 compute、memory、occupancy 或 launch bound。head_dim 等盲区仍是关键热点时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带严格 guard、测试与 fallback 的 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未融合好的等价操作，优先减少 global memory 读写、reshape/shuffle、repeat、cat、permute、packed layout materialize 和全量 gather；重点检查 QKV+norm+RoPE、attention output projection、scale-shift-gate、residual、MLP activation 与通信 overlap，一次只合入一个可归因改动。

7. 使用独立 latent/timestep/context 和 E2E prompt 验收精度、组件速度、E2E 与多卡通信。DiT/block cosine 至少 0.995、normalized MSE 不超过 1e-4，最终 PSNR/SSIM 在预设容差内；20 次 warmup、100 次计时且组件与 E2E 都超过方差才接受，否则回到第 4 步。
