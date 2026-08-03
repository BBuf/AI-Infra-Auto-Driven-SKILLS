# Diffusion distributed runtime 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载一个实际要优化的 checkpoint，并先跑能够驻留的最小 GPU baseline。下面用 Wan T2V 作为可替换的独立例子；国内机器可使用 HF_ENDPOINT 或已验证的 ModelScope 镜像。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/distributed-runtime
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model wan-t2v --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 checkpoint、prompt、seed、resolution、frames、steps、dtype、卡数和 rank mapping，保存单卡或最小卡数 eager reference。对合法 TP、Ulysses/SP、CFG parallel、FSDP/offload、async A2A 组合分别比较输出 cosine/MSE、图像或逐帧 PSNR/SSIM、cache reset 和 100 次连续请求，禁止静默 fallback。

3. 分析 distributed runtime 架构，画出 DiT、VAE、encoder 在每个 rank 的 shard、all-gather、reduce-scatter、A2A、broadcast、CFG split、offload 和 cache 生命周期；记录每处 tensor shape、dtype、bytes、stream、同步点、机内/跨机 topology 与合法约束。

4. 在同一 workload 上 profile torch.compile 后各组件、collective 和 kernel 耗时，报告 E2E、denoise、VAE、通信占比、overlap、等待、峰值显存、首轮与 steady-state；用 eager trace识别 compile graph break、额外 gather、rank shape 漂移或串行化。

5. （并行）对真实 collective 和计算 shape 调研 NCCL、SGLang、DeepEP、FlashInfer、PyTorch distributed 与已有 fused communication kernel，并用 ncu-report skill和通信 trace判断瓶颈。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 world-size/topology/dtype/shape guard、超时诊断和 fallback 的 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未实现的通信/计算 fuse，优先减少 global memory 读写、重复 shard/gather、reshape/shuffle、packed/unpacked layout 和 host sync；重点检查 A2A 与 QKV、CFG concat、norm/modulation、VAE tile gather/blend 的 overlap 或数学等价融合。通信不是热点时不做投机改动。

7. 用独立 workload 在至少两种卡数验收精度、E2E、峰值显存、speedup 和 parallel efficiency；输出必须与 reference 在预设容差内，无 rank divergence、stale cache、显存增长或 fallback。20 次 warmup、100 次计时且通信与 E2E 收益都超过方差才接受，否则回到第 4 步。
