# LTX-2 / LTX-2.3 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 Lightricks/LTX-2.3，并分别为 one-stage、two-stage、HQ、TI2V 和 CFG parallel 建立 baseline；LTX-2 也补一轮独立 baseline。国内机器可使用 HF_ENDPOINT，或用确认存在的 ModelScope 镜像下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Lightricks/LTX-2.3
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/ltx
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model ltx23-ti2v-two-stage --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model ltx23-one-stage --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model ltx23-hq-two-stage --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、输入图、seed、分辨率、121 帧、fps、steps、stage 切换、dtype 和 topology，保存 eager video/audio reference。比较逐帧 PSNR/SSIM、最差帧、TI2V 输入保持、时间一致性、waveform MSE、采样率和 A/V sync；one/two-stage/HQ 分别建质量基线。

3. 分析 LTX pipeline，拆分 condition encoder、DiT split RoPE、attention/MLP、residual-gate add、双阶段调度、CFG parallel、AutoencoderKLLTX2Video、audio VAE/vocoder、upsampler GroupNorm+SiLU、postprocess 和通信，记录真实 shape 和调用次数。

4. profile torch.compile 后各 stage、组件和 kernel，重点定位 split RoPE、attention、GEMM、A2A、residual-gate add、causal Conv3d、upsampler、tile/stream overlap、audio decode、mux 和 graph break；保留 eager trace对照。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model ltx23-ti2v-two-stage --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）对关键 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch、audio codec 和 CUTLASS/Triton 的已有 kernel，使用 ncu-report skill判断优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 stage/causal/dtype/shape/device guard 与 fallback 的实现。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍不理想的等价 fuse，优先减少 global memory 读写、reshape/shuffle、RoPE/layout materialize、stage 间 copy 和 tile 临时张量；重点检查 QKV/RoPE/norm、residual-gate、GroupNorm+SiLU、upsample+conv、tile overlap/blend 与 audio resample。

7. 用独立 TI2V、T2V 和音频样本验收 one-stage、two-stage、HQ、CFG parallel 的精度、速度、显存与 scaling。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且 A/V 同步；20 次 warmup、100 次计时并确认收益超过方差，否则回到第 4 步。
