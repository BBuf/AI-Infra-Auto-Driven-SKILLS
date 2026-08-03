# MOVA 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，分别下载 OpenMOSS-Team/MOVA-360p 和 MOVA-720p，使用人物条件与 4 GPU 建立 native baseline。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope 镜像下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download OpenMOSS-Team/MOVA-720p
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/mova
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model mova-720p --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定人物输入、prompt、seed、360p/720p、帧数、fps、steps、dtype 和 topology，保存 eager 视频与音频 reference。分别检查身份保持、逐帧 PSNR/SSIM、最差帧、waveform cosine/MSE、sample rate/channels、duration 和 A/V sync，禁止只验证容器可播放。

3. 分析 MOVA pipeline，拆分人物 condition encoder、DiT、video/audio branches、Wan-style causal 3D VAE、DAC audio VAE、scheduler、postprocess、mux 和分布式通信，记录 360p/720p 每阶段真实 shape、dtype、调用次数和条件 packing。

4. profile torch.compile 后完整 E2E、video/audio stage 与所有 kernel，重点定位 attention、GEMM、A2A、人物条件、causal Conv3d、tile blend、DAC Conv1d、resample、mux、CPU sync 与 graph break；保留 eager trace对照。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model mova-720p --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch、audio codec 和 CUTLASS/Triton 的已有实现，并用 ncu-report skill判断优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 resolution/audio/dtype/shape/device guard 与 fallback 的关键 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未融合好的数学等价路径，优先减少 global memory 读写、人物条件 packing、reshape/shuffle、audio/video layout 转换、tile 临时张量和 mux 前 copy；重点检查 modulation、norm/activation、residual、causal Conv、upsample+conv、DAC/resample 与 tile blend。

7. 用独立人物输入和 prompt 验收 360p/720p 的精度、速度、显存和 scaling。视频 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且身份不回归，audio normalized MSE 不超过 1e-4并保持 A/V sync；20 次 warmup、100 次计时且 video/audio 组件与 E2E 都超过方差才接受，否则回到第 4 步。
