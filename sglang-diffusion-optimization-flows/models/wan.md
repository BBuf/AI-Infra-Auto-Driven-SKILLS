# Wan / FastWan / TurboWan 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，按目标下载 Wan2.1/2.2 T2V、I2V、TI2V、FastWan、TurboWan、Wan-Fun 或 NVIDIA NVFP4 checkpoint，并先跑 T2V/TI2V/I2V 三个 native baseline。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope checkpoint 下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/wan
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model wan-t2v --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model wan-ti2v --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model wan-i2v --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model fastwan22-ti2v-5b --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、输入图、seed、resolution、frames、fps、steps、dtype 和 topology，保存 eager T2V/I2V/TI2V reference。比较逐帧 PSNR/SSIM、最差帧、输入图保持、运动、首尾和闪烁；1.3B/5B/A14B、蒸馏/少步数、NVFP4 分别建立质量预算。

3. 分析 Wan native pipeline，拆分 text/image encoder、DiT attention/MLP/modulation、packed layout、scheduler、AutoencoderKLWan causal 3D encode/decode、temporal/spatial/patch tiling、postprocess、CFG/Ulysses 通信与 offload，记录真实 shape 和合法并行组合。

4. profile torch.compile 后各组件和 kernel 耗时，重点定位 attention、GEMM、A2A/all-gather、packed layout、modulation、causal Conv3d、norm、resblock、upsample、tile blend、offload copy 与 graph break；保留相同输入的 eager trace。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model wan-ti2v --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断计算、访存或通信空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 task/model/dtype/tile/topology/device guard 与 fallback 的关键 kernel。

6. （并行）研究 compile 后仍未融合好的数学等价路径，优先减少 global memory 读写、reshape/shuffle、packed layout 转换、CFG materialize、跨 rank gather 和 tile 临时张量；重点检查 QKV/RoPE/norm、modulation、residual、causal Conv3d 单帧路径、upsample+conv 与 tile overlap/blend。

7. 用独立 T2V/I2V/TI2V 输入验收 1.3B/5B/A14B 及目标 variant 的精度、速度、显存和多卡 scaling。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/flicker；20 次 warmup、100 次计时且收益超过方差才接受，否则回到第 4 步。
