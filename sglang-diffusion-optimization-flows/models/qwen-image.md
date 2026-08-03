# Qwen-Image / FireRed 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，根据目标下载 Qwen-Image、Edit、2512/2509/2511、Layered、NVIDIA NVFP4 或 FireRed Edit checkpoint，并分别建立 native baseline。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope 镜像下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Qwen/Qwen-Image
   hf download Qwen/Qwen-Image-Edit
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/qwen-image
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model qwen --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model qwen-edit --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、输入图、seed、分辨率、steps、guidance、dtype 和 topology，保存 T2I/Edit/Layered 的 eager output、encode 与 latent reference。比较 PSNR、SSIM、LPIPS、文字和编辑保持；Layered 另验层数、alpha、ordering，NVFP4 与 FireRed 分别建立质量预算。

3. 分析 native pipeline，拆分 Qwen condition encoder、DiT attention/MLP/modulation、fused QK norm/RoPE、packed sequence、多图 condition packing、CFG/TP、scheduler、AutoencoderKLQwenImage causal VAE、tiling 和 postprocess，记录各 variant 真实 shape。

4. profile torch.compile 后各 stage、组件和 kernel，重点定位 attention、GEMM、QK norm/RoPE、modulation、多图 packing、CFG、causal Conv3d、resblock、upsample、tile blend、A2A 与 graph break；保留相同输入的 eager trace。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model qwen-edit --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 variant/input-count/layer/dtype/shape/device guard 与 fallback 的关键 kernel。

6. （并行）研究 compile 后仍未融合好的数学等价操作，优先减少 global memory 读写、reshape/shuffle、packed layout 转换、多图 condition materialize、CFG copy 和 tile 临时张量；重点检查 QK norm+RoPE、modulation、residual、causal Conv3d 单帧路径、upsample+conv 与 tile blend。

7. 用独立 T2I、Edit、Layered 和 FireRed 样本验收精度、速度、显存与多卡 scaling。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且编辑/层语义不回归；20 次 warmup、100 次计时且无 fallback/cache 污染才接受，否则回到第 4 步。
