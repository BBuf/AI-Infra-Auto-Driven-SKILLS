# JoyAI Image Edit 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 jdopensource/JoyAI-Image-Edit-Diffusers，并使用 native backend、2 GPU CFG parallel 建立 baseline。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope 镜像下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download jdopensource/JoyAI-Image-Edit-Diffusers
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/joyai-image-edit
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model joyai-edit --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定输入图、编辑 prompt、seed、1024x1024、40 steps、guidance 4、dtype 和 topology，保存 eager output、输入 encode 与 decoder latent。比较 PSNR、SSIM、LPIPS，同时用身份/结构保持和指令完成度样本验收；CFG parallel 与纯 SP 分别建 reference。

3. 分析 native edit pipeline，拆分输入图 preprocess/encode、condition encoder、condition packing、DiT attention/MLP/modulation、CFG、scheduler、Wan-style causal 3D VAE decode、postprocess 和通信，记录每个阶段真实 shape 与 cache 状态。

4. profile torch.compile 后各组件与 kernel 耗时，重点定位 edit encode、packed condition、attention、GEMM、A2A、causal Conv3d、resblock、upsample、tile blend、layout copy 和 graph break；用相同输入保留 eager trace。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model joyai-edit --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实 edit shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 现有实现，再用 ncu-report skill分析热点。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带输入数、CFG、dtype、tile 和 device guard 的 kernel，并保留 fallback。

6. （并行）研究 compile 后仍未融合好的等价操作，优先减少 global memory 读写、condition packing 的 reshape/shuffle、permute/cat、CFG materialize 和 VAE tile 临时张量；重点检查 modulation、norm/activation、residual、causal Conv3d 单帧路径、upsample+conv 和 tile blend。

7. 使用未参与开发的输入图和编辑 prompt 验收精度、指令完成度、速度、显存与多卡 scaling。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且身份/结构不回归；20 次 warmup、100 次计时并确认无 fallback，收益稳定才接受，否则回到第 4 步。
