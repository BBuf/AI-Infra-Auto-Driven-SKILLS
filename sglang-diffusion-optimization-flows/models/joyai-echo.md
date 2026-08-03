# JoyAI-Echo 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 jdopensource/JoyAI-Echo 并使用 native backend 建立固定短视频 baseline。国内机器可使用 HF_ENDPOINT，或把验证过的 ModelScope checkpoint 下载到本地目录。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=/data/models/JoyAI-Echo
   hf download jdopensource/JoyAI-Echo --local-dir "$MODEL_DIR"
   BENCH_DIR=/tmp/sglang-diffusion-bench/joyai-echo
   mkdir -p "$BENCH_DIR"
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A curious raccoon" --width 640 --height 384 --num-frames 33 --num-inference-steps 8 --seed 42 --num-gpus 2 --ulysses-degree 2 --enable-memory-bank false --save-output --enable-torch-compile --warmup --perf-dump-path "$BENCH_DIR/baseline.json"
   ~~~

2. 固定 prompt、seed、分辨率、帧数、fps、steps、audio window、memory-bank 参数、dtype 和 topology，保存 eager 视频与音频 reference。比较逐帧 PSNR/SSIM、最差帧、闪烁、waveform cosine/MSE、采样率、声道、duration 和 A/V sync；再做长视频 cache reset 与显存有界测试。

3. 分析 pipeline，拆分 condition encoder、DiT、paired audio-video memory bank、96-window audio selection、mel preprocess、LTX-2.3 video VAE、LTX audio VAE/vocoder、scheduler、AV decode 和 mux，记录每个 stage 的 shape、dtype、调用次数和 cache key。

4. 分别 profile torch.compile 开/关的完整 E2E 和各 stage，按 kernel 汇总 attention、GEMM、memory-bank copy、mel、Conv1d/Conv3d、resblock、upsampler、vocoder、resample 和 mux；重点观察首次与 steady-state latency、graph break 和 CPU sync。

   ~~~bash
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A curious raccoon" --width 640 --height 384 --num-frames 33 --num-inference-steps 8 --seed 42 --num-gpus 2 --ulysses-degree 2 --enable-memory-bank false --save-output --enable-torch-compile --warmup --profile --profile-all-stages --perf-dump-path "$BENCH_DIR/compile-profile.json"
   ~~~

5. （并行）对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch、audio codec 和 CUTLASS/Triton 已有实现，使用 ncu-report skill判断优化空间。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带窗口、cache、dtype、shape guard 与 fallback 的关键 kernel。

6. （并行）研究 compile 后仍未融合好的等价操作，优先减少 global memory 读写、audio/video layout 转换、reshape/shuffle、重复 window materialize 和 CPU/GPU 往返；重点检查 modulation、norm/activation、residual、upsample+conv、mel/resample 批量化与 tile blend，保持 causal 和同步语义。

7. 用独立音视频 prompt、memory-bank on/off 与长视频 soak 验收速度、显存和正确性。视频 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/flicker，audio normalized MSE 不超过 1e-4并保持 sample count/channel/A-V sync；20 次 warmup、100 次计时且组件与 E2E 都超过方差才接受，否则回到第 4 步。
