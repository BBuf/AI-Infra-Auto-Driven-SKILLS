# LongLive 2.0 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 Diffusers 或官方 layout checkpoint 并用 native backend 建立 4-step causal DMD baseline。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope 镜像下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=/data/models/LongLive-2.0-5B
   hf download Rabinovich/LongLive-2.0-5B-Diffusers --local-dir "$MODEL_DIR"
   BENCH_DIR=/tmp/sglang-diffusion-bench/longlive2
   mkdir -p "$BENCH_DIR"
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A long continuous walk through a city" --width 832 --height 480 --num-frames 61 --num-inference-steps 4 --guidance-scale 1.0 --seed 42 --save-output --enable-torch-compile --warmup --perf-dump-path "$BENCH_DIR/baseline.json"
   ~~~

2. 固定 prompt、seed、分辨率、fps、steps、num_frames_per_block、dtype 和 topology，保存 eager reference；latent frame 数必须可被 block size 整除。检查逐帧 PSNR/SSIM、chunk seam、运动连续、首帧条件、cache reset、长程漂移和显存有界，官方/转换 layout 分别验收。

3. 分析 native pipeline，拆分 condition encoder、causal DiT、block cache、4-step DMD scheduler、I2V 首帧 encode、跨 chunk memory、Wan-style causal 3D VAE decode、postprocess 和通信，记录每阶段 shape、dtype、cache key 与调用次数。

4. profile torch.compile 后首 chunk 和 steady-state 的完整 E2E、组件与 kernel，重点定位 attention、cache copy、A2A、causal Conv3d、resblock、upsample、tile blend、layout materialize、CPU sync 和 graph break；保留 eager trace对照。

5. （并行）针对真实 chunk shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 的已有实现，再用 ncu-report skill判断优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 block/cache/dtype/shape/device guard 与 fallback 的关键 kernel。

6. （并行）研究 compile 后仍未完成的等价 fuse，优先减少 global memory 读写、cache materialize、reshape/shuffle、layout copy 和跨 chunk gather；重点检查 modulation、norm/activation、residual、causal Conv3d、upsample+conv 与 tile overlap/blend，保持 DMD 和 causal 边界。

7. 用独立长视频和连续 session 验收性能、显存、cache 与正确性。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/flicker；20 次 warmup、100 次计时，首 chunk 和 steady-state 都稳定获益才接受，否则回到第 4 步。
