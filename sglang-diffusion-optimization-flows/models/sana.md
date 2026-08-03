# SANA / SANA-WM 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，分别下载 SANA 图像 checkpoint 和 Efficient-Large-Model/SANA-WM_streaming；两者只共享名称，必须分别建 baseline。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope 镜像下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
   hf download Efficient-Large-Model/SANA-WM_streaming
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/sana
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model sana-1.5-1.6b --label baseline --output-dir "$BENCH_DIR"
   sglang generate --backend sglang --model-path Efficient-Large-Model/SANA-WM_streaming --prompt "The subject slowly turns toward the camera" --image-path /tmp/sana-wm-input.png --width 640 --height 384 --num-frames 17 --fps 16 --num-inference-steps 12 --guidance-scale 4.5 --seed 0 --save-output --enable-torch-compile --warmup --perf-dump-path "$BENCH_DIR/sana-wm-baseline.json"
   ~~~

2. SANA 图像固定 prompt、seed、512/1024、steps 和 dtype，用重建集比较 PSNR、SSIM、LPIPS、文字和细线；SANA-WM 固定输入图、prompt、seed、chunk、帧数、fps 和 topology，比较逐帧 PSNR/SSIM、首帧、chunk seam、steady-state FPS、cache reset 和长时显存。600M/1.6B/4.8B、bidirectional/streaming 分开建 reference。

3. 分析两条 native pipeline。SANA 拆分 encoder、DiT、scheduler、AutoencoderDC 和 postprocess；SANA-WM 拆分 encoder、causal DiT/cache、scheduler、AutoencoderKLLTX2Video、streaming/tile state 和通信。分别记录真实 shape、dtype、调用次数与 cache 合同。

4. profile torch.compile 后各自的组件与 kernel 耗时，SANA 重点检查 attention/GEMM、DC-AE conv/channel mixing/upsample；SANA-WM 重点检查 causal attention/cache、A2A、causal Conv3d、upsampler、chunk/tile blend、CPU sync 和 graph break；都保留 eager trace。

5. （并行）针对两条 pipeline 的真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 现有实现，并用 ncu-report skill分析空间。需要新 kernel 时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 model/stream/cache/dtype/shape/device guard 与 fallback 的实现。

6. （并行）研究 compile 后仍未处理好的等价 fuse，优先减少 global memory 读写、reshape/shuffle、cache/materialize、layout 转换和 tile 临时张量；SANA 重点看 norm/activation、residual、upsample+conv，SANA-WM 重点看 causal cache、Conv3d、upsampler 与 chunk blend。

7. 用独立图像与长视频输入验收两条 pipeline。component cosine 至少 0.999、normalized MSE 不超过 1e-4，图像/逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002，streaming 无 seam/flicker且显存有界；20 次 warmup、100 次计时，组件与 E2E 都超过方差才接受，否则回到第 4 步。
