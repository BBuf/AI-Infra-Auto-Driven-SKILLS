# VAE semantic rewrite 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载实际目标 VAE 的 checkpoint 并把 encode/decode 单独跑起来。下面用 FLUX.1-dev AutoencoderKL 作为独立例子；gated repo 设置 HF_TOKEN，国内机器可使用 HF_ENDPOINT 或验证过的 ModelScope 镜像。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   hf download black-forest-labs/FLUX.1-dev
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-rewrite
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 从真实 pipeline 保存 latent，并在 ImageNet-val 或对应视频/音频/shape 重建集上建立 eager reference。记录 component cosine、normalized MSE、PSNR、SSIM、LPIPS 或领域指标；固定 checkpoint、shape、dtype、tile/cache、seed 和 topology，不要求 bit-identical，但先定义合理容差。

3. 分析 decoder 架构并为每个候选写等价式和适用前提：nearest 2x upsample+3x3 Conv 到 ConvTranspose2d、causal Conv3d 单帧到 Conv2d、attention output projection 折叠进 value projection、bias/residual 到 GroupNorm statistics，以及 attention head_dim 盲区。记录 dtype/layout/device/shape/cache/train guard 和可回滚的原路径。

4. profile torch.compile 后各组件和 kernel 耗时，确认候选位于真实热点且收益上限值得实现；用 decode-only harness 做 20 次 warmup、100 次计时，记录 global memory traffic、launch、layout、tile overlap、graph break 和 E2E 占比。torch.compile 已经完成的优化不重复包装。

5. （并行）对关键 shape 调研 PyTorch、Diffusers、SGLang、cuDNN、FlashInfer、FlashAttention、CUTLASS/Triton 的现有实现，并用 ncu-report skill证明剩余空间。需要新 kernel 时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带完整 guard、幂等安装、restore/rollback、测试和 fallback 的实现。

6. （并行）优先实现数学等价且能移除大中间张量的 rewrite，再研究 kernel 内 fuse；重点减少 global memory 读写、reshape/shuffle、layout conversion、tile materialize 和重复统计。离线合成权重使用 FP32，并记录 checkpoint/source/shape/dtype/topology 指纹；一次只提交一个候选。

7. 使用未参与开发的重建样本独立验收精度、组件速度、E2E 和 1000-call soak。component cosine 至少 0.999、normalized MSE 不超过 1e-4，图像/视频 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且边界/缓存语义正确；收益或精度不达标就拒绝候选并回到第 4 步。
