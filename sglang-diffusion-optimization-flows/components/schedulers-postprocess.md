# Diffusion scheduler / postprocess / delivery 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载一个实际 checkpoint 并跑包含 scheduler、postprocess 和 delivery 的完整 native baseline。下面用 Wan T2V 作为独立例子；国内机器可使用 HF_ENDPOINT 或验证过的 ModelScope 镜像。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/scheduler-postprocess
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model wan-t2v --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、seed、timesteps、sigmas、flow shifts、generator state、dtype、颜色范围、fps/timestamp、sample rate/channels、codec/container 或 mesh contract，保存每一步 latent 和最终 eager reference。输出可打开只是最低门槛，还要比较对应 PSNR/SSIM、waveform 或 mesh 指标。

3. 分析 scheduler 与 delivery 架构，拆分 scheduler step/random、latent cast/copy、safety/guardrail、image/video postprocess、interpolation/upscale、audio resample、mux、codec 和 mesh export，记录 CPU/GPU 边界、同步点、tensor shape、调用次数与生产开关。

4. profile torch.compile 后每个 scheduler step 和 postprocess stage，定位 Python loop、CPU sync、重复 tensor 构造、device transfer、pointwise update、frame/audio copy、codec/mux 和 graph break；只有阶段占 E2E 至少约 3% 或阻断吞吐时才进入 kernel 优化。

5. （并行）针对真实 scheduler/postprocess shape 调研 SGLang、PyTorch、TorchCodec、FFmpeg、Diffusers 和 CUTLASS/Triton 已有实现，用 ncu-report skill判断 GPU 热点空间。需要新 kernel 时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 scheduler/dtype/shape/device/format guard、测试和 fallback 的实现。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未融合好的等价操作，优先减少 global memory 读写、latent cast/copy、reshape/shuffle、frame/audio materialize 和 CPU/GPU 往返；重点检查 scheduler pointwise update 批量化、颜色转换、upscale/interpolation、resample 与 mux 前 copy。近似 upscale/interpolation 单列质量预算。

7. 用独立输入完整验收每一步 latent、最终媒体/mesh、stage latency 和 E2E。timesteps/sigmas/shifts 与媒体格式合同必须保持，图像/视频 PSNR、SSIM、音频 waveform 或 mesh 指标在预设容差内；20 次 warmup、100 次计时且 stage 与 E2E 收益都超过方差才接受，否则回到第 4 步。
