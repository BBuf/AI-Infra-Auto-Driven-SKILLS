# AutoencoderKLLTX2Video VAE 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 Lightricks/LTX-2.3，并通过 native pipeline 把 runtime/models/vaes/ltx_2_vae.py::AutoencoderKLLTX2Video 单独跑起来。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope checkpoint 下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Lightricks/LTX-2.3
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-ltx-video
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model ltx23-ti2v-two-stage --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 从 LTX-2.3 two-stage、SANA-WM streaming 和 JoyAI-Echo 各保存真实 latent，在固定视频重建集上分别建立 eager reference。记录 cosine、normalized MSE、逐帧 PSNR/SSIM、最差帧、chunk/tile seam、时间漂移、历史状态和长视频显存；不同 config 不共享数值结论。

3. 分析 AutoencoderKLLTX2Video 与 causal entry class，拆分 causal Conv3d、resblock、attention、upsampler GroupNorm+SiLU、tile/stream overlap、history state、blend 和 decode postprocess；记录 channels、frames、stride、padding、dtype、layout、tile 与调用次数。

4. 用保存的 latent 建 decode-only 和 streaming harness，profile torch.compile 后各组件与 kernel 耗时，先确认已有 upsampler fusion、tiling 和 parallel decode 是否命中，再定位 causal Conv3d、norm/activation、attention、upsample、tile blend、state copy 和 graph break；20 次 warmup、100 次计时。

5. （并行）针对真实视频 shape 调研 SGLang、Diffusers、PyTorch/cuDNN、FlashAttention、CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断空间。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 causal/history/config/dtype/frames/tile/device guard、测试和 fallback 的 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未融合好的数学等价路径，优先减少 global memory 读写、reshape/shuffle、history/cache materialize、temporal/spatial layout 转换和 tile 临时张量；重点检查 causal Conv3d、GroupNorm+SiLU、residual、upsample+conv 与 tile/stream overlap blend。

7. 用独立 latent 和 LTX/SANA-WM/JoyAI-Echo E2E 验收精度、速度、显存与 streaming 状态。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/时间漂移；所有 config 稳定获益才接受，否则回到第 4 步。
