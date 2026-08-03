# AutoencoderKLQwenImage VAE 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 Qwen/Qwen-Image，并通过 native pipeline 把 runtime/models/vaes/autoencoder_kl_qwenimage.py::AutoencoderKLQwenImage 单独跑起来。国内机器可使用 HF_ENDPOINT 或已经验证存在的 ModelScope checkpoint。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Qwen/Qwen-Image
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-qwen-causal-3d
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model qwen --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 保存 T2I、Edit 和 Layered 的真实 encode/decode latent，在固定图像重建集上建立 eager reference。记录 component cosine、normalized MSE、PSNR、SSIM、LPIPS 和 tile seam；Qwen、Krea-2、FireRed 的 config、latent scaling 和输入层数分别验收，100 次 edit soak 检查缓存污染。

3. 分析 AutoencoderKLQwenImage 架构，拆分 causal Conv3d、resblock、attention、norm/activation、upsample、tiling/parallel decode、tile blend 与 postprocess；记录 temporal length、channels、stride、padding、dtype、layout、tile 和 feature cache 语义。

4. 用保存的 latent 建 encode/decode harness，profile torch.compile 后各组件与 kernel 耗时，定位 causal Conv3d、resblock、attention、GroupNorm/SiLU、upsample、tile blend、layout conversion、cache copy 和 graph break；20 次 warmup、100 次计时并保留 eager trace。

5. （并行）针对真实 image/edit/layered shape 调研 SGLang、Diffusers、PyTorch/cuDNN、FlashAttention、CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 temporal/cache/layer/config/dtype/tile/device guard、测试和 fallback 的 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未融合好的数学等价路径，优先减少 global memory 读写、reshape/shuffle、temporal layout 转换、cache/tile materialize；仅在 T=1 且无历史 cache 时证明 causal Conv3d 到 Conv2d 等价，并检查 norm+activation、residual、upsample+conv、attention projection 和 tile blend。

7. 用独立 Qwen、Krea-2、FireRed 和 Layered latent/E2E 验收。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 tile seam/cache 污染；所有 config 的组件与 E2E 都稳定获益才接受，否则回到第 4 步。
