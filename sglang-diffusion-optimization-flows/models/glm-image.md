# GLM-Image 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境并下载 zai-org/GLM-Image；国内机器可设置 HF_ENDPOINT，或用确认存在的 ModelScope 镜像下载后传本地目录。使用 registry detector 路径建立 native SGLang baseline。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download zai-org/GLM-Image
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/glm-image
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model glm-image --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 1024px prompt、seed、steps、guidance、dtype 和 GPU topology，保存 eager 输出、标准化前后 latent 与 VAE 输入。用固定 ImageNet-val 子集或等价重建集比较 PSNR、SSIM、LPIPS和色偏，并记录固定 seed E2E 感知结果。

3. 分析 GLM-Image pipeline，拆分 GLM condition encoder、DiT attention/MLP/modulation、scheduler、latent mean/std 标准化、AutoencoderKL 2D encode/decode、spatial parallel decode 和 postprocess，记录每个阶段真实 shape、dtype、tile 与并行合同。

4. profile torch.compile 后的完整 E2E 和所有 stage，定位 encoder、attention、GEMM、norm、标准化 broadcast、decoder conv/resblock、mid-block attention、upsample、tile blend、通信与 graph break；同输入保留 eager trace。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model glm-image --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 的已有 kernel，并用 ncu-report skill 判断 compute、memory、occupancy 或 launch bound。必要时启动 kernel design sub agent，使用 ultra 模式与 KernelWiki、ncu-report skill 开发带标准化/config/device guard 和 fallback 的专用 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未完成的数学等价 fuse，优先减少 global memory 读写、reshape/shuffle、permute/cat 和重复 broadcast；重点检查 modulation、latent mean/std、norm+activation、residual、upsample+conv 与 tile blend。保持颜色范围和标准化语义不变。

7. 用独立 prompt 和重建样本验收精度、速度、显存及 spatial parallel。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002，并检查无色偏和 tile seam；20 次 warmup、100 次计时且无 fallback 才接受，否则回到第 4 步。
