# Ideogram 4 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，选择 Ideogram 4 FP8、NF4、Fast、Instant 或 Comfy-Org checkpoint 下载并建立各自 baseline。国内机器可使用 HF_ENDPOINT，或把确认存在的 ModelScope 镜像下载到本地；gated repo 先设置 HF_TOKEN。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   hf download ideogram-ai/ideogram-4-fp8
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/ideogram4
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model ideogram4-fp8 --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定含小字号文字、复杂布局和普通照片的 prompt 集、seed、分辨率、steps、dtype 和 topology，保存 eager output 与 latent。比较 PSNR、SSIM、LPIPS、OCR 字符准确率、拼写、布局和人工 artifact；FP8/NF4、Fast/Instant 分别建立质量预算。

3. 分析 native pipeline，拆分长 prompt encoder、DiT joint attention/MLP/modulation、FlashAttention backend、scheduler、AutoencoderKLFlux2 batch norm/resblock/upsample/tile、postprocess 和多卡通信，记录真实 shape 与量化 dtype。

4. profile torch.compile 后的完整 E2E 和所有 stage，按组件及 kernel 汇总耗时，重点定位长 prompt encoder、attention head shape、GEMM、modulation、batch norm、upsample、tile blend、A2A、dequant 和 graph break；保留 eager trace对照。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model ideogram4-fp8 --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 已有 kernel，并使用 ncu-report skill确认性能空间。必要时启动 kernel design sub agent，使用 ultra 模式和 KernelWiki、ncu-report skill开发带 dtype/quant/config/device guard、fallback 及文字样本测试的关键 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未完成的等价 fuse，优先减少 global memory 读写、reshape/shuffle、permute/cat、dequant materialize 和 tile 临时张量；重点检查 QKV/RoPE/norm、modulation、residual、batch norm/activation、upsample+conv 与 tile blend。

7. 用独立文字和照片 prompt 验收各 variant 的精度、速度、显存和多卡 scaling。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002，OCR/拼写和布局不得超出既定质量预算；20 次 warmup、100 次计时且 E2E 稳定获益才接受，否则回到第 4 步。
