# HunyuanVideo / FastHunyuan 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境并下载 HunyuanVideo；FastHunyuan 使用 FastVideo/FastHunyuan-diffusers 另建 baseline。国内机器可使用 HF_ENDPOINT，或把验证过的 ModelScope checkpoint 下载到本地目录。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download hunyuanvideo-community/HunyuanVideo
   hf download FastVideo/FastHunyuan-diffusers
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/hunyuanvideo
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model hunyuanvideo --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、seed、分辨率、帧数、fps、steps、guidance、dtype 和 topology，分别保存原模型与 Fast 版本的 eager reference。使用固定视频集比较逐帧 PSNR/SSIM、最差帧、闪烁、运动和首尾一致性；蒸馏带来的质量变化不得归因给 lossless kernel。

3. 分析 native pipeline，拆分 dual text encoder、DiT attention/MLP/modulation、RoPE、scheduler、AutoencoderKLHunyuanVideo encode/decode、upsampler、tiling、postprocess 和通信，记录生产 shape、dtype与调用次数。

4. profile torch.compile 后各组件与 kernel 耗时，重点定位 text encoder、长序列 attention、GEMM、A2A、decoder conv/resblock/attention、upsampler GroupNorm+SiLU、tile blend 和 graph break；保留 eager trace作为数值和性能对照。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model hunyuanvideo --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 现有 kernel，再用 ncu-report skill确认优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发 attention、upsampler 或 VAE 专用 kernel，并保留严格 guard 和 fallback。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未解决的 fuse，优先减少 global memory 读写、reshape/shuffle、permute、tile materialize 和通信等待；重点验证 QKV/RoPE/norm、modulation、residual、GroupNorm+SiLU、upsample+conv 和 tile overlap/blend 的数学等价融合。

7. 用独立 prompt、视频和 topology 验收原模型与 Fast 版本的精度、速度、显存和 scaling。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002并且无 seam/flicker；20 次 warmup、100 次计时且收益超过方差才接受，否则回到第 4 步。
