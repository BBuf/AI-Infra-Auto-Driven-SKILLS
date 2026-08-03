# Cosmos3 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境并下载 checkpoint。国内机器可保留 HF_ENDPOINT，也可把已经确认存在的 ModelScope 镜像下载到本地后将模型路径替换为本地目录。先跑 Nano T2I、Nano T2V 和 Super T2V baseline。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download nvidia/Cosmos3-Nano
   hf download nvidia/Cosmos3-Super
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/cosmos3
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model cosmos3-nano-t2i --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model cosmos3-nano-t2v --label baseline --output-dir "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model cosmos3-super-t2v --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 prompt、seed、分辨率、帧数、fps、steps、dtype 和 GPU topology 保存 eager reference。T2I 使用固定图像集比较 PSNR、SSIM、LPIPS；T2V/I2V 使用固定视频集比较逐帧 PSNR/SSIM、最差帧、闪烁和首尾帧；实际命中 AVAE 时再检查 waveform、采样率和音视频同步。隔离 benchmark 可关闭 guardrail，但生产验收必须打开。

3. 分析 Cosmos3 native pipeline 架构，拆分文本/图像 encoder、DiT、scheduler、Wan-style z_dim=48 video VAE、可选 AVAE、guardrail 和 postprocess，分别记录 Nano/Super 以及 T2I/T2V/I2V 的 tensor shape、dtype、调用次数和并行方式。

4. 在相同输入上 profile torch.compile 后的完整 E2E 和所有 stage，按组件及 kernel 汇总耗时，重点定位 attention、MLP、modulation、A2A、causal Conv3d、tile blend、audio decode 与 mux；同时保留 eager trace，确认 compile 没有隐藏 graph break 或 fallback。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model cosmos3-super-t2v --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch、CUTLASS/Triton 中是否已有高性能 kernel；用 ncu-report skill 采集 NCU 证据判断 memory、compute、occupancy 或 launch bound。没有合适实现时启动 kernel design sub agent，使用 ultra 模式并结合 KernelWiki 和 ncu-report skill 开发带完整 dtype/shape/device guard 与 fallback 的关键 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 torch.compile 后仍未解决的等价 fuse，优先减少 global memory 读写、layout 转换、reshape、shuffle、cat/permute 和跨 rank materialize；重点验证 modulation、norm/activation、residual、upsample+conv、causal Conv3d 单帧退化、tile overlap/blend 是否能数学等价重写。一次只合入一个可归因改动。

7. 用未参与开发的固定输入独立验收精度、速度、峰值显存和多卡 scaling。图像 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002，视频还必须无 seam/flicker，component cosine 至少 0.999、normalized MSE 不超过 1e-4；20 次 warmup、100 次计时且组件和 E2E 收益都超过运行方差才接受，否则回到第 4 步继续执行第 5、6 步。
