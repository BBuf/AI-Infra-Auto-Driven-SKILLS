# Diffusion condition encoder 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载实际模型 checkpoint 并跑包含真实 encoder 的 native baseline。下面用 Qwen-Image 作为独立例子；国内机器可使用 HF_ENDPOINT 或已验证的 ModelScope 镜像。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download Qwen/Qwen-Image
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/encoders
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model qwen --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定真实短/长 prompt、负 prompt、图像或音频条件、最大序列长度、dtype 和 batch，保存 tokenizer/preprocess 输出和 eager embedding reference。比较 encoder output cosine、normalized MSE、NaN/Inf，验证 CFG、多条件、动态 prompt、cache on/off 和连续请求 cache reset。

3. 分析 encoder 架构，拆分 tokenizer/preprocess、CPU 到 GPU、T5/CLIP/Qwen/Gemma/Mistral/PaliGemma 或图像/audio encoder forward、attention/MLP、projection、packing、prompt/prefix cache 与 offload；记录 hidden shape、mask、dtype、transfer bytes 和 cache key。

4. profile torch.compile 后 tokenizer、transfer、forward、projection/packing 和 cache 的组件/kernel 耗时，区分 Python/CPU 开销与 GPU attention/GEMM，检查 graph break、同步、重复条件、offload copy 和短 prompt launch overhead；只有 GPU encoder 占 E2E 明显比例才下钻 kernel。

5. （并行）针对真实 sequence/hidden/head shape 调研 Transformers、SGLang、FlashInfer、FlashAttention、PyTorch 和 CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 encoder/input/mask/dtype/shape/device guard、cache 测试和 fallback 的 kernel。

6. （并行）研究 compile 后仍未解决的等价 fuse，优先减少 global memory 读写、token/condition packing、reshape/shuffle、mask materialize、projection copy 和 CPU/GPU 往返；重点检查 QKV+norm+RoPE、MLP、projection+packing、批量 encoder 与重复条件消除。cache key 必须覆盖所有改变 embedding 的输入。

7. 用独立 prompt、图片/音频和动态 session 验收 embedding、E2E、p50/p95、显存和 cache。encoder output cosine 至少 0.98、normalized MSE 在预设容差内，无 NaN/Inf、stale cache 或语义串扰；20 次 warmup、100 次计时且 encoder 与完整 E2E 均超过方差才接受，否则回到第 4 步。
