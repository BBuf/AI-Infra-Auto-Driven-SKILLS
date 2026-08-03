# MiniMax-H3 Video VAE 优化 flow（第一优先级）

workflow 草稿

1. 基于已经合入 MiniMax-H3 支持的最新 SGLang main 配置环境，下载根 checkpoint，并通过 T2VA native pipeline 把 runtime/models/vaes/minimax_h3.py::MiniMaxH3VideoVAE 和 minimax_h3_video_vae 实现单独跑起来。Hugging Face 国内镜像或源码登记的 ModelScope ID 二选一。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=$(hf download MiniMaxAI/MiniMax-H3)
   # ModelScope 替代：modelscope download --model MiniMax/MiniMax-H3 --local_dir /data/models/MiniMax-H3
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-minimax-h3-video
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model minimax-h3-t2va --label eager-baseline --output-dir "$BENCH_DIR"
   ~~~

2. 用 H3 eager T2VA 保存最终 video latent，并从 FL2VA/Ref2VA 保存 encode 输入，建立唯一 lossless reference；当前 torch.compile 结果不能充当 reference。固定 1344x768、124 帧、24 fps、tile/overlap、BF16、TP2+Ulysses2，比较 component cosine/MSE、逐帧 PSNR/SSIM、最差帧、seam、flicker、帧数和时间戳。

3. 分析 MiniMaxH3VideoVAE 架构，拆分 encode、默认 overlapping tiled decode、conv、resblock、attention、norm/activation、upsample、tile blend 和 postprocess，并从 MiniMaxH3VAEDecodingStage 中剥离 audio VAE 与 mux。记录 channels、frames、stride、padding、dtype、layout 和 tile recipe；released contract 禁止 spatial、spatial-shard 和 patch decode。

4. 用保存的 latent 建 VAE-only harness，先 profile eager，再用相同输入单独 profile torch.compile 候选，按组件和 kernel 定位 causal/spatial conv、resblock、attention、norm/activation、upsample、tile overlap/blend、layout copy 和 graph break；compile 数值变化单独记录，20 次 warmup、100 次计时。

5. （并行）针对真实 H3 VAE shape 调研 SGLang、Diffusers、PyTorch/cuDNN、FlashAttention 和 CUTLASS/Triton 已有实现，用 ncu-report skill判断 memory、compute、occupancy 或 launch bound。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 H3 config/dtype/channels/frames/tile/device guard、exact test 和 fallback 的 kernel。

6. （并行）研究 eager/compile 后仍未融合好的数学等价操作，优先减少 global memory 读写、reshape/shuffle、temporal/spatial layout 转换、tile materialize 和 overlap 重复计算；重点证明 norm+activation、bias/residual、upsample+conv、attention output projection 和 tile blend 的等价条件。不得为了速度绕过 released tile contract。

7. 用独立 T2VA、FL2VA、Ref2VA latent 和联合音视频 E2E 验收。component cosine 至少 0.999、normalized MSE 不超过 1e-4，逐帧 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002，无 seam/flicker且 A/V sync 不变；组件和 E2E 收益超过方差才接受，否则回到第 4 步继续执行第 5、6 步。
