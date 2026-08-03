# MiniMax-H3 优化 flow（第一优先级）

workflow 草稿

1. 基于已经合入 MiniMax-H3 支持的最新 SGLang main 配置 diffusion 环境，下载根 checkpoint，不要把 fl2va/ref2va 子目录当成 model path。Hugging Face 国内镜像与源码登记的 ModelScope ID 二选一；先用 4 GPU、TP2+Ulysses2 跑 T2VA eager baseline。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=$(hf download MiniMaxAI/MiniMax-H3)
   # ModelScope 替代方式：
   # MODEL_DIR=/data/models/MiniMax-H3
   # modelscope download --model MiniMax/MiniMax-H3 --local_dir "$MODEL_DIR"
   # export SGLANG_USE_MODELSCOPE=true
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/minimax-h3
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model minimax-h3-t2va --label eager-baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定 seed 1101、1344x768 resolved canvas、5 秒、124 帧、24 fps、50 steps、video flow shift 12、audio flow shift 3、BF16/FP32 eager、TP2+Ulysses2，保存联合音视频 reference。分别跑 T2VA、FL2VA 和 Ref2VA；FL2VA 固定首尾 keyframe，Ref2VA 覆盖 image/video/audio/video_audio reference。视频检查帧数/时间戳、逐帧 PSNR/SSIM、最差帧、seam/flicker；音频检查 32 kHz stereo、sample count、waveform MSE、频谱、duration 和 A/V sync。当前 torch.compile 会改变 H3 数值输出，不能用作 lossless reference。

3. 分析 H3 native pipeline，拆分 Qwen3-VL condition encoder、DiT、video/audio dual stream、MiniMaxH3VideoVAE overlapping tiled encode/decode、MiniMaxH3AudioVAE、专用 Euler ancestral eta=0 scheduler、H.264/AAC mux 和 TP/Ulysses 通信。记录真实 attention 的 tokens、heads、head_dim、RoPE dim 96、QK norm、packed QKV、indexed scale/shift/gate、USP merge、2-rank IPC A2A、batched TP AdaLN 与 final projection shape。released video VAE 禁止 spatial、spatial-shard 和 patch decode。

4. 先 profile eager 的完整 T2VA 各 stage 和 kernel，再用完全相同输入单独 profile torch.compile 候选，按 DiT、video VAE、audio VAE、encoder、scheduler、通信和 mux 汇总耗时；compile 输出只用于定位，任何数值变化单独记录。重点检查 attention、GEMM、A2A、modulation、causal Conv3d、tile blend、audio Conv1d/vocoder、resample、mux、graph break 和 CPU sync。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model minimax-h3-t2va --label eager-profile --output-dir "$BENCH_DIR" --no-torch-compile
   # eager profile：将 helper 打印的命令原样重跑并追加 --profile --profile-all-stages
   # compile 候选：再复制同一命令，把 --enable-torch-compile=false 替换为 --enable-torch-compile，并追加相同 profile flags
   ~~~

5. （并行）针对真实 H3 热点 shape 先检查源码已有的 indexed modulation、head_dim 128 fused QK norm+RoPE、packed Ulysses QKV、usp_merge_heads、IPC A2A、batched TP AdaLN 和 dead-row removal 是否命中，再调研 SGLang、FlashInfer、FlashAttention、PyTorch 和 CUTLASS/Triton 是否有更快实现；用 ncu-report skill确认 compute、memory、occupancy、launch 或通信瓶颈。需要新 kernel 时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 H3 config/dtype/shape/topology guard、exact test 与 fallback 的实现。

6. （并行）研究 eager/compile 后仍不理想的数学等价 fuse，优先减少 global memory 读写、reshape/shuffle、packed/unpacked QKV、scale-shift-gate materialize、跨 rank gather、tile overlap/blend 和 audio/video 中间张量；重点检查 QK norm+RoPE、modulation、residual、norm/activation、causal Conv、upsample+conv、vocoder/resample 与 mux 前 copy。Cache-DiT、online FP8 和少步数均作为近似路线单独验收，一次只合入一个可归因改动。

7. 用开发阶段未使用的 prompt、首尾帧和 reference 资产独立验收 T2VA、FL2VA、Ref2VA，并分别测试 TP2+Ulysses2、Ulysses4；B200/B300 可追加 Ulysses8。component cosine 至少 0.999、normalized MSE 不超过 1e-4，视频 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002且无 seam/flicker，音频 sample/channel/duration/A-V sync 完全满足合同；20 次 warmup、100 次计时且目标组件和 E2E 都超过方差才接受，否则回到第 4 步继续执行第 5、6 步。
