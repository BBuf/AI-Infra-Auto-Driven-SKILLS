# Audio VAE / vocoder 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，选择且只选择一个实际实现建立独立 baseline：MiniMaxH3AudioVAE、AutoencoderKLLTX2Audio、MOVA DAC 或 Cosmos3 AVAE。下面优先使用 MiniMax-H3；国内机器可使用 HF_ENDPOINT，或用源码登记的 ModelScope ID 下载。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=$(hf download MiniMaxAI/MiniMax-H3)
   # ModelScope 替代：modelscope download --model MiniMax/MiniMax-H3 --local_dir /data/models/MiniMax-H3
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/audio-vae
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model minimax-h3-t2va --label eager-baseline --output-dir "$BENCH_DIR"
   ~~~

2. 从真实 pipeline 保存进入 audio decoder 的 latent，并固定 sample rate、duration、channels、mel bins/hop、dtype 和 chunk/window，建立 eager waveform reference。比较 waveform cosine、normalized MSE、sample count、采样率、声道、峰值、响度、频谱以及 A/V duration/sync；覆盖静音、短音频、非整 chunk 和连续请求。

3. 分析选中实现的架构，拆分 audio VAE、vocoder、mel preprocess、Conv1d/transpose conv、norm/activation、overlap-add、resample、layout、CPU I/O 和 mux；记录真实 tensor shape、padding/causal 语义、调用次数与 codec contract，不跨实现复用数值结论。

4. 对 isolate harness 和完整 E2E 分别 profile eager 与 torch.compile，按组件和 kernel 汇总耗时，定位 Conv1d、transpose conv、norm、activation、overlap-add、layout conversion、resample、mux、CPU sync 和 graph break；H3 保持 eager 为唯一 lossless reference。

5. （并行）针对真实 audio shape 调研 PyTorch、TorchAudio、TorchCodec、Diffusers、SGLang、cuDNN 和 CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 implementation/sample-rate/window/dtype/shape/device guard、测试和 fallback 的 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未融合好的数学等价操作，优先减少 global memory 读写、重复 resample/cast/permute、mel/window materialize、overlap-add 临时张量和 mux 前 copy；重点检查 conv+bias+activation、norm+activation、vocoder、resample 批量化和 layout 消除，保持 padding/causal 语义。

7. 用独立 latent 和联合音视频样本验收 component、waveform 和 E2E。audio cosine 至少 0.999、normalized MSE 不超过 1e-4，sample rate/channels/sample count/duration/A-V sync 完全符合合同；20 次 warmup、100 次计时和 1000 次 soak 均稳定且组件/E2E 收益超过方差才接受，否则回到第 4 步。
