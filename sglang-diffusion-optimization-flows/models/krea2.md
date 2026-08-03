# Krea-2 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 krea/Krea-2 并用 native backend 跑固定 1024px baseline。国内机器可使用 HF_ENDPOINT，或用确认存在的 ModelScope 对应 checkpoint 下载到本地目录。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=/data/models/Krea-2
   hf download krea/Krea-2 --local-dir "$MODEL_DIR"
   BENCH_DIR=/tmp/sglang-diffusion-bench/krea2
   mkdir -p "$BENCH_DIR"
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A cinematic mountain landscape" --width 1024 --height 1024 --seed 42 --save-output --enable-torch-compile --warmup --perf-dump-path "$BENCH_DIR/baseline.json"
   ~~~

2. 固定 Raw/Turbo checkpoint 配置、prompt、seed、分辨率、steps、guidance、dtype 和 topology，保存 eager 输出及 VAE latent。用固定照片、文字和细节样本比较 PSNR、SSIM、LPIPS；Cache-DiT on/off 单独建质量 reference，禁止拿 Qwen-Image 输出代替。

3. 分析 Krea-2 native pipeline，拆分 condition encoder、DiT attention/MLP/modulation、Cache-DiT、scheduler、AutoencoderKLQwenImage causal VAE encode/decode、tiling 和 postprocess，记录 checkpoint mapping、latent scaling、真实 shape 与 cache key/reset。

4. profile torch.compile 后各组件及 kernel 耗时，重点定位 attention、GEMM、modulation、cache hit/miss、causal Conv3d、resblock、upsample、tile blend、layout copy 和 graph break；保留相同输入的 eager trace。

   ~~~bash
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A cinematic mountain landscape" --width 1024 --height 1024 --seed 42 --save-output --enable-torch-compile --warmup --profile --profile-all-stages --perf-dump-path "$BENCH_DIR/compile-profile.json"
   ~~~

5. （并行）针对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 的已有 kernel，使用 ncu-report skill确认瓶颈。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发 attention、cache 或 VAE kernel，保留 config/dtype/cache/device guard 与 fallback。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未完成的等价 fuse，优先减少 global memory 读写、reshape/shuffle、condition/cache materialize、layout conversion 和 tile 临时张量；重点检查 modulation、norm/activation、residual、causal Conv3d 单帧路径、upsample+conv 和 tile blend。Cache-DiT 作为近似路线分表。

7. 用独立 prompt 和 Raw/Turbo 配置验收精度、速度、显存以及 cache 连续请求正确性。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002；20 次 warmup、100 次计时且无 stale cache/fallback才接受，否则回到第 4 步。
