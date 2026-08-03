# Stable Diffusion 3 / 3.5 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，设置 HF_TOKEN 后下载 SD3 Medium、SD3.5 Medium/Large 或 Diffusers layout checkpoint，分别建立 native baseline。国内机器可使用 HF_ENDPOINT，或把验证过的 ModelScope checkpoint 下载到本地。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=/data/models/sd3.5-medium
   hf download stabilityai/stable-diffusion-3.5-medium-diffusers --local-dir "$MODEL_DIR"
   BENCH_DIR=/tmp/sglang-diffusion-bench/sd35
   mkdir -p "$BENCH_DIR"
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A studio photograph of a red fox" --width 1024 --height 1024 --seed 42 --save-output --enable-torch-compile --warmup --perf-dump-path "$BENCH_DIR/baseline.json"
   ~~~

2. 固定 prompt、seed、1024px、steps、guidance、dtype 和 topology，保存 eager 图像、CLIP/T5 condition、DiT 输出和 VAE latent。使用固定 ImageNet-val 子集或重建集比较 PSNR、SSIM、LPIPS、颜色和文字；Medium/Large、裸 checkpoint/Diffusers layout 分别建 reference。

3. 分析 native pipeline，拆分 CLIP/T5 encoder、MMDiT joint attention/MLP/modulation、scheduler、AutoencoderKL 2D conv/resblock/mid attention/upsample/tile 和 postprocess，记录 config、latent scaling、真实 shape 与 backend dispatch。

4. profile torch.compile 后的完整 E2E 和所有 stage，重点定位 encoder、joint attention、GEMM、modulation、decoder conv/resblock、GroupNorm/SiLU、mid attention、upsample、tile blend 和 graph break；用 eager trace确认没有 Diffusers backend fallback。

   ~~~bash
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A studio photograph of a red fox" --width 1024 --height 1024 --seed 42 --save-output --enable-torch-compile --warmup --profile --profile-all-stages --perf-dump-path "$BENCH_DIR/compile-profile.json"
   ~~~

5. （并行）针对真实热点 shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断优化空间。必要时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 model-size/layout/dtype/shape/device guard、测试和 fallback 的关键 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未完成的数学等价 fuse，优先减少 global memory 读写、reshape/shuffle、CLIP/T5 condition materialize 和 VAE tile 临时张量；重点检查 QKV/RoPE/norm、modulation、residual、GroupNorm+SiLU、upsample+conv 与 tile overlap/blend。

7. 用独立 prompt 和重建样本验收 Medium/Large 与两种 layout 的精度、速度、显存和 native loader。component cosine 至少 0.999、normalized MSE 不超过 1e-4，PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002；20 次 warmup、100 次计时且无 fallback、组件和 E2E 都稳定获益才接受，否则回到第 4 步。
