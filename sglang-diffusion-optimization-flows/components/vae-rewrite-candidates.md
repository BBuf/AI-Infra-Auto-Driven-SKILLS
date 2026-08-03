# VAE semantic rewrite 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载实际目标 VAE 的 checkpoint 并把 encode/decode 单独跑起来。下面用 FLUX.1-dev AutoencoderKL 作为独立例子；gated repo 设置 HF_TOKEN，国内机器可使用 HF_ENDPOINT 或验证过的 ModelScope 镜像。

   ~~~bash
   export HF_TOKEN=<your-token>
   export HF_ENDPOINT=https://hf-mirror.com
   hf download black-forest-labs/FLUX.1-dev
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-rewrite
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model flux --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 测试对应重建数据集上的精度，建立优化前的精度基线。

3. 分析 decoder 架构。

4. profile torch.compile 后各组件耗时和各种 kernel 耗时，定位关键 kernel 和可以 fuse 的部分。

5. （并行）对于关键 kernel，首先调研是否已经存在对应 GPU 架构和参数下的高性能实现，之后使用 ncu-report skill profile 是否还有优化空间。需要开发时启动 kernel design sub agent，使用 ultra 模式，并结合 KernelWiki 和 ncu-report skill。

6. （并行）研究 compile 后仍未处理好的 fuse 机会，重点减少 global memory 读写、reshape 和 shuffle。优先研究数学等价操作的融合，例如 upsampling 与 convolution；其次研究 kernel 内部融合以减少访存。

7. 用独立输入验收改进后的精度和速度。精度验收不要求 bit-identical，使用合理误差范围。如果结果还不够好就回到第 4 步，再次执行第 5、6 步。
