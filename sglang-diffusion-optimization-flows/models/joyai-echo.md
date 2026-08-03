# JoyAI-Echo 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 jdopensource/JoyAI-Echo 并使用 native backend 建立固定短视频 baseline。国内机器可使用 HF_ENDPOINT，或把验证过的 ModelScope checkpoint 下载到本地目录。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=/data/models/JoyAI-Echo
   hf download jdopensource/JoyAI-Echo --local-dir "$MODEL_DIR"
   BENCH_DIR=/tmp/sglang-diffusion-bench/joyai-echo
   mkdir -p "$BENCH_DIR"
   sglang generate --backend sglang --model-path "$MODEL_DIR" --prompt "A curious raccoon" --width 640 --height 384 --num-frames 33 --num-inference-steps 8 --seed 42 --num-gpus 2 --ulysses-degree 2 --enable-memory-bank false --save-output --enable-torch-compile --warmup --perf-dump-path "$BENCH_DIR/baseline.json"
   ~~~

2. 使用固定输入建立模型端到端精度基线。从 SGLang Diffusion 的 `sglang-diffusion-benchmark-profile` skill 中查找并执行该模型的 benchmark 命令；如果没有现成 preset，则按该 skill 的命令格式建立基线。

3. 分析模型架构。

4. profile torch.compile 后各组件耗时和各种 kernel 耗时，定位关键 kernel 和可以 fuse 的部分。

5. （并行）对于关键 kernel，首先调研是否已经存在对应 GPU 架构和参数下的高性能实现，之后使用 ncu-report skill profile 是否还有优化空间。需要开发时启动 kernel design sub agent，使用 ultra 模式，并结合 KernelWiki 和 ncu-report skill。如果 profile 明确确认模型受 attention 限制，则 fork 当前 SGLang 使用的 FlashAttention，针对真实 shape 修改并验证。

6. （并行）研究 compile 后仍未处理好的 fuse 机会，重点减少 global memory 读写、reshape 和 shuffle。优先研究数学等价操作的融合，例如 upsampling 与 convolution；其次研究 kernel 内部融合以减少访存。

7. 用独立输入验收改进后的精度和速度。精度验收不要求 bit-identical，使用合理误差范围。如果结果还不够好就回到第 4 步，再次执行第 5、6 步。
