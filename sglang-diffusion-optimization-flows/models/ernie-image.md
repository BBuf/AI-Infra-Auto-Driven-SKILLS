# ERNIE-Image 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 ERNIE-Image-Turbo 并建立 native SGLang baseline；Base checkpoint 也至少完成加载和固定 seed smoke。国内机器可使用 HF_ENDPOINT，或用已验证的 ModelScope 对应仓库下载到本地目录。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download baidu/ERNIE-Image-Turbo
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/ernie-image
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model ernie-image-turbo --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 使用固定输入建立模型端到端质量基线。从 SGLang Diffusion 的 `sglang-diffusion-benchmark-profile` skill 中查找并执行该模型的 benchmark 命令；如果没有现成 preset，则按该 skill 的命令格式建立基线。保存基线输出，并按 Sol Engine 的 quality-gated 方式记录与输出模态匹配的质量指标（图像和视频使用对齐 LPIPS）和 Agent 内置质量评审结果。

3. 分析模型架构。

4. 保持与 benchmark 命令相同的 torch.compile 设置进行 profile。benchmark 启用了 torch.compile 就 profile compile 后的模型；benchmark 没有启用 torch.compile 就 profile 未 compile 的模型。统计各组件和各种 kernel 耗时，定位关键 kernel 和可以 fuse 的部分。

5. （并行）对于关键 kernel，首先调研是否已经存在对应 GPU 架构和参数下的高性能实现，之后使用 ncu-report skill profile 是否还有优化空间。需要开发时启动 kernel design sub agent，使用 ultra 模式，并结合 KernelWiki 和 ncu-report skill。如果 profile 明确确认模型受 attention 限制，则 fork 当前 SGLang 使用的 FlashAttention，针对真实 shape 修改并验证。

6. （并行）研究当前执行模式下仍未处理好的 fuse 机会，重点减少 global memory 读写、reshape 和 shuffle。优先研究数学等价操作的融合，例如 upsampling 与 convolution；其次研究 kernel 内部融合以减少访存。

7. 用独立输入验收改进后的质量和速度，并重复与基线相同的 Sol Engine quality-gated 评测。不要求 bitwise match；只要质量指标和 Agent 内置质量评审都在预先设定的可接受范围内，就认为精度通过。如果结果还不够好就回到第 4 步，再次执行第 5、6 步。
