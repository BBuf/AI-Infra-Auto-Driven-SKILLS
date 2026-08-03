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

2. 使用固定输入建立 VAE 组件质量基线。保存基线输出，并按 Sol Engine（[NVlabs/Sana `sol-engine` 分支](https://github.com/NVlabs/Sana/tree/sol-engine)）的 quality-gated 方式记录与输出模态匹配的质量指标（图像和视频使用对齐 LPIPS）和 Agent 内置质量评审结果。

3. 分析 decoder 架构。

4. 保持与 benchmark 命令相同的 torch.compile 设置进行 profile。benchmark 启用了 torch.compile 就 profile compile 后的模型；benchmark 没有启用 torch.compile 就 profile 未 compile 的模型。统计各组件和各种 kernel 耗时，定位关键 kernel 和可以 fuse 的部分。

5. （并行）对于关键 kernel，首先调研是否已经存在对应 GPU 架构和参数下的高性能实现，之后使用 ncu-report skill profile 是否还有优化空间。需要开发时启动 kernel design sub agent，使用 ultra 模式，并结合 KernelWiki 和 ncu-report skill。

6. （并行）研究当前执行模式下仍未处理好的 fuse 机会，重点减少 global memory 读写、reshape 和 shuffle。优先研究数学等价操作的融合，例如 upsampling 与 convolution；其次研究 kernel 内部融合以减少访存。

7. 用独立输入验收改进后的质量和速度，并重复与基线相同的 Sol Engine quality-gated 评测。不要求 bitwise match；只要质量指标和 Agent 内置质量评审都在预先设定的可接受范围内，就认为精度通过。如果结果还不够好就回到第 4 步，再次执行第 5、6 步。
