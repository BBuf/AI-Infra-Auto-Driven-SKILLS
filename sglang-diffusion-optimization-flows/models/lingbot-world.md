# LingBot World 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 LingBot World fast checkpoint，先跑 native offline smoke，再运行仓内 realtime consistency case。国内机器可使用 HF_ENDPOINT，或把验证过的 ModelScope checkpoint 下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   MODEL_DIR=/data/models/lingbot-world-fast
   hf download robbyant/lingbot-world-fast-diffusers --local-dir "$MODEL_DIR"
   sglang generate --backend sglang --model-path "$MODEL_DIR" --pipeline-class-name LingBotWorldCausalDMDPipeline --prompt "A first-person walk along a beach" --image-path /tmp/lingbot-input.png --width 832 --height 480 --num-frames 9 --fps 16 --num-inference-steps 4 --guidance-scale 1.0 --seed 42 --text-encoder-cpu-offload --warmup false --save-output
   pytest -q python/sglang/multimodal_gen/test/server/test_server_1_gpu.py -k lingbot_world_realtime_plastic_beach -s
   ~~~

2. 固定 chunk、camera、prompt events、seed、resolution、fps、steps、dtype 和 topology，分别保存 V1/V2 首 chunk 与 steady-state reference。检查逐帧 PSNR/SSIM、chunk seam、prompt/camera 切换、stale frame、运动连续性、cache reset、长时显存和 steady-state FPS。

3. 分析 realtime pipeline，拆分 condition encoder、camera conditioner、DiT、bounded sink-window KV cache、dynamic condition/cross-attention cache、Wan-style causal VAE、lazy black-frame encode、预分配 writer、scheduler 和 postprocess，记录 cache key、窗口、shape 与事件状态机。

4. profile torch.compile 后首 chunk、steady-state chunk 和事件切换的组件/kernel 耗时，重点定位 attention、cache copy、A2A、camera packing、causal Conv3d、tile blend、writer、CPU sync 与 graph break；报告首帧延迟和稳定 FPS。

5. （并行）针对真实 chunk shape 调研 SGLang、FlashInfer、FlashAttention、Diffusers、PyTorch 和 CUTLASS/Triton 已有实现，用 ncu-report skill分析热点。需要新 kernel 时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 window/cache/event/dtype/device guard 与 fallback 的实现。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未解决的等价 fuse，优先减少 global memory 读写、cache materialize、reshape/shuffle、condition packing、跨 rank gather 与帧缓冲复制；重点检查 modulation、norm/activation、residual、causal Conv3d、upsample+conv 和 tile blend，保持 bounded cache 与事件语义。

7. 用独立长时间交互序列验收 V1/V2 的首 chunk 延迟、steady-state FPS、显存和输出正确性。component cosine 至少 0.999、normalized MSE 不超过 1e-4，视频 PSNR 下降不超过 0.10 dB、SSIM 下降不超过 0.002，任何 stale frame、串扰、无界 cache 或 seam 都直接拒绝；性能不够则回到第 4 步。
