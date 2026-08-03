# Hunyuan3D ShapeVAE 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 tencent/Hunyuan3D-2，并通过 native shape pipeline 把 runtime/models/vaes/hunyuan3d_vae.py::ShapeVAE 单独跑起来。国内机器可使用 HF_ENDPOINT 或已验证的 ModelScope 镜像。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download tencent/Hunyuan3D-2
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/vae-hunyuan3d-shape
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model hunyuan3d-shape --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 保存真实 point、latent、ShapeVAE 输出和 mesh reference，固定输入图、seed、point 数、dtype 和 topology。比较 latent cosine/MSE、顶点/面数量、有限值、包围盒、抽样 Chamfer、mesh 可加载性和拓扑；不能用图像 PSNR 代替 shape correctness。

3. 分析 ShapeVAE 架构，拆分 point/latent packing、attention、GEMM、norm、projection、scatter/gather、decode、surface extraction 与 mesh export；记录真实 point count、hidden shape、dtype、padding、index layout、空 tensor 和 CPU/GPU 边界。

4. 用保存的 point/latent 建 component harness，profile torch.compile 后 ShapeVAE 各层与 kernel 耗时，并将 surface/mesh 后处理单独归因；定位 attention、GEMM、scatter/gather、index materialize、layout copy、CPU sync 和 graph break，20 次 warmup、100 次计时。

5. （并行）针对真实 point/latent shape 调研 PyTorch、SGLang、Kaolin、PyTorch3D、CUTLASS/Triton 已有 kernel，并用 ncu-report skill判断优化空间。需要新实现时启动 kernel design sub agent，以 ultra 模式结合 KernelWiki 和 ncu-report skill开发带 point-count/index/dtype/shape/device guard、测试和 fallback 的 kernel。

   如果 trace/NCU 明确证明 attention-bound，则必须立即 fork 当前 SGLang 所依赖版本的 FlashAttention，在 fork 中针对真实 head_dim、token、layout 和 GPU 架构修改 kernel 与 dispatch；特别覆盖 FlashAttention/cuDNN 当前不支持的 head_dim 384/512 等盲区，并让 SGLang 显式指向该 fork。不得只停留在调研或另写旁路原型；所有非目标 shape 保持 fail-closed 回退，最后用原模型、相同输入、NCU 与端到端精度/性能共同验收。

6. （并行）研究 compile 后仍未融合好的数学等价操作，优先减少 global memory 读写、重复 scatter/gather、reshape/shuffle、index materialize 和 CPU/GPU 往返；重点检查 projection、norm/activation、residual、point packing 与可批量的 surface postprocess，不改变 mesh topology。

7. 用独立输入图、point cloud 和 mesh 样本验收精度、速度、显存与输出可用性。component cosine 至少 0.999、normalized MSE 不超过 1e-4，顶点/面、包围盒和 Chamfer 在预设容差内且无拓扑破损；ShapeVAE stage 与完整 mesh E2E 均超过方差才接受，否则回到第 4 步。
