# Hunyuan3D-2 优化 flow

workflow 草稿

1. 在最新 SGLang main 根目录配置 diffusion 环境，下载 tencent/Hunyuan3D-2 并先关闭 paint 建立 shape/mesh native baseline。国内机器可使用 HF_ENDPOINT，或用确认存在的 ModelScope 镜像下载到本地。

   ~~~bash
   export HF_ENDPOINT=https://hf-mirror.com
   hf download tencent/Hunyuan3D-2
   BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
   BENCH_DIR=/tmp/sglang-diffusion-bench/hunyuan3d
   mkdir -p "$BENCH_DIR"
   PYTHONPATH=python python3 "$BENCH_PY" --model hunyuan3d-shape --label baseline --output-dir "$BENCH_DIR"
   ~~~

2. 固定输入图、seed、point 数、steps、dtype 和 topology，保存 eager point/latent/mesh reference。检查 latent、顶点/面数量、有限值、包围盒、抽样 Chamfer、mesh 可加载性和明显拓扑破损；shape 稳定后再开启 paint，并对 paint 输出另做 PSNR/SSIM/LPIPS。

3. 分析 pipeline 架构，拆分图像 condition encoder、shape DiT、ShapeVAE decode、surface extraction、mesh export、可选 paint 的 multiview UNet/AutoencoderKL 和 postprocess，记录各阶段 shape、dtype、CPU/GPU 边界和调用次数。

4. profile torch.compile 后各组件与 kernel 耗时，分清 attention/GEMM/scatter、ShapeVAE 和 marching-cubes 类后处理，定位 layout、indexing、空 tensor、CPU sync、mesh export 和 graph break；保持相同 eager trace作为对照。

   ~~~bash
   PYTHONPATH=python python3 "$BENCH_PY" --model hunyuan3d-shape --label compile-baseline --output-dir "$BENCH_DIR"
   # 将 helper 打印的 sglang generate 命令原样重跑，并追加：--profile --profile-all-stages
   ~~~

5. （并行）针对真实点数和 latent shape 调研 SGLang、PyTorch、Diffusers、Kaolin、CUTLASS/Triton 中已有高性能实现，并用 ncu-report skill分析热点 kernel。需要新实现时启动 kernel design sub agent，使用 ultra 模式和 KernelWiki、ncu-report skill开发带 point-count、dtype、layout、device guard 与 fallback 的 kernel。

6. （并行）研究 compile 后仍未处理好的等价 fuse，优先减少 global memory 读写、重复 gather/scatter、reshape/shuffle、index materialize 和 CPU/GPU 往返；重点检查 projection、norm/activation、residual、point packing 与 surface 后处理批量化。不得用改变网格拓扑的近似替换冒充 lossless fuse。

7. 用独立输入图和 mesh 样本验收精度、速度、显存及输出可用性。component cosine 至少 0.999、normalized MSE 不超过 1e-4，顶点/面、包围盒和 Chamfer 在预设容差内且无拓扑破损；20 次 warmup、100 次计时，shape stage 与完整 mesh E2E 都超过方差才接受，否则回到第 4 步。
