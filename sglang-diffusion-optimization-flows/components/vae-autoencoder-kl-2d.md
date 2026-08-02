# AutoencoderKL 2D VAE flow

实现：`runtime/models/vaes/autoencoder.py::AutoencoderKL`。FLUX.1、Z-Image、SD3 与
GLM-Image 复用 runtime 类，但 latent channels、scaling/shift、block config 不同，
分别跑代表 shape；GLM 还要保留 latent mean/std 标准化合同。

1. 从对应 pipeline 保存进入 decoder 前的 latent；单独 warmup 20 次、计时 100 次。
2. profile decoder conv/resblock、GroupNorm/SiLU、mid-block attention、upsample 和 tiling
   blend；先比较 eager/compile 及已有 parallel tiling。
3. 优先研究 upsample + conv 等价 rewrite、GroupNorm + SiLU、bias/residual 融合和 tile
   边界访存；所有专用 dispatch 保留原实现 fallback。
4. 固定重建集比较 baseline：cosine `>=0.999`、normalized MSE `<=1e-4`、PSNR drop
   `<=0.10 dB`、SSIM drop `<=0.002`。
5. 分别报告 untiled 与生产 tiled shape；tile seam/artifact 检查不能被整体 PSNR 代替。
6. GLM 另比较标准化前后 latent，并检查 mean/std broadcast 与颜色偏移。

执行前后还要按 [`../common/execution-contract.md`](../common/execution-contract.md)
做 E2E 验收。
