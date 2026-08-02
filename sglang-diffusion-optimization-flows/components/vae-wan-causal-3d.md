# Wan causal 3D VAE flow

实现：`runtime/models/vaes/wanvae.py::AutoencoderKLWan`。Wan、Cosmos3、Helios、
JoyAI Image Edit、MOVA、LingBot World 和 LongLive 2 复用它；Cosmos3 将 `z_dim`
改为 48，因此必须保留独立代表 shape。

1. 选三类代表输入：Wan 720p decode、I2V encode+decode、Cosmos3 z=48；保存 latent。
2. profile causal Conv3d、resblock、RMS/GroupNorm、attention、upsample、temporal/spatial
   tiling 和跨 rank gather/blend。
3. 分别扫描 full/spatial/temporal/patch parallel decode 的合法组合；模型禁止的模式
   必须 fail closed，不能因更快而绕过合同。
4. 优先减少 tile overlap、layout/reshape 和 global memory round-trip；单帧 Conv2d
   rewrite 只在严格等价 shape guard 下启用。
5. 用视频 latent cosine `>=0.999`、frame MSE `<=1e-4`、PSNR/SSIM 容差，并逐帧
   检查 seam、闪烁、首尾帧；固定 seed 跑各代表模型 E2E。

同一 runtime 只定义这一份 flow；各模型文件只补 config/shape 和业务 correctness。
