# FLUX.2 VAE flow

实现：`runtime/models/vaes/autoencoder_kl_flux2.py::AutoencoderKLFlux2`；复用于
FLUX.2/Klein、Ideogram 4 和 ERNIE-Image。ERNIE checkpoint 的
`vae/config.json::_class_name` 是 `AutoencoderKLFlux2`，但 latent/patch/config shape
仍单独验收。

1. 从 FLUX.2、Ideogram 与 ERNIE 各保存一个真实 latent shape，不能只跑一个 checkpoint。
2. 分离 encode/decode、batch norm、resblock、attention、upsample、tile blend；记录
   NVFP4/FP8 模型是否仍以同一 VAE dtype 运行。
3. 先查 parallel tiling 和 compile graph；再按热点考虑 norm/activation、upsample +
   conv、tile overlap 的读写削减。
4. fail-closed guard 至少包含 dtype、channels、stride、tile 参数和设备能力。
5. 使用 VAE cosine `>=0.999`、MSE `<=1e-4`、PSNR/SSIM 容差及固定 seed E2E 验收。

共享代码只优化一次，但三个 config/shape 都必须通过。
