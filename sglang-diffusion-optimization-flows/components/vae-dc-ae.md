# SANA DC-AE flow

实现：`runtime/models/vaes/autoencoder_dc.py::AutoencoderDC`，内部使用 Diffusers
`AutoencoderDC`，服务 SANA 图像 checkpoint。

1. 保存 512px 与 1024px latent，分开计时 wrapper、inner model 和 postprocess。
2. profile encoder/decoder conv、channel mixing、upsample；当前 tiling 支持要以源码
   和 runtime log 为准，不要假设与其他 VAE 一致。
3. 先评估 compile、contiguous/layout 和 wrapper 开销，再进入专用 kernel。
4. 用公共 VAE 容差；因 32x compression，重点看细线、文字、小物体和色偏。
5. 同时跑 SANA 600M 与 1.6B 的代表 E2E，确认优化不依赖 DiT 规模。
