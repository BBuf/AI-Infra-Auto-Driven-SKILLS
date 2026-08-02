# Qwen Image causal VAE flow

实现：`runtime/models/vaes/autoencoder_kl_qwenimage.py::AutoencoderKLQwenImage`；
复用于 Qwen-Image、Qwen edit/layered、Krea-2 与 FireRed edit。

1. 保存 T2I 与 edit 的真实 latent；分别 profile encode、decode、causal Conv3d、
   resblock、attention、upsample、tiling/parallel decode。
2. 单图/单帧 shape 先证明 causal Conv3d 到 Conv2d 的等价条件，再做专用 dispatch；
   多层/layered 输入必须走原 3D correctness 覆盖。
3. 检查 layout conversion、tile blend 和 GroupNorm/SiLU 是否比 conv 更值得 fuse。
4. Krea/Qwen/FireRed 至少各跑一条固定 seed E2E；共享 runtime 不代表 config、latent
   scaling 和 encoder 路径相同。
5. 使用公共 VAE 容差与 tile seam 检查，100 次 edit soak 验证输入不会污染缓存。
