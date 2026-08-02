# HunyuanVideo VAE flow

实现：`runtime/models/vaes/hunyuanvae.py::AutoencoderKLHunyuanVideo`，覆盖
HunyuanVideo 与 FastHunyuan。

1. 保存生产分辨率/帧数 latent，分别 profile decoder conv/resblock、attention、
   upsampler GroupNorm + SiLU、temporal/spatial tiling。
2. 先确认已有 Hunyuan/LTX upsampler fusion 是否命中；不命中先查 dtype/shape guard。
3. 对 decode-only 与 I/O/后处理分开计时；避免把视频编码器耗时计入 VAE。
4. 使用公共 VAE 容差并做逐帧 seam/flicker 检查；FastHunyuan 的少步数质量变化
   不得归因给 VAE。
5. component 与 E2E 均超过运行方差且至少 3% 才接受。
