# LTX video VAE flow

实现：`runtime/models/vaes/ltx_2_vae.py::AutoencoderKLLTX2Video`（含 causal 注册名）；
LTX-2/2.3、SANA-WM、JoyAI-Echo 复用，variant/config 必须分别验收。

1. 保存 LTX-2.3 双阶段、SANA-WM streaming、JoyEcho 的真实 latent shape。
2. profile causal Conv3d、resblock、attention、upsampler GroupNorm + SiLU、tile/stream
   overlap 和 decode 后处理。
3. 先检查已有 upsampler fusion、tiling 与 parallel decode；streaming candidate 必须
   保持 chunk 边界、历史状态和 causal 语义。
4. 对普通与 causal EntryClass 都跑 fallback/加载测试。
5. 使用公共 VAE 容差；长视频额外逐 chunk 比较 seam、时间漂移和显存是否有界。
