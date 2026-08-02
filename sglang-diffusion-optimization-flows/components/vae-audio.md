# Audio VAE / vocoder flow

这些实现都输出音频，但源码不同，不共享数值或性能结论：

- H3：`MiniMaxH3AudioVAE` / `minimax_h3_audio_vae/`；
- LTX：`AutoencoderKLLTX2Audio` 及 vocoder；
- MOVA：`DacVAEConfig` 对应 DAC audio VAE；
- Cosmos3：`cosmos3_avae.py::Cosmos3AVAEAudioTokenizer`（仅实际 pipeline 命中时跑）。

对选中的一个实现执行：

1. 固定 sample rate、duration、channels、mel bins/hop 和真实 latent，warmup 20、
   计时 100 次；分离 VAE、vocoder、resample、mux。
2. profile Conv1d/transpose conv、norm/activation、overlap-add、layout 和 CPU I/O。
3. 先消除重复 resample/cast/permute，再考虑等价 fuse；边界 padding/causal 语义
   必须写进 guard。
4. 比较 waveform cosine、normalized MSE、duration/sample rate/channel、峰值和响度；
   做静音、短音频、非整 chunk 与 100 次连续请求。
5. 音视频模型必须验证 A/V duration 与同步，不得只用容器“可播放”作为正确性。

H3 audio candidate 必须在 [`../models/minimax-h3.md`](../models/minimax-h3.md) 的
eager T2VA 合同里复验。
