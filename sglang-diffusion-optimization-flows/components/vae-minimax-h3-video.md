# MiniMax-H3 video VAE flow（优先）

实现：`runtime/models/vaes/minimax_h3.py::MiniMaxH3VideoVAE` 与
`runtime/models/vaes/minimax_h3_video_vae/`。这是 H3 独有实现，不能与 Wan/LTX VAE
合并。继承 [`../common/execution-contract.md`](../common/execution-contract.md)。

1. 用 H3 eager T2VA preset 保存最终 video latent；另从 FL2VA/Ref2VA 保存 encode
   输入。不要用 `torch.compile` 结果作正确性 reference。
2. 单独 profile overlapping tiled decode、encode、conv/resblock/attention、tile blend；
   从聚合的 `MiniMaxH3VAEDecodingStage` 中剥离 audio VAE 和 mux/后处理时间。
3. 当前 released contract 拒绝 spatial、spatial-shard 和 patch decode；不得为了速度
   强开这些模式。候选先优化默认 overlapping tile recipe。
4. 对 tile overlap/边界、layout、norm/activation、upsample + conv 依次取 NCU 证据；
   专用 kernel 必须 guard H3 config/dtype/shape 并保留原实现 fallback。
5. component：cosine `>=0.999`、MSE `<=1e-4`；视频：PSNR drop `<=0.10 dB`、
   SSIM drop `<=0.002`、无 seam/flicker。再与音频一起跑同步 E2E。
6. 分别验 T2VA、FL2VA、Ref2VA；4-GPU TP2/Ulysses2 是首个并行验收 shape。

每轮只改视频 VAE 或 audio VAE 之一，避免无法归因。
