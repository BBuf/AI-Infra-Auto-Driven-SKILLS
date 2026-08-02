# DiT / attention / modulation 共享 flow

继承 [`../common/execution-contract.md`](../common/execution-contract.md)。这个 flow
适用于所有 diffusion transformer，但每个候选只针对 profile 中的真实 shape。

1. 从 `--profile-all-stages` 中分离 self/cross/joint attention、MLP、norm、
   modulation、RoPE 和通信；记录 `(B, heads, tokens, head_dim, dtype)`。
2. 检查当前 dispatch 是否命中 fused QK norm、QK norm + RoPE、packed QKV/KV、
   varlen USP、fused scale/shift/gate、residual-gate add 等已有 fast path。
3. attention-bound 时先比较 FlashInfer/FlashAttention/SDPA 与已有专用 backend；
   head-dim 不受支持时保留 fail-closed guard，再进入 KernelWiki + NCU flow。
4. modulation/reshape-bound 时优先消除 materialize、repeat、cat、permute 和全量
   gather；不要为一个低占比 pointwise op 单独写 kernel。
5. 分布式候选同时报告计算、A2A/all-gather 和 overlap，不能只报单 rank kernel。
6. 用固定 latent/timestep/context 比较原 block 与候选，DiT cosine `>=0.995`、
   normalized MSE `<=1e-4`；再跑固定 seed E2E。

结束时保留 shape 表、dispatch 证据、trace/NCU、精度与 E2E 表。一个模型上命中的
专用 shape 不能直接当作另一模型已完成。
