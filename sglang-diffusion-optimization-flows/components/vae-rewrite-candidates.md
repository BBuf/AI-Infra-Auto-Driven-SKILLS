# VAE semantic rewrite candidate checklist

本清单来自 `vae-decoder-acceleration-report.md` 的已验证思路，但不是收益承诺。每个
VAE flow 先用真实 graph/shape 证明前提，再决定复用、改写或记录阴性结论。

## 按优先级检查

1. `nearest 2x upsample + Conv2d(3x3,s1,p1)`：证明相位/tap 映射后，在加载期合成
   `ConvTranspose2d(k4,s2,p1)` 权重，消除 4x 中间张量。
2. causal `Conv3d` 的单帧路径：仅当 `T=1`、无 feature cache/历史帧、推理态时，
   取最后一个 temporal weight slice 退化为 `Conv2d`；视频/streaming 必须回退。
3. attention output projection：若 `softmax` attention 行和为 1 且 projection 只作用
   channel，证明后把 `W_o/b_o` 离线折叠进 value projection，消除 `proj_out`。
4. residual/conv bias → GroupNorm statistics：直接从表达式在 FP32 累积 mean/variance，
   折叠 add/bias/SiLU；需要物化残差时用一次 materialize+stats pass。
5. attention head-dim 盲区：只有 SDPA/Flash 后端未覆盖且 profile 是热点时，再按
   GPU ISA 分家设计专用 kernel；SM100 与 SM120 只共享数学，不强共享实现。

## 上线合同

- 每个 rewrite 写等价式、适用前提、dtype/layout/device/shape/cache/train guard；
- 不满足任一 guard 就走保留的原实现，安装幂等且可 restore/rollback；
- 离线合成权重用 FP32，并记录 checkpoint/source/shape/dtype/topology 指纹；
- baseline 与 candidate 走同一公开 decode 边界、checkpoint、latent 和 BF16 策略；
- 至少 20 warmup + 100 timing；正式 VAE release 做 1,000-call soak；
- component 精度与 ImageNet-val/固定重建集质量独立于性能验收；
- profile 证明可归属收益理论上限太小时，记录 `rejected`，不要为了“有改动”上线。

`torch.compile` 是非 H3 模型的优化下界。这里的候选价值来自数学重组、调用合同、
加载期权重合成或 ISA 级 kernel；不能只把已有 eager 算子重新包一层。
