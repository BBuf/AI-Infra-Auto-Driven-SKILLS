# Z-Image flow

覆盖 `Tongyi-MAI/Z-Image` 与 `Tongyi-MAI/Z-Image-Turbo`。

```bash
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 2)
PYTHONPATH=python python3 "$BENCH_PY" --model zimage --label baseline --output-dir "$BENCH_DIR"
# base 用 1 GPU：--model zimage-base
```

执行 [`../components/dit-attention.md`](../components/dit-attention.md)、
[`../components/vae-autoencoder-kl-2d.md`](../components/vae-autoencoder-kl-2d.md) 和 encoder
flow。先确认源码已有 bf16 Triton norm/tanh residual 是否命中，再研究 attention/MLP。
base 与 Turbo 的采样步数/质量不同，分别建 baseline；共享 VAE 只优化一次。
