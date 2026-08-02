# ERNIE-Image flow

覆盖 `baidu/ERNIE-Image` 与 Turbo。

```bash
MODEL_DIR=$(hf download baidu/ERNIE-Image-Turbo)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model ernie-image-turbo --label baseline --output-dir "$BENCH_DIR"
```

执行 DiT、[`../components/vae-flux2.md`](../components/vae-flux2.md) 的 ERNIE shape 和
encoder flow。checkpoint 的 VAE `_class_name` 是 `AutoencoderKLFlux2`，因此不重复
定义 VAE；但 ERNIE 的 latent/patch/config 仍独立验收。Turbo 与 base 的 step/质量
分表；base 至少完成加载、native backend、固定 seed correctness 与一轮 perf dump。
