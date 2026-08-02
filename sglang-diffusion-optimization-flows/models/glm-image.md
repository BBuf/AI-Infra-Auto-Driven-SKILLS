# GLM-Image flow

registry 使用 detector 识别 GLM-Image；当前 skill preset 的公开 ID 是
`zai-org/GLM-Image`。

```bash
MODEL_DIR=$(hf download zai-org/GLM-Image)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model glm-image --label baseline --output-dir "$BENCH_DIR"
```

执行 DiT、[`../components/vae-autoencoder-kl-2d.md`](../components/vae-autoencoder-kl-2d.md)
的 GLM shape 和 encoder flow。重点检查 GLM condition encoder、attention、latent
mean/std 标准化、2D block config 与 spatial parallel decode。虽与 FLUX.1/SD3 共用
`AutoencoderKL` runtime，也必须固定 1024px prompt 做独立 correctness/E2E。
