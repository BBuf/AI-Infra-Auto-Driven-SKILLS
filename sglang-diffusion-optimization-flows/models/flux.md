# FLUX family flow

覆盖 `black-forest-labs/FLUX.1-dev`、FLUX.2 dev/NVFP4、Klein 4B/9B 与 Klein Base。
这些 gated repo 需要 `HF_TOKEN`。按公共下载协议下载一个目标 checkpoint。

```bash
export HF_TOKEN=<token>
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 2)
PYTHONPATH=python python3 "$BENCH_PY" --model flux --label baseline --output-dir "$BENCH_DIR"
PYTHONPATH=python python3 "$BENCH_PY" --model flux2 --label baseline --output-dir "$BENCH_DIR"
# 4B 覆盖：--model flux2-klein 或 flux2-klein-base（各 1 GPU）
```

执行 [`../components/dit-attention.md`](../components/dit-attention.md)；FLUX.1 VAE
走 [`../components/vae-autoencoder-kl-2d.md`](../components/vae-autoencoder-kl-2d.md)，
FLUX.2/Klein 走 [`../components/vae-flux2.md`](../components/vae-flux2.md)，不可混用。
再查 [`../components/encoders.md`](../components/encoders.md) 和分布式 flow。

profile 优先检查 joint attention、double/single blocks、modulation、RoPE、packed QKV，
以及已有 Nunchaku GELU MLP/NVFP4 fast path。NVFP4 是量化支线；9B 和 Base 至少补
smoke/E2E，不能只拿 4B 数字代表全部 Klein。
