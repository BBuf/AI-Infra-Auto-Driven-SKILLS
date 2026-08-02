# Stable Diffusion 3 / 3.5 flow

覆盖 SD3 Medium、SD3.5 Medium/Large 及 `-diffusers` layout；gated repo 先设置
`HF_TOKEN`。

```bash
MODEL_ID=stabilityai/stable-diffusion-3.5-medium-diffusers
hf download "$MODEL_ID" --local-dir /data/models/sd3.5-medium
sglang generate --backend sglang --model-path /data/models/sd3.5-medium \
  --prompt "A studio photograph of a red fox" --width 1024 --height 1024 \
  --seed 42 --save-output --enable-torch-compile --warmup \
  --perf-dump-path "$BENCH_DIR/sd35-medium-baseline.json"
```

执行 DiT、[`../components/vae-autoencoder-kl-2d.md`](../components/vae-autoencoder-kl-2d.md)
的 SD3 config shape，以及 encoder flow。分离 CLIP/T5、joint attention、MLP 与 VAE；
Medium/Large 至少各 smoke。裸 checkpoint 与 Diffusers layout 的加载映射、输出和
fallback guard 都要验，禁止出现 Diffusers backend fallback。
