# Krea-2 flow

checkpoint：`krea/Krea-2`。

```bash
hf download krea/Krea-2 --local-dir /data/models/Krea-2
sglang generate --backend sglang --model-path /data/models/Krea-2 \
  --prompt "A cinematic mountain landscape" --width 1024 --height 1024 \
  --seed 42 --save-output --enable-torch-compile --warmup \
  --perf-dump-path "$BENCH_DIR/krea2-baseline.json"
```

执行 DiT、[`../components/vae-qwen-causal-3d.md`](../components/vae-qwen-causal-3d.md)
和 encoder flow。Cache-DiT 集成会因 Raw/Turbo 配置不同而变化，先确认 checkpoint
mapping，再把 cache 路线单列为 approximate。重点 profile attention/modulation、
condition encoder 与 Qwen VAE；不得拿 Qwen-Image E2E 结果代替 Krea-2。
