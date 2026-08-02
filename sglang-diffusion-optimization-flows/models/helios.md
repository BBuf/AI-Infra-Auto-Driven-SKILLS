# Helios flow

覆盖 `BestWishYsh/Helios-Base`、Mid、Distilled。

```bash
MODEL_DIR=$(hf download BestWishYsh/Helios-Base)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model helios --label baseline --output-dir "$BENCH_DIR"
```

执行 DiT、[`../components/vae-wan-causal-3d.md`](../components/vae-wan-causal-3d.md)、
encoder/distributed flow。重点 profile Helios 专用 denoising stage、block cache/causal
状态、attention 和 Wan decode；Base/Mid/Distilled 的调度与质量分别建基线。长视频
验帧数、首尾/闪烁、cache reset 和显存有界。
