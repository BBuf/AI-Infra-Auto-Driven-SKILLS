# HunyuanVideo / FastHunyuan flow

```bash
MODEL_DIR=$(hf download hunyuanvideo-community/HunyuanVideo)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model hunyuanvideo --label baseline --output-dir "$BENCH_DIR"
```

Fast 版本替换为 `FastVideo/FastHunyuan-diffusers`，使用同一固定 shape 单独建基线。
执行 [`../components/dit-attention.md`](../components/dit-attention.md)、
[`../components/vae-hunyuan-video.md`](../components/vae-hunyuan-video.md)、encoder 和
distributed flow。重点 profile dual text encoder、attention/MLP、upsampler fusion 与
长序列通信。FastHunyuan 的步数/蒸馏质量单列，不能作为原模型 lossless candidate。
