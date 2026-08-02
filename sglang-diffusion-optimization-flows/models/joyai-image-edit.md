# JoyAI Image Edit flow

checkpoint：`jdopensource/JoyAI-Image-Edit-Diffusers`。

```bash
MODEL_DIR=$(hf download jdopensource/JoyAI-Image-Edit-Diffusers)
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 2)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model joyai-edit --label baseline --output-dir "$BENCH_DIR"
```

preset 用 cat edit、1024²、40 steps、guidance 4、2 GPU CFG parallel。执行 DiT、
[`../components/vae-wan-causal-3d.md`](../components/vae-wan-causal-3d.md)、encoder 与
distributed flow。重点比较 CFG parallel/纯 SP、编辑输入 encode、condition packing、
Wan VAE decode；correctness 同时要求保留原图身份/结构并完成 prompt 编辑。
