# Ideogram 4 flow

覆盖 `fal/ideogram-v4-fast`、instant、`ideogram-ai/ideogram-4-fp8`、NF4 与
`Comfy-Org/Ideogram-4` native mapping。

```bash
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 2)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model ideogram4-fp8 --label baseline --output-dir "$BENCH_DIR"
```

执行 DiT、[`../components/vae-flux2.md`](../components/vae-flux2.md)、encoder 与
distributed flow。profile 文字渲染相关的长 prompt encoder、attention、FlashAttention
backend、VAE batch norm。FP8/NF4、Fast/Instant 是不同近似路线；至少用含小字号文字、
布局和普通照片的固定 prompt 集，OCR/拼写、PSNR 类指标与人工 artifact 一起报告。
