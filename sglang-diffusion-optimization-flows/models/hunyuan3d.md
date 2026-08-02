# Hunyuan3D-2 flow

```bash
MODEL_DIR=$(hf download tencent/Hunyuan3D-2)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model hunyuan3d-shape --label baseline --output-dir "$BENCH_DIR"
```

preset 使用 `$ASSET_DIR/cat.png` 且关闭 paint，先隔离 shape path。执行
[`../components/vae-hunyuan3d-shape.md`](../components/vae-hunyuan3d-shape.md)、DiT 与
encoder flow。分别记录 condition encode、shape denoise、ShapeVAE decode、surface/
mesh export；只在 shape 稳定后再打开 paint。正确性包括 latent、顶点/面、有限值、
包围盒、抽样 Chamfer 和 mesh 可加载性，不能用图像 PSNR 代替。

paint path 内部另有 `runtime/.../hunyuan3d/paint.py` 直接加载的 Diffusers
`AutoencoderKL` 与 multiview UNet；它不是 backend fallback，也不与 SGLang native
`AutoencoderKL` flow 合并。paint 成为热点后按 scheduler/postprocess flow 单独归因。
