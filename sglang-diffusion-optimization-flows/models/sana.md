# SANA / SANA-WM flow

SANA 图像与 SANA-WM world model 只共享名字，不共享 DiT/VAE。分别执行。

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model sana-1.5-1.6b --label baseline --output-dir "$BENCH_DIR"

# SANA-WM 无 helper preset；使用 GPU case 的短 TI2V shape：
sglang generate --backend sglang \
  --model-path Efficient-Large-Model/SANA-WM_streaming \
  --prompt "The subject slowly turns toward the camera" \
  --image-path "$ASSET_DIR/cat.png" \
  --width 640 --height 384 --num-frames 17 --fps 16 \
  --num-inference-steps 12 --guidance-scale 4.5 --seed 0 --save-output \
  --enable-torch-compile --warmup \
  --perf-dump-path "$BENCH_DIR/sana-wm-streaming-baseline.json"
```

SANA 图像跑 DiT、[`../components/vae-dc-ae.md`](../components/vae-dc-ae.md) 和 encoder
flow；覆盖 600M/1.6B、512/1024 与 SANA1.5 4.8B smoke。SANA-WM 跑 DiT、
[`../components/vae-ltx-video.md`](../components/vae-ltx-video.md)、encoder/distributed，
并分别验 bidirectional、streaming。streaming 重点看 causal cache、chunk seam、首帧
延迟、steady-state FPS、显存有界；不能只报整段离线 E2E。
