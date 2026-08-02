# LongLive 2.0 flow

覆盖 `Rabinovich/LongLive-2.0-5B-Diffusers` 与官方
`Efficient-Large-Model/LongLive-2.0-5B` layout；native config 继承 Wan2.2 TI2V 5B，
使用 causal 4-step DMD。

```bash
MODEL_ID=Rabinovich/LongLive-2.0-5B-Diffusers
hf download "$MODEL_ID" --local-dir /data/models/LongLive-2.0-5B
sglang generate --backend sglang --model-path /data/models/LongLive-2.0-5B \
  --prompt "A long continuous walk through a city" \
  --width 832 --height 480 --num-frames 61 \
  --num-inference-steps 4 --guidance-scale 1.0 --seed 42 --save-output \
  --enable-torch-compile --warmup \
  --perf-dump-path "$BENCH_DIR/longlive2-baseline.json"
```

记录调整后的帧数：latent frame 必须能被 `num_frames_per_block` 整除。执行 DiT、
[`../components/vae-wan-causal-3d.md`](../components/vae-wan-causal-3d.md)、encoder/
distributed flow。重点 profile causal block cache、4-step DMD、I2V
首帧 encode、跨 chunk memory 与 decode；官方/转换 layout 都验 native loader。
长程 correctness 检查 chunk seam、运动连续、首帧条件、cache reset、显存有界；少步
DMD 不与普通 Wan quality 混比。
