# JoyAI-Echo flow

checkpoint：`jdopensource/JoyAI-Echo`。它复用 LTX-2.3 video VAE，但 DiT、memory
bank、audio-window 条件路径独立。

```bash
MODEL_ID=jdopensource/JoyAI-Echo
MODEL_DIR=/data/models/JoyAI-Echo
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sglang generate --backend sglang --model-path "$MODEL_DIR" \
  --prompt "A curious raccoon" --width 640 --height 384 --num-frames 33 \
  --num-inference-steps 8 --seed 42 --num-gpus 2 --ulysses-degree 2 \
  --enable-memory-bank false --save-output \
  --enable-torch-compile --warmup \
  --perf-dump-path "$BENCH_DIR/joyai-echo-baseline.json"
```

先以 checkpoint 默认 sampling shape 记录合同，再固定显式参数。执行 DiT、
[`../components/vae-ltx-video.md`](../components/vae-ltx-video.md)、
[`../components/vae-audio.md`](../components/vae-audio.md) 的 LTX audio VAE/vocoder 子项、
encoder 和 distributed flow。重点 profile paired audio-video memory bank（max size 7/
fix frames 3）、96-window audio selection、mel preprocess、late-layer path 和 AV decode；
缓存 key/reset、音视频同步、长视频显存有界是强制 correctness。
