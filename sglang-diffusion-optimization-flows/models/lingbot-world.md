# LingBot World realtime flow

覆盖 `IPostYellow/lingbot-world-fast-diffusers`、`robbyant/...fast...` 与 World V2
causal fast。它是交互式/实时 workload，离线单段 E2E 不足以验收。

```bash
MODEL_ID=robbyant/lingbot-world-fast-diffusers
hf download "$MODEL_ID" --local-dir /data/models/lingbot-world-fast
sglang generate --backend sglang --model-path /data/models/lingbot-world-fast \
  --pipeline-class-name LingBotWorldCausalDMDPipeline \
  --prompt "A first-person walk along a beach" \
  --image-path "$ASSET_DIR/cat.png" \
  --width 832 --height 480 --num-frames 9 --fps 16 \
  --num-inference-steps 4 --guidance-scale 1.0 --seed 42 \
  --text-encoder-cpu-offload --warmup false --save-output
```

上面只做 native offline smoke；正式 benchmark 使用源码 realtime consistency case
`lingbot_world_realtime_plastic_beach` 固定 chunk/camera/prompt 合同，再做性能修改。

```bash
pytest -q python/sglang/multimodal_gen/test/server/test_server_1_gpu.py \
  -k lingbot_world_realtime_plastic_beach -s
```

执行 DiT、[`../components/vae-wan-causal-3d.md`](../components/vae-wan-causal-3d.md)、
encoder/distributed flow。重点 profile bounded sink-window KV cache、
prompt/camera event reset、dynamic condition cache、cross-attention cache、Ulysses camera
conditioner、lazy black-frame VAE encode 和预分配 writer。

报告首 chunk、steady-state chunk/FPS、事件切换延迟、长时间显存；V1/V2 都跑一致性
harness。任何 stale frame、无界 cache 或 prompt/camera 串扰都直接 rejected。
