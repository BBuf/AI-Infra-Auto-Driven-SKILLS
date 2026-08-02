# Cosmos3 flow

覆盖 `nvidia/Cosmos3-Nano`、Super、Super-Text2Image、Super-Image2Video；同一 native
pipeline 根据 `num_frames`/`image_path` 分派 T2I/T2V/I2V。

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model cosmos3-super-t2v --label baseline --output-dir "$BENCH_DIR"
PYTHONPATH=python python3 "$BENCH_PY" \
  --model cosmos3-nano-t2i --label baseline --output-dir "$BENCH_DIR"
PYTHONPATH=python python3 "$BENCH_PY" \
  --model cosmos3-nano-t2v --label baseline --output-dir "$BENCH_DIR"
```

benchmark 隔离时 helper 会禁用 guardrails，生产结论必须说明该差异。执行 DiT、
[`../components/vae-wan-causal-3d.md`](../components/vae-wan-causal-3d.md)（Cosmos 的
`z_dim=48` 独立 shape）、encoder、distributed flow；若 trace 实际出现 AVAE，再跑
audio flow 的 Cosmos 子项。Nano/Super 与三种任务都要独立 smoke；不要跨任务复用
latency/quality reference。
