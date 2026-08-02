# Wan / FastWan / TurboWan flow

覆盖 registry 中 Wan2.1/2.2 T2V、I2V、TI2V、FastWan、TurboWan、Wan-Fun 和 NVIDIA
NVFP4。先以 nightly 三个 shape 建基线：

```bash
PYTHONPATH=python python3 "$BENCH_PY" --model wan-t2v --label baseline --output-dir "$BENCH_DIR"
PYTHONPATH=python python3 "$BENCH_PY" --model wan-ti2v --label baseline --output-dir "$BENCH_DIR"
PYTHONPATH=python python3 "$BENCH_PY" --model wan-i2v --label baseline --output-dir "$BENCH_DIR"
# FastWan 5B：--model fastwan22-ti2v-5b
```

执行 [`../components/dit-attention.md`](../components/dit-attention.md)、
[`../components/vae-wan-causal-3d.md`](../components/vae-wan-causal-3d.md)、encoder 与
distributed flow。4 卡 nightly topology 是 CFG + Ulysses2；纯 latency 另测 Ulysses4，
8 卡比较 Ulysses8 与 CFG+Ulysses4。I2V/TI2V 必须保留输入图和 encode 正确性。

优先 profile attention/A2A、packed layout、modulation、causal VAE 与 text encoder
offload；少步/蒸馏和 NVFP4 单列质量路线。至少补 1.3B、5B、A14B 各一条 smoke。
