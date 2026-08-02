# Qwen-Image / FireRed flow

覆盖 Qwen-Image、2512、Edit、Edit-2509/2511、Layered、NVIDIA NVFP4，以及共用
Qwen Edit Plus native path 的 FireRed 1.0/1.1。

```bash
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 2)
PYTHONPATH=python python3 "$BENCH_PY" --model qwen --label baseline --output-dir "$BENCH_DIR"
PYTHONPATH=python python3 "$BENCH_PY" --model qwen-edit --label baseline --output-dir "$BENCH_DIR"
# 另有 qwen-image、qwen-edit-2509、firered-edit-1.0、firered-edit-1.1 preset
```

下载 ID 直接取 README registry 表；edit preset 依赖 `$ASSET_DIR/cat.png`。执行
[`../components/dit-attention.md`](../components/dit-attention.md)、
[`../components/vae-qwen-causal-3d.md`](../components/vae-qwen-causal-3d.md) 与 encoder
flow。重点检查 Qwen modulation、fused QK norm/RoPE、packed sequence、edit 多图条件
和 CFG/TP。Layered 要单独验证层数、alpha/ordering；NVFP4 单列质量预算。
