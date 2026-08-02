# LTX-2 / LTX-2.3 flow

```bash
MODEL_DIR=$(hf download Lightricks/LTX-2.3)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model ltx23-ti2v-two-stage --label baseline --output-dir "$BENCH_DIR"
# 补跑：ltx2、ltx23-one-stage、ltx23-two-stage、ltx23-hq-two-stage、
#       ltx23-two-stage-cfg-parallel
```

执行 [`../components/dit-attention.md`](../components/dit-attention.md)、
[`../components/vae-ltx-video.md`](../components/vae-ltx-video.md)、
[`../components/vae-audio.md`](../components/vae-audio.md) 的 LTX 子项，以及 encoder/
distributed flow。重点检查 split RoPE、residual-gate add、upsampler GroupNorm+SiLU、
双阶段切换与 CFG parallel。

one-stage、two-stage、HQ 的 shape/VRAM/质量分表；TI2V 验输入图与 121 帧时间一致性，
音频路径另验 waveform。JoyAI-Echo 有独立 flow，不在这里重复定义。
