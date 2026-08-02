# MOVA flow

覆盖 `OpenMOSS-Team/MOVA-360p` 与 `OpenMOSS-Team/MOVA-720p` detector path。MOVA
同时生成视频和音频，并含人物条件。

```bash
MODEL_DIR=$(hf download OpenMOSS-Team/MOVA-720p)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model mova-720p --label baseline --output-dir "$BENCH_DIR"
```

preset 使用 `$ASSET_DIR/mova_single_person.jpg` 和 4 GPU。执行 DiT、
[`../components/vae-wan-causal-3d.md`](../components/vae-wan-causal-3d.md)、audio flow
的 MOVA DAC 子项、encoder/distributed flow。分别计时视频 VAE、DAC/audio、人物条件、
denoise 与 mux；360p/720p 都要验。正确性分别看身份/视频帧、waveform 与 A/V sync，
禁止只检查输出文件能播放。
