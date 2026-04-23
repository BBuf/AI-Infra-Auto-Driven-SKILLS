# SGLang Qwen VLM/Omni/ASR Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, and Qwen3-ASR.

## Summary

Qwen multimodal optimization is dominated by processors, ViT attention/cache, mRoPE/3D mRoPE, encoder DP/PP, feature transfer, audio encoder behavior, and streaming ASR, rather than pure language-model forward.

## Code Surfaces

- `python/sglang/srt/models/qwen2_5_vl.py`
- `python/sglang/srt/models/qwen3_vl.py`
- `python/sglang/srt/models/qwen3_vl_moe.py`
- `python/sglang/srt/models/qwen3_omni_moe.py`
- `python/sglang/srt/models/qwen3_asr.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `python/sglang/srt/multimodal/processors/qwen_audio.py`
- `python/sglang/srt/multimodal/processors/qwen3_asr.py`

## Merged PRs

Key merged PRs: `#10911`, `#10985`, `#12333`, `#13724`, `#13736`, `#14292`, `#14907`, `#15320`, `#16366`, `#17624`, `#18024`, `#18185`, `#19003`, `#19291`, `#19333`, `#20759`, `#20788`, `#21458`, `#22038`, `#22073`, `#22089`, and `#22230`.

## Open Radar

Track `#12662`, `#23220`, `#23469`, and the broader Qwen multimodal follow-up PRs listed in the skill reference.

## Cookbook Evidence

Track `sgl-cookbook#76`, `#84`, `#102`, `#110`, and `#124`.

## Next Work

Add streaming ASR chunk-boundary tests, separate Qwen3-VL encoder DP/PP, chunked ViT cache, and EAGLE3 regression lanes, and always record media size, frame count, processor output shape, and feature-transfer time.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. This pass adds `#12554`, `#12703`, `#15205`, `#16785`, `#19693`, `#21469`, `#21849`, `#22052`, `#22266`, `#22431`, `#22839`, `#22848`, `#23115`, `#23304`, and `#23469`. Extra test surfaces include `test/manual/models/test_qwen3_asr.py` and NPU Qwen3-VL/Qwen3-ASR registered tests; official Qwen3-VL docs add FP8/BF16, image/video, device-cache, and concurrent-call requirements.
