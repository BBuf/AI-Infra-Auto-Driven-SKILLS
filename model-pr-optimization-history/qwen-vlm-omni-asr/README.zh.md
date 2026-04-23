# SGLang Qwen VLM/Omni/ASR 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 Qwen2.5-VL、Qwen3-VL、Qwen3-VL-MoE、Qwen3-Omni、Qwen3-ASR。

结论：Qwen 多模态优化的风险主要在 processor、ViT attention/cache、mRoPE/3D mRoPE、encoder DP/PP、feature transfer、audio encoder 和 streaming ASR，而不是纯语言模型 forward。

## 代码面

- `python/sglang/srt/models/qwen2_5_vl.py`
- `python/sglang/srt/models/qwen3_vl.py`
- `python/sglang/srt/models/qwen3_vl_moe.py`
- `python/sglang/srt/models/qwen3_omni_moe.py`
- `python/sglang/srt/models/qwen3_asr.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `python/sglang/srt/multimodal/processors/qwen_audio.py`
- `python/sglang/srt/multimodal/processors/qwen3_asr.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen2.5-VL.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-VL.mdx`

## 已合入主线 PR

- `#10911`：Qwen3-Omni thinker-only。
- `#10985`：Qwen3-VL MRotaryEmbedding 参数修复。
- `#12333`：Qwen3-VL PP。
- `#13724`、`#17624`：Qwen3-VL vision model DP / DP size>1。
- `#13736`：Qwen-VL `repeat_interleave` 替换。
- `#14292`：position embedding cache。
- `#14907`：chunked ViT attention。
- `#15320`：Qwen3-VL ViT PCG。
- `#16366`：Qwen3-VL video memory。
- `#18024`：Qwen3-VL weight loading prefix mapping。
- `#18185`：Qwen3-Omni audio encoder。
- `#19003`：FlashInfer CUDNN prefill as ViT backend。
- `#19291`、`#19333`：Qwen3-VL quant_config 和 visual loading。
- `#20759`、`#20788`：DP encoder hang/position embedding TP。
- `#21458`：AMD Qwen3-VL decode fusion：QK-norm、3D mRoPE、KV store。
- `#22038`：chunk-aware ViT encoding cache/lazy transfer。
- `#22073`：Qwen3-ASR 支持。
- `#22089`：chunk-based streaming ASR。
- `#22230`：Qwen3-VL EAGLE3。

## Open PR 雷达

- `#12662`：CPU Qwen3-VL/Qwen3-Omni。
- `#23220`：Qwen3-VL-MoE encoder_only。
- `#23469`：NPU Qwen3-ASR。
- 另需跟踪 `#12261`、`#13918`、`#14886`、`#16491`、`#16571`、`#16996`、`#17202`、`#17276`、`#18721`、`#18771`、`#19242`、`#19693`、`#20857`、`#22052`、`#22848` 等 Qwen multimodal follow-up。

## sgl-cookbook / 文档

- `sgl-cookbook#76`：Qwen3-VL AMD MI300X。
- `sgl-cookbook#84`：Qwen2.5-VL AMD MI300X。
- `sgl-cookbook#102`：Qwen3-VL MI355X。
- `sgl-cookbook#110`：Qwen2.5-VL MI355X/MI325X。
- `sgl-cookbook#124`：Qwen3-VL MI325X。

## 下一步优化建议

1. 给 Qwen3-ASR 建 streaming chunk 边界测试。
2. 给 Qwen3-VL 的 encoder DP/PP、chunked ViT cache、EAGLE3 分别建独立回归。
3. 多模态性能测试必须记录媒体尺寸、帧数、processor 输出 shape 和 feature transfer 时间。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。本轮补入 `#12554`、`#12703`、`#15205`、`#16785`、`#19693`、`#21469`、`#21849`、`#22052`、`#22266`、`#22431`、`#22839`、`#22848`、`#23115`、`#23304`、`#23469`。测试入口补充 `test/manual/models/test_qwen3_asr.py` 和 NPU Qwen3-VL/Qwen3-ASR registered tests；官方 Qwen3-VL 文档中的 FP8/BF16、多图/视频、device cache 和 concurrent-call 参数也已纳入。
