# SGLang Gemma 4 支持与 PR 历史

本文记录 SGLang 中与 Gemma 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## 主要代码面

- `sglang/python/sglang/srt/models/gemma4_causal.py`
- `sglang/python/sglang/srt/models/gemma4_mm.py`
- `sglang/python/sglang/srt/models/gemma4_vision.py`
- `sglang/python/sglang/srt/models/gemma4_audio.py`

## 已合入 PR

- [#21952](https://github.com/sgl-project/sglang/pull/21952) `New Model: Gemma 4`：Initial Gemma 4 support in SGLang.
- [#22079](https://github.com/sgl-project/sglang/pull/22079) `Gemma4 nvfp4 fix`：Fixed the NVFP4 launch path.
- [#22408](https://github.com/sgl-project/sglang/pull/22408) `Adding Gemma 4 to Nightly CI`：Added model-family regression coverage.

## 配套 skill

- `skills/model-optimization/sglang/sglang-gemma4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-gemma4-optimization/references/pr-history.md`
