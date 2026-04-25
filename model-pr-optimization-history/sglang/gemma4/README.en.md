# SGLang Gemma 4 Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Gemma 4.

- Status: 当前 mainline 已支持

## Key Conclusions

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/gemma4_causal.py`
- `sglang/python/sglang/srt/models/gemma4_mm.py`
- `sglang/python/sglang/srt/models/gemma4_vision.py`
- `sglang/python/sglang/srt/models/gemma4_audio.py`

## Landed PRs

- [#21952](https://github.com/sgl-project/sglang/pull/21952) `New Model: Gemma 4`: Initial Gemma 4 support in SGLang.
- [#22079](https://github.com/sgl-project/sglang/pull/22079) `Gemma4 nvfp4 fix`: Fixed the NVFP4 launch path.
- [#22408](https://github.com/sgl-project/sglang/pull/22408) `Adding Gemma 4 to Nightly CI`: Added model-family regression coverage.

## Matching Skill

- `skills/model-optimization/sglang/sglang-gemma4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-gemma4-optimization/references/pr-history.md`
