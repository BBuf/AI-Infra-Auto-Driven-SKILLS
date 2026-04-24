# SGLang Gemma 4 PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Gemma 4 text, MoE, multimodal, reasoning, tool use, and quantized MoE serving.

## Landed PRs

### PR #21952 - New Model: Gemma 4

- Link: https://github.com/sgl-project/sglang/pull/21952
- Why it mattered: Initial Gemma 4 support in SGLang.
- Runtime path: sglang/python/sglang/srt/models/gemma4_causal.py, sglang/python/sglang/srt/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22079 - Gemma4 nvfp4 fix

- Link: https://github.com/sgl-project/sglang/pull/22079
- Why it mattered: Fixed the NVFP4 launch path.
- Runtime path: sglang/python/sglang/srt/models/gemma4_causal.py, sglang/python/sglang/srt/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22408 - Adding Gemma 4 to Nightly CI

- Link: https://github.com/sgl-project/sglang/pull/22408
- Why it mattered: Added model-family regression coverage.
- Runtime path: sglang/python/sglang/srt/models/gemma4_causal.py, sglang/python/sglang/srt/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
