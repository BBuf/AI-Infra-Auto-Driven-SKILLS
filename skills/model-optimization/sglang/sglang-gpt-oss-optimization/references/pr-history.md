# SGLang GPT-OSS PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: OpenAI GPT-OSS MoE, MXFP4/FP8 quantization, DP/EP, reasoning parser, tool calling, and Eagle/spec decode.

## Landed PRs

### PR #8843 - Support mxfp4 for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8843
- Why it mattered: Added the headline quantized checkpoint path.
- Runtime path: sglang/python/sglang/srt/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #8944 - Expert Parallelism for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8944
- Why it mattered: Scaled GPT-OSS beyond pure tensor parallel.
- Runtime path: sglang/python/sglang/srt/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #9043 - Implement Native GPT-OSS Tool Call Support

- Link: https://github.com/sgl-project/sglang/pull/9043
- Why it mattered: Added native tool parser support instead of Harmony integration.
- Runtime path: sglang/python/sglang/srt/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #9359 - Support DP attention with GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9359
- Why it mattered: Enabled larger topologies via DP attention.
- Runtime path: sglang/python/sglang/srt/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #14920 - GPT-OSS Eagle v2 support

- Link: https://github.com/sgl-project/sglang/pull/14920
- Why it mattered: Added speculative decoding support.
- Runtime path: sglang/python/sglang/srt/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #18988 - Support fp8 online quantization for gpt-oss bf16

- Link: https://github.com/sgl-project/sglang/pull/18988
- Why it mattered: Extended quantization coverage to online FP8.
- Runtime path: sglang/python/sglang/srt/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
