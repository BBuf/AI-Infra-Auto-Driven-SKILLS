# SGLang Llama 4 PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Llama4 text and multimodal runtime, FP8/FP4 quantization, router behavior, long-context attention, and Eagle support.

## Landed PRs

### PR #5092 - Add Llama4 support

- Link: https://github.com/sgl-project/sglang/pull/5092
- Why it mattered: Initial Llama4 landing in SGLang.
- Runtime path: sglang/python/sglang/srt/models/llama4.py, sglang/python/sglang/srt/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #5194 - Support Llama4 fp8 inference

- Link: https://github.com/sgl-project/sglang/pull/5194
- Why it mattered: Enabled the first production quantized lane.
- Runtime path: sglang/python/sglang/srt/models/llama4.py, sglang/python/sglang/srt/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #6162 - Fix Llama4 gibberish output with long context and CUDA graph

- Link: https://github.com/sgl-project/sglang/pull/6162
- Why it mattered: Closed a major correctness bug.
- Runtime path: sglang/python/sglang/srt/models/llama4.py, sglang/python/sglang/srt/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #7129 - Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang

- Link: https://github.com/sgl-project/sglang/pull/7129
- Why it mattered: Added the ModelOpt checkpoint path.
- Runtime path: sglang/python/sglang/srt/models/llama4.py, sglang/python/sglang/srt/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #13421 - Add Llama4 attention backend auto-selection

- Link: https://github.com/sgl-project/sglang/pull/13421
- Why it mattered: Stabilized backend choice for real deployments.
- Runtime path: sglang/python/sglang/srt/models/llama4.py, sglang/python/sglang/srt/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
