# SGLang InternVL3.5 PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: InternVL3.5 multimodal processor, video support, ViT DP / CUDA graph, and non-CUDA backend compatibility.

## Landed PRs

### PR #5350 - Support InternVL3

- Link: https://github.com/sgl-project/sglang/pull/5350
- Why it mattered: Initial InternVL family support that later carried 3.5.
- Runtime path: sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #13640 - Support Piecewise CUDA Graph for InternVL

- Link: https://github.com/sgl-project/sglang/pull/13640
- Why it mattered: Added graph capture support on the encoder path.
- Runtime path: sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #13925 - Support InternVL Vision Encoder Data Parallelism

- Link: https://github.com/sgl-project/sglang/pull/13925
- Why it mattered: Opened the multi-GPU ViT path.
- Runtime path: sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15942 - Support Video for InternVL3_5

- Link: https://github.com/sgl-project/sglang/pull/15942
- Why it mattered: Extended support to 3.5 video use cases.
- Runtime path: sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #19127 - Support processor and embedding inputs for InternVL

- Link: https://github.com/sgl-project/sglang/pull/19127
- Why it mattered: Hardened processor / embed input interoperability.
- Runtime path: sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
