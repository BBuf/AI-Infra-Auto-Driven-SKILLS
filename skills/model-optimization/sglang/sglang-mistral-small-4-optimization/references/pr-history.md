# SGLang Mistral Small 4 PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Mistral Small 4, Leanstral, and closely related Mistral Large 3 / Ministral serving behavior, including multimodal and EAGLE paths.

## Landed PRs

### PR #14213 - Add Mistral Large 3 support

- Link: https://github.com/sgl-project/sglang/pull/14213
- Why it mattered: Historical base runtime reused by later Small 4 work.
- Runtime path: sglang/python/sglang/srt/models/mistral_large_3.py, sglang/python/sglang/srt/models/mistral_large_3_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #14466 - Add Mistral Large 3 Eagle Support

- Link: https://github.com/sgl-project/sglang/pull/14466
- Why it mattered: Enabled speculative decode on the underlying family.
- Runtime path: sglang/python/sglang/srt/models/mistral_large_3.py, sglang/python/sglang/srt/models/mistral_large_3_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15049 - Mistral Large 3 NVFP4 TRTLLM MoE support

- Link: https://github.com/sgl-project/sglang/pull/15049
- Why it mattered: Added the first serious quantized MoE path.
- Runtime path: sglang/python/sglang/srt/models/mistral_large_3.py, sglang/python/sglang/srt/models/mistral_large_3_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #20708 - Add Mistral Small 4 support

- Link: https://github.com/sgl-project/sglang/pull/20708
- Why it mattered: Brought Mistral Small 4 / Pixtral-style runtime into mainline.
- Runtime path: sglang/python/sglang/srt/models/mistral_large_3.py, sglang/python/sglang/srt/models/mistral_large_3_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #21620 - Mistral Small 4 fails to start due to config/weight format mismatch

- Link: https://github.com/sgl-project/sglang/pull/21620
- Why it mattered: Closed a startup regression after launch.
- Runtime path: sglang/python/sglang/srt/models/mistral_large_3.py, sglang/python/sglang/srt/models/mistral_large_3_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
