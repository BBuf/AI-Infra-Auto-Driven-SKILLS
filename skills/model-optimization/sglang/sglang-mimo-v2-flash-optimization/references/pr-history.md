# SGLang MiMo-V2-Flash PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: MiMo-V2-Flash inference-centric MoE runtime, flashinfer fused all-reduce, overlap, and reasoning parser behavior.

## Landed PRs

### PR #15207 - MiMo-V2-Flash day0 support

- Link: https://github.com/sgl-project/sglang/pull/15207
- Why it mattered: Initial MiMo-V2-Flash landing.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- Link: https://github.com/sgl-project/sglang/pull/15464
- Why it mattered: Targeted decode-side communication cost.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15488 - Respect `--swa-full-tokens-ratio`

- Link: https://github.com/sgl-project/sglang/pull/15488
- Why it mattered: Fixed a concrete runtime flag integration bug.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #17634 - Support two batch overlap

- Link: https://github.com/sgl-project/sglang/pull/17634
- Why it mattered: Added overlap / throughput optimization.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #21414 - Add mimo reasoning parser

- Link: https://github.com/sgl-project/sglang/pull/21414
- Why it mattered: Completed the parser path for thinking outputs.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
