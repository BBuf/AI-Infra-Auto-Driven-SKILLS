# SGLang Step3.5 / Step3-VL PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Step3.5-Flash and Step3-VL-10B serving, MTP, MoE all-reduce, tool/reasoning parser, and processor evolution.

## Landed PRs

### PR #8583 - Support Step3V

- Link: https://github.com/sgl-project/sglang/pull/8583
- Why it mattered: Initial Step3 visual model support.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #8699 - Support DP Attention for step3_vl

- Link: https://github.com/sgl-project/sglang/pull/8699
- Why it mattered: Enabled multi-GPU VL serving.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #9695 - Add step3 tool parser

- Link: https://github.com/sgl-project/sglang/pull/9695
- Why it mattered: Added tool-call parsing.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #18564 - Implement the standard multi-layer MTP for step3p5

- Link: https://github.com/sgl-project/sglang/pull/18564
- Why it mattered: Added Step3.5 draft-model support.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22773 - Optimize allreduce in MoE layers

- Link: https://github.com/sgl-project/sglang/pull/22773
- Why it mattered: Targeted the Step3.5 MoE hot path.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
