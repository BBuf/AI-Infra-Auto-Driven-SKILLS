# SGLang Nemotron Super / Nano Hybrid PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: NemotronH, Nemotron 3 Super, Nemotron Nano hybrid Mamba+Attention+MoE, MTP, NVFP4, and VL adjacencies.

## Landed PRs

### PR #16172 - NemotronH PP support

- Link: https://github.com/sgl-project/sglang/pull/16172
- Why it mattered: Opened pipeline parallelism on NemotronH.
- Runtime path: sglang/python/sglang/srt/models/nemotron_h.py, sglang/python/sglang/srt/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #16227 - Add latent MoE support

- Link: https://github.com/sgl-project/sglang/pull/16227
- Why it mattered: Added the hybrid latent-MoE path.
- Runtime path: sglang/python/sglang/srt/models/nemotron_h.py, sglang/python/sglang/srt/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #19903 - Enable Piecewise CUDA Graph for NemotronH Hybrid Models

- Link: https://github.com/sgl-project/sglang/pull/19903
- Why it mattered: Improved hybrid serving efficiency.
- Runtime path: sglang/python/sglang/srt/models/nemotron_h.py, sglang/python/sglang/srt/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #20407 - Support Nemotron 3 Super NVFP4

- Link: https://github.com/sgl-project/sglang/pull/20407
- Why it mattered: Added the key quantized Super checkpoint path.
- Runtime path: sglang/python/sglang/srt/models/nemotron_h.py, sglang/python/sglang/srt/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #20575 - Add Nemotron 3 Super CI tests for BF16 and NVFP4

- Link: https://github.com/sgl-project/sglang/pull/20575
- Why it mattered: Added regression coverage for the production checkpoint variants.
- Runtime path: sglang/python/sglang/srt/models/nemotron_h.py, sglang/python/sglang/srt/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
