# vLLM DeepSeek V3.1 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: DeepSeek V3.1 parser, scale-format, DeepGEMM, and reasoning-tooling deltas layered on top of the base DeepSeek V3 runtime.

This family inherits the base runtime context from `deepseek-v3-r1`; this file records only the delta that is specific to `deepseek-v31`.

## Landed PRs

### PR #23454 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/vllm-project/vllm/pull/23454
- Why it mattered: Added the first V3.1-specific tool-call parser surface to vLLM.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23666 - Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- Link: https://github.com/vllm-project/vllm/pull/23666
- Why it mattered: Tuned the scale-format path used by DeepGEMM-based DeepSeek V3.1 kernels.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25589 - Add DeepSeek-V3.1 reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/25589
- Why it mattered: Separated V3.1 reasoning output handling from generic DeepSeek parsing.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #32361 - Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- Link: https://github.com/vllm-project/vllm/pull/32361
- Why it mattered: Patched a concrete shape mismatch between newer checkpoints and DeepGEMM assumptions.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
