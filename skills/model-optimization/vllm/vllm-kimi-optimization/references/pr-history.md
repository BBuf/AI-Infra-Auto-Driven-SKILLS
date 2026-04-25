# vLLM Kimi K2 / K2.5 / Linear / Audio / VL PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Kimi-VL, Kimi-Linear, Kimi-K2.5, Kimi-Audio, parser aliases, and quantized MLA behavior in vLLM.

## Landed PRs

### PR #16387 - Add Kimi-VL model support

- Link: https://github.com/vllm-project/vllm/pull/16387
- Why it mattered: Landed the original Kimi-VL multimodal runtime.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #27809 - Introduce Kimi Linear to vLLM

- Link: https://github.com/vllm-project/vllm/pull/27809
- Why it mattered: Added the linear-attention Kimi family instead of only the VL path.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33131 - Kimi-K2.5

- Link: https://github.com/vllm-project/vllm/pull/33131
- Why it mattered: Brought the K2.5 generation into mainline.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33876 - Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- Link: https://github.com/vllm-project/vllm/pull/33876
- Why it mattered: Closed a concrete launch blocker for quantized K2.5 checkpoints.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36127 - Add support for moonshotai/Kimi-Audio-7B-Instruct

- Link: https://github.com/vllm-project/vllm/pull/36127
- Why it mattered: Extended the family to audio-conditioned serving.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37438 - Add Kimi-K2.5 reasoning/tool parser aliases

- Link: https://github.com/vllm-project/vllm/pull/37438
- Why it mattered: Aligned parser aliases and tool-call IDs with the newer model outputs.
- Runtime path: vllm/vllm/model_executor/models/kimi_vl.py, vllm/vllm/model_executor/models/kimi_linear.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
