# vLLM MiMo-V2-Flash PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: MiMo-V2-Flash inference-centric MoE runtime, MTP behavior, and the transition from older MiMo checkpoints in vLLM.

## Landed PRs

### PR #17433 - Support MiMo-7B inference with MTP

- Link: https://github.com/vllm-project/vllm/pull/17433
- Why it mattered: Historical base for the MiMo family.
- Runtime path: vllm/vllm/model_executor/models/mimo_v2_flash.py, vllm/vllm/model_executor/models/mimo.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25136 - Fix MTP inference path for MiMo-7B model

- Link: https://github.com/vllm-project/vllm/pull/25136
- Why it mattered: Closed a concrete draft-path bug.
- Runtime path: vllm/vllm/model_executor/models/mimo_v2_flash.py, vllm/vllm/model_executor/models/mimo.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #30836 - Add MiMo-V2-Flash support

- Link: https://github.com/vllm-project/vllm/pull/30836
- Why it mattered: Landed the dedicated V2-Flash runtime.
- Runtime path: vllm/vllm/model_executor/models/mimo_v2_flash.py, vllm/vllm/model_executor/models/mimo.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
