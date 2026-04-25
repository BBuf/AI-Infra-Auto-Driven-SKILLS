# vLLM Mixtral Quark / INT4-FP8 MoE PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported or only adjacent architectures landed on current mainline
- Scope: Mixtral MoE, expert parallelism, FP8 / ModelOpt quantization, and EPLB in vLLM, which together form the nearest equivalent to Quark INT4-FP8 Mixtral serving.

## Landed PRs

### PR #2011 - Mixtral 8x7B support

- Link: https://github.com/vllm-project/vllm/pull/2011
- Why it mattered: Initial Mixtral model-family support.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #2090 - Optimize Mixtral with expert parallelism

- Link: https://github.com/vllm-project/vllm/pull/2090
- Why it mattered: Added early expert-parallel scaling instead of pure TP execution.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #2542 - Fused MOE for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/2542
- Why it mattered: Brought fused-MoE kernels into the Mixtral serving path.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #4527 - Support MoE FP8 checkpoints for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/4527
- Why it mattered: Added the first serious FP8 checkpoint path for Mixtral MoE.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15961 - Support ModelOpt quantization of Mixtral model

- Link: https://github.com/vllm-project/vllm/pull/15961
- Why it mattered: Extended the family to NVIDIA ModelOpt quantization flows.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22842 - Support EPLB for Mixtral Model

- Link: https://github.com/vllm-project/vllm/pull/22842
- Why it mattered: Added expert-parallel load balancing to the Mixtral family.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
