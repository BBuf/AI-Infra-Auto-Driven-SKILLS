# vLLM GLM-4.5 / 4.5V PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: GLM-4.5 text, GLM-4.5V, GLM-4.5-Air, shared MoE routing, and tool/reasoning parser behavior in vLLM.


## Landed PRs

### PR #22171 - Modify the organization of GLM series

- Link: https://github.com/vllm-project/vllm/pull/22171
- Why it mattered: Reworked the family layout so 4.5-era models reused a cleaner GLM structure.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22460 - not tie_word_embeddings for glm-4.5 and glm-4.5v

- Link: https://github.com/vllm-project/vllm/pull/22460
- Why it mattered: Aligned the loader with the real 4.5 checkpoint contract instead of forcing tied embeddings.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22832 - Modify the gate implementation of glm4_moe

- Link: https://github.com/vllm-project/vllm/pull/22832
- Why it mattered: Changed the GLM4.5 MoE gating path used by text and VL variants.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23695 - Add triton fused moe config for GLM-4.5-Air-FP8 on B200

- Link: https://github.com/vllm-project/vllm/pull/23695
- Why it mattered: Added a production kernel-tuning lane for the 4.5 Air FP8 deployment path.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #24589 - Add documentation for GLM-4.5 series tool-calling and reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/24589
- Why it mattered: Codified the parser choices needed to serve 4.5 reasoning / tool checkpoints correctly.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
