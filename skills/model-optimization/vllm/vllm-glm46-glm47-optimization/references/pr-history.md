# vLLM GLM-4.6 / 4.7 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: GLM-4.6, GLM-4.6V, GLM-4.7, GLM-4.7-Flash, GLM-Lite, and the parser / quant / fused-MoE deltas after the 4.5 generation.


## Landed PRs

### PR #26818 - Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on B200

- Link: https://github.com/vllm-project/vllm/pull/26818
- Why it mattered: Added fused-MoE tuning configs for the new Blackwell deployment lane.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #30210 - Fix glm46 awq marlin moe compatibility

- Link: https://github.com/vllm-project/vllm/pull/30210
- Why it mattered: Closed an incompatibility between GLM-4.6 AWQ checkpoints and Marlin MoE assumptions.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #30876 - GLM-4.7 Tool Parser and Doc Update

- Link: https://github.com/vllm-project/vllm/pull/30876
- Why it mattered: Brought parser behavior and docs up to date for 4.7 / 4.7-Flash.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #31386 - GLM Model support for GLM-Lite

- Link: https://github.com/vllm-project/vllm/pull/31386
- Why it mattered: Extended the same runtime family to the Lite checkpoint line.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37386 - Improve tool call parsing and content normalization for glm47

- Link: https://github.com/vllm-project/vllm/pull/37386
- Why it mattered: Fixed concrete parsing errors that surfaced in newer GLM-4.7 outputs.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
