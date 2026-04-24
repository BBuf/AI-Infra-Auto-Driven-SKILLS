# vLLM DeepSeek V3 / R1 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: DeepSeek V3 and DeepSeek R1 MLA, MoE, packed-module loading, LoRA, MTP/Eagle, and quantized ROCm/CUDA validation paths.


## Landed PRs

### PR #22352 - Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`

- Link: https://github.com/vllm-project/vllm/pull/22352
- Why it mattered: Fixed quantized and packed-weight loading for DeepSeek V2/V3/R1 style checkpoints.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/models/deepseek_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23971 - Add LoRA support for DeepSeek models (V2, V3, R1-0528)

- Link: https://github.com/vllm-project/vllm/pull/23971
- Why it mattered: Enabled adapter injection on the DeepSeek family rather than only base dense models.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/models/deepseek_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #29545 - Fix DeepSeek R1 MTP weight loading

- Link: https://github.com/vllm-project/vllm/pull/29545
- Why it mattered: Hardened R1 NextN / MTP draft loading after launch failures on draft weights.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/models/deepseek_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36247 - Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x

- Link: https://github.com/vllm-project/vllm/pull/36247
- Why it mattered: Closed a production ROCm gap for compressed-tensors DeepSeek-R1 deployment.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/models/deepseek_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
