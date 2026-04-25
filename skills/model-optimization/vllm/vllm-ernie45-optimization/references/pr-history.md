# vLLM Ernie4.5 / Ernie4.5-VL PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Baidu Ernie4.5 text/VL/MoE runtime, vision rotary, and long-input stability.

## Landed PRs

### PR #20220 - Add Ernie4.5 and Ernie4.5MoE Model Support

- Link: https://github.com/vllm-project/vllm/pull/20220
- Why it mattered: Landed text and MoE support.
- Runtime path: vllm/vllm/model_executor/models/ernie45.py, vllm/vllm/model_executor/models/ernie45_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #21717 - Fix Ernie4_5_MoeForCausalLM shared experts

- Link: https://github.com/vllm-project/vllm/pull/21717
- Why it mattered: Fixed shared-expert correctness.
- Runtime path: vllm/vllm/model_executor/models/ernie45.py, vllm/vllm/model_executor/models/ernie45_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22514 - Add Ernie4.5 VL Model Support

- Link: https://github.com/vllm-project/vllm/pull/22514
- Why it mattered: Added the multimodal Ernie4.5-VL lane.
- Runtime path: vllm/vllm/model_executor/models/ernie45.py, vllm/vllm/model_executor/models/ernie45_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #24074 - Fix Ernie4.5-VL hanging on long inputs

- Link: https://github.com/vllm-project/vllm/pull/24074
- Why it mattered: Closed a production long-input stall.
- Runtime path: vllm/vllm/model_executor/models/ernie45.py, vllm/vllm/model_executor/models/ernie45_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #31274 - Support video metadata for timestamp rendering

- Link: https://github.com/vllm-project/vllm/pull/31274
- Why it mattered: Improved VL video output fidelity.
- Runtime path: vllm/vllm/model_executor/models/ernie45.py, vllm/vllm/model_executor/models/ernie45_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
