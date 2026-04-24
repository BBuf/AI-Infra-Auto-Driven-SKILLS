# vLLM Qwen3.5 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Qwen3.5 dense / MoE / GDN runtime, MTP, FP8 and NVFP4 quantization, LoRA, and Eagle3 in vLLM.


## Landed PRs

### PR #34110 - Adding Support for Qwen3.5 Models

- Link: https://github.com/vllm-project/vllm/pull/34110
- Why it mattered: Landed the Qwen3.5 runtime family.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #34697 - Redo Qwen3.5/Qwen3-Next GDN projector fusion

- Link: https://github.com/vllm-project/vllm/pull/34697
- Why it mattered: Reworked an earlier fusion that had to be reverted.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #35289 - Fix Qwen3.5 FP8 quantization tuple shard_id weight loading

- Link: https://github.com/vllm-project/vllm/pull/35289
- Why it mattered: Closed a concrete FP8 weight-loading failure.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36658 - Add Eagle3 support for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/36658
- Why it mattered: Enabled the draft-model fast path.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37975 - Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/37975
- Why it mattered: Reduced duplicated GDN logic across related families.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #39181 - Fix EP precision for Qwen3.5, Qwen3-Next

- Link: https://github.com/vllm-project/vllm/pull/39181
- Why it mattered: Patched a serving-precision bug under expert parallelism.
- Runtime path: vllm/vllm/model_executor/models/qwen3_5.py, vllm/vllm/model_executor/models/qwen3_5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
