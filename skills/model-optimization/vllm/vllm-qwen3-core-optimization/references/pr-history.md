# vLLM Qwen3 Core PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Qwen3 dense, Qwen3 MoE, embeddings/rerankers, GGUF/GPTQ/ModelOpt quant paths, and Eagle3 speculative decoding in vLLM.


## Landed PRs

### PR #15289 - Add Qwen3 and Qwen3MoE

- Link: https://github.com/vllm-project/vllm/pull/15289
- Why it mattered: Initial Qwen3 dense and MoE support landed here.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/model_executor/models/qwen3_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #19260 - Support Qwen3 Embedding & Reranker

- Link: https://github.com/vllm-project/vllm/pull/19260
- Why it mattered: Extended the family to bidirectional embedding / reranker models.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/model_executor/models/qwen3_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #19598 - Skip loading extra parameters for modelopt Qwen3 MoE model

- Link: https://github.com/vllm-project/vllm/pull/19598
- Why it mattered: Fixed a concrete ModelOpt launch failure on Qwen3 MoE.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/model_executor/models/qwen3_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22017 - KeyError for Qwen3-MoE with GPTQ on ROCm

- Link: https://github.com/vllm-project/vllm/pull/22017
- Why it mattered: Closed a GPTQ loading failure in the Qwen3 MoE path.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/model_executor/models/qwen3_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22785 - Fix GGUF loader for Qwen3 MoE

- Link: https://github.com/vllm-project/vllm/pull/22785
- Why it mattered: Made the Qwen3 MoE loader accept GGUF weights again.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/model_executor/models/qwen3_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23490 - Fix Qwen3 MoE GPTQ inference

- Link: https://github.com/vllm-project/vllm/pull/23490
- Why it mattered: Patched runtime correctness after GPTQ startup succeeded.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/model_executor/models/qwen3_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #26485 - Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE

- Link: https://github.com/vllm-project/vllm/pull/26485
- Why it mattered: Enabled the draft-model path on top of the base Qwen3 MoE runtime.
- Runtime path: vllm/vllm/model_executor/models/qwen3.py, vllm/vllm/model_executor/models/qwen3_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
