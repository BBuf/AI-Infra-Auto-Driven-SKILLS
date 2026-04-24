# vLLM DeepSeek V3.2 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: DeepSeek V3.2 sparse-MLA / DSA runtime, indexer, tool parser, MTP fallback, and long-context decode kernels in vLLM.


## Landed PRs

### PR #25896 - Support DeepSeek-V3.2

- Link: https://github.com/vllm-project/vllm/pull/25896
- Why it mattered: Landed the initial V3.2 model registration, sparse-attention runtime, and benchmark hooks.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/rotary_embedding/mrope.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25999 - Support indexer prefill chunking

- Link: https://github.com/vllm-project/vllm/pull/25999
- Why it mattered: Made the V3.2 sparse indexer work with chunked prefill instead of eager-only behavior.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/rotary_embedding/mrope.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #26670 - Add AMD GPU support on DeepSeek v3.2 and SparseMLA

- Link: https://github.com/vllm-project/vllm/pull/26670
- Why it mattered: Opened the ROCm SparseMLA lane for V3.2 deployments.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/rotary_embedding/mrope.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #29848 - Add DeepSeek-V3.2 tool parser

- Link: https://github.com/vllm-project/vllm/pull/29848
- Why it mattered: Added the parser surface that cookbook-style V3.2 reasoning deployments depend on.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/rotary_embedding/mrope.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33090 - Fix DeepseekV32 `AssertionError: num_kv_heads == 1`

- Link: https://github.com/vllm-project/vllm/pull/33090
- Why it mattered: Removed a hard failure triggered by newer V3.2 attention shapes.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/rotary_embedding/mrope.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37421 - Persistent TopK scheduler for DeepSeek-V3.2 decode

- Link: https://github.com/vllm-project/vllm/pull/37421
- Why it mattered: Modernized the decode scheduler with a CUDAGraph-safe persistent TopK kernel.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/rotary_embedding/mrope.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
