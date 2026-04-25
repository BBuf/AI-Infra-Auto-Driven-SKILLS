# vLLM Llama 4 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Llama4 text and multimodal runtime, FP8/FP4 quantization, router behavior, long-context attention, and Eagle support.

## Landed PRs

### PR #16104 - Support Llama4 in vLLM

- Link: https://github.com/vllm-project/vllm/pull/16104
- Why it mattered: Initial Llama4 landing.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #20419 - Enable ModelOpt Llama4 fp8 checkpoint deployment

- Link: https://github.com/vllm-project/vllm/pull/20419
- Why it mattered: Added ModelOpt FP8 coverage.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #20591 - Llama4 EAGLE Support

- Link: https://github.com/vllm-project/vllm/pull/20591
- Why it mattered: Opened speculative decoding for Llama4.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22511 - Fix Llama4 FlashInfer FP4 MoE issues

- Link: https://github.com/vllm-project/vllm/pull/22511
- Why it mattered: Stabilized the FP4 MoE path.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25889 - Fix misplaced dtype cast in Llama4VisionRotaryEmbedding

- Link: https://github.com/vllm-project/vllm/pull/25889
- Why it mattered: Patched a multimodal rotary bug.
- Runtime path: vllm/vllm/model_executor/models/llama4.py, vllm/vllm/model_executor/models/mllama4.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
