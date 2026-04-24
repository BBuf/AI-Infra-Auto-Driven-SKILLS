# vLLM GPT-OSS PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: OpenAI GPT-OSS MoE, MXFP4/FP8 quantization, DP/EP, reasoning parser, tool calling, and Eagle/spec decode.

## Landed PRs

### PR #22327 - Add GPT-OSS model code and config

- Link: https://github.com/vllm-project/vllm/pull/22327
- Why it mattered: Initial GPT-OSS landing in vLLM.
- Runtime path: vllm/vllm/model_executor/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23819 - Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE

- Link: https://github.com/vllm-project/vllm/pull/23819
- Why it mattered: Opened large-scale GPT-OSS serving topologies.
- Runtime path: vllm/vllm/model_executor/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25246 - Enable Eagle3 speculative decoding for GPT-OSS model

- Link: https://github.com/vllm-project/vllm/pull/25246
- Why it mattered: Added draft-model acceleration.
- Runtime path: vllm/vllm/model_executor/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25515 - Structure_Tag support for gpt-oss tool-call in cot

- Link: https://github.com/vllm-project/vllm/pull/25515
- Why it mattered: Improved tool calling in reasoning-mode outputs.
- Runtime path: vllm/vllm/model_executor/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #30647 - Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE

- Link: https://github.com/vllm-project/vllm/pull/30647
- Why it mattered: Targeted the hot MXFP4/MXFP8 path for throughput.
- Runtime path: vllm/vllm/model_executor/models/gpt_oss.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
