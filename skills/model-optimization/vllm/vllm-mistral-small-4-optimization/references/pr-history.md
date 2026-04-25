# vLLM Mistral Small 4 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Mistral Small 4, Leanstral, and closely related Mistral Large 3 / Ministral serving behavior, including multimodal and MoE execution.

## Landed PRs

### PR #29757 - Add Mistral Large 3 and Ministral 3

- Link: https://github.com/vllm-project/vllm/pull/29757
- Why it mattered: Landed the runtime family that Mistral Small 4 deployments build on in vLLM.
- Runtime path: vllm/vllm/model_executor/models/mistral_large_3.py, vllm/vllm/model_executor/models/mistral_large_3_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33174 - Add support for Mistral Large 3 inference with Flashinfer MoE

- Link: https://github.com/vllm-project/vllm/pull/33174
- Why it mattered: Improved the practical MoE serving path for the same family.
- Runtime path: vllm/vllm/model_executor/models/mistral_large_3.py, vllm/vllm/model_executor/models/mistral_large_3_eagle.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
