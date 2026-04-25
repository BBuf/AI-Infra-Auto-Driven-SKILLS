# vLLM InternVL3.5 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: InternVL3.5 multimodal processor, video support, ViT DP / torch.compile, and backend compatibility.

## Landed PRs

### PR #6514 - Initialize support for InternVL2 series models

- Link: https://github.com/vllm-project/vllm/pull/6514
- Why it mattered: Historical base for current InternVL runtime code.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #18499 - Initialize video input support for InternVL models

- Link: https://github.com/vllm-project/vllm/pull/18499
- Why it mattered: Added video processing to the family.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23658 - Enable video support for InternVL3.5 models

- Link: https://github.com/vllm-project/vllm/pull/23658
- Why it mattered: Carried video support into the 3.5 checkpoints.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23742 - Enable native HF format InternVL support

- Link: https://github.com/vllm-project/vllm/pull/23742
- Why it mattered: Removed reliance on ad hoc checkpoint rewrites.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #38049 - Add torch.compile support for InternVL vision encoder

- Link: https://github.com/vllm-project/vllm/pull/38049
- Why it mattered: Modernized the encoder execution path.
- Runtime path: vllm/vllm/model_executor/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
