# vLLM GLM VLM / OCR PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: GLM-4V, GLM-4.1V, GLM-OCR, GLM visual processor, MRoPE, video, and OCR MTP behavior in vLLM.


## Landed PRs

### PR #9242 - Add GLM-4v support

- Link: https://github.com/vllm-project/vllm/pull/9242
- Why it mattered: Landed the original GLM4V multimodal model path.
- Runtime path: vllm/vllm/model_executor/models/glm4v.py, vllm/vllm/model_executor/models/glm4_1v.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #19331 - Add GLM4.1V model

- Link: https://github.com/vllm-project/vllm/pull/19331
- Why it mattered: Extended the family to the newer GLM4.1V checkpoint layout and vision stack.
- Runtime path: vllm/vllm/model_executor/models/glm4v.py, vllm/vllm/model_executor/models/glm4_1v.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #27860 - Fix broken MRoPE for GLM-4.1V/GLM-4.5V

- Link: https://github.com/vllm-project/vllm/pull/27860
- Why it mattered: Closed a positional-embedding bug with large practical accuracy impact on vision inputs.
- Runtime path: vllm/vllm/model_executor/models/glm4v.py, vllm/vllm/model_executor/models/glm4_1v.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33005 - GLM-OCR with MTP Support

- Link: https://github.com/vllm-project/vllm/pull/33005
- Why it mattered: Added OCR-specific draft / MTP support rather than text-only OCR loading.
- Runtime path: vllm/vllm/model_executor/models/glm4v.py, vllm/vllm/model_executor/models/glm4_1v.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33350 - Fix broken GLM-OCR initialization

- Link: https://github.com/vllm-project/vllm/pull/33350
- Why it mattered: Fixed startup failures in the GLM-OCR path after the first bring-up.
- Runtime path: vllm/vllm/model_executor/models/glm4v.py, vllm/vllm/model_executor/models/glm4_1v.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37962 - GLM OCR Patch Merger context_dim

- Link: https://github.com/vllm-project/vllm/pull/37962
- Why it mattered: Updated the patch-merger contract for newer OCR checkpoints.
- Runtime path: vllm/vllm/model_executor/models/glm4v.py, vllm/vllm/model_executor/models/glm4_1v.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
