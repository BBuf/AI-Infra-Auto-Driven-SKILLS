# vLLM Nemotron Super / Nano Hybrid PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: NemotronH, Nemotron 3 Super, Nemotron Nano hybrid Mamba+Attention+MoE, MTP, NVFP4, and VL adjacencies.

## Landed PRs

### PR #18863 - NemotronH support

- Link: https://github.com/vllm-project/vllm/pull/18863
- Why it mattered: Initial NemotronH landing in vLLM.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25863 - Add MoE support for NemotronH

- Link: https://github.com/vllm-project/vllm/pull/25863
- Why it mattered: Extended the hybrid family to routed experts.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33726 - Nemotron-H MTP and Mamba Speculative Decoding Support

- Link: https://github.com/vllm-project/vllm/pull/33726
- Why it mattered: Opened the MTP / spec-decode path.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36803 - E2E Nemotron-3-Super tests

- Link: https://github.com/vllm-project/vllm/pull/36803
- Why it mattered: Added direct Super-family regression coverage.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37803 - Enable NemotronHPuzzle + NemotronHMTP

- Link: https://github.com/vllm-project/vllm/pull/37803
- Why it mattered: Expanded hybrid and MTP coverage for the family.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
