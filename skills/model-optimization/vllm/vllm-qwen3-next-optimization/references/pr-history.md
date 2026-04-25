# vLLM Qwen3-Next PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: Qwen3-Next GDN attention, MTP, packed module naming, PP, and cross-hardware tuned MoE configuration in vLLM.


## Landed PRs

### PR #24709 - Fix Qwen3-Next PP

- Link: https://github.com/vllm-project/vllm/pull/24709
- Why it mattered: Corrected pipeline-parallel execution on Qwen3-Next.
- Runtime path: vllm/vllm/model_executor/models/qwen3_next.py, vllm/vllm/model_executor/models/qwen3_next_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #24957 - Fix the varlen issue in qwen3-next MTP implementation

- Link: https://github.com/vllm-project/vllm/pull/24957
- Why it mattered: Removed a concrete MTP correctness bug on variable-length batches.
- Runtime path: vllm/vllm/model_executor/models/qwen3_next.py, vllm/vllm/model_executor/models/qwen3_next_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #24960 - Add prefixes to shared_expert in qwen3-next

- Link: https://github.com/vllm-project/vllm/pull/24960
- Why it mattered: Fixed ignored-parameter and quantized weight loading for shared experts.
- Runtime path: vllm/vllm/model_executor/models/qwen3_next.py, vllm/vllm/model_executor/models/qwen3_next_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25743 - Fix cuda graph capture bug in GDN metadata and a stride bug

- Link: https://github.com/vllm-project/vllm/pull/25743
- Why it mattered: Stabilized GDN execution under CUDA graphs.
- Runtime path: vllm/vllm/model_executor/models/qwen3_next.py, vllm/vllm/model_executor/models/qwen3_next_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #31722 - Speed-up of GDN attention decode part

- Link: https://github.com/vllm-project/vllm/pull/31722
- Why it mattered: Improved decode throughput on the GDN attention path.
- Runtime path: vllm/vllm/model_executor/models/qwen3_next.py, vllm/vllm/model_executor/models/qwen3_next_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33657 - Initial support for GDN attention on Qwen3-next/Qwen3.5 (XPU)

- Link: https://github.com/vllm-project/vllm/pull/33657
- Why it mattered: Extended the family beyond CUDA with XPU GDN coverage.
- Runtime path: vllm/vllm/model_executor/models/qwen3_next.py, vllm/vllm/model_executor/models/qwen3_next_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
