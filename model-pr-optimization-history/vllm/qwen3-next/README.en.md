# vLLM Qwen3-Next Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen3-Next.

- Status: supported on current mainline

## Key Conclusions

- Qwen3-Next is its own runtime family because of Gated DeltaNet attention and its MTP path.
- The practical risks are PP, MTP varlen handling, quantized shared-expert naming, and GDN-specific CUDA graph bugs.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3_next.py`
- `vllm/vllm/model_executor/models/qwen3_next_mtp.py`

## Landed PRs

- [#24709](https://github.com/vllm-project/vllm/pull/24709) `Fix Qwen3-Next PP`: Corrected pipeline-parallel execution on Qwen3-Next.
- [#24957](https://github.com/vllm-project/vllm/pull/24957) `Fix the varlen issue in qwen3-next MTP implementation`: Removed a concrete MTP correctness bug on variable-length batches.
- [#24960](https://github.com/vllm-project/vllm/pull/24960) `Add prefixes to shared_expert in qwen3-next`: Fixed ignored-parameter and quantized weight loading for shared experts.
- [#25743](https://github.com/vllm-project/vllm/pull/25743) `Fix cuda graph capture bug in GDN metadata and a stride bug`: Stabilized GDN execution under CUDA graphs.
- [#31722](https://github.com/vllm-project/vllm/pull/31722) `Speed-up of GDN attention decode part`: Improved decode throughput on the GDN attention path.
- [#33657](https://github.com/vllm-project/vllm/pull/33657) `Initial support for GDN attention on Qwen3-next/Qwen3.5 (XPU)`: Extended the family beyond CUDA with XPU GDN coverage.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-next-optimization/references/pr-history.md`
