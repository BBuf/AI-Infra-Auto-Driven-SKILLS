# vLLM GPT-OSS Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for GPT-OSS.

- Status: supported on current mainline

## Key Conclusions

- GPT-OSS is a flagship MoE family in vLLM.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/gpt_oss.py`

## Landed PRs

- [#22327](https://github.com/vllm-project/vllm/pull/22327) `Add GPT-OSS model code and config`: Initial GPT-OSS landing in vLLM.
- [#23819](https://github.com/vllm-project/vllm/pull/23819) `Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE`: Opened large-scale GPT-OSS serving topologies.
- [#25246](https://github.com/vllm-project/vllm/pull/25246) `Enable Eagle3 speculative decoding for GPT-OSS model`: Added draft-model acceleration.
- [#25515](https://github.com/vllm-project/vllm/pull/25515) `Structure_Tag support for gpt-oss tool-call in cot`: Improved tool calling in reasoning-mode outputs.
- [#30647](https://github.com/vllm-project/vllm/pull/30647) `Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE`: Targeted the hot MXFP4/MXFP8 path for throughput.

## Matching Skill

- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/references/pr-history.md`
