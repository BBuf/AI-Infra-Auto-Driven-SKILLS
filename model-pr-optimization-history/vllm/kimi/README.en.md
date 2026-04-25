# vLLM Kimi K2 / K2.5 / Linear / Audio / VL Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Kimi K2 / K2.5 / Linear / Audio / VL.

- Status: supported on current mainline

## Key Conclusions

- The Kimi family in vLLM spans vision, linear-attention, K2.5, and audio checkpoints.
- The most fragile areas are MLA plus FP8/NVFP4 loading, processor evolution, and parser alias compatibility between K2 and K2.5.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/kimi_vl.py`
- `vllm/vllm/model_executor/models/kimi_linear.py`
- `vllm/vllm/model_executor/models/kimi_k25.py`
- `vllm/vllm/model_executor/models/kimi_audio.py`

## Landed PRs

- [#16387](https://github.com/vllm-project/vllm/pull/16387) `Add Kimi-VL model support`: Landed the original Kimi-VL multimodal runtime.
- [#27809](https://github.com/vllm-project/vllm/pull/27809) `Introduce Kimi Linear to vLLM`: Added the linear-attention Kimi family instead of only the VL path.
- [#33131](https://github.com/vllm-project/vllm/pull/33131) `Kimi-K2.5`: Brought the K2.5 generation into mainline.
- [#33876](https://github.com/vllm-project/vllm/pull/33876) `Fix Kimi-K2.5 NVFP4 checkpoints weight loading`: Closed a concrete launch blocker for quantized K2.5 checkpoints.
- [#36127](https://github.com/vllm-project/vllm/pull/36127) `Add support for moonshotai/Kimi-Audio-7B-Instruct`: Extended the family to audio-conditioned serving.
- [#37438](https://github.com/vllm-project/vllm/pull/37438) `Add Kimi-K2.5 reasoning/tool parser aliases`: Aligned parser aliases and tool-call IDs with the newer model outputs.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-kimi-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-kimi-optimization/references/pr-history.md`
