# vLLM Hunyuan 3 Preview Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Hunyuan 3 Preview.

- Status: partially supported or only adjacent architectures landed on current mainline

## Key Conclusions

- vLLM does not currently expose a dedicated Hunyuan 3 Preview model alias.
- The closest landed evidence is the Hunyuan dense, Hunyuan OCR, and HunyuanVL / Eagle work already in tree.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/hunyuan_v1.py`
- `vllm/vllm/model_executor/models/hunyuan_vision.py`

## Landed PRs

- [#21368](https://github.com/vllm-project/vllm/pull/21368) `Add Hunyuan V1 Dense Model support`: Brought the dense Hunyuan line into vLLM mainline.
- [#29327](https://github.com/vllm-project/vllm/pull/29327) `Add HunyuanOCR support`: Extended the family to OCR workloads instead of text-only generation.
- [#33035](https://github.com/vllm-project/vllm/pull/33035) `Eagle3 support for HunyuanVL & Hunyuan`: Added speculative decoding support on top of the Hunyuan family.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-hunyuan3-preview-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-hunyuan3-preview-optimization/references/pr-history.md`
