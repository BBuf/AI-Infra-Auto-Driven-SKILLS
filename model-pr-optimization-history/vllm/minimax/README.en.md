# vLLM MiniMax M1 / M2 / VL Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for MiniMax M1 / M2 / VL.

- Status: supported on current mainline

## Key Conclusions

- MiniMax support evolved from the text-01 path into M1/M2 and VL variants.
- Today the key production surfaces are linear-attention correctness, VL processor behavior, LoRA, and Eagle3 on M2.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/minimax_text_01.py`
- `vllm/vllm/model_executor/models/minimax_m2.py`
- `vllm/vllm/model_executor/models/minimax_vl_01.py`

## Landed PRs

- [#13454](https://github.com/vllm-project/vllm/pull/13454) `Support MiniMaxText01 model inference`: Landed the original MiniMax text runtime.
- [#16328](https://github.com/vllm-project/vllm/pull/16328) `support MiniMax-VL-01 model`: Added the multimodal MiniMax-VL path.
- [#19677](https://github.com/vllm-project/vllm/pull/19677) `Add support for MiniMaxM1ForCausalLM`: Connected the M1 checkpoint alias to the shared MiniMax runtime.
- [#27535](https://github.com/vllm-project/vllm/pull/27535) `Support MiniMax-M2 Model`: Brought the M2 generation into mainline.
- [#32763](https://github.com/vllm-project/vllm/pull/32763) `Complete LoRA support for MiniMaxM2`: Finished missing adapter wiring in the M2 family.
- [#37512](https://github.com/vllm-project/vllm/pull/37512) `MiniMax-M2: add Eagle3 speculative decoding support`: Enabled the draft-model acceleration path for MiniMax M2.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-minimax-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-minimax-optimization/references/pr-history.md`
