# vLLM MiniMax M1 / M2 / VL PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: MiniMaxText01, MiniMax-M1, MiniMax-M2, MiniMax-VL-01, LoRA, and Eagle3 support in vLLM.

## Landed PRs

### PR #13454 - Support MiniMaxText01 model inference

- Link: https://github.com/vllm-project/vllm/pull/13454
- Why it mattered: Landed the original MiniMax text runtime.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #16328 - support MiniMax-VL-01 model

- Link: https://github.com/vllm-project/vllm/pull/16328
- Why it mattered: Added the multimodal MiniMax-VL path.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #19677 - Add support for MiniMaxM1ForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/19677
- Why it mattered: Connected the M1 checkpoint alias to the shared MiniMax runtime.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #27535 - Support MiniMax-M2 Model

- Link: https://github.com/vllm-project/vllm/pull/27535
- Why it mattered: Brought the M2 generation into mainline.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #32763 - Complete LoRA support for MiniMaxM2

- Link: https://github.com/vllm-project/vllm/pull/32763
- Why it mattered: Finished missing adapter wiring in the M2 family.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37512 - MiniMax-M2: add Eagle3 speculative decoding support

- Link: https://github.com/vllm-project/vllm/pull/37512
- Why it mattered: Enabled the draft-model acceleration path for MiniMax M2.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
