# vLLM Llama 4 Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Llama 4.

- Status: supported on current mainline

## Key Conclusions

- Llama4 is mature on the vLLM side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/llama4.py`
- `vllm/vllm/model_executor/models/mllama4.py`
- `vllm/vllm/model_executor/models/llama4_eagle.py`

## Landed PRs

- [#16104](https://github.com/vllm-project/vllm/pull/16104) `Support Llama4 in vLLM`: Initial Llama4 landing.
- [#20419](https://github.com/vllm-project/vllm/pull/20419) `Enable ModelOpt Llama4 fp8 checkpoint deployment`: Added ModelOpt FP8 coverage.
- [#20591](https://github.com/vllm-project/vllm/pull/20591) `Llama4 EAGLE Support`: Opened speculative decoding for Llama4.
- [#22511](https://github.com/vllm-project/vllm/pull/22511) `Fix Llama4 FlashInfer FP4 MoE issues`: Stabilized the FP4 MoE path.
- [#25889](https://github.com/vllm-project/vllm/pull/25889) `Fix misplaced dtype cast in Llama4VisionRotaryEmbedding`: Patched a multimodal rotary bug.

## Matching Skill

- `skills/model-optimization/vllm/vllm-llama4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-llama4-optimization/references/pr-history.md`
