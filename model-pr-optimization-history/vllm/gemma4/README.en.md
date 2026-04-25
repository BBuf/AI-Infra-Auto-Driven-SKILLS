# vLLM Gemma 4 Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Gemma 4.

- Status: supported on current mainline

## Key Conclusions

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/gemma4.py`
- `vllm/vllm/model_executor/models/gemma4_mm.py`

## Landed PRs

- [#38826](https://github.com/vllm-project/vllm/pull/38826) `Implement Google Gemma 4 architecture support`: Initial Gemma 4 text/MoE/multimodal landing.
- [#38879](https://github.com/vllm-project/vllm/pull/38879) `Enable Fast Prefill Optimization`: Added YOCO KV-sharing based fast prefill for Gemma4.
- [#39045](https://github.com/vllm-project/vllm/pull/39045) `Support quantized MoE`: Extended Gemma4 to quantized MoE checkpoints.
- [#38844](https://github.com/vllm-project/vllm/pull/38844) `Enable Gemma4ForCausalLM to load LoRA adapters correctly`: Fixed adapter naming/load behavior.
- [#39450](https://github.com/vllm-project/vllm/pull/39450) `Add Gemma4 Eagle3 support`: Enabled speculative decode for Gemma4.

## Matching Skill

- `skills/model-optimization/vllm/vllm-gemma4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gemma4-optimization/references/pr-history.md`
