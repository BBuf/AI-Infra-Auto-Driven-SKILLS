# vLLM Gemma 4 支持与 PR 历史

本文记录 vLLM 中与 Gemma 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## 主要代码面

- `vllm/vllm/model_executor/models/gemma4.py`
- `vllm/vllm/model_executor/models/gemma4_mm.py`

## 已合入 PR

- [#38826](https://github.com/vllm-project/vllm/pull/38826) `Implement Google Gemma 4 architecture support`：Initial Gemma 4 text/MoE/multimodal landing.
- [#38879](https://github.com/vllm-project/vllm/pull/38879) `Enable Fast Prefill Optimization`：Added YOCO KV-sharing based fast prefill for Gemma4.
- [#39045](https://github.com/vllm-project/vllm/pull/39045) `Support quantized MoE`：Extended Gemma4 to quantized MoE checkpoints.
- [#38844](https://github.com/vllm-project/vllm/pull/38844) `Enable Gemma4ForCausalLM to load LoRA adapters correctly`：Fixed adapter naming/load behavior.
- [#39450](https://github.com/vllm-project/vllm/pull/39450) `Add Gemma4 Eagle3 support`：Enabled speculative decode for Gemma4.

## 配套 skill

- `skills/model-optimization/vllm/vllm-gemma4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gemma4-optimization/references/pr-history.md`
