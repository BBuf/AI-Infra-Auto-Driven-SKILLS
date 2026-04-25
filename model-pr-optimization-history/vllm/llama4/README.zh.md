# vLLM Llama 4 支持与 PR 历史

本文记录 vLLM 中与 Llama 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Llama4 is mature on the vLLM side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## 主要代码面

- `vllm/vllm/model_executor/models/llama4.py`
- `vllm/vllm/model_executor/models/mllama4.py`
- `vllm/vllm/model_executor/models/llama4_eagle.py`

## 已合入 PR

- [#16104](https://github.com/vllm-project/vllm/pull/16104) `Support Llama4 in vLLM`：Initial Llama4 landing.
- [#20419](https://github.com/vllm-project/vllm/pull/20419) `Enable ModelOpt Llama4 fp8 checkpoint deployment`：Added ModelOpt FP8 coverage.
- [#20591](https://github.com/vllm-project/vllm/pull/20591) `Llama4 EAGLE Support`：Opened speculative decoding for Llama4.
- [#22511](https://github.com/vllm-project/vllm/pull/22511) `Fix Llama4 FlashInfer FP4 MoE issues`：Stabilized the FP4 MoE path.
- [#25889](https://github.com/vllm-project/vllm/pull/25889) `Fix misplaced dtype cast in Llama4VisionRotaryEmbedding`：Patched a multimodal rotary bug.

## 配套 skill

- `skills/model-optimization/vllm/vllm-llama4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-llama4-optimization/references/pr-history.md`
