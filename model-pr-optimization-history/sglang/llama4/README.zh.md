# SGLang Llama 4 支持与 PR 历史

本文记录 SGLang 中与 Llama 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Llama4 is mature on the SGLang side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## 主要代码面

- `sglang/python/sglang/srt/models/llama4.py`
- `sglang/python/sglang/srt/models/mllama4.py`

## 已合入 PR

- [#5092](https://github.com/sgl-project/sglang/pull/5092) `Add Llama4 support`：Initial Llama4 landing in SGLang.
- [#5194](https://github.com/sgl-project/sglang/pull/5194) `Support Llama4 fp8 inference`：Enabled the first production quantized lane.
- [#6162](https://github.com/sgl-project/sglang/pull/6162) `Fix Llama4 gibberish output with long context and CUDA graph`：Closed a major correctness bug.
- [#7129](https://github.com/sgl-project/sglang/pull/7129) `Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang`：Added the ModelOpt checkpoint path.
- [#13421](https://github.com/sgl-project/sglang/pull/13421) `Add Llama4 attention backend auto-selection`：Stabilized backend choice for real deployments.

## 配套 skill

- `skills/model-optimization/sglang/sglang-llama4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-llama4-optimization/references/pr-history.md`
