# vLLM DeepSeek V3.1 支持与 PR 历史

本文记录 vLLM 中与 DeepSeek V3.1 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持
- 该家族继承 `deepseek-v3-r1` 的基础 runtime，这里只记录增量 PR。

## 核心结论

- V3.1 mostly reuses the base V3 runtime and adds parser plus scale-format correctness work.
- The practical blast radius is in tool calling, DeepGEMM scale handling, and reasoning-parser behavior.

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`

## 已合入 PR

- [#23454](https://github.com/vllm-project/vllm/pull/23454) `Support DeepSeek-V3.1 tool call`：Added the first V3.1-specific tool-call parser surface to vLLM.
- [#23666](https://github.com/vllm-project/vllm/pull/23666) `Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt`：Tuned the scale-format path used by DeepGEMM-based DeepSeek V3.1 kernels.
- [#25589](https://github.com/vllm-project/vllm/pull/25589) `Add DeepSeek-V3.1 reasoning parser`：Separated V3.1 reasoning output handling from generic DeepSeek parsing.
- [#32361](https://github.com/vllm-project/vllm/pull/32361) `Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes`：Patched a concrete shape mismatch between newer checkpoints and DeepGEMM assumptions.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-deepseek-v31-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v31-optimization/references/pr-history.md`
