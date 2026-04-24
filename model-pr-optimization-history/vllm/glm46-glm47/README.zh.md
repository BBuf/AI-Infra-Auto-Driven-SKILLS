# vLLM GLM-4.6 / 4.7 支持与 PR 历史

本文记录 vLLM 中与 GLM-4.6 / 4.7 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- The 4.6/4.7 generation mainly extends the 4.5 base with new tuning tables, parser behavior, and Lite variants.
- AWQ / Marlin compatibility and content-normalization in tool parsing are the recurring pitfalls.

## 主要代码面

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## 已合入 PR

- [#26818](https://github.com/vllm-project/vllm/pull/26818) `Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on B200`：Added fused-MoE tuning configs for the new Blackwell deployment lane.
- [#30210](https://github.com/vllm-project/vllm/pull/30210) `Fix glm46 awq marlin moe compatibility`：Closed an incompatibility between GLM-4.6 AWQ checkpoints and Marlin MoE assumptions.
- [#30876](https://github.com/vllm-project/vllm/pull/30876) `GLM-4.7 Tool Parser and Doc Update`：Brought parser behavior and docs up to date for 4.7 / 4.7-Flash.
- [#31386](https://github.com/vllm-project/vllm/pull/31386) `GLM Model support for GLM-Lite`：Extended the same runtime family to the Lite checkpoint line.
- [#37386](https://github.com/vllm-project/vllm/pull/37386) `Improve tool call parsing and content normalization for glm47`：Fixed concrete parsing errors that surfaced in newer GLM-4.7 outputs.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-glm46-glm47-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm46-glm47-optimization/references/pr-history.md`
