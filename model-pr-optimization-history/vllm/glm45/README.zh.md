# vLLM GLM-4.5 / 4.5V 支持与 PR 历史

本文记录 vLLM 中与 GLM-4.5 / 4.5V 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- The GLM-4.5 lane is where vLLM reorganized the GLM family around text, MoE, and vision variants.
- Most regressions are in MoE gate behavior, tie-word-embedding policy, and vendor-specific fused MoE tuning.

## 主要代码面

- `vllm/vllm/model_executor/models/glm4.py`
- `vllm/vllm/model_executor/models/glm4_moe.py`
- `vllm/vllm/model_executor/models/glm4v.py`

## 已合入 PR

- [#22171](https://github.com/vllm-project/vllm/pull/22171) `Modify the organization of GLM series`：Reworked the family layout so 4.5-era models reused a cleaner GLM structure.
- [#22460](https://github.com/vllm-project/vllm/pull/22460) `not tie_word_embeddings for glm-4.5 and glm-4.5v`：Aligned the loader with the real 4.5 checkpoint contract instead of forcing tied embeddings.
- [#22832](https://github.com/vllm-project/vllm/pull/22832) `Modify the gate implementation of glm4_moe`：Changed the GLM4.5 MoE gating path used by text and VL variants.
- [#23695](https://github.com/vllm-project/vllm/pull/23695) `Add triton fused moe config for GLM-4.5-Air-FP8 on B200`：Added a production kernel-tuning lane for the 4.5 Air FP8 deployment path.
- [#24589](https://github.com/vllm-project/vllm/pull/24589) `Add documentation for GLM-4.5 series tool-calling and reasoning parser`：Codified the parser choices needed to serve 4.5 reasoning / tool checkpoints correctly.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-glm45-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm45-optimization/references/pr-history.md`
