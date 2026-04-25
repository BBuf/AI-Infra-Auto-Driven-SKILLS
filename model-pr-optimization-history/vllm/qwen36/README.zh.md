# vLLM Qwen3.6 支持与 PR 历史

本文记录 vLLM 中与 Qwen3.6 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 不支持

## 核心结论

- Qwen3.6 should not be treated as automatically covered just because Qwen3 / Qwen3.5 are supported.
- At the current checked commit, there is no dedicated `Qwen3.6` model module or registry alias.

## 主要代码面

- `vllm/vllm/model_executor/models/registry.py`

## 已合入 PR

- 当前 dossier 中没有已记录的 merged PR。

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen36-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen36-optimization/references/pr-history.md`
