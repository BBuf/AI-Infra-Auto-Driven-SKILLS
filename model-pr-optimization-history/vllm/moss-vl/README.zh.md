# vLLM MOSS-VL 支持与 PR 历史

本文记录 vLLM 中与 MOSS-VL 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 不支持

## 核心结论

- No `moss` or `moss_vl` model module is present in current vLLM mainline.
- Keep the family explicitly marked unsupported until that changes.

## 主要代码面

- `vllm/vllm/model_executor/models/registry.py`

## 已合入 PR

- 当前 dossier 中没有已记录的 merged PR。

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-moss-vl-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-moss-vl-optimization/references/pr-history.md`
