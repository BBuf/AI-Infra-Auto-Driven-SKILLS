# vLLM Qwen-Image 支持与 PR 历史

本文记录 vLLM 中与 Qwen-Image 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 不支持

## 核心结论

- vLLM current mainline does not ship a Qwen-Image diffusion model runtime.
- The family should stay marked unsupported rather than being backfilled from Qwen text/VL support.

## 主要代码面

- `vllm/vllm/model_executor/models/registry.py`

## 已合入 PR

- 当前 dossier 中没有已记录的 merged PR。

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen-image-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen-image-optimization/references/pr-history.md`
