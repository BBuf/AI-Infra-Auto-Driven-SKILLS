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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `MOSS-VL`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 公开 PR 检索结论

- 本轮在 `vllm-project/vllm` 中按 `MOSS-VL, MOSS VL, mossvl` 等关键词检索，未确认到可归入该模型支持或优化主线的公开 PR。
- 当前文档因此只保留 no-match 结论；后续若出现模型文件、processor、kernel 或 benchmark PR，应按本节验收口径补齐卡片。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
