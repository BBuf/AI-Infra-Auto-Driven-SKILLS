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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Qwen-Image`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 公开 PR 检索结论

- 本轮在 `vllm-project/vllm` 中按 `Qwen-Image, Qwen Image diffusion, qwen image generation` 等关键词检索，未确认到可归入该模型支持或优化主线的公开 PR。
- 当前文档因此只保留 no-match 结论；后续若出现模型文件、processor、kernel 或 benchmark PR，应按本节验收口径补齐卡片。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
