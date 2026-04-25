# vLLM GLM VLM / OCR 支持与 PR 历史

本文记录 vLLM 中与 GLM VLM / OCR 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- GLM visual/OCR support in vLLM spans classic GLM4V, newer GLM4.1V, and GLM-OCR-specific processor paths.
- The main failures are processor-schema drift, MRoPE/video position handling, and OCR-specific weight or patch-merger mismatches.

## 主要代码面

- `vllm/vllm/model_executor/models/glm4v.py`
- `vllm/vllm/model_executor/models/glm4_1v.py`
- `vllm/vllm/model_executor/models/glm_ocr.py`

## 已合入 PR

- [#9242](https://github.com/vllm-project/vllm/pull/9242) `Add GLM-4v support`：Landed the original GLM4V multimodal model path.
- [#19331](https://github.com/vllm-project/vllm/pull/19331) `Add GLM4.1V model`：Extended the family to the newer GLM4.1V checkpoint layout and vision stack.
- [#27860](https://github.com/vllm-project/vllm/pull/27860) `Fix broken MRoPE for GLM-4.1V/GLM-4.5V`：Closed a positional-embedding bug with large practical accuracy impact on vision inputs.
- [#33005](https://github.com/vllm-project/vllm/pull/33005) `GLM-OCR with MTP Support`：Added OCR-specific draft / MTP support rather than text-only OCR loading.
- [#33350](https://github.com/vllm-project/vllm/pull/33350) `Fix broken GLM-OCR initialization`：Fixed startup failures in the GLM-OCR path after the first bring-up.
- [#37962](https://github.com/vllm-project/vllm/pull/37962) `GLM OCR Patch Merger context_dim`：Updated the patch-merger contract for newer OCR checkpoints.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-glm-vlm-ocr-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-glm-vlm-ocr-optimization/references/pr-history.md`
