# vLLM Hunyuan 3 Preview 支持与 PR 历史

本文记录 vLLM 中与 Hunyuan 3 Preview 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 仅部分支持，或只有相邻架构已落地

## 核心结论

- vLLM does not currently expose a dedicated Hunyuan 3 Preview model alias.
- The closest landed evidence is the Hunyuan dense, Hunyuan OCR, and HunyuanVL / Eagle work already in tree.

## 主要代码面

- `vllm/vllm/model_executor/models/hunyuan_v1.py`
- `vllm/vllm/model_executor/models/hunyuan_vision.py`

## 已合入 PR

- [#21368](https://github.com/vllm-project/vllm/pull/21368) `Add Hunyuan V1 Dense Model support`：Brought the dense Hunyuan line into vLLM mainline.
- [#29327](https://github.com/vllm-project/vllm/pull/29327) `Add HunyuanOCR support`：Extended the family to OCR workloads instead of text-only generation.
- [#33035](https://github.com/vllm-project/vllm/pull/33035) `Eagle3 support for HunyuanVL & Hunyuan`：Added speculative decoding support on top of the Hunyuan family.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-hunyuan3-preview-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-hunyuan3-preview-optimization/references/pr-history.md`
