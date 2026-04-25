# vLLM Mixtral Quark / INT4-FP8 MoE 支持与 PR 历史

本文记录 vLLM 中与 Mixtral Quark / INT4-FP8 MoE 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 仅部分支持，或只有相邻架构已落地

## 核心结论

- vLLM has rich Mixtral MoE support, but not every Quark-branded checkpoint path is called out by name.
- The closest production evidence is the Mixtral fused-MoE, FP8, ModelOpt, and EPLB work already merged.

## 主要代码面

- `vllm/vllm/model_executor/models/mixtral.py`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`

## 已合入 PR

- [#2011](https://github.com/vllm-project/vllm/pull/2011) `Mixtral 8x7B support`：Initial Mixtral model-family support.
- [#2090](https://github.com/vllm-project/vllm/pull/2090) `Optimize Mixtral with expert parallelism`：Added early expert-parallel scaling instead of pure TP execution.
- [#2542](https://github.com/vllm-project/vllm/pull/2542) `Fused MOE for Mixtral`：Brought fused-MoE kernels into the Mixtral serving path.
- [#4527](https://github.com/vllm-project/vllm/pull/4527) `Support MoE FP8 checkpoints for Mixtral`：Added the first serious FP8 checkpoint path for Mixtral MoE.
- [#15961](https://github.com/vllm-project/vllm/pull/15961) `Support ModelOpt quantization of Mixtral model`：Extended the family to NVIDIA ModelOpt quantization flows.
- [#22842](https://github.com/vllm-project/vllm/pull/22842) `Support EPLB for Mixtral Model`：Added expert-parallel load balancing to the Mixtral family.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-mixtral-quark-int4fp8-moe-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mixtral-quark-int4fp8-moe-optimization/references/pr-history.md`
