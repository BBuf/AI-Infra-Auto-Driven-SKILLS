# vLLM InternVL3.5 支持与 PR 历史

本文记录 vLLM 中与 InternVL3.5 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- InternVL3.5 is mostly a processor / encoder / video problem in vLLM.
- Video handling, native HF loading, and backend compatibility dominate the risk surface.

## 主要代码面

- `vllm/vllm/model_executor/models/internvl.py`

## 已合入 PR

- [#6514](https://github.com/vllm-project/vllm/pull/6514) `Initialize support for InternVL2 series models`：Historical base for current InternVL runtime code.
- [#18499](https://github.com/vllm-project/vllm/pull/18499) `Initialize video input support for InternVL models`：Added video processing to the family.
- [#23658](https://github.com/vllm-project/vllm/pull/23658) `Enable video support for InternVL3.5 models`：Carried video support into the 3.5 checkpoints.
- [#23742](https://github.com/vllm-project/vllm/pull/23742) `Enable native HF format InternVL support`：Removed reliance on ad hoc checkpoint rewrites.
- [#38049](https://github.com/vllm-project/vllm/pull/38049) `Add torch.compile support for InternVL vision encoder`：Modernized the encoder execution path.

## 配套 skill

- `skills/model-optimization/vllm/vllm-internvl35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-internvl35-optimization/references/pr-history.md`
