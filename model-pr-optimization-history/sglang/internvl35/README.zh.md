# SGLang InternVL3.5 支持与 PR 历史

本文记录 SGLang 中与 InternVL3.5 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- InternVL3.5 is mostly a processor / encoder / video problem in SGLang.
- Video handling, data-parallel vision execution, and backend compatibility dominate the risk surface.

## 主要代码面

- `sglang/python/sglang/srt/models/internvl.py`

## 已合入 PR

- [#5350](https://github.com/sgl-project/sglang/pull/5350) `Support InternVL3`：Initial InternVL family support that later carried 3.5.
- [#13640](https://github.com/sgl-project/sglang/pull/13640) `Support Piecewise CUDA Graph for InternVL`：Added graph capture support on the encoder path.
- [#13925](https://github.com/sgl-project/sglang/pull/13925) `Support InternVL Vision Encoder Data Parallelism`：Opened the multi-GPU ViT path.
- [#15942](https://github.com/sgl-project/sglang/pull/15942) `Support Video for InternVL3_5`：Extended support to 3.5 video use cases.
- [#19127](https://github.com/sgl-project/sglang/pull/19127) `Support processor and embedding inputs for InternVL`：Hardened processor / embed input interoperability.

## 配套 skill

- `skills/model-optimization/sglang/sglang-internvl35-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-internvl35-optimization/references/pr-history.md`
