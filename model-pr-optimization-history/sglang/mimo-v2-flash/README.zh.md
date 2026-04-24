# SGLang MiMo-V2-Flash 支持与 PR 历史

本文记录 SGLang 中与 MiMo-V2-Flash 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- MiMo-V2-Flash is primarily a throughput-oriented MoE serving family.
- All-reduce fusion, overlap, and reasoning behavior matter more than generic text-only loader work.

## 主要代码面

- `sglang/python/sglang/srt/models/mimo_v2_flash.py`
- `sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py`

## 已合入 PR

- [#15207](https://github.com/sgl-project/sglang/pull/15207) `MiMo-V2-Flash day0 support`：Initial MiMo-V2-Flash landing.
- [#15464](https://github.com/sgl-project/sglang/pull/15464) `Optimize MiMo-V2-Flash by flashinfer fused allreduce`：Targeted decode-side communication cost.
- [#15488](https://github.com/sgl-project/sglang/pull/15488) `Respect `--swa-full-tokens-ratio``：Fixed a concrete runtime flag integration bug.
- [#17634](https://github.com/sgl-project/sglang/pull/17634) `Support two batch overlap`：Added overlap / throughput optimization.
- [#21414](https://github.com/sgl-project/sglang/pull/21414) `Add mimo reasoning parser`：Completed the parser path for thinking outputs.

## 配套 skill

- `skills/model-optimization/sglang/sglang-mimo-v2-flash-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-mimo-v2-flash-optimization/references/pr-history.md`
