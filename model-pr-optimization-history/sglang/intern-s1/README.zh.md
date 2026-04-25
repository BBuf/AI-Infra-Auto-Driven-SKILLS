# SGLang Intern-S1 支持与 PR 历史

本文记录 SGLang 中与 Intern-S1 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Intern-S1 leans heavily on shared InternVL processor code in SGLang.
- Most regressions come from processor compatibility, parser behavior, and video-aware serving rather than the text stack alone.

## 主要代码面

- `sglang/python/sglang/srt/models/interns1.py`
- `sglang/python/sglang/srt/models/internvl.py`

## 已合入 PR

- [#9381](https://github.com/sgl-project/sglang/pull/9381) `InternS1 image token updates in InternVL processor`：Aligned the shared processor with Intern-S1 image semantics.
- [#12367](https://github.com/sgl-project/sglang/pull/12367) `Fix Intern-S1 accuracy and `/generate` input_ids support`：Closed early correctness gaps.
- [#14866](https://github.com/sgl-project/sglang/pull/14866) `Add tool calling and reasoning parser support for Intern-S1`：Added parser support that cookbook usage depends on.
- [#17040](https://github.com/sgl-project/sglang/pull/17040) `Support InternS1 text_config in InternVL processor`：Improved sub-config compatibility in shared processors.

## 配套 skill

- `skills/model-optimization/sglang/sglang-intern-s1-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-intern-s1-optimization/references/pr-history.md`
