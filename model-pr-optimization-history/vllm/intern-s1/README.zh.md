# vLLM Intern-S1 支持与 PR 历史

本文记录 vLLM 中与 Intern-S1 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Intern-S1 leans heavily on shared InternVL processor code in vLLM.
- Most regressions come from processor compatibility and video-aware serving rather than the text stack alone.

## 主要代码面

- `vllm/vllm/model_executor/models/interns1.py`
- `vllm/vllm/model_executor/models/interns1_pro.py`

## 已合入 PR

- [#21628](https://github.com/vllm-project/vllm/pull/21628) `Support Intern-S1`：Initial Intern-S1 support in vLLM.
- [#21671](https://github.com/vllm-project/vllm/pull/21671) `Add video support for Intern-S1`：Extended the family beyond static images.
- [#22417](https://github.com/vllm-project/vllm/pull/22417) `Fix wrong method name in Intern-S1 image processor`：Patched a processor bug after bring-up.
- [#33636](https://github.com/vllm-project/vllm/pull/33636) `Intern-S1-Pro`：Added the Pro generation / alias path.

## 配套 skill

- `skills/model-optimization/vllm/vllm-intern-s1-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-intern-s1-optimization/references/pr-history.md`
