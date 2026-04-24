# SGLang Intern-S1 Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Intern-S1.

- Status: 当前 mainline 已支持

## Key Conclusions

- Intern-S1 leans heavily on shared InternVL processor code in SGLang.
- Most regressions come from processor compatibility, parser behavior, and video-aware serving rather than the text stack alone.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/interns1.py`
- `sglang/python/sglang/srt/models/internvl.py`

## Landed PRs

- [#9381](https://github.com/sgl-project/sglang/pull/9381) `InternS1 image token updates in InternVL processor`: Aligned the shared processor with Intern-S1 image semantics.
- [#12367](https://github.com/sgl-project/sglang/pull/12367) `Fix Intern-S1 accuracy and `/generate` input_ids support`: Closed early correctness gaps.
- [#14866](https://github.com/sgl-project/sglang/pull/14866) `Add tool calling and reasoning parser support for Intern-S1`: Added parser support that cookbook usage depends on.
- [#17040](https://github.com/sgl-project/sglang/pull/17040) `Support InternS1 text_config in InternVL processor`: Improved sub-config compatibility in shared processors.

## Matching Skill

- `skills/model-optimization/sglang/sglang-intern-s1-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-intern-s1-optimization/references/pr-history.md`
