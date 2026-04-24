# SGLang Step3.5 / Step3-VL Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Step3.5 / Step3-VL.

- Status: 当前 mainline 已支持

## Key Conclusions

- Step3.5 is split between text/MTP and VL processor work.
- All-reduce efficiency and parser behavior are the main axes to track.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/step3p5.py`
- `sglang/python/sglang/srt/models/step3p5_mtp.py`
- `sglang/python/sglang/srt/models/step3_vl.py`
- `sglang/python/sglang/srt/models/step3_vl_10b.py`

## Landed PRs

- [#8583](https://github.com/sgl-project/sglang/pull/8583) `Support Step3V`: Initial Step3 visual model support.
- [#8699](https://github.com/sgl-project/sglang/pull/8699) `Support DP Attention for step3_vl`: Enabled multi-GPU VL serving.
- [#9695](https://github.com/sgl-project/sglang/pull/9695) `Add step3 tool parser`: Added tool-call parsing.
- [#18564](https://github.com/sgl-project/sglang/pull/18564) `Implement the standard multi-layer MTP for step3p5`: Added Step3.5 draft-model support.
- [#22773](https://github.com/sgl-project/sglang/pull/22773) `Optimize allreduce in MoE layers`: Targeted the Step3.5 MoE hot path.

## Matching Skill

- `skills/model-optimization/sglang/sglang-step35-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-step35-optimization/references/pr-history.md`
