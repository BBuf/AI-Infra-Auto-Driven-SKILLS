# SGLang Mistral Small 4 Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Mistral Small 4.

- Status: 当前 mainline 已支持

## Key Conclusions

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- Startup format mismatches, multimodal projector behavior, and Eagle / MoE integration are the main risk areas.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/mistral_large_3.py`
- `sglang/python/sglang/srt/models/mistral_large_3_eagle.py`
- `sglang/python/sglang/srt/models/mistral.py`
- `sglang/python/sglang/srt/models/ministral3.py`

## Landed PRs

- [#14213](https://github.com/sgl-project/sglang/pull/14213) `Add Mistral Large 3 support`: Historical base runtime reused by later Small 4 work.
- [#14466](https://github.com/sgl-project/sglang/pull/14466) `Add Mistral Large 3 Eagle Support`: Enabled speculative decode on the underlying family.
- [#15049](https://github.com/sgl-project/sglang/pull/15049) `Mistral Large 3 NVFP4 TRTLLM MoE support`: Added the first serious quantized MoE path.
- [#20708](https://github.com/sgl-project/sglang/pull/20708) `Add Mistral Small 4 support`: Brought Mistral Small 4 / Pixtral-style runtime into mainline.
- [#21620](https://github.com/sgl-project/sglang/pull/21620) `Mistral Small 4 fails to start due to config/weight format mismatch`: Closed a startup regression after launch.

## Matching Skill

- `skills/model-optimization/sglang/sglang-mistral-small-4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-mistral-small-4-optimization/references/pr-history.md`
