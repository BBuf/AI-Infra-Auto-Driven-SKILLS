# SGLang MiMo-V2-Flash Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for MiMo-V2-Flash.

- Status: 当前 mainline 已支持

## Key Conclusions

- MiMo-V2-Flash is primarily a throughput-oriented MoE serving family.
- All-reduce fusion, overlap, and reasoning behavior matter more than generic text-only loader work.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/mimo_v2_flash.py`
- `sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py`

## Landed PRs

- [#15207](https://github.com/sgl-project/sglang/pull/15207) `MiMo-V2-Flash day0 support`: Initial MiMo-V2-Flash landing.
- [#15464](https://github.com/sgl-project/sglang/pull/15464) `Optimize MiMo-V2-Flash by flashinfer fused allreduce`: Targeted decode-side communication cost.
- [#15488](https://github.com/sgl-project/sglang/pull/15488) `Respect `--swa-full-tokens-ratio``: Fixed a concrete runtime flag integration bug.
- [#17634](https://github.com/sgl-project/sglang/pull/17634) `Support two batch overlap`: Added overlap / throughput optimization.
- [#21414](https://github.com/sgl-project/sglang/pull/21414) `Add mimo reasoning parser`: Completed the parser path for thinking outputs.

## Matching Skill

- `skills/model-optimization/sglang/sglang-mimo-v2-flash-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-mimo-v2-flash-optimization/references/pr-history.md`
