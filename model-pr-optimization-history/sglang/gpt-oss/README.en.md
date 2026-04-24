# SGLang GPT-OSS Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for GPT-OSS.

- Status: 当前 mainline 已支持

## Key Conclusions

- GPT-OSS is a flagship MoE family in SGLang.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/gpt_oss.py`

## Landed PRs

- [#8843](https://github.com/sgl-project/sglang/pull/8843) `Support mxfp4 for GPT-OSS`: Added the headline quantized checkpoint path.
- [#8944](https://github.com/sgl-project/sglang/pull/8944) `Expert Parallelism for GPT-OSS`: Scaled GPT-OSS beyond pure tensor parallel.
- [#9043](https://github.com/sgl-project/sglang/pull/9043) `Implement Native GPT-OSS Tool Call Support`: Added native tool parser support instead of Harmony integration.
- [#9359](https://github.com/sgl-project/sglang/pull/9359) `Support DP attention with GPT-OSS`: Enabled larger topologies via DP attention.
- [#14920](https://github.com/sgl-project/sglang/pull/14920) `GPT-OSS Eagle v2 support`: Added speculative decoding support.
- [#18988](https://github.com/sgl-project/sglang/pull/18988) `Support fp8 online quantization for gpt-oss bf16`: Extended quantization coverage to online FP8.

## Matching Skill

- `skills/model-optimization/sglang/sglang-gpt-oss-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-gpt-oss-optimization/references/pr-history.md`
