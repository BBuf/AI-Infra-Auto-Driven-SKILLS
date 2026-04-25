# SGLang Nemotron Super / Nano Hybrid Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Nemotron Super / Nano Hybrid.

- Status: 当前 mainline 已支持

## Key Conclusions

- Nemotron is a hybrid family: attention, Mamba, MoE, MTP, and VL all intersect.
- Because of that, graph execution, cache dtype, and quantized MoE correctness are the primary risk areas.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/nemotron_h.py`
- `sglang/python/sglang/srt/models/nemotron_h_mtp.py`
- `sglang/python/sglang/srt/models/nano_nemotron_vl.py`

## Landed PRs

- [#16172](https://github.com/sgl-project/sglang/pull/16172) `NemotronH PP support`: Opened pipeline parallelism on NemotronH.
- [#16227](https://github.com/sgl-project/sglang/pull/16227) `Add latent MoE support`: Added the hybrid latent-MoE path.
- [#19903](https://github.com/sgl-project/sglang/pull/19903) `Enable Piecewise CUDA Graph for NemotronH Hybrid Models`: Improved hybrid serving efficiency.
- [#20407](https://github.com/sgl-project/sglang/pull/20407) `Support Nemotron 3 Super NVFP4`: Added the key quantized Super checkpoint path.
- [#20575](https://github.com/sgl-project/sglang/pull/20575) `Add Nemotron 3 Super CI tests for BF16 and NVFP4`: Added regression coverage for the production checkpoint variants.

## Matching Skill

- `skills/model-optimization/sglang/sglang-nemotron-super-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-nemotron-super-optimization/references/pr-history.md`
