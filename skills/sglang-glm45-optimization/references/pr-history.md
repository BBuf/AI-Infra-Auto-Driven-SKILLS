# GLM-4.5 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: `glm4.py`, `glm4_moe.py`, `glm4_moe_lite.py`, GLM-4.5 parser/docs/snippets.
- Searched PR terms: `GLM-4.5`, `GLM45`, `glm45`, `glm4_moe`, `GLM-4.5-Air`.

## Runtime Surfaces

- `python/sglang/srt/models/glm4.py`
- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `docs/basic_usage/glm45.md`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx`
- `docs_new/src/snippets/autoregressive/glm-45-deployment.jsx`

## Merged/Current-Main PRs

- [#11665](https://github.com/sgl-project/sglang/pull/11665): disable reduce scatter for GLM-4.5 path.
- [#11692](https://github.com/sgl-project/sglang/pull/11692): fix missing A2A backend initialization in GLM-4.5 MoE block.
- [#11800](https://github.com/sgl-project/sglang/pull/11800): refactor GLM-4.5/4.5V.
- [#12497](https://github.com/sgl-project/sglang/pull/12497): remove assertion for padding NVFP4 weight scales.
- [#17714](https://github.com/sgl-project/sglang/pull/17714): GLM45 tool interruption support.
- [#11017](https://github.com/sgl-project/sglang/pull/11017): GLM-4.5 documentation update.

## Open PR Radar

- [#13711](https://github.com/sgl-project/sglang/pull/13711): fused-MoE TP2 RTX Pro 6000 configs for GLM-4.5-Air and GLM-4.5V.
- [#19106](https://github.com/sgl-project/sglang/pull/19106): GLM4 MoE Lite compressed-tensors.
- [#19728](https://github.com/sgl-project/sglang/pull/19728): ROCm GLM-4.5V-FP8 startup.
- [#20917](https://github.com/sgl-project/sglang/pull/20917): `enable_thinking` for Qwen3/GLM45.
- [#23067](https://github.com/sgl-project/sglang/pull/23067): GLM45 detector `continue_final_message`.

## Cookbook Evidence

- sgl-cookbook [#92](https://github.com/sgl-project/sgl-cookbook/pull/92): GLM-4.5 AMD MI300X/MI325X/MI355X support.
- sgl-cookbook [#95](https://github.com/sgl-project/sgl-cookbook/pull/95): GLM-4.5V AMD support, VLM lane but adjacent to GLM-4.5 release.

## Validation Notes

- GLM-4.5 is the reference point for GLM MoE A2A and reduce-scatter behavior.
- Keep GLM-4.5V and GLM-OCR issues in the VLM/OCR skill unless the shared text MoE code is changed.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional merged/current-main PRs from GLM4-MoE path history: [#8224](https://github.com/sgl-project/sglang/pull/8224), [#8456](https://github.com/sgl-project/sglang/pull/8456), [#8647](https://github.com/sgl-project/sglang/pull/8647), [#8729](https://github.com/sgl-project/sglang/pull/8729), [#8804](https://github.com/sgl-project/sglang/pull/8804), [#8883](https://github.com/sgl-project/sglang/pull/8883), [#9136](https://github.com/sgl-project/sglang/pull/9136), [#9223](https://github.com/sgl-project/sglang/pull/9223), [#9264](https://github.com/sgl-project/sglang/pull/9264), [#10008](https://github.com/sgl-project/sglang/pull/10008), [#11847](https://github.com/sgl-project/sglang/pull/11847), [#12162](https://github.com/sgl-project/sglang/pull/12162), [#12572](https://github.com/sgl-project/sglang/pull/12572), [#12834](https://github.com/sgl-project/sglang/pull/12834), [#12957](https://github.com/sgl-project/sglang/pull/12957), [#14668](https://github.com/sgl-project/sglang/pull/14668).

Parser/tooling additions: [#12456](https://github.com/sgl-project/sglang/pull/12456), [#13989](https://github.com/sgl-project/sglang/pull/13989), [#15333](https://github.com/sgl-project/sglang/pull/15333), [#15753](https://github.com/sgl-project/sglang/pull/15753), [#15754](https://github.com/sgl-project/sglang/pull/15754), [#20543](https://github.com/sgl-project/sglang/pull/20543).

External evidence: official SGLang docs cover GLM-4.5/4.6/4.7 launch, EAGLE flags, parser split (`glm45` vs `glm47`), and thinking budget via `--enable-custom-logit-processor`; the LMSYS GLM-4.5 blog documents day-one SGLang support, 128k context, native function calling, and MTP.
