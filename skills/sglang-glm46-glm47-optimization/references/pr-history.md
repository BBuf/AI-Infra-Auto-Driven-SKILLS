# GLM-4.6/4.7 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_nextn.py`, GLM-4.7 detector, GLM-4.6/4.7 docs/snippets.
- Searched PR terms: `GLM-4.6`, `GLM46`, `glm46`, `GLM-4.7`, `GLM47`, `GLM-4.7-Flash`, `glm47`.

## Runtime Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx`
- `docs_new/src/snippets/autoregressive/glm-46-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-47-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-47-flash-deployment.jsx`
- `test/registered/8-gpu-models/test_glm_46.py`

## Merged/Current-Main PRs

- [#13786](https://github.com/sgl-project/sglang/pull/13786): overlap GLM MoE GEMMs in two streams.
- [#13873](https://github.com/sgl-project/sglang/pull/13873): GLM-4.6 shared-expert fusion.
- [#14585](https://github.com/sgl-project/sglang/pull/14585): GLM-4.6V accuracy/launch bug; adjacent GLM4 MoE work.
- [#15333](https://github.com/sgl-project/sglang/pull/15333): GLM-4.7 tool parser and docs.
- [#15520](https://github.com/sgl-project/sglang/pull/15520): model-gateway tool parser for GLM-4.7.
- [#15753](https://github.com/sgl-project/sglang/pull/15753): GLM detector complex schema.
- [#15754](https://github.com/sgl-project/sglang/pull/15754): GLM detector empty function name.
- [#17166](https://github.com/sgl-project/sglang/pull/17166): GLM-4.7 NVFP4 and MTP.
- [#17247](https://github.com/sgl-project/sglang/pull/17247): GLM-4.7-Flash model.
- [#21403](https://github.com/sgl-project/sglang/pull/21403): AMD RMSNorm + FP8 per-token quant fusion for GLM-4.7-FP8.
- [#21534](https://github.com/sgl-project/sglang/pull/21534): AMD GLM-4.7-FP8 accuracy CI.
- [#21851](https://github.com/sgl-project/sglang/pull/21851): GLM-4.7/Flash loading and import format.
- [#22509](https://github.com/sgl-project/sglang/pull/22509): NPU GLM-4.7-Flash fix.
- [#22720](https://github.com/sgl-project/sglang/pull/22720): detect gfx95 quant format for GLM4.7-Flash.
- [#22823](https://github.com/sgl-project/sglang/pull/22823): preserve `quant_config` for GLM NextN draft.

## Open PR Radar

- [#11951](https://github.com/sgl-project/sglang/pull/11951): GLM4.6 tool-call streaming parser.
- [#17869](https://github.com/sgl-project/sglang/pull/17869): NPU GLM-4.7-Flash.
- [#18930](https://github.com/sgl-project/sglang/pull/18930): AMD MTP tests for GLM-4.7.
- [#19040](https://github.com/sgl-project/sglang/pull/19040): GLM4-MoE-Lite config and A2A MoE for GLM-4.7-Flash.
- [#19106](https://github.com/sgl-project/sglang/pull/19106): GLM4 MoE Lite compressed-tensors.
- [#22315](https://github.com/sgl-project/sglang/pull/22315): GLM-4.7-FP8 EAGLE accept length due draft quant config.
- [#22801](https://github.com/sgl-project/sglang/pull/22801): NPU dual-stream and DeepEP GLM-4.7-Flash.

## Cookbook Evidence

- sgl-cookbook [#93](https://github.com/sgl-project/sgl-cookbook/pull/93): GLM-4.6 AMD MI300X/MI325X/MI355X support.
- sgl-cookbook [#94](https://github.com/sgl-project/sgl-cookbook/pull/94): GLM-4.7 AMD MI300X/MI325X/MI355X support.
- sgl-cookbook [#109](https://github.com/sgl-project/sgl-cookbook/pull/109): GLM-4.7-Flash docs and benchmark.

## Validation Notes

- GLM-4.7 parser changes need complex schema and streaming coverage.
- Flash/lite loading changes must check quant config propagation into draft/MTP layers.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional merged/current-main PRs: [#13989](https://github.com/sgl-project/sglang/pull/13989), [#14668](https://github.com/sgl-project/sglang/pull/14668), [#19246](https://github.com/sgl-project/sglang/pull/19246), [#20543](https://github.com/sgl-project/sglang/pull/20543), [#21135](https://github.com/sgl-project/sglang/pull/21135), [#21660](https://github.com/sgl-project/sglang/pull/21660).

Production-blog evidence: the LMSYS/Novita GLM4-MoE optimization blog ties this lane to shared-expert fusion ([#13873](https://github.com/sgl-project/sglang/pull/13873)), QKNorm fusion, async transfer, suffix decoding, and speculative/EAGLE deployment configs. Treat suffix decoding and shared-expert fusion as separate validation toggles.
