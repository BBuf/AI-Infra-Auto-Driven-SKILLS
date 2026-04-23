# GLM-5/5.1 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: GLM MoE/NextN files, NSA indexer/backend files, GLM-5 docs/snippets, registered GLM-5 tests.
- Searched PR terms: `GLM-5`, `GLM5`, `GLM-5.1`, `GLM51`, `glm5`, `glm51`, `GlmMoeDsa`.

## Runtime Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/layers/attention/nsa/`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`
- `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`
- `test/registered/8-gpu-models/test_glm_51_fp8.py`
- `test/registered/gb300/test_glm5_fp8.py`
- `test/registered/gb300/test_glm5_nvfp4.py`

## Merged/Current-Main PRs

- [#18521](https://github.com/sgl-project/sglang/pull/18521): support `GlmMoeDsaForCausalLM`.
- [#18804](https://github.com/sgl-project/sglang/pull/18804): GLM-5 fused shared expert fix.
- [#18911](https://github.com/sgl-project/sglang/pull/18911): AMD GLM-5 day-0 test.
- [#20062](https://github.com/sgl-project/sglang/pull/20062): V3.2/GLM5 dense-attention threshold environment.
- [#21710](https://github.com/sgl-project/sglang/pull/21710): AMD GLM-5-FP8 performance.
- [#21773](https://github.com/sgl-project/sglang/pull/21773): AMD GLM-5-MXFP4 accuracy/performance.
- [#22179](https://github.com/sgl-project/sglang/pull/22179): GLM-5 and DeepSeek V3.2 docs improvement.
- [#22285](https://github.com/sgl-project/sglang/pull/22285): CI tests for GLM-5.
- [#22314](https://github.com/sgl-project/sglang/pull/22314): AMD GLM-5 FP8 KV dispatch.
- [#22336](https://github.com/sgl-project/sglang/pull/22336): AMD GLM-5.1-FP8 accuracy/performance.
- [#22399](https://github.com/sgl-project/sglang/pull/22399): GLM-5.1 nightly tests and Qwen3.5 model update.
- [#22543](https://github.com/sgl-project/sglang/pull/22543): GLM-5/5.1 MXFP4 checkpoint compatibility.
- [#22595](https://github.com/sgl-project/sglang/pull/22595): normalize tool-message content for GLM5.1 chat template.
- [#22712](https://github.com/sgl-project/sglang/pull/22712): NPU GLM5 guide update.
- [#22850](https://github.com/sgl-project/sglang/pull/22850): AMD NSA indexer kernel reduction.
- [#23219](https://github.com/sgl-project/sglang/pull/23219): AMD MTP for GLM-5-MXFP4; touches shared `deepseek_nextn.py`.

## Open PR Radar

- [#20275](https://github.com/sgl-project/sglang/pull/20275): thinking budget logit processor.
- [#21332](https://github.com/sgl-project/sglang/pull/21332): TRTLLM MHA B200.
- [#21529](https://github.com/sgl-project/sglang/pull/21529): ROCm MXFP4/Quark W4A4.
- [#21889](https://github.com/sgl-project/sglang/pull/21889): FP4 KV NSA TileLang.
- [#22409](https://github.com/sgl-project/sglang/pull/22409): GLM-5.1-MXFP4 MI30x/MI35x accuracy/perf.
- [#22473](https://github.com/sgl-project/sglang/pull/22473): dense MLA decode fallback.
- [#22488](https://github.com/sgl-project/sglang/pull/22488): GLM-5 256-expert JIT fused gate.
- [#22638](https://github.com/sgl-project/sglang/pull/22638): AITER decode backend auto-detect.
- [#22851](https://github.com/sgl-project/sglang/pull/22851): NSA top-k backend.
- [#22977](https://github.com/sgl-project/sglang/pull/22977): GLM5.1 tool-result template.
- [#23037](https://github.com/sgl-project/sglang/pull/23037): EAGLE CUDA graph IMA PD+DP+MTP for GLM-5.1.
- [#23346](https://github.com/sgl-project/sglang/pull/23346): decode state retract-resume for GLM-5.1.
- [#23351](https://github.com/sgl-project/sglang/pull/23351): PCG with NSA.

## Cookbook Evidence

- sgl-cookbook [#152](https://github.com/sgl-project/sgl-cookbook/pull/152): initial GLM-5 cookbook.
- sgl-cookbook [#175](https://github.com/sgl-project/sgl-cookbook/pull/175): AMD MI300X/MI325X/MI355X support for GLM-5.
- sgl-cookbook [#206](https://github.com/sgl-project/sgl-cookbook/pull/206): GLM-5 FP8 B200 optimized recipe.
- sgl-cookbook [#208](https://github.com/sgl-project/sgl-cookbook/pull/208): GLM-5 cookbook update.
- sgl-cookbook [#211](https://github.com/sgl-project/sgl-cookbook/pull/211): GLM-5 docs update.
- sgl-cookbook [#228](https://github.com/sgl-project/sgl-cookbook/pull/228): GLM-5.1 model.
- sgl-cookbook [#231](https://github.com/sgl-project/sgl-cookbook/pull/231): GLM-5.1-FP8 AMD support.
- sgl-cookbook [#233](https://github.com/sgl-project/sgl-cookbook/pull/233): GLM-5 NVFP4 B200 support.
- sgl-cookbook [#237](https://github.com/sgl-project/sgl-cookbook/pull/237): FP8 KV cache accuracy caution for GLM-5/Qwen3.5.

## Validation Notes

- GLM-5/5.1 is the most active GLM optimization lane; re-check open PR states before implementation.
- Shared NSA/MTP files can affect DeepSeek V3.2 and GLM simultaneously.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional PR/doc points: [#21405](https://github.com/sgl-project/sglang/pull/21405) for IndexCache, [#21716](https://github.com/sgl-project/sglang/pull/21716), [#22179](https://github.com/sgl-project/sglang/pull/22179), [#22238](https://github.com/sgl-project/sglang/pull/22238), [#22372](https://github.com/sgl-project/sglang/pull/22372), [#19656](https://github.com/sgl-project/sglang/pull/19656), and [#21487](https://github.com/sgl-project/sglang/pull/21487) for GB300/Blackwell CI relevance across shared quant/test paths.

Official-doc flags and caveats to keep in the playbook: GLM-5 launch by replacing the DeepSeek V3.2 model path with `zai-org/GLM-5-FP8`, `--tool-call-parser glm47`, `--reasoning-parser glm45`, NSA backends, PD disaggregation, `--nsa-prefill-cp-mode`, `--page-size 64`, and FP8 KV-cache accuracy cautions.

Public-blog evidence: the HiSparse blog lists GLM-5.1 as a supported DSA family and uses `--enable-hisparse` plus `--hisparse-config` for high-concurrency long-context decode; the ModelOpt blog is relevant for NVFP4/MXFP4/FP8 deployment framing.
