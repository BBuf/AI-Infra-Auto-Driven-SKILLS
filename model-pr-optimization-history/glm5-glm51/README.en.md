# SGLang GLM-5/5.1 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: GLM-5, GLM-5.1, GlmMoeDsa, NSA/DSA, FP8/MXFP4/NVFP4, NextN/MTP, tool templates, AMD, GB300, and NPU.

## Summary

GLM-5/5.1 is the most active GLM optimization lane. It ties together GLM MoE, DSA/NSA indexer work, DeepSeek-style NextN/MTP, quantized checkpoints, dense-attention threshold behavior, EAGLE/PCG, tool-result templates, and AMD/Blackwell/NPU validation.

## Code Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/layers/attention/nsa/`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`
- `test/registered/8-gpu-models/test_glm_51_fp8.py`
- `test/registered/gb300/test_glm5_fp8.py`
- `test/registered/gb300/test_glm5_nvfp4.py`

## Merged PRs

Key merged PRs: `#18521`, `#18804`, `#18911`, `#20062`, `#21710`, `#21773`, `#22179`, `#22285`, `#22314`, `#22336`, `#22399`, `#22543`, `#22595`, `#22712`, `#22850`, and `#23219`.

## Open Radar

Track `#20275`, `#21332`, `#21529`, `#21889`, `#22409`, `#22473`, `#22488`, `#22638`, `#22851`, `#22977`, `#23037`, `#23346`, and `#23351`.

## Cookbook Evidence

Track `sgl-cookbook#152`, `#175`, `#206`, `#208`, `#211`, `#228`, `#231`, `#233`, and `#237`.

## Next Work

Label shared NSA/MTP impacts on DeepSeek V3.2, keep separate compatibility tests for MXFP4/NVFP4/FP8 checkpoints, and validate tool-template behavior with OpenAI chat completion tests.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. This pass adds `#21405` for IndexCache plus `#21716`, `#22238`, `#22372`, `#19656`, and `#21487` for docs/platform/GB300/DSA relevance. Official DeepSeek V3.2/GLM-5 docs require `glm47` tool parser, `glm45` reasoning parser, NSA backends, PD disaggregation, context parallel modes, `--page-size 64`, and FP8 KV-cache cautions.

Public evidence adds the HiSparse and ModelOpt blogs: GLM-5.1 is a DSA/HiSparse target, so high-concurrency long-context work needs separate `--enable-hisparse` and `--hisparse-config` validation; FP8/NVFP4/MXFP4 notes should name the ModelOpt/Quark checkpoint source.
