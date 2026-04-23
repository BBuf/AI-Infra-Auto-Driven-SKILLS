# SGLang GLM-5/5.1 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 GLM-5、GLM-5.1、GlmMoeDsa、NSA/DSA、FP8/MXFP4/NVFP4、NextN/MTP、tool template、AMD/GB300/NPU。

结论：GLM-5/5.1 是当前最活跃的 GLM 优化 lane。它同时连到 GLM MoE、DSA/NSA indexer、DeepSeek-style NextN/MTP、FP8/MXFP4/NVFP4、dense-attention threshold、EAGLE/PCG、tool-result template 和 AMD/Blackwell/NPU validation。

## 代码面

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

## 已合入主线 PR

- `#18521`：支持 `GlmMoeDsaForCausalLM`。
- `#18804`：GLM-5 fused shared expert 修复。
- `#18911`：AMD GLM-5 day-0 test。
- `#20062`：V3.2/GLM5 dense attention threshold env。
- `#21710`：AMD GLM-5-FP8 perf。
- `#21773`：AMD GLM-5-MXFP4 acc/perf。
- `#22179`：GLM-5 / DeepSeek V3.2 docs。
- `#22285`：GLM-5 CI tests。
- `#22314`：AMD GLM-5 FP8 KV dispatch。
- `#22336`：AMD GLM-5.1-FP8 acc/perf。
- `#22399`：GLM-5.1 nightly tests 和 Qwen3.5 model update。
- `#22543`：GLM-5/5.1 MXFP4 checkpoint compatibility。
- `#22595`：GLM5.1 chat template tool message content normalization。
- `#22712`：NPU GLM5 guide。
- `#22850`：AMD NSA indexer kernel reduction。
- `#23219`：AMD GLM-5-MXFP4 MTP，触及共享 `deepseek_nextn.py`。

## Open PR 雷达

- `#20275`：thinking budget logit processor。
- `#21332`：B200 TRTLLM MHA。
- `#21529`：ROCm MXFP4 / Quark W4A4。
- `#21889`：FP4 KV NSA TileLang。
- `#22409`：GLM-5.1-MXFP4 MI30x/MI35x。
- `#22473`：dense MLA decode fallback。
- `#22488`：GLM-5 256-expert JIT fused gate。
- `#22638`：AITER decode backend auto-detect。
- `#22851`：NSA top-k backend。
- `#22977`：GLM5.1 tool-result template。
- `#23037`：GLM-5.1 EAGLE CUDA graph IMA PD+DP+MTP。
- `#23346`：GLM-5.1 decode state retract-resume。
- `#23351`：NSA PCG。

## sgl-cookbook / 文档

- `sgl-cookbook#152`、`#175`、`#206`、`#208`、`#211`：GLM-5 cookbook、AMD、B200 FP8、文档更新。
- `sgl-cookbook#228`、`#231`：GLM-5.1 和 GLM-5.1-FP8 AMD。
- `sgl-cookbook#233`：GLM-5 NVFP4 B200。
- `sgl-cookbook#237`：GLM-5/Qwen3.5 FP8 KV caution。

## 下一步优化建议

1. GLM-5/5.1 改动先标明是否会影响 DeepSeek V3.2 的 shared NSA/MTP 文件。
2. MXFP4/NVFP4/FP8 每条都要配独立 checkpoint 兼容测试。
3. tool template 与 tool-result normalization 需要 OpenAI chat completion 测试，不应只跑模型启动。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。本轮补入 `#21405` IndexCache，以及 `#21716`、`#22238`、`#22372`、`#19656`、`#21487` 等 docs/platform/GB300/DSA 相关项。官方 DeepSeek V3.2/GLM-5 文档要求保留 `glm47` tool parser、`glm45` reasoning parser、NSA backend、PD disaggregation、context parallel、`--page-size 64` 和 FP8 KV caution。

外部资料补充 HiSparse 与 ModelOpt 博客：GLM-5.1 属于 DSA/HiSparse 支持对象，长上下文高并发优化要单独验证 `--enable-hisparse` 与 `--hisparse-config`；FP8/NVFP4/MXFP4 说明要和 ModelOpt/Quark checkpoint 来源绑定。
