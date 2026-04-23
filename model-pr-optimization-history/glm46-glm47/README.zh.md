# SGLang GLM-4.6/4.7 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 GLM-4.6、GLM-4.7、GLM-4.7-Flash、GLM4-MoE-Lite、GLM-4.7 parser、NVFP4/MTP、shared expert fusion、dual-stream MoE。

结论：GLM-4.6/4.7 的优化核心是 MoE fusion + parser/loading 正确性。GLM-4.6 引入 shared expert fusion 和 MoE GEMM overlap；GLM-4.7 引入 parser、NVFP4/MTP、Flash/lite import format 和更多 AMD/NPU 路径。

## 代码面

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx`
- `test/registered/8-gpu-models/test_glm_46.py`

## 已合入主线 PR

- `#13786`：GLM MoE GEMM 双流 overlap。
- `#13873`：GLM-4.6 shared expert fusion。
- `#14585`：GLM-4.6V accuracy/launch bug，触及 GLM4 MoE adjacent。
- `#15333`：GLM-4.7 tool parser 和文档。
- `#15520`：model-gateway GLM-4.7 parser。
- `#15753`、`#15754`：复杂 schema、空函数名 parser 修复。
- `#17166`：GLM-4.7 NVFP4 + MTP。
- `#17247`：GLM-4.7-Flash。
- `#21403`：AMD GLM-4.7-FP8 RMSNorm + FP8 per-token quant fusion。
- `#21534`：AMD GLM-4.7-FP8 accuracy CI。
- `#21851`：GLM-4.7/Flash loading/import format。
- `#22509`：NPU GLM-4.7-Flash。
- `#22720`：gfx95 quant format detection for GLM4.7-Flash。
- `#22823`：GLM NextN draft 保留 `quant_config`。

## Open PR 雷达

- `#11951`：GLM4.6 tool-call streaming parser。
- `#17869`：NPU GLM-4.7-Flash。
- `#18930`：AMD GLM-4.7 MTP tests。
- `#19040`：GLM4-MoE-Lite config 和 GLM-4.7-Flash A2A MoE。
- `#19106`：GLM4 MoE Lite compressed-tensors。
- `#22315`：GLM-4.7-FP8 EAGLE accept length / draft quant_config。
- `#22801`：NPU dual-stream/DeepEP GLM-4.7-Flash。

## sgl-cookbook / 文档

- `sgl-cookbook#93`：GLM-4.6 AMD MI300X/MI325X/MI355X。
- `sgl-cookbook#94`：GLM-4.7 AMD MI300X/MI325X/MI355X。
- `sgl-cookbook#109`：GLM-4.7-Flash 文档与 benchmark。

## 下一步优化建议

1. GLM-4.7 parser 应补 streaming + complex schema + empty name 三类测试。
2. GLM-4.7-Flash loading 需要量化 checkpoint 与 draft/MTP 的 `quant_config` 传播测试。
3. shared expert fusion 改动必须同时给 logits 对齐和 perf profile。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。本轮补入 `#13989`、`#14668`、`#19246`、`#20543`、`#21135`、`#21660`。LMSYS/Novita GLM4-MoE 生产优化博客把这条 lane 和 shared-expert fusion、QKNorm fusion、async transfer、suffix decoding、EAGLE/spec decoding 绑定在一起，后续验证时要把 suffix decoding 与 shared-expert fusion 分开。
