# SGLang Qwen3.5 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 Qwen3.5 dense/MoE、Qwen3.5 MTP、FP8、NVFP4、MXFP4、PP/EPLB、GDN fusion、AMD/NPU/Blackwell 部署。

结论：Qwen3.5 是目前 Qwen 文本模型里最活跃的工程 lane 之一。它的主要风险集中在 weight loading、PP split、tied embedding、MTP + EPLB、GDN projection fusion、shared expert fusion、quantization guard，以及 cookbook 中 H200/B200/AMD/GB300 命令同步。

## 代码面

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen3_5_mtp.py`
- `python/sglang/srt/configs/qwen3_5.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`
- `test/registered/8-gpu-models/test_qwen35.py`
- `test/registered/gb300/test_qwen35_fp8.py`
- `test/registered/gb300/test_qwen35_nvfp4.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`

## 已合入主线 PR

- `#18489`：初始 Qwen3.5 支持。
- `#18538`：重构 Qwen3.5 MTP 类。
- `#18937`：NVFP4 checkpoint。
- `#19070`：dense Qwen3.5 TP>1 精度 bug。
- `#19220`：PCG 修复。
- `#19411`：Qwen3.5-27B repeat bug。
- `#19670`：PP 支持。
- `#19767`：MTP + EPLB。
- `#19889`：TRTLLM all-reduce fusion。
- `#19961`：linear attention `a_log` 改 FP32。
- `#20386`、`#21019`、`#22312`：GDN projection 和非连续 B/A tensor 支持。
- `#20736`：AMD shared expert + router expert fusion。
- `#21070`：PP layer splitting 修复。
- `#21234`：AMD MXFP4 Qwen3.5-397B。
- `#21347`：PP tied embedding weight loading。
- `#21448`：MoE loading 和 Mamba cache sharding in PP。
- `#21669`：AMD Qwen3.5-397B FP8 nightly perf。
- `#22913`：拆分 Qwen3.5 测试和分区。

## Open PR 雷达

- `#19484`：CPU Qwen3.5。
- `#20074`：DeltaNet input projection fusion。
- `#20565`：H200 TP4 fused MoE Triton config。
- `#21668`：Intel XPU。
- `#22618`：compressed-tensors NVFP4 linear-attention guard。
- `#23146`：AMD EAGLE speculative FP8/MXFP4 AITER。
- `#23147`：random MTP init。
- `#23331`：adaptive spec 与 hybrid GDN 冲突。
- `#23462`：NPU MTP。

## sgl-cookbook / 文档

- `sgl-cookbook#164`、`#168`、`#169`：Qwen3.5、FP8/NVFP4、B200 configs。
- `sgl-cookbook#177`、`#214`：H200 FP8/MTP。
- `sgl-cookbook#179`：AMD MI300X/MI325X/MI355X。
- `sgl-cookbook#207`、`#230`、`#237`：B200 FlashInfer all-reduce、FP4/NVFP4 generator、FP8 KV caution。

## 下一步优化建议

1. 把 Qwen3.5 的 PP、MTP、EPLB、quant 四维组合写成固定回归矩阵。
2. open 的 DeltaNet/GDN fusion 与 AMD EAGLE 应先用 BF16/quant 双基线做精度验证。
3. Cookbook 的 FP8 KV caution 要和 server 默认值同步，不要只改一边。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。本轮补入 `#18926`、`#19391`、`#20864`、`#21487`、`#21849`、`#22145`、`#22240`、`#22431`、`#22493`、`#22908`、`#22948`。

新增 open radar：`#19585`、`#19781`、`#19956`、`#19974`、`#20029`、`#20255`、`#20370`、`#20448`、`#20667`、`#20789`、`#20918`、`#21185`、`#21217`、`#21464`、`#22027`、`#22325`、`#22502`、`#22867`、`#22885`、`#23062`、`#23468`、`#23471`。外部资料补充了 SGLang Qwen3.5 官方文档与 AMD Day-0 文章，确认 AMD/Triton/GDN/shared-expert/vision kernel 路线。
