# SGLang Qwen3-Next 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 Qwen3-Next、Qwen3-Next MTP、Qwen3-Coder-Next 共享的 hybrid Gated Delta Network/Mamba/RadixLinearAttention 路径。

结论：Qwen3-Next 是独立优化问题。它不是普通 Qwen3 MoE，而是 hybrid GDN/Mamba state、RadixLinearAttention、MTP、quantized loading、CPU offload、GDN projection fusion、FlashInfer all-reduce、AMD/NPU/Blackwell 后端共同组成的运行面。

## 代码面

- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_next_mtp.py`
- `python/sglang/srt/configs/qwen3_next.py`
- `test/registered/4-gpu-models/test_qwen3_next_models.py`
- `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py`
- `test/registered/models/test_qwen3_next_models_fp4.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx`

## 已合入主线 PR

- `#10912`：Qwen3-Next 等 hybrid model 的 PD 支持。
- `#11487`、`#11969`、`#12508`、`#12525`、`#13081`、`#13708`、`#14607`、`#14855`、`#16164`、`#16863`、`#17016`、`#17373`、`#17570`、`#17613`、`#17627`、`#17660`：早期 Qwen3-Next/hybrid runtime 演进。
- `#18224`：ModelOpt Qwen3-Next-Coder NVFP4。
- `#18355`：AMD Qwen3-Coder-Next 支持，复用 Next 架构。
- `#18489`：Qwen3.5 支持期间同步整理 Next-adjacent 类。
- `#19220`：Qwen3.5/Qwen3-Next PCG 修复。
- `#19321`：融合 Qwen3-Next GDN `qkvz_proj` 与 `ba_proj`。
- `#19767`：MTP/EPLB 相关变化。
- `#21019`：融合 GDN projection 中 split/reshape/cat。
- `#21313`、`#21496`、`#21662`：Qwen3-Next weight loading 与 FP8 loader 修复/回滚。
- `#22073`：Qwen3-ASR 支持，触及部分 Next-family hybrid plumbing。
- `#22458`：修复 Qwen3-Next MTP NCCL AllGather hang。
- `#22664`：Qwen3-Next 自动启用 FlashInfer all-reduce。

## Open PR 雷达

- `#10657`：Qwen3-Next EAGLE3。
- `#12892`：spec 解码避免 SSM/conv state copy。
- `#13964`：Qwen3-Next kernel perf。
- `#14502`：Qwen3-Next PCG。
- `#16488`：TBO 支持。
- `#17981`、`#17983`：Blackwell decode/MTP、prefill/GDN Gluon/cumsum。
- `#19812`：Qwen3.5/Qwen3-Next MTP + EPLB 兼容。
- `#20397`：NPU MTP。
- `#23474`：hybrid linear-attn 模型 CPU offload。

## sgl-cookbook / 文档

- `sgl-cookbook#100`：AMD MI300X/MI355X。
- `sgl-cookbook#123`：AMD MI325X。
- `sgl-cookbook#143`：Qwen3-Coder-Next 文档，复用 Next 部署开关。

## 下一步优化建议

1. 先补 Mamba/radix-cache 与 MTP 的组合测试。
2. GDN fusion 继续做 logits 对齐 + kernel profile 双验证。
3. CPU offload、PCG、EAGLE3 和 Blackwell kernel 应拆成互不混淆的实验 lane。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。本轮补入官方文档提到的初始支持 `#10233`，以及 `#10322`、`#10379`、`#10392`、`#10466`、`#10622`、`#15631`、`#17373`、`#17627`、`#18917`、`#19434`、`#22358`。

新增 open radar：`#21684`、`#21698`、`#22876`、`#23075`、`#23273`。官方文档要求保留 Mamba cache/SSM/scheduler 参数：`--max-mamba-cache-size`、`--mamba-ssm-dtype`、`--mamba-full-memory-ratio`、`--mamba-scheduler-strategy extra_buffer`、`--page-size 64`，以及 NEXTN/EAGLE、`tool-call-parser qwen`、`reasoning-parser qwen3`。
