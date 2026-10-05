# TokenSpeed Qwen3.5 模型 PR 优化历史

## 2026-08-23 源码 head 刷新

已复核 TokenSpeed 上游 main：
`lightseekorg/tokenspeed@2706143a8669d50a8f56466b9d340b86922b8f2d`。
已读完上一 head `d73bf0454422092f306d5575e803a08fd35ac41c`
之后的 2-commit 完整增量；两项都只是 Kimi K3 文档变化，因此不新增 Qwen3.5 卡片。

结果：Qwen3.5 新增原生优化 DFlash，修正 mixed-precision GDN/MoE FP8 权重加载，并补齐 topology-safe 的多机 collective 与 staging 行为。

## 2026-06-27 PR 补漏复核

已按 TokenSpeed 上游 `HEAD@d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8` 复核。这个文件按 SGLang/vLLM 同样的格式记录模型相关 PR、已读 diff、实现文件、代码摘录与验证风险。

本轮筛选规则：GitHub merged PR、标题/文件路径命中 `Qwen3.5`、`qwen3_5`、`Qwen3Moe`、`VLM`、`PD`、`moe`、`activation`、`rotary`、`flashinfer_trtllm` 等；过滤纯格式化和无模型路径的基础设施 PR。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/qwen3_5_config.py` | [#485](https://github.com/lightseekorg/tokenspeed/pull/485) |
| `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` | [#485](https://github.com/lightseekorg/tokenspeed/pull/485) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` | [#1096](https://github.com/lightseekorg/tokenspeed/pull/1096) |
| `python/tokenspeed/runtime/models/qwen3_5.py` | [#196](https://github.com/lightseekorg/tokenspeed/pull/196), [#198](https://github.com/lightseekorg/tokenspeed/pull/198), [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#228](https://github.com/lightseekorg/tokenspeed/pull/228), [#354](https://github.com/lightseekorg/tokenspeed/pull/354), [#429](https://github.com/lightseekorg/tokenspeed/pull/429), [#485](https://github.com/lightseekorg/tokenspeed/pull/485), [#510](https://github.com/lightseekorg/tokenspeed/pull/510), [#766](https://github.com/lightseekorg/tokenspeed/pull/766), [#828](https://github.com/lightseekorg/tokenspeed/pull/828), [#928](https://github.com/lightseekorg/tokenspeed/pull/928), [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `python/tokenspeed/runtime/models/qwen3_5_moe.py` | [#235](https://github.com/lightseekorg/tokenspeed/pull/235), [#433](https://github.com/lightseekorg/tokenspeed/pull/433), [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `python/tokenspeed/runtime/models/qwen3_5_nextn.py` | [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#429](https://github.com/lightseekorg/tokenspeed/pull/429), [#582](https://github.com/lightseekorg/tokenspeed/pull/582), [#777](https://github.com/lightseekorg/tokenspeed/pull/777), [#828](https://github.com/lightseekorg/tokenspeed/pull/828) |
| `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) |
| `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) |
| `test/ci/eval/qwen3.5-35b-a3b-fp8-deepep-tp2dp2ep4-evalscope-gsm8k.yaml` | 无直接 PR 号提交 |
| `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml` | [#776](https://github.com/lightseekorg/tokenspeed/pull/776) |
| `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` | [#776](https://github.com/lightseekorg/tokenspeed/pull/776) |
| `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` | [#426](https://github.com/lightseekorg/tokenspeed/pull/426) |
| `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` | [#250](https://github.com/lightseekorg/tokenspeed/pull/250) |
| `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` | [#195](https://github.com/lightseekorg/tokenspeed/pull/195), [#229](https://github.com/lightseekorg/tokenspeed/pull/229), [#241](https://github.com/lightseekorg/tokenspeed/pull/241), [#245](https://github.com/lightseekorg/tokenspeed/pull/245), [#257](https://github.com/lightseekorg/tokenspeed/pull/257) |
| `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` | [#264](https://github.com/lightseekorg/tokenspeed/pull/264) |
| `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml` | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) |
| `test/ci/ut/qwen3.5-397b-a17b-nvfp4-pd-1p1d.yaml` | [#400](https://github.com/lightseekorg/tokenspeed/pull/400) |
| `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh` | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) |
| `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` | [#389](https://github.com/lightseekorg/tokenspeed/pull/389) |
| `test/runtime/distributed/test_qwen35_epd_1e1p2d.py` | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) |
| `test/runtime/distributed/test_qwen35_pd_1p1d.py` | [#400](https://github.com/lightseekorg/tokenspeed/pull/400) |
| `test/runtime/models/test_qwen35_vlm_e2e.py` | 无直接 PR 号提交 |
| `test/runtime/models/test_qwen3_5_fused_qkvzba.py` | [#928](https://github.com/lightseekorg/tokenspeed/pull/928), [#1043](https://github.com/lightseekorg/tokenspeed/pull/1043) |
| `test/runtime/models/test_qwen3_5_shared_expert_dp.py` | [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `test/runtime/test_qwen35_gdn_replay.py` | [#1096](https://github.com/lightseekorg/tokenspeed/pull/1096) |
| `test/runtime/test_qwen35_nextn_quantization.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 30
- 原文档显式引用补充 PR 数: 5
- 当前文档总 PR 数: 35
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-19 | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) | merged | feat(qwen3): add Qwen3 MoE causal LM support | `qwen3_moe.py`, `qwen3_moe_config.py`, HF utils, model tests |
| 2026-05-20 | [#189](https://github.com/lightseekorg/tokenspeed/pull/189) | merged | Fix Qwen3 FP8 MoE activation scale layout | `ops/moe/triton.py`, `test_moe_triton.py` |
| 2026-05-22 | [#196](https://github.com/lightseekorg/tokenspeed/pull/196) | merged | perf(qwen3.5): fuse q/k GemmaRMSNorm into one triton launch | `runtime/models/qwen3_5.py`, `test_layernorm.py` |
| 2026-05-23 | [#198](https://github.com/lightseekorg/tokenspeed/pull/198) | merged | perf(qwen3.5): fuse attn_output_gate sigmoid+mul | `qwen3_5.py`, `activation/triton.py`, `test_activation.py` |
| 2026-05-23 | [#195](https://github.com/lightseekorg/tokenspeed/pull/195) | merged | Add qwen3.5-397b-a17b nvfp4 perf CI task | `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` |
| 2026-05-23 | [#228](https://github.com/lightseekorg/tokenspeed/pull/228) | merged | Perf[Qwen3.5]: some kernel fuse optimizations. | `python/tokenspeed/runtime/models/qwen3_5.py` |
| 2026-05-24 | [#235](https://github.com/lightseekorg/tokenspeed/pull/235) | merged | perf[Qwen3.5]: fuse small kernels in MoE block. | `python/tokenspeed/runtime/models/qwen3_5_moe.py` |
| 2026-05-24 | [#229](https://github.com/lightseekorg/tokenspeed/pull/229) | merged | perf(qwen3.5): reduce prefill memcpy sync and mamba update overhead | `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`, `python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py`, `python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py` |
| 2026-05-24 | [#241](https://github.com/lightseekorg/tokenspeed/pull/241) | merged | chore(qwen3.5): adjust qwen perf tps threshold | `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` |
| 2026-05-25 | [#245](https://github.com/lightseekorg/tokenspeed/pull/245) | merged | ci(perf-qwen3.5-agentic): route to gb200-4gpu-perf runner | `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` |
| 2026-05-25 | [#250](https://github.com/lightseekorg/tokenspeed/pull/250) | merged | ci(perf): add Qwen3.5-NVFP4 agentic perf on b200-8gpu | `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` |
| 2026-05-28 | [#264](https://github.com/lightseekorg/tokenspeed/pull/264) | merged | ci(perf): add 1m perf bench for qwen3.5 | `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` |
| 2026-05-28 | [#217](https://github.com/lightseekorg/tokenspeed/pull/217) | merged | perf(Spec Decode): skip dead-position compute in draft catch-up step(decode) | `python/tokenspeed/runtime/models/qwen3_5.py` |
| 2026-05-29 | [#257](https://github.com/lightseekorg/tokenspeed/pull/257) | merged | ci(perf): add qwen3.5 agentic perf ci bs16 case | `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` |
| 2026-06-01 | [#309](https://github.com/lightseekorg/tokenspeed/pull/309) | merged | fix(dp): fix qwen 3.5 data parallel bug | `comm_manager.py`, `vocab_parallel_embedding.py` |
| 2026-06-09 | [#400](https://github.com/lightseekorg/tokenspeed/pull/400) | merged | ci(qwen3.5): add qwen3.5 397b pd ci (1p1d) | PD YAML, distributed smoke test |
| 2026-06-09 | [#389](https://github.com/lightseekorg/tokenspeed/pull/389) | merged | Add PD Qwen3.5 HTTP worker | `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` |
| 2026-06-11 | [#426](https://github.com/lightseekorg/tokenspeed/pull/426) | merged | Add Qwen3.5 PD AIME25 eval CI | `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` |
| 2026-06-13 | [#433](https://github.com/lightseekorg/tokenspeed/pull/433) | merged | perf(qwen3.5): use nvfp4_gemm_swiglu_nvfp4_quant in shared experts | `python/tokenspeed/runtime/models/qwen3_5_moe.py` |
| 2026-06-17 | [#429](https://github.com/lightseekorg/tokenspeed/pull/429) | merged | refactor(spec-decode): simplify Qwen3.5 NextN attention path for #217 (2/3) | `python/tokenspeed/runtime/models/qwen3_5_nextn.py`, `python/tokenspeed/runtime/models/qwen3_5.py` |
| 2026-06-20 | [#485](https://github.com/lightseekorg/tokenspeed/pull/485) | merged | Sanitize Qwen3.5 rope parameters | `python/tokenspeed/runtime/configs/qwen3_5_config.py`, `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` |
| 2026-06-23 | [#354](https://github.com/lightseekorg/tokenspeed/pull/354) | merged | feat(video): generalize multimodal runtime support and add Qwen3.5 video | multimodal runtime, MRoPE, `qwen3_5.py` |
| 2026-06-25 | [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | merged | perf(kernel): optimize Qwen vision QKV rotary layout | packed rotary kernel, Qwen3.5 VLM E2E test |
| 2026-07-03 | [#582](https://github.com/lightseekorg/tokenspeed/pull/582) | merged | ci: disable distributed_argmax on qwen3.5 MTP drafter | `python/tokenspeed/runtime/models/qwen3_5_nextn.py` |
| 2026-07-15 | [#510](https://github.com/lightseekorg/tokenspeed/pull/510) | merged | feat: support Qwen3.5 DFlash and its optimizations | `python/tokenspeed/runtime/models/qwen3_5.py` |
| 2026-07-18 | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) | merged | ci(eval): add Qwen3.5 aggregate and EPD OCRBench coverage | `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh`, `test/runtime/distributed/test_qwen35_epd_1e1p2d.py`, `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` |
| 2026-07-22 | [#766](https://github.com/lightseekorg/tokenspeed/pull/766) | merged | fix(qwen3.5): Fix Qwen3.5 FP8 weight loading | `python/tokenspeed/runtime/models/qwen3_5.py` |
| 2026-07-23 | [#776](https://github.com/lightseekorg/tokenspeed/pull/776) | merged | ci: disable thinking and allow longer context for qwen3.5 eval/pd cases | `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml`, `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` |
| 2026-07-26 | [#777](https://github.com/lightseekorg/tokenspeed/pull/777) | merged | fix(qwen3.5): allow MTP to be quantized | `python/tokenspeed/runtime/models/qwen3_5_nextn.py` |
| 2026-07-27 | [#780](https://github.com/lightseekorg/tokenspeed/pull/780) | merged | fix(multi-node): harden Qwen3.5 multi-node execution | `python/tokenspeed/runtime/layers/logits_processor.py`, `python/tokenspeed/runtime/execution/model_executor.py`, `test/runtime/execution/test_input_buffer_mamba_staging.py` |
| 2026-08-04 | [#928](https://github.com/lightseekorg/tokenspeed/pull/928) | merged | fix(qwen3.5): support non-pow2 ratios in fused_qkvzba_split_reshape_cat_contiguous path | `test/runtime/models/test_qwen3_5_fused_qkvzba.py`, `python/tokenspeed/runtime/models/qwen3_5.py` |
| 2026-08-05 | [#828](https://github.com/lightseekorg/tokenspeed/pull/828) | merged | feat(qwen3.5): support text-only qwen3.5 config | `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py` |
| 2026-08-11 | [#1043](https://github.com/lightseekorg/tokenspeed/pull/1043) | merged | [ci][stability] Stabilize Qwen3.5 performance guards | `test/runtime/models/test_qwen3_5_fused_qkvzba.py` |
| 2026-08-15 | [#1096](https://github.com/lightseekorg/tokenspeed/pull/1096) | merged | feat: add Qwen3.5 GDN ReplaySSM | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py`, `test/runtime/test_qwen35_gdn_replay.py` |
| 2026-09-22 | [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) | merged | fix(qwen3.5 moe): shard Qwen shared experts over the MoE reduction group | `test/runtime/models/test_qwen3_5_shared_expert_dp.py`, `python/tokenspeed/runtime/models/qwen3_5_moe.py`, `python/tokenspeed/runtime/models/qwen3_5.py` |

## 逐 PR diff 审计卡

### PR #181 - feat(qwen3): add Qwen3 MoE causal LM support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/181
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+610/-0，本地 patch 790 行。
- 动机: TokenSpeed 需要 Qwen3 MoE causal LM runtime；这条路径与 Qwen3.5 的 MoE block 复用关系很近，是后续 Qwen3.5 fast path 的可迁移来源。
- 实现要点: 新增 `Qwen3MoeConfig`、`Qwen3MoeForCausalLM` 和 HF config 映射，并在模型测试里把 Qwen3 MoE 接进 runtime。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+from tokenspeed.runtime.models.qwen3_5 import Qwen3_5MoeSparseMoeBlock
+class Qwen3MoeForCausalLM(nn.Module):
```

- 已读文件: `docs/recipes/models.md`, `qwen3_moe_config.py`, `qwen3_moe.py`, `hf_transformers_utils.py`, `test_qwen3_moe_models.py`
- 验证与风险: 对 SGLang SOTA loop 的价值不是直接复制 Qwen3 模型，而是查看 Qwen3.5 MoE block 复用、权重命名和 HF config adapter 的边界。

### PR #189 - Fix Qwen3 FP8 MoE activation scale layout

- 链接: https://github.com/lightseekorg/tokenspeed/pull/189
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+107/-14，本地 patch 174 行。
- 动机: FP8 MoE activation scale 的内存布局和 fused MoE kernel 预期不一致，会影响 Qwen/Qwen3.5 MoE 后端的精度或错误读取。
- 实现要点: 调整 `fused_moe_kernel` / `invoke_fused_moe_kernel` 的 scale 参数准备，并在 Triton MoE 测试里覆盖布局。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+def _normalize_fp8_group_scale_layout(
+    A: torch.Tensor,
+    A_scale: torch.Tensor,
+    expected_scale_k: int,
+) -> torch.Tensor:
+            A_scale = _normalize_fp8_group_scale_layout(A, A_scale, expected_scale_k)
```

- 已读文件: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton.py`, `tokenspeed-kernel/test/ops/test_moe_triton.py`
- 验证与风险: 对比 SGLang 的 FP8/DeepGEMM/MoE 路径时，要把 scale tensor layout 纳入 profiler 前置检查；否则同名 backend 可能走了不同 layout contract。

### PR #196 - perf(qwen3.5): fuse q/k GemmaRMSNorm into one Triton launch

- 链接: https://github.com/lightseekorg/tokenspeed/pull/196
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+87/-12，本地 patch 155 行。
- 动机: Qwen3.5 attention 前处理原来对 Q/K 分别跑 `GemmaRMSNorm`，带来两次 launch 与额外 memory traffic。
- 实现要点: 在 `Qwen3_5AttentionDecoderLayer._apply_qk_norm` 中改用 `qk_rmsnorm`，同时补 layernorm 单测。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
-q = self.q_norm(q)
-k = self.k_norm(k)
+q, k = qk_rmsnorm(q, k, q_gamma, k_gamma, eps)
```

- 已读文件: `python/tokenspeed/runtime/models/qwen3_5.py`, `tokenspeed-kernel/test/ops/test_layernorm.py`
- 验证与风险: 这是 SGLang Qwen3.5/Gemma-style norm fusion 的直接竞品证据；需要在 SGLang loop 里同时看 launch 数、BF16 rounding 顺序和 q/k stride。

### PR #198 - perf(qwen3.5): fuse attn_output_gate sigmoid+mul

- 链接: https://github.com/lightseekorg/tokenspeed/pull/198
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 3 个文件，+234/-3，本地 patch 323 行。
- 动机: Qwen3.5 `attn_output_gate` 原路径包含 reshape、sigmoid、mul，decode 阶段会表现成小 kernel 与一次 gate contiguous copy。
- 实现要点: 新增 Triton `sigmoid_mul`，直接读取 `torch.chunk(q_gate, 2, dim=-1)` 产生的 3D strided gate view，并在 `self_attention` 里原地更新 `attn_output`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
-attn_output = attn_output * torch.sigmoid(gate)
+sigmoid_mul(attn_output, gate)
```

- 已读文件: `python/tokenspeed/runtime/models/qwen3_5.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/triton.py`, `tokenspeed-kernel/test/ops/test_activation.py`
- 验证与风险: 对 SGLang 的启发是把 gate layout 当成 kernel API 的一部分测试；如果 SGLang profiler 表里出现 sigmoid/mul/copy 簇，这是优先候选融合。

### PR #195 - Add qwen3.5-397b-a17b nvfp4 perf CI task

- 链接: https://github.com/lightseekorg/tokenspeed/pull/195
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；关联提交 `5a2db4b1d4bf`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+106/-3，可读 patch 126 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` added +98/-0 (98 lines); hunks: -0,0 +1,98。
- 代码 diff 细节:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` added +98/-0 (98 lines); hunks: -0,0 +1,98
- 关键代码摘录:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml
@@ -0,0 +1,98 @@
+api_version: ci.tokenspeed.io/v1
+name: perf-qwen3.5-397b-a17b-nvfp4-agentic
+type: perf
+triggers:
+  - per-commit
+  - manual
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` added +98/-0
- 验证与风险: diff 自带测试面 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`, `test/ci_system/pipeline.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #228 - Perf[Qwen3.5]: some kernel fuse optimizations.

- 链接: https://github.com/lightseekorg/tokenspeed/pull/228
- 状态/时间: merged / 2026-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5.py`；关联提交 `5454c826a876`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+520/-81，可读 patch 807 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5.py` modified +86/-71 (157 lines); hunks: -31,7 +31,10; -43,7 +46,6; symbols: __init__, _get_split_sizes_for_param, _make_packed_weight_loader，涉及 `__init__, _get_split_sizes_for_param, _make_packed_weight_loader`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +86/-71 (157 lines); hunks: -31,7 +31,10; -43,7 +46,6; symbols: __init__, _get_split_sizes_for_param, _make_packed_weight_loader
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5.py
@@ -31,7 +31,10 @@
-from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
+from tokenspeed_kernel.ops.layernorm.triton import (
+    fused_qk_rmsnorm_rope_gate,
+    qk_rmsnorm,
+)
@@ -43,7 +46,6 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +86/-71
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_layernorm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #235 - perf[Qwen3.5]: fuse small kernels in MoE block.

- 链接: https://github.com/lightseekorg/tokenspeed/pull/235
- 状态/时间: merged / 2026-05-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5_moe.py`；关联提交 `acf6ba45433b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+176/-17，可读 patch 245 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +19/-15 (34 lines); hunks: -24,7 +24,7; -263,14 +263,6 @@ def _forward_tp(; symbols: _forward_tp, _forward_deepep，涉及 `_forward_tp, _forward_deepep`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +19/-15 (34 lines); hunks: -24,7 +24,7; -263,14 +263,6 @@ def _forward_tp(; symbols: _forward_tp, _forward_deepep
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5_moe.py
@@ -24,7 +24,7 @@
-import torch.nn.functional as F
+from tokenspeed_kernel.ops.activation.triton import fused_gate_sigmoid_mul_add
@@ -263,14 +263,6 @@ def _forward_tp(
-                    if (
-                        hidden_states.shape[0] > 0
-                        and self.shared_expert_gate is not None
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +19/-15
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_activation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #229 - perf(qwen3.5): reduce prefill memcpy sync and mamba update overhead

- 链接: https://github.com/lightseekorg/tokenspeed/pull/229
- 状态/时间: merged / 2026-05-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；关联提交 `8d2d78292dd1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+458/-140，可读 patch 851 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +4/-4 (8 lines); hunks: -92,7 +92,7 @@ report:；`python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` modified +142/-47 (189 lines); hunks: -41,6 +41,10; -55,6 +59,7 @@ class MambaForwardMetadata:; symbols: MambaForwardMetadata, get_mamba_indices, _build_mtp_output_indices_kernel, get_mtp_output_indices，涉及 `MambaForwardMetadata, get_mamba_indices, _build_mtp_output_indices_kernel`；`python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py` modified +94/-6 (100 lines); hunks: -168,26 +168,29 @@ def _mamba_state_snapshot_kernel(; -242,6 +245,91 @@ def fused_mamba_state_snapshot(; symbols: _mamba_state_snapshot_kernel, fused_mamba_state_snapshot, fused_mamba_state_copy, _mamba_state_zero_kernel，涉及 `_mamba_state_snapshot_kernel, fused_mamba_state_snapshot, fused_mamba_state_copy`；`python/tokenspeed/runtime/layers/attention/backends/trtllm.py` modified +81/-6 (87 lines); hunks: -30,6 +30,8; -562,6 +564,37 @@ def _init_multi_token_metadata_capture(; symbols: _init_multi_token_metadata_capture, _replay_gather_page_table, init_forward_metadata_replay_cuda_graph，涉及 `_init_multi_token_metadata_capture, _replay_gather_page_table, init_forward_metadata_replay_cuda_graph`。
- 代码 diff 细节:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +4/-4 (8 lines); hunks: -92,7 +92,7 @@ report:
  - `python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` modified +142/-47 (189 lines); hunks: -41,6 +41,10; -55,6 +59,7 @@ class MambaForwardMetadata:; symbols: MambaForwardMetadata, get_mamba_indices, _build_mtp_output_indices_kernel, get_mtp_output_indices
  - `python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py` modified +94/-6 (100 lines); hunks: -168,26 +168,29 @@ def _mamba_state_snapshot_kernel(; -242,6 +245,91 @@ def fused_mamba_state_snapshot(; symbols: _mamba_state_snapshot_kernel, fused_mamba_state_snapshot, fused_mamba_state_copy, _mamba_state_zero_kernel
  - `python/tokenspeed/runtime/layers/attention/backends/trtllm.py` modified +81/-6 (87 lines); hunks: -30,6 +30,8; -562,6 +564,37 @@ def _init_multi_token_metadata_capture(; symbols: _init_multi_token_metadata_capture, _replay_gather_page_table, init_forward_metadata_replay_cuda_graph
  - `python/tokenspeed/runtime/execution/model_executor.py` modified +49/-20 (69 lines); hunks: -502,6 +502,47 @@ def accumulate_decode_stats(self, results: ModelExecutionRe...; -549,27 +590,15 @@ def _snapshot_mamba_checkpoints(; symbols: accumulate_decode_stats, _compute_mtp_snapshot_indices, _snapshot_mamba_checkpoints
- 关键代码摘录:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml
@@ -92,7 +92,7 @@ report:
-  1:  [440, 9300]
-  2:  [340, 14500]
-  4:  [250, 21000]
-  8:  [149, 27000]
+  1:  [530, 11600]
+  2:  [440, 17800]
diff -- python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py
@@ -41,6 +41,10 @@
+from tokenspeed.runtime.layers.attention.linear.index import (
+    set_total_chunks_hint,
+    set_total_chunks_hint_uniform,
+)
@@ -55,6 +59,7 @@ class MambaForwardMetadata:
+    extend_seq_lens_cpu: Optional[torch.Tensor] = None
diff -- python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py
@@ -168,26 +168,29 @@ def _mamba_state_snapshot_kernel(
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +4/-4
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` modified +142/-47; `python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py` modified +94/-6; `python/tokenspeed/runtime/layers/attention/backends/trtllm.py` modified +81/-6; `python/tokenspeed/runtime/execution/model_executor.py` modified +49/-20; `python/tokenspeed/runtime/layers/attention/linear/index.py` modified +47/-10; `python/tokenspeed/runtime/layers/attention/linear/causal_conv1d.py` modified +18/-21
- 验证与风险: diff 自带测试面 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #241 - chore(qwen3.5): adjust qwen perf tps threshold

- 链接: https://github.com/lightseekorg/tokenspeed/pull/241
- 状态/时间: merged / 2026-05-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；关联提交 `2859f547dc8d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 8 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -93,6 +93,6 @@ perf_threshold: 0.9。
- 代码 diff 细节:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -93,6 +93,6 @@ perf_threshold: 0.9
- 关键代码摘录:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml
@@ -93,6 +93,6 @@ perf_threshold: 0.9
-  2:  [440, 17800]
+  2:  [420, 17800]
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #245 - ci(perf-qwen3.5-agentic): route to gb200-4gpu-perf runner

- 链接: https://github.com/lightseekorg/tokenspeed/pull/245
- 状态/时间: merged / 2026-05-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；关联提交 `282f80a6fdd0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -6,7 +6,7 @@ triggers:。
- 代码 diff 细节:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -6,7 +6,7 @@ triggers:
- 关键代码摘录:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml
@@ -6,7 +6,7 @@ triggers:
-    - gb200-4gpu
+    - gb200-4gpu-perf
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #250 - ci(perf): add Qwen3.5-NVFP4 agentic perf on b200-8gpu

- 链接: https://github.com/lightseekorg/tokenspeed/pull/250
- 状态/时间: merged / 2026-05-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml`；关联提交 `9683592b32c0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+93/-0，可读 patch 94 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` added +93/-0 (93 lines); hunks: -0,0 +1,93。
- 代码 diff 细节:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` added +93/-0 (93 lines); hunks: -0,0 +1,93
- 关键代码摘录:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml
@@ -0,0 +1,93 @@
+api_version: ci.tokenspeed.io/v1
+name: perf-qwen3.5-397b-a17b-nvfp4-agentic-b200-8gpu
+type: perf
+triggers:
+  - per-commit
+  - manual
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` added +93/-0
- 验证与风险: diff 自带测试面 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #264 - ci(perf): add 1m perf bench for qwen3.5

- 链接: https://github.com/lightseekorg/tokenspeed/pull/264
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml`；关联提交 `6b792807c1c9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+353/-0，可读 patch 355 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` added +87/-0 (87 lines); hunks: -0,0 +1,87。
- 代码 diff 细节:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` added +87/-0 (87 lines); hunks: -0,0 +1,87
- 关键代码摘录:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml
@@ -0,0 +1,87 @@
+api_version: ci.tokenspeed.io/v1
+name: perf-qwen3.5-397b-a17b-nvfp4-longctx
+type: perf
+triggers:
+  - manual
+runner:
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` added +87/-0
- 验证与风险: diff 自带测试面 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml`, `test/long_context_benchmark/tokenspeed/collect_outputs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #217 - perf(Spec Decode): skip dead-position compute in draft catch-up step(decode)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/217
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`；关联提交 `27e99fb86fa7`, `a700be07ddef`, `a9bc2188501c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+109/-38，可读 patch 286 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -747,6 +747,10 @@ def self_attention(; -774,6 +778,9 @@ def forward(; symbols: self_attention, forward，涉及 `self_attention, forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -747,6 +747,10 @@ def self_attention(; -774,6 +778,9 @@ def forward(; symbols: self_attention, forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5.py
@@ -747,6 +747,10 @@ def self_attention(
+        if ctx.draft_first_step_reduce:
+            # Slice attn_output to [bs, H] so o_proj runs on live rows only.
+            attn_output = attn_output.index_select(0, ctx.gather_ids)
@@ -774,6 +778,9 @@ def forward(
+            if ctx.draft_first_step_reduce:
+                # Gather residual to self_attention's [bs, H].
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/distributed/comm_manager.py`, `python/tokenspeed/runtime/execution/context.py`, `python/tokenspeed/runtime/execution/drafter/eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #257 - ci(perf): add qwen3.5 agentic perf ci bs16 case

- 链接: https://github.com/lightseekorg/tokenspeed/pull/257
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；关联提交 `ca4641e39495`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 16 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +3/-2 (5 lines); hunks: -74,8 +74,8 @@ perf:; -96,3 +96,4 @@ perf_reference:。
- 代码 diff 细节:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +3/-2 (5 lines); hunks: -74,8 +74,8 @@ perf:; -96,3 +96,4 @@ perf_reference:
- 关键代码摘录:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml
@@ -74,8 +74,8 @@ perf:
-    --number 4 8 8 16
-    --parallel 1 2 4 8
+    --number 4 8 8 16 32
+    --parallel 1 2 4 8 16
@@ -96,3 +96,4 @@ perf_reference:
+  16: [78, 29000]
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +3/-2
- 验证与风险: diff 自带测试面 `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #309 - fix(dp): fix qwen 3.5 data parallel bug

- 链接: https://github.com/lightseekorg/tokenspeed/pull/309
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+13/-1，本地 patch 46 行。
- 动机: Qwen3.5 data parallel 下 vocab parallel embedding 的 mask/clamp 行为和 TP>1 路径不一致，可能让 padding 或越界 token 进入 embedding lookup。
- 实现要点: 在 embedding 前按 DP mask 路径 clamp input，并同步修正 distributed comm manager 的 rank 参数。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+masked_input = torch.clamp(masked_input, min=0, max=self.num_embeddings - 1)
```

- 已读文件: `python/tokenspeed/runtime/distributed/comm_manager.py`, `python/tokenspeed/runtime/layers/vocab_parallel_embedding.py`
- 验证与风险: SGLang 对齐 TokenSpeed DP/EP benchmark 时，应检查 vocab mask、padding token 和 TP/DP rank 对齐，不要只看 kernel profile。

### PR #400 - ci(qwen3.5): add Qwen3.5 397B PD CI (1p1d)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/400
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+169/-0，本地 patch 345 行。
- 动机: TokenSpeed 把 `nvidia/Qwen3.5-397B-A17B-NVFP4` 的 prefill/decode disaggregation 变成固定 CI lane，避免 PD 路径只靠人工命令验证。
- 实现要点: 新增 `test_qwen35_pd_1p1d.py`，启动 PD serve 脚本后通过 OpenAI `/v1/models` 和 `/v1/chat/completions` 做 smoke。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+MODEL = os.environ.get("MODEL", "nvidia/Qwen3.5-397B-A17B-NVFP4")
+pytest test/runtime/distributed/test_qwen35_pd_1p1d.py -v
```

- 已读文件: `test/ci/ut/qwen3.5-397b-a17b-nvfp4-pd-1p1d.yaml`, `test/runtime/distributed/test_qwen35_pd_1p1d.py`
- 验证与风险: SGLang SOTA loop 若比较 PD/disagg 场景，需要把 TokenSpeed 的 1P1D lane 作为独立 workload，不应混到单体 serving 的公平对比里。

### PR #389 - Add PD Qwen3.5 HTTP worker

- 链接: https://github.com/lightseekorg/tokenspeed/pull/389
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh`；关联提交 `2b3c8eb15faf`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+558/-0，可读 patch 560 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` added +201/-0 (201 lines); hunks: -0,0 +1,201。
- 代码 diff 细节:
  - `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` added +201/-0 (201 lines); hunks: -0,0 +1,201
- 关键代码摘录:

```diff
diff -- test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh
@@ -0,0 +1,201 @@
+#!/usr/bin/env bash
+set -euo pipefail
+MODEL=${MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}
+SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
+PREFILL_GPUS=${PREFILL_GPUS:-0,1}
+DECODE_GPUS=${DECODE_GPUS:-2,3}
```

- 提取文件（未人工审阅）:
  - tests: `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` added +201/-0
- 验证与风险: diff 自带测试面 `test/ci_system/pd_http_worker.py`, `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #426 - Add Qwen3.5 PD AIME25 eval CI

- 链接: https://github.com/lightseekorg/tokenspeed/pull/426
- 状态/时间: merged / 2026-06-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml`；关联提交 `da785595ad56`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+38/-0，可读 patch 39 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` added +38/-0 (38 lines); hunks: -0,0 +1,38。
- 代码 diff 细节:
  - `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` added +38/-0 (38 lines); hunks: -0,0 +1,38
- 关键代码摘录:

```diff
diff -- test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml
@@ -0,0 +1,38 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-qwen3.5-397b-a17b-nvfp4-pd-1p1d-aime25
+type: eval
+triggers:
+  - per-commit
+  - manual
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` added +38/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #433 - perf(qwen3.5): use nvfp4_gemm_swiglu_nvfp4_quant in shared experts

- 链接: https://github.com/lightseekorg/tokenspeed/pull/433
- 状态/时间: merged / 2026-06-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5_moe.py`；关联提交 `6c4765f5cabc`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+37/-1，可读 patch 69 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +37/-1 (38 lines); hunks: -25,6 +25,11; -33,6 +38,7; symbols: _is_moe_layer, __init__, forward，涉及 `_is_moe_layer, __init__, forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +37/-1 (38 lines); hunks: -25,6 +25,11; -33,6 +38,7; symbols: _is_moe_layer, __init__, forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5_moe.py
@@ -25,6 +25,11 @@
+from tokenspeed_kernel.ops.gemm.cute_dsl import (
+    nvfp4_gemm_swiglu_nvfp4_quant,
+)
+from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
+from tokenspeed_kernel.platform import current_platform
@@ -33,6 +38,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +37/-1
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/qwen3_5_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #429 - refactor(spec-decode): simplify Qwen3.5 NextN attention path for #217 (2/3)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/429
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`；关联提交 `a700be07ddef`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+276/-114，可读 patch 561 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +103/-2 (105 lines); hunks: -22,13 +22,16; -38,12 +41,110; symbols: Qwen3_5DraftAttentionDecoderLayer, _attn, docstring, _apply_correction，涉及 `Qwen3_5DraftAttentionDecoderLayer, _attn, docstring`；`python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-27 (75 lines); hunks: -710,16 +710,13 @@ def _apply_qk_norm(; -737,23 +734,48 @@ def self_attention(; symbols: _apply_qk_norm, self_attention, _project_qkv_rope, _attn，涉及 `_apply_qk_norm, self_attention, _project_qkv_rope`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +103/-2 (105 lines); hunks: -22,13 +22,16; -38,12 +41,110; symbols: Qwen3_5DraftAttentionDecoderLayer, _attn, docstring, _apply_correction
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-27 (75 lines); hunks: -710,16 +710,13 @@ def _apply_qk_norm(; -737,23 +734,48 @@ def self_attention(; symbols: _apply_qk_norm, self_attention, _project_qkv_rope, _attn
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5_nextn.py
@@ -22,13 +22,16 @@
+from dataclasses import replace
+from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
+from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
@@ -38,12 +41,110 @@
-from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForCausalLM
+from tokenspeed.runtime.models.qwen3_5 import (
diff -- python/tokenspeed/runtime/models/qwen3_5.py
@@ -710,16 +710,13 @@ def _apply_qk_norm(
-    def self_attention(
+    def _project_qkv_rope(
-        ctx: ForwardContext,
-        out_cache_loc: torch.Tensor,
-    ) -> torch.Tensor:
-        """Full attention forward pass."""
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +103/-2; `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-27
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/execution/drafter/eagle.py`, `python/tokenspeed/runtime/execution/model_executor.py`, `python/tokenspeed/runtime/layers/attention/backends/base.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #485 - Sanitize Qwen3.5 rope parameters

- 链接: https://github.com/lightseekorg/tokenspeed/pull/485
- 状态/时间: merged / 2026-06-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/qwen3_5_config.py`, `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py`, `python/tokenspeed/runtime/models/qwen3_5.py`；关联提交 `c9a1c7fd7c73`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+55/-22，可读 patch 170 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/configs/qwen3_5_config.py` modified +18/-8 (26 lines); hunks: -25,6 +25,18; -41,16 +53,14 @@ def __init__(; symbols: _to_transformers_rope_parameters, Qwen3_5VisionConfig, __init__, Qwen3_5Config，涉及 `_to_transformers_rope_parameters, Qwen3_5VisionConfig, __init__`；`python/tokenspeed/runtime/models/qwen3_5.py` modified +3/-7 (10 lines); hunks: -40,6 +40,7; -591,10 +592,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` modified +3/-3 (6 lines); hunks: -90,7 +90,7 @@ class Qwen3_5BaseTextConfig(PretrainedConfig):; -201,7 +201,7 @@ def __init__(; symbols: Qwen3_5BaseTextConfig, __init__，涉及 `Qwen3_5BaseTextConfig, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/configs/qwen3_5_config.py` modified +18/-8 (26 lines); hunks: -25,6 +25,18; -41,16 +53,14 @@ def __init__(; symbols: _to_transformers_rope_parameters, Qwen3_5VisionConfig, __init__, Qwen3_5Config
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +3/-7 (10 lines); hunks: -40,6 +40,7; -591,10 +592,7 @@ def __init__(; symbols: __init__
  - `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` modified +3/-3 (6 lines); hunks: -90,7 +90,7 @@ class Qwen3_5BaseTextConfig(PretrainedConfig):; -201,7 +201,7 @@ def __init__(; symbols: Qwen3_5BaseTextConfig, __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/configs/qwen3_5_config.py
@@ -25,6 +25,18 @@
+_MROPE_EXTENSION_KEYS = frozenset({"mrope_section", "mrope_interleaved"})
+def _to_transformers_rope_parameters(rope_config):
+    if not isinstance(rope_config, dict):
+        return rope_config
+    return {
+        key: value
diff -- python/tokenspeed/runtime/models/qwen3_5.py
@@ -40,6 +40,7 @@
+from tokenspeed.runtime.configs.utils import get_rope_parameters
@@ -591,10 +592,7 @@ def __init__(
-        if hasattr(config, "rope_parameters"):
-            self.rope_scaling = getattr(config, "rope_parameters", None)
-        else:
-            self.rope_scaling = getattr(config, "rope_scaling", None)
diff -- python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py
@@ -90,7 +90,7 @@ class Qwen3_5BaseTextConfig(PretrainedConfig):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/configs/qwen3_5_config.py` modified +18/-8; `python/tokenspeed/runtime/models/qwen3_5.py` modified +3/-7; `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` modified +3/-3
- 验证与风险: diff 自带测试面 `test/runtime/test_resolve_architecture.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #354 - feat(video): generalize multimodal runtime support and add Qwen3.5 video

- 链接: https://github.com/lightseekorg/tokenspeed/pull/354
- 状态/时间: merged / 2026-06-23
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 19 个文件，+982/-266，本地 patch 2,500 行。
- 动机: Qwen3.5 video/image path 需要统一多模态 runtime、encoder output budget、M-RoPE decode position 以及 CUDA graph capture，而不是每个模型单独处理。
- 实现要点: 抽象 multimodal adapter、budget graph、metadata sequence budget；在 generation output / input processor / model executor 中加入 MRoPE position delta cache 与 decode override。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+mrope_position_delta_scalar: Optional[int] = None
+        if not is_prefill:
+            return self._build_decode_mrope_positions_override(
```

- 已读文件: `generation_output_processor.py`, `input_processor.py`, `model_executor.py`, `runtime/multimodal/*`, `runtime/models/qwen3_5.py`, `runtime/models/kimi_k25.py`
- 验证与风险: 多模态 profile 要拆 encoder capture、MRoPE build、output D2H、decode forward；单看 LLM decode kernel 会漏掉 TokenSpeed 的实际优化面。

### PR #456 - perf(kernel): optimize Qwen vision QKV rotary layout

- 链接: https://github.com/lightseekorg/tokenspeed/pull/456
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 6 个文件，+452/-35，本地 patch 816 行。
- 动机: Qwen3.5 VLM vision attention 的 packed QKV + rotary 原路径会拆分、搬运和再 materialize；PR 把 NeoX rotary 也接入 packed rotary kernel。
- 实现要点: 新增 `packed_qkv_neox_rotary`，在 `mm_encoder_attention.py` 根据 position embedding 模式选择 packed rotary；新增 Blackwell Qwen3.5 VLM E2E smoke。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+            q, k, v = packed_qkv_neox_rotary(
+                qkv,
+                self.q_size,
+__all__ = ["packed_qkv_complex_rotary", "packed_qkv_neox_rotary"]
```

- 已读文件: `mm_encoder_attention.py`, `runtime/models/qwen3_5.py`, `qkv_rotary.py`, `test_qwen35_vlm_e2e.py`, `trtllm_fp8.py`
- 验证与风险: 对 SGLang VLM/vision encoder 优化，优先检查 QKV split + rotary + V copy 是否已经融合；如果没有，TokenSpeed 这里是明确的竞品实现证据。


### PR #582 - ci: disable distributed_argmax on qwen3.5 MTP drafter

- 链接: https://github.com/lightseekorg/tokenspeed/pull/582
- 状态/时间: merged / 2026-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5_nextn.py`；关联提交 `547e7b700ae8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+9/-3，可读 patch 40 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-1 (1 lines); hunks: -206,7 +206,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-1 (1 lines); hunks: -206,7 +206,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5_nextn.py
@@ -206,7 +206,6 @@ def __init__(
-            do_argmax=True,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/execution/drafter/eagle.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #510 - 支持 Qwen3.5 DFlash 并优化 runtime

- 链接: https://github.com/lightseekorg/tokenspeed/pull/510
- 状态/时间: merged / 2026-07-15
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 2,169 行 diff，12 个文件，+1597/-81。
- 动机: Qwen3.5 缺少原生 DFlash；直接移植仍会分别支付 hidden-state projection、KV materialization、prepare-decode、QK norm/RoPE 和 draft-cache 多次 launch。
- 实现要点: 捕获选定 target layer，在 auxiliary stream 上增量投影；加入融合 KV RMSNorm/RoPE/scatter 与 FP8 draft-cache，融合 prepare-decode bookkeeping，支持 FA4 非因果 draft attention，并修复 draft RoPE 配置。
- 代码 diff 细节: DFlash drafter 缓存 KV buffer pointer、把每层 KV 直接写入 pool，用 event 同步 incremental projection，并为 draft model 新增融合 QK-RMSNorm+RoPE Triton kernel。
- 关键代码摘录:

```diff
+_fused_norm_rope_scatter_kernel[(total_ctx, num_kv_heads, n_layers)](
+    kv, k_norm_weight, eps, cos_sin_cache, positions, loc, k_ptrs, v_ptrs, ...)
+self.drafter._prepare_incremental_proj(
+    ctx.input_num_tokens, positions, out_cache_loc)
```

- 已读文件: runtime：`execution/drafter/{dflash,_dflash_fused_kv}.py`、`cache_loc_kernel.py`、CUDA-graph/model executor/runner、`models/{dflash,qwen3_5}.py`、MHA config；kernel/测试：FlashAttention registry、`layernorm/triton.py`、`test_layernorm.py`。
- 验证与风险: 必须记录 target/draft checkpoint、draft attention backend、KV dtype、capture layer 和 speculative token 数；验证 FP8 scale、非因果 FA4 window、auxiliary-stream 顺序、CUDA-graph replay，以及 RoPE 修复后的 acceptance rate。

### PR #549 - ci(eval): add Qwen3.5 aggregate and EPD OCRBench coverage

- 链接: https://github.com/lightseekorg/tokenspeed/pull/549
- 状态/时间: merged / 2026-07-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml`, `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml`, `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml`, `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh`, `test/runtime/distributed/test_qwen35_epd_1e1p2d.py`；关联提交 `8207c4121ca1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+595/-0，可读 patch 600 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh` added +248/-0 (248 lines); hunks: -0,0 +1,248；`test/runtime/distributed/test_qwen35_epd_1e1p2d.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: _truetype_font, _image_data_uri, _wait_for_server, _chat，涉及 `_truetype_font, _image_data_uri, _wait_for_server`；`test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` added +61/-0 (61 lines); hunks: -0,0 +1,61；`test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` added +37/-0 (37 lines); hunks: -0,0 +1,37。
- 代码 diff 细节:
  - `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh` added +248/-0 (248 lines); hunks: -0,0 +1,248
  - `test/runtime/distributed/test_qwen35_epd_1e1p2d.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: _truetype_font, _image_data_uri, _wait_for_server, _chat
  - `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` added +61/-0 (61 lines); hunks: -0,0 +1,61
  - `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` added +37/-0 (37 lines); hunks: -0,0 +1,37
  - `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml` added +18/-0 (18 lines); hunks: -0,0 +1,18
- 关键代码摘录:

```diff
diff -- test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh
@@ -0,0 +1,248 @@
+#!/usr/bin/env bash
+# EPD (encode-prefill-decode) 1E-1P-2D smoke/eval topology on a single node.
+#
+# Serves nvidia/Qwen3.5-122B-A10B-NVFP4 (override via MODEL). EPD mode is
+# gRPC-only (the SMG gateway refuses HTTP connection_mode for EPD), so the
+# workers are gRPC servicers launched via `python3 -m smg_grpc_servicer.tokenspeed`
diff -- test/runtime/distributed/test_qwen35_epd_1e1p2d.py
@@ -0,0 +1,231 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml
@@ -0,0 +1,61 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh` added +248/-0; `test/runtime/distributed/test_qwen35_epd_1e1p2d.py` added +231/-0; `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` added +61/-0; `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` added +37/-0; `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml` added +18/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml`, `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml`, `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml`, `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #766 - 修复 Qwen3.5 FP8 权重加载

- 链接: https://github.com/lightseekorg/tokenspeed/pull/766
- 状态/时间: merged / 2026-07-22
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 283 行 diff，2 个文件，+118/-67。
- 动机: Qwen3.5-35B-A3B FP8 checkpoint 量化 GDN `qkv/z`，但让 `b/a` 保持 BF16；把 6 个 projection 全塞进一个 quantized linear 会输出乱码，MoE kernel 在 TP 下还拿到了未分片 intermediate size。
- 实现要点: 从 `ignored_layers` 推导各 GDN projection group 的量化属性，仅在量化不同的时候拆成 `in_proj_qkvz` 与 `in_proj_ba`，按所选布局调整 checkpoint mapping，并从 TP-sharded `w2_weight` 推导 MoE intermediate size。
- 代码 diff 细节: 全量化或全不量化 checkpoint 继续使用单一 fused projection；只有 mixed checkpoint 承担 split-linear 成本。
- 关键代码摘录:

```diff
+self._split_in_proj = quant_config is not None and (qkvz_unquant != ba_unquant)
+if self._split_in_proj:
+    self.in_proj_qkvz = MergedColumnParallelLinear(..., quant_config=...)
+    self.in_proj_ba = MergedColumnParallelLinear(..., quant_config=...)
+intermediate_size = w.w2_weight.shape[-1]
```

- 已读文件: runtime：`python/tokenspeed/runtime/models/qwen3_5.py`；kernel wrapper：`tokenspeed_kernel/ops/moe/flashinfer/trtllm_fp8.py`。
- 验证与风险: 需在 TP、EP、hybrid 下覆盖 BF16、全 FP8 和 mixed ignored-layer checkpoint；PR 报告 Qwen3.5-35B-A3B-FP8 在 TP2 与 DP4EP4 输出正确。

### PR #776 - ci: disable thinking and allow longer context for qwen3.5 eval/pd cases

- 链接: https://github.com/lightseekorg/tokenspeed/pull/776
- 状态/时间: merged / 2026-07-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml`, `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml`；关联提交 `670663cb53a2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+3/-0，可读 patch 24 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -9,6 +9,7 @@ runner:；`test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -10,6 +10,7 @@ runner:。
- 代码 diff 细节:
  - `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -9,6 +9,7 @@ runner:
  - `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -10,6 +10,7 @@ runner:
- 关键代码摘录:

```diff
diff -- test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml
@@ -9,6 +9,7 @@ runner:
+  TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN: "1"
diff -- test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml
@@ -10,6 +10,7 @@ runner:
+  TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN: "1"
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml` modified +1/-0; `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` modified +1/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml`, `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml`, `test/ci_system/pd_http_worker.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #777 - fix(qwen3.5): allow MTP to be quantized

- 链接: https://github.com/lightseekorg/tokenspeed/pull/777
- 状态/时间: merged / 2026-07-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5_nextn.py`；关联提交 `8e2911dad09e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+0/-4，可读 patch 11 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-4 (4 lines); hunks: -163,10 +163,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-4 (4 lines); hunks: -163,10 +163,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5_nextn.py
@@ -163,10 +163,6 @@ def __init__(
-        # The MTP model is unquantized in the nvfp4 checkpoint.
-        if quant_config and quant_config.get_name() == "nvfp4":
-            quant_config = None
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-4
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/qwen3_5_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #780 - 加固 Qwen3.5 多机执行

- 链接: https://github.com/lightseekorg/tokenspeed/pull/780
- 状态/时间: merged / 2026-07-28
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 555 行 diff，8 个文件，+347/-35。
- 动机: Qwen3.5 多机布局可能跨节点选择 CUDA-IPC/symmetric-memory collective；overlap H2D copy 尚未完成时，还可能复用 persistent pinned Mamba/GDN staging buffer。
- 实现要点: 检测跨节点 process group，对 all-reduce、all-gather、token collective、logits gather/argmax 强制回退 NCCL；把 Mamba index 改为逐 step bulk pinned staging，并补充一致 NCCL 设置的文档。
- 代码 diff 细节: node-local group 继续走低延迟 RSAG/custom path，只有 cross-node group 回退；测试断言 backend selection 与 staging 不复用。
- 关键代码摘录:

```diff
+if self._group_spans_nodes(group):
+    return self._nccl.token_all_gather(tensor, group, scattered_num_tokens)
+(...mamba staging...) = self._bulk_pinned(
+    (batch_size, torch.int32), (batch_size, torch.int32), ...)
```

- 已读文件: runtime：`distributed/comm_backend/auto.py`、`execution/{input_buffer,model_executor}.py`、`layers/logits_processor.py`；测试/文档：`test_comm_ops.py`、`test_input_buffer_mamba_staging.py`、`test_logits_processor.py`、`docs/serving/parallelism.md`。
- 验证与风险: 验证 node-rank mapping 与 NCCL transport 一致性，确保 custom collective 仍只用于 node-local，并结合 GDN state copy-on-write 与 TP logits 路径跑 overlap/CUDA-graph replay。

### PR #928 - fix(qwen3.5): support non-pow2 ratios in fused_qkvzba_split_reshape_cat_contiguous path

- 链接: https://github.com/lightseekorg/tokenspeed/pull/928
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5.py`, `test/runtime/models/test_qwen3_5_fused_qkvzba.py`；关联提交 `a1c09236006d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+334/-36，可读 patch 395 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_qwen3_5_fused_qkvzba.py` added +282/-0 (282 lines); hunks: -0,0 +1,282; symbols: _legacy_wide_arange_kernel, _legacy_launch, _make_inputs, FusedQkvzbaTest，涉及 `_legacy_wide_arange_kernel, _legacy_launch, _make_inputs`；`python/tokenspeed/runtime/models/qwen3_5.py` modified +52/-36 (88 lines); hunks: -467,7 +467,7 @@ def forward(; -1765,26 +1765,6 @@ def fused_qkvzba_split_reshape_cat_contiguous_kernel(; symbols: forward, fused_qkvzba_split_reshape_cat_contiguous_kernel，涉及 `forward, fused_qkvzba_split_reshape_cat_contiguous_kernel`。
- 代码 diff 细节:
  - `test/runtime/models/test_qwen3_5_fused_qkvzba.py` added +282/-0 (282 lines); hunks: -0,0 +1,282; symbols: _legacy_wide_arange_kernel, _legacy_launch, _make_inputs, FusedQkvzbaTest
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +52/-36 (88 lines); hunks: -467,7 +467,7 @@ def forward(; -1765,26 +1765,6 @@ def fused_qkvzba_split_reshape_cat_contiguous_kernel(; symbols: forward, fused_qkvzba_split_reshape_cat_contiguous_kernel
- 关键代码摘录:

```diff
diff -- test/runtime/models/test_qwen3_5_fused_qkvzba.py
@@ -0,0 +1,282 @@
+"""Correctness + perf guard for fused_qkvzba_split_reshape_cat_contiguous.
+The kernel's v/z access was reworked from one V_PER_GROUP * HEAD_V arange
+into a per-head static_range loop so any integer head ratio works (Triton
+arange requires a power-of-2 span; ratio 3 * 128 = 384 is not one).
+- correctness: bit-exact vs a plain torch split reference for ratios
+  1/2/3/4, plus 16-byte alignment of every output (flashinfer's CuteDSL
diff -- python/tokenspeed/runtime/models/qwen3_5.py
@@ -467,7 +467,7 @@ def forward(
-        if self.num_v_heads // self.num_k_heads in [1, 2, 4]:
+        if self.num_v_heads % self.num_k_heads == 0:
@@ -1765,26 +1765,6 @@ def fused_qkvzba_split_reshape_cat_contiguous_kernel(
-    # v for head group i_qk: in the all_v region
-    blk_v_ptr = (
-        mixed_qkvz
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_qwen3_5_fused_qkvzba.py` added +282/-0
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +52/-36
- 验证与风险: diff 自带测试面 `test/runtime/models/test_qwen3_5_fused_qkvzba.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #828 - feat(qwen3.5): support text-only qwen3.5 config

- 链接: https://github.com/lightseekorg/tokenspeed/pull/828
- 状态/时间: merged / 2026-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`；关联提交 `4498a6149b19`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+63/-4，可读 patch 142 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-3 (51 lines); hunks: -39,6 +39,7; -1134,7 +1135,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Qwen3_5MoeForCausalLM, Qwen3_5MoeModel, for，涉及 `load_weights, Qwen3_5MoeForCausalLM, Qwen3_5MoeModel`；`python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +5/-0 (5 lines); hunks: -436,7 +436,12 @@ def __init__(; symbols: __init__, Qwen3_5MoeForCausalLMNextN，涉及 `__init__, Qwen3_5MoeForCausalLMNextN`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-3 (51 lines); hunks: -39,6 +39,7; -1134,7 +1135,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Qwen3_5MoeForCausalLM, Qwen3_5MoeModel, for
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +5/-0 (5 lines); hunks: -436,7 +436,12 @@ def __init__(; symbols: __init__, Qwen3_5MoeForCausalLMNextN
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5.py
@@ -39,6 +39,7 @@
+    Qwen3_5MoeConfig,
@@ -1134,7 +1135,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
-class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
+class Qwen3_5MoeModel(Qwen3_5ForCausalLM):
+    """MoE backbone (internal). The ``Qwen3_5MoeForCausalLM`` name is taken by
+    the registry entry class for text-only flat checkpoints below."""
diff -- python/tokenspeed/runtime/models/qwen3_5_nextn.py
@@ -436,7 +436,12 @@ def __init__(
+class Qwen3_5MoeForCausalLMNextN(Qwen3_5MoeForConditionalGenerationNextN):
+    """MTP draft head for text-only flat checkpoints."""
+    Qwen3_5MoeForCausalLMNextN,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-3; `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/configs/__init__.py`, `python/tokenspeed/runtime/layers/attention/registry.py`, `python/tokenspeed/runtime/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1043 - [ci][stability] Stabilize Qwen3.5 performance guards

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1043
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/models/test_qwen3_5_fused_qkvzba.py`；关联提交 `059b512a18e6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+11/-4，可读 patch 31 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_qwen3_5_fused_qkvzba.py` modified +11/-4 (15 lines); hunks: -234,10 +234,13 @@ def test_perf_no_regression_vs_legacy_wide_arange(self):; -268,8 +271,12 @@ def torch_fallback():; symbols: test_perf_no_regression_vs_legacy_wide_arange, torch_fallback，涉及 `test_perf_no_regression_vs_legacy_wide_arange, torch_fallback`。
- 代码 diff 细节:
  - `test/runtime/models/test_qwen3_5_fused_qkvzba.py` modified +11/-4 (15 lines); hunks: -234,10 +234,13 @@ def test_perf_no_regression_vs_legacy_wide_arange(self):; -268,8 +271,12 @@ def torch_fallback():; symbols: test_perf_no_regression_vs_legacy_wide_arange, torch_fallback
- 关键代码摘录:

```diff
diff -- test/runtime/models/test_qwen3_5_fused_qkvzba.py
@@ -234,10 +234,13 @@ def test_perf_no_regression_vs_legacy_wide_arange(self):
-                lambda a=args: fused(*a), warmup=50, rep=200
+                lambda a=args: fused(*a), warmup=50, rep=200, return_mode="median"
-                lambda a=args: _legacy_launch(*a), warmup=50, rep=200
+                lambda a=args: _legacy_launch(*a),
+                warmup=50,
+                rep=200,
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_qwen3_5_fused_qkvzba.py` modified +11/-4
- 验证与风险: diff 自带测试面 `test/runtime/models/test_qwen3_5_fused_qkvzba.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1096 - feat: add Qwen3.5 GDN ReplaySSM

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1096
- 状态/时间: merged / 2026-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py`, `test/runtime/test_qwen35_gdn_replay.py`；关联提交 `cd78b4e0b238`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+1747/-85，可读 patch 2276 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` modified +15/-1 (16 lines); hunks: -1,5 +1,7; -162,6 +164,7 @@ def prepare_qwen35_cache(; symbols: prepare_qwen35_cache，涉及 `prepare_qwen35_cache`；`test/runtime/test_qwen35_gdn_replay.py` added +409/-0 (409 lines); hunks: -0,0 +1,409; symbols: _config, _make_backend, _inputs, _forward_verify，涉及 `_config, _make_backend, _inputs`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` modified +15/-1 (16 lines); hunks: -1,5 +1,7; -162,6 +164,7 @@ def prepare_qwen35_cache(; symbols: prepare_qwen35_cache
  - `test/runtime/test_qwen35_gdn_replay.py` added +409/-0 (409 lines); hunks: -0,0 +1,409; symbols: _config, _make_backend, _inputs, _forward_verify
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py
@@ -1,5 +1,7 @@
+import torch
@@ -162,6 +164,7 @@ def prepare_qwen35_cache(
+    replay_ssm = False
@@ -185,11 +188,22 @@ def prepare_qwen35_cache(
+        replay_ssm = (
+            getattr(server_args, "enable_replay_ssm", False)
diff -- test/runtime/test_qwen35_gdn_replay.py
@@ -0,0 +1,409 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` modified +15/-1
  - tests: `test/runtime/test_qwen35_gdn_replay.py` added +409/-0
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_gdn_qkv_split_fused.py`, `test/runtime/test_cache_setup.py`, `test/runtime/test_cli_config_compat.py`, `test/runtime/test_gdn_state_paging.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1713 - fix(qwen3.5 moe): shard Qwen shared experts over the MoE reduction group

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1713
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_moe.py`, `test/runtime/models/test_qwen3_5_shared_expert_dp.py`；关联提交 `0722ff403d8b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+88/-1，可读 patch 169 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_qwen3_5_shared_expert_dp.py` modified +71/-0 (71 lines); hunks: -11,7 +11,9; -36,6 +38,7 @@ def _mlp(world_size: int, replicate: bool) -> Qwen3_5MoeMLP:; symbols: _mlp, forward, test_shared_expert_shards_match_moe_reduction, test_dense_dp_and_deepep_shared_still_replicate，涉及 `_mlp, forward, test_shared_expert_shards_match_moe_reduction`；`python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +13/-1 (14 lines); hunks: -23,6 +23,8; -114,6 +116,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/tokenspeed/runtime/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -592,6 +592,7 @@ def __init__(; -773,6 +774,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `test/runtime/models/test_qwen3_5_shared_expert_dp.py` modified +71/-0 (71 lines); hunks: -11,7 +11,9; -36,6 +38,7 @@ def _mlp(world_size: int, replicate: bool) -> Qwen3_5MoeMLP:; symbols: _mlp, forward, test_shared_expert_shards_match_moe_reduction, test_dense_dp_and_deepep_shared_still_replicate
  - `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +13/-1 (14 lines); hunks: -23,6 +23,8; -114,6 +116,8 @@ def __init__(; symbols: __init__
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -592,6 +592,7 @@ def __init__(; -773,6 +774,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- test/runtime/models/test_qwen3_5_shared_expert_dp.py
@@ -11,7 +11,9 @@
+import pytest
+import torch.nn.functional as F
@@ -36,6 +38,7 @@ def _mlp(world_size: int, replicate: bool) -> Qwen3_5MoeMLP:
+        parallelism="moe_shared",
@@ -122,5 +125,73 @@ def forward(self, x):
+@pytest.mark.parametrize("moe_tp,moe_ep", [(4, 1), (1, 4), (2, 2)])
diff -- python/tokenspeed/runtime/models/qwen3_5_moe.py
@@ -23,6 +23,8 @@
+from typing import Literal
@@ -114,6 +116,8 @@ def __init__(
+        *,
+        parallelism: Literal["dense", "moe_shared"],
@@ -138,10 +142,17 @@ def __init__(
-        if mapping.dense.has_tp and not replicate_weights:
diff -- python/tokenspeed/runtime/models/qwen3_5.py
@@ -592,6 +592,7 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_qwen3_5_shared_expert_dp.py` modified +71/-0
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +13/-1; `python/tokenspeed/runtime/models/qwen3_5.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_qwen3_5_shared_expert_dp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
