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

| 文件 | 关联 PR |
| --- | --- |
| `python/tokenspeed/runtime/models/qwen3_5.py` | [#196](https://github.com/lightseekorg/tokenspeed/pull/196), [#198](https://github.com/lightseekorg/tokenspeed/pull/198), [#354](https://github.com/lightseekorg/tokenspeed/pull/354), [#456](https://github.com/lightseekorg/tokenspeed/pull/456) |
| `python/tokenspeed/runtime/configs/qwen3_moe_config.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) |
| `python/tokenspeed/runtime/models/qwen3_moe.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) |
| `python/tokenspeed/runtime/layers/vocab_parallel_embedding.py` | [#309](https://github.com/lightseekorg/tokenspeed/pull/309) |
| `python/tokenspeed/runtime/distributed/comm_manager.py` | [#309](https://github.com/lightseekorg/tokenspeed/pull/309) |
| `python/tokenspeed/runtime/multimodal/*` | [#354](https://github.com/lightseekorg/tokenspeed/pull/354) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/triton.py` | [#198](https://github.com/lightseekorg/tokenspeed/pull/198) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/qkv_rotary.py` | [#456](https://github.com/lightseekorg/tokenspeed/pull/456) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton.py` | [#189](https://github.com/lightseekorg/tokenspeed/pull/189) |
| `test/runtime/models/test_qwen35_vlm_e2e.py` | [#456](https://github.com/lightseekorg/tokenspeed/pull/456) |
| `test/runtime/distributed/test_qwen35_pd_1p1d.py` | [#400](https://github.com/lightseekorg/tokenspeed/pull/400) |

## PR 覆盖总览

- 本轮审计 PR 数: 8
- 文件反查命令: `git log --name-only -- <model-files>`
- diff 来源: `gh pr diff` / GitHub Pull Request files API patch，本地缓存 `/tmp/model_pr_diffs/tokenspeed/pr*.diff`
- 已读 patch 行数: 5,149
- TokenSpeed Qwen3.5 关键形态: Qwen3/Qwen3.5 MoE runtime、Q/K RMSNorm 融合、attention gate 融合、多模态 MRoPE/视频路径、packed QKV rotary、PD disaggregation CI。

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-19 | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) | merged | feat(qwen3): add Qwen3 MoE causal LM support | `qwen3_moe.py`, `qwen3_moe_config.py`, HF utils, model tests |
| 2026-05-20 | [#189](https://github.com/lightseekorg/tokenspeed/pull/189) | merged | Fix Qwen3 FP8 MoE activation scale layout | `ops/moe/triton.py`, `test_moe_triton.py` |
| 2026-05-22 | [#196](https://github.com/lightseekorg/tokenspeed/pull/196) | merged | perf(qwen3.5): fuse q/k GemmaRMSNorm into one triton launch | `runtime/models/qwen3_5.py`, `test_layernorm.py` |
| 2026-05-23 | [#198](https://github.com/lightseekorg/tokenspeed/pull/198) | merged | perf(qwen3.5): fuse attn_output_gate sigmoid+mul | `qwen3_5.py`, `activation/triton.py`, `test_activation.py` |
| 2026-06-01 | [#309](https://github.com/lightseekorg/tokenspeed/pull/309) | merged | fix(dp): fix qwen 3.5 data parallel bug | `comm_manager.py`, `vocab_parallel_embedding.py` |
| 2026-06-09 | [#400](https://github.com/lightseekorg/tokenspeed/pull/400) | merged | ci(qwen3.5): add qwen3.5 397b pd ci (1p1d) | PD YAML, distributed smoke test |
| 2026-06-23 | [#354](https://github.com/lightseekorg/tokenspeed/pull/354) | merged | feat(video): generalize multimodal runtime support and add Qwen3.5 video | multimodal runtime, MRoPE, `qwen3_5.py` |
| 2026-06-25 | [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | merged | perf(kernel): optimize Qwen vision QKV rotary layout | packed rotary kernel, Qwen3.5 VLM E2E test |

## 逐 PR diff 审计卡

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
