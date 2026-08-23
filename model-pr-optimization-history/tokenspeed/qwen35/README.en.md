# TokenSpeed Qwen3.5 Model PR Optimization History

## 2026-08-23 Source Head Refresh

Rechecked TokenSpeed upstream main at `lightseekorg/tokenspeed@2706143a8669d50a8f56466b9d340b86922b8f2d`.
The two-commit range after the previous head
`d73bf0454422092f306d5575e803a08fd35ac41c` was read in full. Both commits are
Kimi K3 documentation-only changes, so no new Qwen3.5 card is promoted.

Result: Qwen3.5 gained native optimized DFlash, corrected mixed-precision GDN/MoE FP8 loading, and topology-safe multi-node collective/staging behavior.

## 2026-06-27 PR Backfill Audit

Checked against TokenSpeed upstream `HEAD@d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8`. This page follows the same structure as the SGLang/vLLM histories: model-relevant PR timeline, reviewed diffs, implementation files, short code excerpts, and validation risks.

Filter used in this pass: merged GitHub PRs whose title or files matched `Qwen3.5`, `qwen3_5`, `Qwen3Moe`, `VLM`, `PD`, `moe`, `activation`, `rotary`, or `flashinfer_trtllm`. Pure formatting and unrelated infrastructure PRs were excluded.

## Implementation File Coverage

| File | Related PRs |
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

## PR Coverage Summary

- Reviewed PRs: 8
- File trace command: `git log --name-only -- <model-files>`
- Diff source: `gh pr diff` / GitHub Pull Request files API patches cached under `/tmp/model_pr_diffs/tokenspeed/pr*.diff`
- Reviewed patch lines: 5,149
- Main TokenSpeed Qwen3.5 themes: Qwen3/Qwen3.5 MoE runtime, Q/K RMSNorm fusion, attention-gate fusion, multimodal MRoPE/video runtime, packed QKV rotary, and PD disaggregation CI.

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-19 | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) | merged | feat(qwen3): add Qwen3 MoE causal LM support | `qwen3_moe.py`, `qwen3_moe_config.py`, HF utils, model tests |
| 2026-05-20 | [#189](https://github.com/lightseekorg/tokenspeed/pull/189) | merged | Fix Qwen3 FP8 MoE activation scale layout | `ops/moe/triton.py`, `test_moe_triton.py` |
| 2026-05-22 | [#196](https://github.com/lightseekorg/tokenspeed/pull/196) | merged | perf(qwen3.5): fuse q/k GemmaRMSNorm into one triton launch | `runtime/models/qwen3_5.py`, `test_layernorm.py` |
| 2026-05-23 | [#198](https://github.com/lightseekorg/tokenspeed/pull/198) | merged | perf(qwen3.5): fuse attn_output_gate sigmoid+mul | `qwen3_5.py`, `activation/triton.py`, `test_activation.py` |
| 2026-06-01 | [#309](https://github.com/lightseekorg/tokenspeed/pull/309) | merged | fix(dp): fix qwen 3.5 data parallel bug | `comm_manager.py`, `vocab_parallel_embedding.py` |
| 2026-06-09 | [#400](https://github.com/lightseekorg/tokenspeed/pull/400) | merged | ci(qwen3.5): add qwen3.5 397b pd ci (1p1d) | PD YAML, distributed smoke test |
| 2026-06-23 | [#354](https://github.com/lightseekorg/tokenspeed/pull/354) | merged | feat(video): generalize multimodal runtime support and add Qwen3.5 video | multimodal runtime, MRoPE, `qwen3_5.py` |
| 2026-06-25 | [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | merged | perf(kernel): optimize Qwen vision QKV rotary layout | packed rotary kernel, Qwen3.5 VLM E2E test |

## Per-PR Diff Audit Cards

### PR #510 - Support Qwen3.5 DFlash and optimize its runtime

- Link: https://github.com/lightseekorg/tokenspeed/pull/510
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 2,169-line diff, 12 files, +1597/-81.
- Motivation: Qwen3.5 lacked a native DFlash path, and a direct implementation would still pay separate hidden-state projection, KV materialization, prepare-decode, QK normalization/RoPE, and draft-cache launches.
- Key implementation: captures selected target layers, incrementally projects them on an auxiliary stream, adds fused KV RMSNorm/RoPE/scatter with FP8 draft-cache support, fuses prepare-decode bookkeeping, enables FA4 non-causal draft attention, and fixes the draft RoPE configuration.
- Code diff details: the DFlash drafter caches KV buffer pointers, writes per-layer KV directly into the pool, synchronizes the incremental projection with an event, and adds a fused QK-RMSNorm+RoPE Triton kernel for the draft model.
- Key code excerpts:

```diff
+_fused_norm_rope_scatter_kernel[(total_ctx, num_kv_heads, n_layers)](
+    kv, k_norm_weight, eps, cos_sin_cache, positions, loc, k_ptrs, v_ptrs, ...)
+self.drafter._prepare_incremental_proj(
+    ctx.input_num_tokens, positions, out_cache_loc)
```

- Reviewed files: runtime: `execution/drafter/{dflash,_dflash_fused_kv}.py`, `cache_loc_kernel.py`, CUDA-graph/model executor/runner, `models/{dflash,qwen3_5}.py`, MHA config; kernels/tests: FlashAttention registry, `layernorm/triton.py`, `test_layernorm.py`.
- Risk and verification: record target/draft checkpoints, draft attention backend, KV dtype, captured layers, and speculative token count; validate FP8 scales, non-causal FA4 windowing, auxiliary-stream ordering, CUDA-graph replay, and acceptance rate after RoPE correction.

### PR #766 - Fix Qwen3.5 FP8 weight loading

- Link: https://github.com/lightseekorg/tokenspeed/pull/766
- Status/date: merged / 2026-07-22
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 283-line diff, 2 files, +118/-67.
- Motivation: Qwen3.5-35B-A3B FP8 checkpoints quantize GDN `qkv/z` but leave `b/a` in BF16, so packing all six projections into one quantized linear produced garbled output; the MoE kernel also received an unsharded intermediate size under TP.
- Key implementation: derives quantization per GDN projection group from `ignored_layers`, splits the module into `in_proj_qkvz` and `in_proj_ba` only when their quantization differs, adapts checkpoint mappings to the selected layout, and derives MoE intermediate size from the TP-sharded `w2_weight`.
- Code diff details: fully quantized or fully unquantized checkpoints retain the single fused projection; only mixed checkpoints pay the split-linear cost.
- Key code excerpts:

```diff
+self._split_in_proj = quant_config is not None and (qkvz_unquant != ba_unquant)
+if self._split_in_proj:
+    self.in_proj_qkvz = MergedColumnParallelLinear(..., quant_config=...)
+    self.in_proj_ba = MergedColumnParallelLinear(..., quant_config=...)
+intermediate_size = w.w2_weight.shape[-1]
```

- Reviewed files: runtime: `python/tokenspeed/runtime/models/qwen3_5.py`; kernel wrapper: `tokenspeed_kernel/ops/moe/flashinfer/trtllm_fp8.py`.
- Risk and verification: test BF16, all-FP8, and mixed ignored-layer checkpoints under TP, EP, and hybrid layouts; the PR reports correct Qwen3.5-35B-A3B-FP8 output on TP2 and DP4EP4.

### PR #780 - Harden Qwen3.5 multi-node execution

- Link: https://github.com/lightseekorg/tokenspeed/pull/780
- Status/date: merged / 2026-07-28
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 555-line diff, 8 files, +347/-35.
- Motivation: Qwen3.5 multi-node layouts could select CUDA-IPC/symmetric-memory collectives across nodes and could reuse persistent pinned Mamba/GDN staging buffers while an overlapped H2D copy was still in flight.
- Key implementation: detects process groups spanning nodes and forces NCCL for all-reduce, all-gather, token collectives, and logits gather/argmax; moves Mamba indices to per-step bulk pinned staging and documents consistent NCCL settings.
- Code diff details: node-local groups retain low-latency RSAG/custom paths, while only cross-node groups fall back; tests assert backend selection and non-reused staging behavior.
- Key code excerpts:

```diff
+if self._group_spans_nodes(group):
+    return self._nccl.token_all_gather(tensor, group, scattered_num_tokens)
+(...mamba staging...) = self._bulk_pinned(
+    (batch_size, torch.int32), (batch_size, torch.int32), ...)
```

- Reviewed files: runtime: `distributed/comm_backend/auto.py`, `execution/{input_buffer,model_executor}.py`, `layers/logits_processor.py`; tests/docs: `test_comm_ops.py`, `test_input_buffer_mamba_staging.py`, `test_logits_processor.py`, `docs/serving/parallelism.md`.
- Risk and verification: validate node-rank mapping and NCCL transport consistency, ensure custom collectives remain node-local, and run overlap/CUDA-graph replay with GDN state copy-on-write plus TP logits paths.

### PR #181 - feat(qwen3): add Qwen3 MoE causal LM support

- Link: https://github.com/lightseekorg/tokenspeed/pull/181
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +610/-0, 790 cached patch lines.
- Motivation: TokenSpeed needed a Qwen3 MoE causal LM runtime. The Qwen3 MoE path is close to Qwen3.5 MoE and exposes reusable sparse-MoE model patterns.
- Key implementation: adds `Qwen3MoeConfig`, `Qwen3MoeForCausalLM`, HF config mapping, and model tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+from tokenspeed.runtime.models.qwen3_5 import Qwen3_5MoeSparseMoeBlock
+class Qwen3MoeForCausalLM(nn.Module):
```

- Reviewed files: `docs/recipes/models.md`, `qwen3_moe_config.py`, `qwen3_moe.py`, `hf_transformers_utils.py`, `test_qwen3_moe_models.py`
- Risk and verification: useful for SGLang as MoE runtime evidence, especially HF config mapping and weight naming.

### PR #189 - Fix Qwen3 FP8 MoE activation scale layout

- Link: https://github.com/lightseekorg/tokenspeed/pull/189
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +107/-14, 174 cached patch lines.
- Motivation: the FP8 MoE activation scale layout did not match the fused MoE kernel contract.
- Key implementation: updates scale handling in `fused_moe_kernel` / `invoke_fused_moe_kernel` and extends Triton MoE tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+def _normalize_fp8_group_scale_layout(
+    A: torch.Tensor,
+    A_scale: torch.Tensor,
+    expected_scale_k: int,
+) -> torch.Tensor:
+            A_scale = _normalize_fp8_group_scale_layout(A, A_scale, expected_scale_k)
```

- Reviewed files: `ops/moe/triton.py`, `test_moe_triton.py`
- Risk and verification: compare scale layout before reading MoE profiler wins across SGLang and TokenSpeed.

### PR #196 - perf(qwen3.5): fuse q/k GemmaRMSNorm into one Triton launch

- Link: https://github.com/lightseekorg/tokenspeed/pull/196
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +87/-12, 155 cached patch lines.
- Motivation: the old attention prep normalized Q and K through separate `GemmaRMSNorm` launches.
- Key implementation: replaces the two-launch path inside `_apply_qk_norm` with `qk_rmsnorm` and adds layernorm tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
-q = self.q_norm(q)
-k = self.k_norm(k)
+q, k = qk_rmsnorm(q, k, q_gamma, k_gamma, eps)
```

- Reviewed files: `runtime/models/qwen3_5.py`, `test_layernorm.py`
- Risk and verification: a direct competitor clue for SGLang Qwen3.5 norm fusion; check launch count, BF16 rounding order, and Q/K strides.

### PR #198 - perf(qwen3.5): fuse attn_output_gate sigmoid+mul

- Link: https://github.com/lightseekorg/tokenspeed/pull/198
- Status/date: merged / 2026-05-23
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 3 files, +234/-3, 323 cached patch lines.
- Motivation: the `attn_output_gate` path used reshape, sigmoid, and multiply as separate work.
- Key implementation: adds Triton `sigmoid_mul` and consumes the 3D strided gate view produced by `torch.chunk`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
-attn_output = attn_output * torch.sigmoid(gate)
+sigmoid_mul(attn_output, gate)
```

- Reviewed files: `runtime/models/qwen3_5.py`, `activation/triton.py`, `test_activation.py`
- Risk and verification: if SGLang traces show a sigmoid/mul/copy cluster, this is the closest TokenSpeed precedent.

### PR #309 - fix(dp): fix qwen 3.5 data parallel bug

- Link: https://github.com/lightseekorg/tokenspeed/pull/309
- Status/date: merged / 2026-06-01
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +13/-1, 46 cached patch lines.
- Motivation: DP vocab-parallel embedding masking differed from the TP>1 mask path.
- Key implementation: clamps masked input before embedding lookup and fixes a distributed comm rank path.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+masked_input = torch.clamp(masked_input, min=0, max=self.num_embeddings - 1)
```

- Reviewed files: `comm_manager.py`, `vocab_parallel_embedding.py`
- Risk and verification: SGLang/TokenSpeed DP comparisons should check token masking and rank layout before blaming kernels.

### PR #400 - ci(qwen3.5): add Qwen3.5 397B PD CI (1p1d)

- Link: https://github.com/lightseekorg/tokenspeed/pull/400
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +169/-0, 345 cached patch lines.
- Motivation: TokenSpeed made `nvidia/Qwen3.5-397B-A17B-NVFP4` prefill/decode disaggregation a fixed CI lane.
- Key implementation: launches a PD serve script and validates `/v1/models` plus `/v1/chat/completions`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+MODEL = os.environ.get("MODEL", "nvidia/Qwen3.5-397B-A17B-NVFP4")
+pytest test/runtime/distributed/test_qwen35_pd_1p1d.py -v
```

- Reviewed files: PD CI YAML, `test_qwen35_pd_1p1d.py`
- Risk and verification: treat PD/disaggregation as a separate workload from monolithic serving in the SOTA loop.

### PR #354 - feat(video): generalize multimodal runtime support and add Qwen3.5 video

- Link: https://github.com/lightseekorg/tokenspeed/pull/354
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 19 files, +982/-266, 2,500 cached patch lines.
- Motivation: Qwen3.5 video/image serving needed unified multimodal runtime support, encoder budgets, MRoPE decode positions, and CUDA graph capture.
- Key implementation: introduces multimodal adapters, budget graphs, metadata sequence budgets, and MRoPE position-delta handling.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+mrope_position_delta_scalar: Optional[int] = None
+        if not is_prefill:
+            return self._build_decode_mrope_positions_override(
```

- Reviewed files: generation/output processors, input processor, model executor, `runtime/multimodal/*`, `qwen3_5.py`, `kimi_k25.py`
- Risk and verification: profile encoder capture, MRoPE construction, output D2H, and LLM decode separately.

### PR #456 - perf(kernel): optimize Qwen vision QKV rotary layout

- Link: https://github.com/lightseekorg/tokenspeed/pull/456
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 6 files, +452/-35, 816 cached patch lines.
- Motivation: Qwen3.5 VLM vision attention had packed-QKV rotary split/materialization overhead.
- Key implementation: adds `packed_qkv_neox_rotary`, wires it into multimodal encoder attention, and adds a Blackwell Qwen3.5 VLM smoke test.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+            q, k, v = packed_qkv_neox_rotary(
+                qkv,
+                self.q_size,
+__all__ = ["packed_qkv_complex_rotary", "packed_qkv_neox_rotary"]
```

- Reviewed files: `mm_encoder_attention.py`, `qwen3_5.py`, `qkv_rotary.py`, `test_qwen35_vlm_e2e.py`, `trtllm_fp8.py`
- Risk and verification: for SGLang VLM work, first check whether QKV split, rotary, and V copy are already fused.
