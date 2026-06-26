# TokenSpeed Qwen3.5 Model PR Optimization History

## 2026-06-26 Latest Source Scan

Rechecked TokenSpeed upstream `lightseekorg/tokenspeed@5aedf69d6b476baa65571011de6ea60fd5a238a8` against the tracked files listed below.
The file-level match used a GitHub mirror `git log --name-only`; PR titles, links, and merge times were batch-verified through the GitHub GraphQL Pull Request API. Previous freshness anchor: `2026-06-26`.

Result: no additional PR-numbered merges touched the tracked files beyond the existing timeline/backfill rows.

## 2026-06-26 PR Backfill Audit

Checked against TokenSpeed upstream `HEAD@5aedf69d6b476baa65571011de6ea60fd5a238a8`. This page follows the same structure as the SGLang/vLLM histories: model-relevant PR timeline, reviewed diffs, implementation files, short code excerpts, and validation risks.

Filter used in this pass: merged GitHub PRs whose title or files matched `Qwen3.5`, `qwen3_5`, `Qwen3Moe`, `VLM`, `PD`, `moe`, `activation`, `rotary`, or `flashinfer_trtllm`. Pure formatting and unrelated infrastructure PRs were excluded.

## Model Implementation File Coverage

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

## PR Coverage Overview

- Reviewed PRs: 8
- Diff source: `gh pr diff` / GitHub PR patches cached under `/tmp/model_pr_diffs/tokenspeed/pr*.diff`
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

### PR #181 - feat(qwen3): add Qwen3 MoE causal LM support

- Link: https://github.com/lightseekorg/tokenspeed/pull/181
- State/time: merged / 2026-05-19
- Diff coverage: 5 files, +610/-0, 790 cached patch lines.
- Motivation: TokenSpeed needed a Qwen3 MoE causal LM runtime. The Qwen3 MoE path is close to Qwen3.5 MoE and exposes reusable sparse-MoE model patterns.
- Key implementation: adds `Qwen3MoeConfig`, `Qwen3MoeForCausalLM`, HF config mapping, and model tests.
- Code excerpt:

```diff
+from tokenspeed.runtime.models.qwen3_5 import Qwen3_5MoeSparseMoeBlock
+class Qwen3MoeForCausalLM(nn.Module):
```

- Reviewed files: `docs/recipes/models.md`, `qwen3_moe_config.py`, `qwen3_moe.py`, `hf_transformers_utils.py`, `test_qwen3_moe_models.py`
- Validation/risk: useful for SGLang as MoE runtime evidence, especially HF config mapping and weight naming.

### PR #189 - Fix Qwen3 FP8 MoE activation scale layout

- Link: https://github.com/lightseekorg/tokenspeed/pull/189
- State/time: merged / 2026-05-20
- Diff coverage: 2 files, +107/-14, 174 cached patch lines.
- Motivation: the FP8 MoE activation scale layout did not match the fused MoE kernel contract.
- Key implementation: updates scale handling in `fused_moe_kernel` / `invoke_fused_moe_kernel` and extends Triton MoE tests.
- Code excerpt:

```diff
+def _normalize_fp8_group_scale_layout(
+    A: torch.Tensor,
+    A_scale: torch.Tensor,
+    expected_scale_k: int,
+) -> torch.Tensor:
+            A_scale = _normalize_fp8_group_scale_layout(A, A_scale, expected_scale_k)
```

- Reviewed files: `ops/moe/triton.py`, `test_moe_triton.py`
- Validation/risk: compare scale layout before reading MoE profiler wins across SGLang and TokenSpeed.

### PR #196 - perf(qwen3.5): fuse q/k GemmaRMSNorm into one Triton launch

- Link: https://github.com/lightseekorg/tokenspeed/pull/196
- State/time: merged / 2026-05-22
- Diff coverage: 2 files, +87/-12, 155 cached patch lines.
- Motivation: the old attention prep normalized Q and K through separate `GemmaRMSNorm` launches.
- Key implementation: replaces the two-launch path inside `_apply_qk_norm` with `qk_rmsnorm` and adds layernorm tests.
- Code excerpt:

```diff
-q = self.q_norm(q)
-k = self.k_norm(k)
+q, k = qk_rmsnorm(q, k, q_gamma, k_gamma, eps)
```

- Reviewed files: `runtime/models/qwen3_5.py`, `test_layernorm.py`
- Validation/risk: a direct competitor clue for SGLang Qwen3.5 norm fusion; check launch count, BF16 rounding order, and Q/K strides.

### PR #198 - perf(qwen3.5): fuse attn_output_gate sigmoid+mul

- Link: https://github.com/lightseekorg/tokenspeed/pull/198
- State/time: merged / 2026-05-23
- Diff coverage: 3 files, +234/-3, 323 cached patch lines.
- Motivation: the `attn_output_gate` path used reshape, sigmoid, and multiply as separate work.
- Key implementation: adds Triton `sigmoid_mul` and consumes the 3D strided gate view produced by `torch.chunk`.
- Code excerpt:

```diff
-attn_output = attn_output * torch.sigmoid(gate)
+sigmoid_mul(attn_output, gate)
```

- Reviewed files: `runtime/models/qwen3_5.py`, `activation/triton.py`, `test_activation.py`
- Validation/risk: if SGLang traces show a sigmoid/mul/copy cluster, this is the closest TokenSpeed precedent.

### PR #309 - fix(dp): fix qwen 3.5 data parallel bug

- Link: https://github.com/lightseekorg/tokenspeed/pull/309
- State/time: merged / 2026-06-01
- Diff coverage: 2 files, +13/-1, 46 cached patch lines.
- Motivation: DP vocab-parallel embedding masking differed from the TP>1 mask path.
- Key implementation: clamps masked input before embedding lookup and fixes a distributed comm rank path.
- Code excerpt:

```diff
+masked_input = torch.clamp(masked_input, min=0, max=self.num_embeddings - 1)
```

- Reviewed files: `comm_manager.py`, `vocab_parallel_embedding.py`
- Validation/risk: SGLang/TokenSpeed DP comparisons should check token masking and rank layout before blaming kernels.

### PR #400 - ci(qwen3.5): add Qwen3.5 397B PD CI (1p1d)

- Link: https://github.com/lightseekorg/tokenspeed/pull/400
- State/time: merged / 2026-06-09
- Diff coverage: 2 files, +169/-0, 345 cached patch lines.
- Motivation: TokenSpeed made `nvidia/Qwen3.5-397B-A17B-NVFP4` prefill/decode disaggregation a fixed CI lane.
- Key implementation: launches a PD serve script and validates `/v1/models` plus `/v1/chat/completions`.
- Code excerpt:

```diff
+MODEL = os.environ.get("MODEL", "nvidia/Qwen3.5-397B-A17B-NVFP4")
+pytest test/runtime/distributed/test_qwen35_pd_1p1d.py -v
```

- Reviewed files: PD CI YAML, `test_qwen35_pd_1p1d.py`
- Validation/risk: treat PD/disaggregation as a separate workload from monolithic serving in the SOTA loop.

### PR #354 - feat(video): generalize multimodal runtime support and add Qwen3.5 video

- Link: https://github.com/lightseekorg/tokenspeed/pull/354
- State/time: merged / 2026-06-23
- Diff coverage: 19 files, +982/-266, 2,500 cached patch lines.
- Motivation: Qwen3.5 video/image serving needed unified multimodal runtime support, encoder budgets, MRoPE decode positions, and CUDA graph capture.
- Key implementation: introduces multimodal adapters, budget graphs, metadata sequence budgets, and MRoPE position-delta handling.
- Code excerpt:

```diff
+mrope_position_delta_scalar: Optional[int] = None
+        if not is_prefill:
+            return self._build_decode_mrope_positions_override(
```

- Reviewed files: generation/output processors, input processor, model executor, `runtime/multimodal/*`, `qwen3_5.py`, `kimi_k25.py`
- Validation/risk: profile encoder capture, MRoPE construction, output D2H, and LLM decode separately.

### PR #456 - perf(kernel): optimize Qwen vision QKV rotary layout

- Link: https://github.com/lightseekorg/tokenspeed/pull/456
- State/time: merged / 2026-06-25
- Diff coverage: 6 files, +452/-35, 816 cached patch lines.
- Motivation: Qwen3.5 VLM vision attention had packed-QKV rotary split/materialization overhead.
- Key implementation: adds `packed_qkv_neox_rotary`, wires it into multimodal encoder attention, and adds a Blackwell Qwen3.5 VLM smoke test.
- Code excerpt:

```diff
+            q, k, v = packed_qkv_neox_rotary(
+                qkv,
+                self.q_size,
+__all__ = ["packed_qkv_complex_rotary", "packed_qkv_neox_rotary"]
```

- Reviewed files: `mm_encoder_attention.py`, `qwen3_5.py`, `qkv_rotary.py`, `test_qwen35_vlm_e2e.py`, `trtllm_fp8.py`
- Validation/risk: for SGLang VLM work, first check whether QKV split, rotary, and V copy are already fused.
