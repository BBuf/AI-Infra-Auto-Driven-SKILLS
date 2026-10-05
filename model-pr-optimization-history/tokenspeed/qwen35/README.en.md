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

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/configs/qwen3_5_config.py` | [#485](https://github.com/lightseekorg/tokenspeed/pull/485) |
| `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` | [#485](https://github.com/lightseekorg/tokenspeed/pull/485) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` | [#1096](https://github.com/lightseekorg/tokenspeed/pull/1096) |
| `python/tokenspeed/runtime/models/qwen3_5.py` | [#196](https://github.com/lightseekorg/tokenspeed/pull/196), [#198](https://github.com/lightseekorg/tokenspeed/pull/198), [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#228](https://github.com/lightseekorg/tokenspeed/pull/228), [#354](https://github.com/lightseekorg/tokenspeed/pull/354), [#429](https://github.com/lightseekorg/tokenspeed/pull/429), [#485](https://github.com/lightseekorg/tokenspeed/pull/485), [#510](https://github.com/lightseekorg/tokenspeed/pull/510), [#766](https://github.com/lightseekorg/tokenspeed/pull/766), [#828](https://github.com/lightseekorg/tokenspeed/pull/828), [#928](https://github.com/lightseekorg/tokenspeed/pull/928), [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `python/tokenspeed/runtime/models/qwen3_5_moe.py` | [#235](https://github.com/lightseekorg/tokenspeed/pull/235), [#433](https://github.com/lightseekorg/tokenspeed/pull/433), [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `python/tokenspeed/runtime/models/qwen3_5_nextn.py` | [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#429](https://github.com/lightseekorg/tokenspeed/pull/429), [#582](https://github.com/lightseekorg/tokenspeed/pull/582), [#777](https://github.com/lightseekorg/tokenspeed/pull/777), [#828](https://github.com/lightseekorg/tokenspeed/pull/828) |
| `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) |
| `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` | [#549](https://github.com/lightseekorg/tokenspeed/pull/549) |
| `test/ci/eval/qwen3.5-35b-a3b-fp8-deepep-tp2dp2ep4-evalscope-gsm8k.yaml` | no direct PR-number commit |
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
| `test/runtime/models/test_qwen35_vlm_e2e.py` | no direct PR-number commit |
| `test/runtime/models/test_qwen3_5_fused_qkvzba.py` | [#928](https://github.com/lightseekorg/tokenspeed/pull/928), [#1043](https://github.com/lightseekorg/tokenspeed/pull/1043) |
| `test/runtime/models/test_qwen3_5_shared_expert_dp.py` | [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `test/runtime/test_qwen35_gdn_replay.py` | [#1096](https://github.com/lightseekorg/tokenspeed/pull/1096) |
| `test/runtime/test_qwen35_nextn_quantization.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 30
- Extra PRs preserved from existing docs: 5
- Total PRs in this document: 35
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

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

### PR #195 - Add qwen3.5-397b-a17b nvfp4 perf CI task

- Link: https://github.com/lightseekorg/tokenspeed/pull/195
- Status/date: merged / 2026-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; associated commits `5a2db4b1d4bf`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +106/-3, 126 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` added +98/-0 (98 lines); hunks: -0,0 +1,98.
- Code diff details:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` added +98/-0 (98 lines); hunks: -0,0 +1,98
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` added +98/-0
- Risk and verification: The diff ships test coverage in `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`, `test/ci_system/pipeline.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #228 - Perf[Qwen3.5]: some kernel fuse optimizations.

- Link: https://github.com/lightseekorg/tokenspeed/pull/228
- Status/date: merged / 2026-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5.py`; associated commits `5454c826a876`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +520/-81, 807 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5.py` modified +86/-71 (157 lines); hunks: -31,7 +31,10; -43,7 +46,6; symbols: __init__, _get_split_sizes_for_param, _make_packed_weight_loader, touching `__init__, _get_split_sizes_for_param, _make_packed_weight_loader`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +86/-71 (157 lines); hunks: -31,7 +31,10; -43,7 +46,6; symbols: __init__, _get_split_sizes_for_param, _make_packed_weight_loader
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +86/-71
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/ops/test_layernorm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #235 - perf[Qwen3.5]: fuse small kernels in MoE block.

- Link: https://github.com/lightseekorg/tokenspeed/pull/235
- Status/date: merged / 2026-05-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5_moe.py`; associated commits `acf6ba45433b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +176/-17, 245 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +19/-15 (34 lines); hunks: -24,7 +24,7; -263,14 +263,6 @@ def _forward_tp(; symbols: _forward_tp, _forward_deepep, touching `_forward_tp, _forward_deepep`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +19/-15 (34 lines); hunks: -24,7 +24,7; -263,14 +263,6 @@ def _forward_tp(; symbols: _forward_tp, _forward_deepep
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +19/-15
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/ops/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #229 - perf(qwen3.5): reduce prefill memcpy sync and mamba update overhead

- Link: https://github.com/lightseekorg/tokenspeed/pull/229
- Status/date: merged / 2026-05-24
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; associated commits `8d2d78292dd1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +458/-140, 851 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +4/-4 (8 lines); hunks: -92,7 +92,7 @@ report:; `python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` modified +142/-47 (189 lines); hunks: -41,6 +41,10; -55,6 +59,7 @@ class MambaForwardMetadata:; symbols: MambaForwardMetadata, get_mamba_indices, _build_mtp_output_indices_kernel, get_mtp_output_indices, touching `MambaForwardMetadata, get_mamba_indices, _build_mtp_output_indices_kernel`; `python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py` modified +94/-6 (100 lines); hunks: -168,26 +168,29 @@ def _mamba_state_snapshot_kernel(; -242,6 +245,91 @@ def fused_mamba_state_snapshot(; symbols: _mamba_state_snapshot_kernel, fused_mamba_state_snapshot, fused_mamba_state_copy, _mamba_state_zero_kernel, touching `_mamba_state_snapshot_kernel, fused_mamba_state_snapshot, fused_mamba_state_copy`; `python/tokenspeed/runtime/layers/attention/backends/trtllm.py` modified +81/-6 (87 lines); hunks: -30,6 +30,8; -562,6 +564,37 @@ def _init_multi_token_metadata_capture(; symbols: _init_multi_token_metadata_capture, _replay_gather_page_table, init_forward_metadata_replay_cuda_graph, touching `_init_multi_token_metadata_capture, _replay_gather_page_table, init_forward_metadata_replay_cuda_graph`.
- Code diff details:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +4/-4 (8 lines); hunks: -92,7 +92,7 @@ report:
  - `python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` modified +142/-47 (189 lines); hunks: -41,6 +41,10; -55,6 +59,7 @@ class MambaForwardMetadata:; symbols: MambaForwardMetadata, get_mamba_indices, _build_mtp_output_indices_kernel, get_mtp_output_indices
  - `python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py` modified +94/-6 (100 lines); hunks: -168,26 +168,29 @@ def _mamba_state_snapshot_kernel(; -242,6 +245,91 @@ def fused_mamba_state_snapshot(; symbols: _mamba_state_snapshot_kernel, fused_mamba_state_snapshot, fused_mamba_state_copy, _mamba_state_zero_kernel
  - `python/tokenspeed/runtime/layers/attention/backends/trtllm.py` modified +81/-6 (87 lines); hunks: -30,6 +30,8; -562,6 +564,37 @@ def _init_multi_token_metadata_capture(; symbols: _init_multi_token_metadata_capture, _replay_gather_page_table, init_forward_metadata_replay_cuda_graph
  - `python/tokenspeed/runtime/execution/model_executor.py` modified +49/-20 (69 lines); hunks: -502,6 +502,47 @@ def accumulate_decode_stats(self, results: ModelExecutionRe...; -549,27 +590,15 @@ def _snapshot_mamba_checkpoints(; symbols: accumulate_decode_stats, _compute_mtp_snapshot_indices, _snapshot_mamba_checkpoints
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +4/-4
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` modified +142/-47; `python/tokenspeed/runtime/layers/attention/linear/mamba_state_scatter_triton.py` modified +94/-6; `python/tokenspeed/runtime/layers/attention/backends/trtllm.py` modified +81/-6; `python/tokenspeed/runtime/execution/model_executor.py` modified +49/-20; `python/tokenspeed/runtime/layers/attention/linear/index.py` modified +47/-10; `python/tokenspeed/runtime/layers/attention/linear/causal_conv1d.py` modified +18/-21
- Risk and verification: The diff ships test coverage in `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #241 - chore(qwen3.5): adjust qwen perf tps threshold

- Link: https://github.com/lightseekorg/tokenspeed/pull/241
- Status/date: merged / 2026-05-24
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; associated commits `2859f547dc8d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 8 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -93,6 +93,6 @@ perf_threshold: 0.9.
- Code diff details:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -93,6 +93,6 @@ perf_threshold: 0.9
- Key code excerpts:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml
@@ -93,6 +93,6 @@ perf_threshold: 0.9
-  2:  [440, 17800]
+  2:  [420, 17800]
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #245 - ci(perf-qwen3.5-agentic): route to gb200-4gpu-perf runner

- Link: https://github.com/lightseekorg/tokenspeed/pull/245
- Status/date: merged / 2026-05-25
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; associated commits `282f80a6fdd0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -6,7 +6,7 @@ triggers:.
- Code diff details:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1 (2 lines); hunks: -6,7 +6,7 @@ triggers:
- Key code excerpts:

```diff
diff -- test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml
@@ -6,7 +6,7 @@ triggers:
-    - gb200-4gpu
+    - gb200-4gpu-perf
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #250 - ci(perf): add Qwen3.5-NVFP4 agentic perf on b200-8gpu

- Link: https://github.com/lightseekorg/tokenspeed/pull/250
- Status/date: merged / 2026-05-25
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml`; associated commits `9683592b32c0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +93/-0, 94 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` added +93/-0 (93 lines); hunks: -0,0 +1,93.
- Code diff details:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` added +93/-0 (93 lines); hunks: -0,0 +1,93
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml` added +93/-0
- Risk and verification: The diff ships test coverage in `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #264 - ci(perf): add 1m perf bench for qwen3.5

- Link: https://github.com/lightseekorg/tokenspeed/pull/264
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml`; associated commits `6b792807c1c9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +353/-0, 355 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` added +87/-0 (87 lines); hunks: -0,0 +1,87.
- Code diff details:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` added +87/-0 (87 lines); hunks: -0,0 +1,87
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml` added +87/-0
- Risk and verification: The diff ships test coverage in `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-longctx.yaml`, `test/long_context_benchmark/tokenspeed/collect_outputs.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #217 - perf(Spec Decode): skip dead-position compute in draft catch-up step(decode)

- Link: https://github.com/lightseekorg/tokenspeed/pull/217
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`; associated commits `27e99fb86fa7`, `a700be07ddef`, `a9bc2188501c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +109/-38, 286 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -747,6 +747,10 @@ def self_attention(; -774,6 +778,9 @@ def forward(; symbols: self_attention, forward, touching `self_attention, forward`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +7/-0 (7 lines); hunks: -747,6 +747,10 @@ def self_attention(; -774,6 +778,9 @@ def forward(; symbols: self_attention, forward
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/distributed/comm_manager.py`, `python/tokenspeed/runtime/execution/context.py`, `python/tokenspeed/runtime/execution/drafter/eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #257 - ci(perf): add qwen3.5 agentic perf ci bs16 case

- Link: https://github.com/lightseekorg/tokenspeed/pull/257
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; associated commits `ca4641e39495`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +3/-2, 16 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +3/-2 (5 lines); hunks: -74,8 +74,8 @@ perf:; -96,3 +96,4 @@ perf_reference:.
- Code diff details:
  - `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +3/-2 (5 lines); hunks: -74,8 +74,8 @@ perf:; -96,3 +96,4 @@ perf_reference:
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml` modified +3/-2
- Risk and verification: The diff ships test coverage in `test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

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

### PR #389 - Add PD Qwen3.5 HTTP worker

- Link: https://github.com/lightseekorg/tokenspeed/pull/389
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh`; associated commits `2b3c8eb15faf`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +558/-0, 560 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` added +201/-0 (201 lines); hunks: -0,0 +1,201.
- Code diff details:
  - `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` added +201/-0 (201 lines); hunks: -0,0 +1,201
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` added +201/-0
- Risk and verification: The diff ships test coverage in `test/ci_system/pd_http_worker.py`, `test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #426 - Add Qwen3.5 PD AIME25 eval CI

- Link: https://github.com/lightseekorg/tokenspeed/pull/426
- Status/date: merged / 2026-06-11
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml`; associated commits `da785595ad56`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +38/-0, 39 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` added +38/-0 (38 lines); hunks: -0,0 +1,38.
- Code diff details:
  - `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` added +38/-0 (38 lines); hunks: -0,0 +1,38
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml` added +38/-0
- Risk and verification: The diff ships test coverage in `test/ci/eval/qwen3.5-397b-a17b-nvfp4-pd-1p1d-evalscope-aime25.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #433 - perf(qwen3.5): use nvfp4_gemm_swiglu_nvfp4_quant in shared experts

- Link: https://github.com/lightseekorg/tokenspeed/pull/433
- Status/date: merged / 2026-06-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5_moe.py`; associated commits `6c4765f5cabc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +37/-1, 69 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +37/-1 (38 lines); hunks: -25,6 +25,11; -33,6 +38,7; symbols: _is_moe_layer, __init__, forward, touching `_is_moe_layer, __init__, forward`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +37/-1 (38 lines); hunks: -25,6 +25,11; -33,6 +38,7; symbols: _is_moe_layer, __init__, forward
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +37/-1
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/models/qwen3_5_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #429 - refactor(spec-decode): simplify Qwen3.5 NextN attention path for #217 (2/3)

- Link: https://github.com/lightseekorg/tokenspeed/pull/429
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`; associated commits `a700be07ddef`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +276/-114, 561 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +103/-2 (105 lines); hunks: -22,13 +22,16; -38,12 +41,110; symbols: Qwen3_5DraftAttentionDecoderLayer, _attn, docstring, _apply_correction, touching `Qwen3_5DraftAttentionDecoderLayer, _attn, docstring`; `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-27 (75 lines); hunks: -710,16 +710,13 @@ def _apply_qk_norm(; -737,23 +734,48 @@ def self_attention(; symbols: _apply_qk_norm, self_attention, _project_qkv_rope, _attn, touching `_apply_qk_norm, self_attention, _project_qkv_rope`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +103/-2 (105 lines); hunks: -22,13 +22,16; -38,12 +41,110; symbols: Qwen3_5DraftAttentionDecoderLayer, _attn, docstring, _apply_correction
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-27 (75 lines); hunks: -710,16 +710,13 @@ def _apply_qk_norm(; -737,23 +734,48 @@ def self_attention(; symbols: _apply_qk_norm, self_attention, _project_qkv_rope, _attn
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +103/-2; `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-27
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/execution/drafter/eagle.py`, `python/tokenspeed/runtime/execution/model_executor.py`, `python/tokenspeed/runtime/layers/attention/backends/base.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #485 - Sanitize Qwen3.5 rope parameters

- Link: https://github.com/lightseekorg/tokenspeed/pull/485
- Status/date: merged / 2026-06-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/qwen3_5_config.py`, `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py`, `python/tokenspeed/runtime/models/qwen3_5.py`; associated commits `c9a1c7fd7c73`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +55/-22, 170 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/configs/qwen3_5_config.py` modified +18/-8 (26 lines); hunks: -25,6 +25,18; -41,16 +53,14 @@ def __init__(; symbols: _to_transformers_rope_parameters, Qwen3_5VisionConfig, __init__, Qwen3_5Config, touching `_to_transformers_rope_parameters, Qwen3_5VisionConfig, __init__`; `python/tokenspeed/runtime/models/qwen3_5.py` modified +3/-7 (10 lines); hunks: -40,6 +40,7; -591,10 +592,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` modified +3/-3 (6 lines); hunks: -90,7 +90,7 @@ class Qwen3_5BaseTextConfig(PretrainedConfig):; -201,7 +201,7 @@ def __init__(; symbols: Qwen3_5BaseTextConfig, __init__, touching `Qwen3_5BaseTextConfig, __init__`.
- Code diff details:
  - `python/tokenspeed/runtime/configs/qwen3_5_config.py` modified +18/-8 (26 lines); hunks: -25,6 +25,18; -41,16 +53,14 @@ def __init__(; symbols: _to_transformers_rope_parameters, Qwen3_5VisionConfig, __init__, Qwen3_5Config
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +3/-7 (10 lines); hunks: -40,6 +40,7; -591,10 +592,7 @@ def __init__(; symbols: __init__
  - `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` modified +3/-3 (6 lines); hunks: -90,7 +90,7 @@ class Qwen3_5BaseTextConfig(PretrainedConfig):; -201,7 +201,7 @@ def __init__(; symbols: Qwen3_5BaseTextConfig, __init__
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/configs/qwen3_5_config.py` modified +18/-8; `python/tokenspeed/runtime/models/qwen3_5.py` modified +3/-7; `python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/runtime/test_resolve_architecture.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

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


### PR #582 - ci: disable distributed_argmax on qwen3.5 MTP drafter

- Link: https://github.com/lightseekorg/tokenspeed/pull/582
- Status/date: merged / 2026-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5_nextn.py`; associated commits `547e7b700ae8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +9/-3, 40 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-1 (1 lines); hunks: -206,7 +206,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-1 (1 lines); hunks: -206,7 +206,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5_nextn.py
@@ -206,7 +206,6 @@ def __init__(
-            do_argmax=True,
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/execution/drafter/eagle.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

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

### PR #549 - ci(eval): add Qwen3.5 aggregate and EPD OCRBench coverage

- Link: https://github.com/lightseekorg/tokenspeed/pull/549
- Status/date: merged / 2026-07-18
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml`, `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml`, `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml`, `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh`, `test/runtime/distributed/test_qwen35_epd_1e1p2d.py`; associated commits `8207c4121ca1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +595/-0, 600 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh` added +248/-0 (248 lines); hunks: -0,0 +1,248; `test/runtime/distributed/test_qwen35_epd_1e1p2d.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: _truetype_font, _image_data_uri, _wait_for_server, _chat, touching `_truetype_font, _image_data_uri, _wait_for_server`; `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` added +61/-0 (61 lines); hunks: -0,0 +1,61; `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` added +37/-0 (37 lines); hunks: -0,0 +1,37.
- Code diff details:
  - `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh` added +248/-0 (248 lines); hunks: -0,0 +1,248
  - `test/runtime/distributed/test_qwen35_epd_1e1p2d.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: _truetype_font, _image_data_uri, _wait_for_server, _chat
  - `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` added +61/-0 (61 lines); hunks: -0,0 +1,61
  - `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` added +37/-0 (37 lines); hunks: -0,0 +1,37
  - `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml` added +18/-0 (18 lines); hunks: -0,0 +1,18
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh` added +248/-0; `test/runtime/distributed/test_qwen35_epd_1e1p2d.py` added +231/-0; `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml` added +61/-0; `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml` added +37/-0; `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml` added +18/-0
- Risk and verification: The diff ships test coverage in `test/ci/eval/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d-evalscope-ocr-bench.yaml`, `test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml`, `test/ci/ut/qwen3.5-122b-a10b-nvfp4-epd-1e1p2d.yaml`, `test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

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

### PR #776 - ci: disable thinking and allow longer context for qwen3.5 eval/pd cases

- Link: https://github.com/lightseekorg/tokenspeed/pull/776
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml`, `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml`; associated commits `670663cb53a2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +3/-0, 24 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -9,6 +9,7 @@ runner:; `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -10,6 +10,7 @@ runner:.
- Code diff details:
  - `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -9,6 +9,7 @@ runner:
  - `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` modified +1/-0 (1 lines); hunks: -10,6 +10,7 @@ runner:
- Key code excerpts:

```diff
diff -- test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml
@@ -9,6 +9,7 @@ runner:
+  TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN: "1"
diff -- test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml
@@ -10,6 +10,7 @@ runner:
+  TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN: "1"
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml` modified +1/-0; `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml`, `test/ci/eval/qwen3.5-397b-a17b-nvfp4-evalscope-aime25.yaml`, `test/ci_system/pd_http_worker.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #777 - fix(qwen3.5): allow MTP to be quantized

- Link: https://github.com/lightseekorg/tokenspeed/pull/777
- Status/date: merged / 2026-07-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5_nextn.py`; associated commits `8e2911dad09e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +0/-4, 11 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-4 (4 lines); hunks: -163,10 +163,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-4 (4 lines); hunks: -163,10 +163,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_5_nextn.py
@@ -163,10 +163,6 @@ def __init__(
-        # The MTP model is unquantized in the nvfp4 checkpoint.
-        if quant_config and quant_config.get_name() == "nvfp4":
-            quant_config = None
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +0/-4
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/models/qwen3_5_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

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

### PR #928 - fix(qwen3.5): support non-pow2 ratios in fused_qkvzba_split_reshape_cat_contiguous path

- Link: https://github.com/lightseekorg/tokenspeed/pull/928
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5.py`, `test/runtime/models/test_qwen3_5_fused_qkvzba.py`; associated commits `a1c09236006d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +334/-36, 395 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_qwen3_5_fused_qkvzba.py` added +282/-0 (282 lines); hunks: -0,0 +1,282; symbols: _legacy_wide_arange_kernel, _legacy_launch, _make_inputs, FusedQkvzbaTest, touching `_legacy_wide_arange_kernel, _legacy_launch, _make_inputs`; `python/tokenspeed/runtime/models/qwen3_5.py` modified +52/-36 (88 lines); hunks: -467,7 +467,7 @@ def forward(; -1765,26 +1765,6 @@ def fused_qkvzba_split_reshape_cat_contiguous_kernel(; symbols: forward, fused_qkvzba_split_reshape_cat_contiguous_kernel, touching `forward, fused_qkvzba_split_reshape_cat_contiguous_kernel`.
- Code diff details:
  - `test/runtime/models/test_qwen3_5_fused_qkvzba.py` added +282/-0 (282 lines); hunks: -0,0 +1,282; symbols: _legacy_wide_arange_kernel, _legacy_launch, _make_inputs, FusedQkvzbaTest
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +52/-36 (88 lines); hunks: -467,7 +467,7 @@ def forward(; -1765,26 +1765,6 @@ def fused_qkvzba_split_reshape_cat_contiguous_kernel(; symbols: forward, fused_qkvzba_split_reshape_cat_contiguous_kernel
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_qwen3_5_fused_qkvzba.py` added +282/-0
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +52/-36
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_qwen3_5_fused_qkvzba.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #828 - feat(qwen3.5): support text-only qwen3.5 config

- Link: https://github.com/lightseekorg/tokenspeed/pull/828
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_nextn.py`; associated commits `4498a6149b19`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +63/-4, 142 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-3 (51 lines); hunks: -39,6 +39,7; -1134,7 +1135,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Qwen3_5MoeForCausalLM, Qwen3_5MoeModel, for, touching `load_weights, Qwen3_5MoeForCausalLM, Qwen3_5MoeModel`; `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +5/-0 (5 lines); hunks: -436,7 +436,12 @@ def __init__(; symbols: __init__, Qwen3_5MoeForCausalLMNextN, touching `__init__, Qwen3_5MoeForCausalLMNextN`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-3 (51 lines); hunks: -39,6 +39,7; -1134,7 +1135,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Qwen3_5MoeForCausalLM, Qwen3_5MoeModel, for
  - `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +5/-0 (5 lines); hunks: -436,7 +436,12 @@ def __init__(; symbols: __init__, Qwen3_5MoeForCausalLMNextN
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_5.py` modified +48/-3; `python/tokenspeed/runtime/models/qwen3_5_nextn.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/configs/__init__.py`, `python/tokenspeed/runtime/layers/attention/registry.py`, `python/tokenspeed/runtime/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #1043 - [ci][stability] Stabilize Qwen3.5 performance guards

- Link: https://github.com/lightseekorg/tokenspeed/pull/1043
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `test/runtime/models/test_qwen3_5_fused_qkvzba.py`; associated commits `059b512a18e6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-4, 31 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_qwen3_5_fused_qkvzba.py` modified +11/-4 (15 lines); hunks: -234,10 +234,13 @@ def test_perf_no_regression_vs_legacy_wide_arange(self):; -268,8 +271,12 @@ def torch_fallback():; symbols: test_perf_no_regression_vs_legacy_wide_arange, torch_fallback, touching `test_perf_no_regression_vs_legacy_wide_arange, torch_fallback`.
- Code diff details:
  - `test/runtime/models/test_qwen3_5_fused_qkvzba.py` modified +11/-4 (15 lines); hunks: -234,10 +234,13 @@ def test_perf_no_regression_vs_legacy_wide_arange(self):; -268,8 +271,12 @@ def torch_fallback():; symbols: test_perf_no_regression_vs_legacy_wide_arange, torch_fallback
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_qwen3_5_fused_qkvzba.py` modified +11/-4
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_qwen3_5_fused_qkvzba.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1096 - feat: add Qwen3.5 GDN ReplaySSM

- Link: https://github.com/lightseekorg/tokenspeed/pull/1096
- Status/date: merged / 2026-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py`, `test/runtime/test_qwen35_gdn_replay.py`; associated commits `cd78b4e0b238`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +1747/-85, 2276 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` modified +15/-1 (16 lines); hunks: -1,5 +1,7; -162,6 +164,7 @@ def prepare_qwen35_cache(; symbols: prepare_qwen35_cache, touching `prepare_qwen35_cache`; `test/runtime/test_qwen35_gdn_replay.py` added +409/-0 (409 lines); hunks: -0,0 +1,409; symbols: _config, _make_backend, _inputs, _forward_verify, touching `_config, _make_backend, _inputs`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` modified +15/-1 (16 lines); hunks: -1,5 +1,7; -162,6 +164,7 @@ def prepare_qwen35_cache(; symbols: prepare_qwen35_cache
  - `test/runtime/test_qwen35_gdn_replay.py` added +409/-0 (409 lines); hunks: -0,0 +1,409; symbols: _config, _make_backend, _inputs, _forward_verify
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py` modified +15/-1
  - tests: `test/runtime/test_qwen35_gdn_replay.py` added +409/-0
- Risk and verification: The diff ships test coverage in `test/runtime/layers/test_gdn_qkv_split_fused.py`, `test/runtime/test_cache_setup.py`, `test/runtime/test_cli_config_compat.py`, `test/runtime/test_gdn_state_paging.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1713 - fix(qwen3.5 moe): shard Qwen shared experts over the MoE reduction group

- Link: https://github.com/lightseekorg/tokenspeed/pull/1713
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_5.py`, `python/tokenspeed/runtime/models/qwen3_5_moe.py`, `test/runtime/models/test_qwen3_5_shared_expert_dp.py`; associated commits `0722ff403d8b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +88/-1, 169 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_qwen3_5_shared_expert_dp.py` modified +71/-0 (71 lines); hunks: -11,7 +11,9; -36,6 +38,7 @@ def _mlp(world_size: int, replicate: bool) -> Qwen3_5MoeMLP:; symbols: _mlp, forward, test_shared_expert_shards_match_moe_reduction, test_dense_dp_and_deepep_shared_still_replicate, touching `_mlp, forward, test_shared_expert_shards_match_moe_reduction`; `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +13/-1 (14 lines); hunks: -23,6 +23,8; -114,6 +116,8 @@ def __init__(; symbols: __init__, touching `__init__`; `python/tokenspeed/runtime/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -592,6 +592,7 @@ def __init__(; -773,6 +774,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `test/runtime/models/test_qwen3_5_shared_expert_dp.py` modified +71/-0 (71 lines); hunks: -11,7 +11,9; -36,6 +38,7 @@ def _mlp(world_size: int, replicate: bool) -> Qwen3_5MoeMLP:; symbols: _mlp, forward, test_shared_expert_shards_match_moe_reduction, test_dense_dp_and_deepep_shared_still_replicate
  - `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +13/-1 (14 lines); hunks: -23,6 +23,8; -114,6 +116,8 @@ def __init__(; symbols: __init__
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -592,6 +592,7 @@ def __init__(; -773,6 +774,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_qwen3_5_shared_expert_dp.py` modified +71/-0
  - runtime: `python/tokenspeed/runtime/models/qwen3_5_moe.py` modified +13/-1; `python/tokenspeed/runtime/models/qwen3_5.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_qwen3_5_shared_expert_dp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
