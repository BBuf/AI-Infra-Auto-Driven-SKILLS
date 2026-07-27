# TokenSpeed Kimi Model PR Optimization History

## 2026-07-28 Source Head Refresh

Rechecked TokenSpeed upstream main at `lightseekorg/tokenspeed@e41aa8b1609a9412d7ed26aa56d910828607950f`.
The two-commit range after the previous head
`d73bf0454422092f306d5575e803a08fd35ac41c` was read in full.

Result: PR #821 adds the Kimi K3 FlatKV/KDA/MLA deployment contract and is
promoted as source guidance with its hardware and validation limitations.
PR #823 only adds the corresponding README news link. The page continues to
cover the DP+EAGLE3 collective-size hang fix, Kimi incremental DFlash capture,
and Kimi-K2.7 EAGLE3.1 model semantics.

| Merged | PR | Runtime signal |
| --- | --- | --- |
| 2026-07-27 | [#821](https://github.com/lightseekorg/tokenspeed/pull/821) | Kimi K3 deployment contract |
| 2026-07-07 | [#596](https://github.com/lightseekorg/tokenspeed/pull/596) | DP + EAGLE3 mixed-step hang |
| 2026-07-25 | [#795](https://github.com/lightseekorg/tokenspeed/pull/795) | Kimi-K2.7 EAGLE3.1 |
| 2026-07-26 | [#797](https://github.com/lightseekorg/tokenspeed/pull/797) | incremental DFlash capture |

## 2026-06-27 PR Backfill Audit

Checked against TokenSpeed upstream `HEAD@d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8`. This uses a SGLang-style timeline plus per-PR diff audit cards for Kimi K2.5/K2.x.

Filter used in this pass: merged PRs whose titles or files matched `Kimi`, `kimi_k25`, `K2.5`, `NVFP4`, `MXFP4`, `MXINT4`, `lm_head`, `top_k/top_p`, `InstantTensor`, `OCR`, `FA4`, `vision`, or `MLA`. Formatting-only and unrelated infrastructure changes were excluded.

## Implementation File Coverage

| File | Related PRs |
| --- | --- |
| `python/tokenspeed/runtime/models/kimi_k25.py` | [#354](https://github.com/lightseekorg/tokenspeed/pull/354), [#418](https://github.com/lightseekorg/tokenspeed/pull/418), [#454](https://github.com/lightseekorg/tokenspeed/pull/454), [#477](https://github.com/lightseekorg/tokenspeed/pull/477) |
| `python/tokenspeed/runtime/layers/logits_processor.py` | [#126](https://github.com/lightseekorg/tokenspeed/pull/126) |
| `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/lm_head_gemm.py` | [#126](https://github.com/lightseekorg/tokenspeed/pull/126) |
| `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/fused_topk_topp/*` | [#184](https://github.com/lightseekorg/tokenspeed/pull/184) |
| `python/tokenspeed/runtime/layers/moe/weights/mxint4.py` | [#444](https://github.com/lightseekorg/tokenspeed/pull/444) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer/trtllm_mxint4.py` | [#444](https://github.com/lightseekorg/tokenspeed/pull/444) |
| `python/tokenspeed/runtime/layers/quantization/*mxfp4*` | [#454](https://github.com/lightseekorg/tokenspeed/pull/454) |
| `python/tokenspeed/runtime/model_loader/*` | [#418](https://github.com/lightseekorg/tokenspeed/pull/418) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/qkv_rotary.py` | [#477](https://github.com/lightseekorg/tokenspeed/pull/477), [#482](https://github.com/lightseekorg/tokenspeed/pull/482) |
| `test/ci/eval/kimi-k2.5-*.yaml` | [#29](https://github.com/lightseekorg/tokenspeed/pull/29), [#253](https://github.com/lightseekorg/tokenspeed/pull/253), [#476](https://github.com/lightseekorg/tokenspeed/pull/476), [#482](https://github.com/lightseekorg/tokenspeed/pull/482) |

## PR Coverage Summary

- Reviewed PRs: 10
- File trace command: `git log --name-only -- <model-files>`
- Diff source: `gh pr diff` / GitHub Pull Request files API patches cached under `/tmp/model_pr_diffs/tokenspeed/pr*.diff`
- Reviewed patch lines: 11,975
- Main TokenSpeed Kimi themes: K2.5 agentic/OCR eval lanes, fused lm_head GEMM, TopK+TopP renormalization, InstantTensor loader, MXINT4/MXFP4 MoE/quantization, and FA4 multimodal attention.

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-08 | [#29](https://github.com/lightseekorg/tokenspeed/pull/29) | merged | Add Kimi K2.5 agentic perf CI task | perf YAML, CI pipeline |
| 2026-05-13 | [#126](https://github.com/lightseekorg/tokenspeed/pull/126) | merged | perf(K2.5): Optimize lm_head | `logits_processor.py`, CUDA `lm_head_gemm` |
| 2026-05-20 | [#184](https://github.com/lightseekorg/tokenspeed/pull/184) | merged | perf(K2.5): optimize top_k_renorm_prob + top_p_renorm_prob | fused sampling CUDA, backend/server args |
| 2026-05-28 | [#253](https://github.com/lightseekorg/tokenspeed/pull/253) | merged | ci(eval): add Kimi-K2.5-NVFP4 ocr_bench task | OCR eval YAML |
| 2026-06-15 | [#418](https://github.com/lightseekorg/tokenspeed/pull/418) | merged | Add InstantTensor weight loader | loader, weight utils, `kimi_k25.py`, docs/CI |
| 2026-06-14 | [#444](https://github.com/lightseekorg/tokenspeed/pull/444) | merged | feat(moe): add trtllm mxint4 MoE path for Kimi-K2.x | MXINT4 weights and FlashInfer TRT-LLM MoE op |
| 2026-06-16 | [#454](https://github.com/lightseekorg/tokenspeed/pull/454) | merged | [AMD] Support Kimi K2.5 MXFP4 serving | MXFP4 layers, dense path, MLA backend, Kimi model |
| 2026-06-19 | [#477](https://github.com/lightseekorg/tokenspeed/pull/477) | merged | perf(kernel): Optimize Kimi Vision FA4 QKV + RoPE | Kimi model, mm attention, packed complex rotary |
| 2026-06-19 | [#482](https://github.com/lightseekorg/tokenspeed/pull/482) | merged | ci: use FA4 mm attention for Kimi OCR eval | OCR eval YAML |
| 2026-06-26 | [#476](https://github.com/lightseekorg/tokenspeed/pull/476) | merged | Add AMD Kimi MXFP4 CI job | AMD eval YAML, MLA metadata unit test |

## Per-PR Diff Audit Cards

### PR #821 - Add the Kimi K3 deployment recipe

- Link: https://github.com/lightseekorg/tokenspeed/pull/821
- Status/date: merged / 2026-07-27
- Trace source: final upstream commit
  `55a8390007e5ace17919d76e5cfaef0c68c79e25`; complete two-commit increment
  and full recipe diff read locally.
- Diff scope read: 1 file, +117/-0.
- Motivation: document the runtime and packaging constraints required to serve
  Kimi K3 instead of treating K2.5 flags as interchangeable.
- Key implementation: requires FlatKV, describes vendor-neutral KDA dispatch
  with NVIDIA FLA-derived or AMD native state layouts, selects MLA backends,
  and gives separate NVIDIA B300 and AMD gfx950 commands.
- Code diff details: the recipe adds FlatKV build/preflight constraints,
  flattened-checkpoint and writable-module-cache requirements, an NVIDIA
  `tokenspeed-situ` sidecar path with Triton fallback, and an AMD Gluon path.
- Key code excerpts:

```diff
+- K3 is FlatKV-only. Build the `tokenspeed_scheduler` extension with
+  `-DTOKENSPEED_FLAT_KVCACHE=ON`
+tokenspeed serve moonshotai/Kimi-K3 \
+  --kv-cache-dtype fp8 \
+  --tensor-parallel-size 8
```

- Reviewed files: `docs/recipes/models.md`; the following README commit only
  links the new K3 announcement.
- Risk and verification: NVIDIA uses a B300/CUDA 13 `tokenspeed-situ` sidecar
  or falls back to Triton on other platforms; the checkpoint must be flattened,
  remote-code caches must be writable, and default FP8 KV scales can affect
  accuracy. This recipe is not a measured cross-framework benchmark.

### PR #596 - Fix Kimi DP EAGLE3 mixed-step hang

- Link: https://github.com/lightseekorg/tokenspeed/pull/596
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 184-line diff, 6 files, +27/-30.
- Motivation: when DP ranks mixed EXTEND and DECODE in one scheduler step, active and idle ranks could size the EAGLE3 first catch-up collective differently and hang.
- Key implementation: makes first-step activation reduction an explicit draft-model capability shared by active EAGLE execution and idle replay, marks Kimi/DeepSeek, Llama, and Qwen3.5 draft models accordingly, and prevents fused `lm_head_gemm` from launching with zero tokens.
- Code diff details: the old class checks are replaced with `draft_first_step_reduce_for_catchup`, so collective sizing follows model behavior rather than a hard-coded model list.
- Key code excerpts:

```diff
+def draft_model_reduces_first_step_catchup(draft_model) -> bool:
+    return bool(getattr(draft_model, "draft_first_step_reduce_for_catchup", False))
+draft_first_step_reduce = step_idx == 0 and (
+    all_decode_or_idle or draft_reduces_first_step_catchup)
```

- Reviewed files: runtime: `execution/drafter/eagle.py`, `execution/model_executor.py`, `models/{deepseek_v3,llama_eagle3,qwen3_5_nextn}.py`, `lm_head_gemm.py`; no separate test file was added.
- Risk and verification: all ranks must derive identical collective row counts for mixed forward modes; the PR reports DP8 + EAGLE3 AIME25 completion at 28/30 after the fix, and zero-row fused lm-head routing must remain a no-op.

### PR #795 - Support EAGLE3.1 for Kimi-K2.7 Code

- Link: https://github.com/lightseekorg/tokenspeed/pull/795
- Status/date: merged / 2026-07-25
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 49-line diff, 1 file, +24/-0.
- Motivation: the Kimi-K2.7 EAGLE3.1 MLA speculator publishes per-input FC normalization and optional normalized auxiliary-output semantics that the shared DeepSeek-style drafter did not implement.
- Key implementation: constructs one RMSNorm per concatenated FC input chunk when `fc_norm` is enabled, normalizes each chunk before the projection, and honors `norm_output` for the auxiliary hidden-state output.
- Code diff details: all behavior is config-gated inside `Eagle3MlaModel`, preserving older EAGLE checkpoints.
- Key code excerpts:

```diff
+if self.fc_norm is not None:
+    chunks = hidden_states.chunk(self.num_fc_input_dim, dim=-1)
+    hidden_states = torch.cat(
+        [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)], dim=-1)
```

- Reviewed files: runtime: `python/tokenspeed/runtime/models/deepseek_v3.py`; validation evidence: PR benchmark and launch recipe for `nvidia/Kimi-K2.7-Code-NVFP4`.
- Risk and verification: keep `fc_norm`/`norm_output` tied to checkpoint config; the reported 4xGB200 1-3-4 run shows 1.36x-1.91x category speedups, which should not be generalized to other acceptance lengths or serving shapes.

### PR #797 - Support incremental DFlash capture for Kimi

- Link: https://github.com/lightseekorg/tokenspeed/pull/797
- Status/date: merged / 2026-07-26
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 166-line diff, 4 files, +61/-7.
- Motivation: Kimi delegated DFlash capture to its DeepSeek-style language model, but the wrapper dropped the incremental projection callback and slot buffers expected by the executor, preventing startup with incremental projection enabled.
- Key implementation: threads callback and slot buffers through `KimiK25ForConditionalGeneration.set_dflash_layers_to_capture`, stores a layer-to-slot map, copies each captured hidden state into its slot, and invokes the incremental projection callback as soon as that layer finishes.
- Code diff details: the model tracks `_dflash_incr_active`; CI plumbing also aligns Slurm server startup timeout with readiness so long Kimi startup does not fail independently.
- Key code excerpts:

```diff
+self.model._dflash_capture_idx_map = {
+    layer_idx: i for i, layer_idx in enumerate(sorted(self.model.layers_to_capture))
+}
+self.model._dflash_incremental_callback(capture_idx, num_tokens)
```

- Reviewed files: runtime: `models/deepseek_v3.py`, `models/kimi_k25.py`; tests/CI: `test/ci_system/{pipeline,test_pipeline}.py`.
- Risk and verification: callback ordering, slot capacity, CUDA-stream lifetime, and `_dflash_incr_active` reset must match the executor; the PR records syntax/pre-commit checks and a live B200 validation lane.

### PR #29 - Add Kimi K2.5 agentic perf CI task

- Link: https://github.com/lightseekorg/tokenspeed/pull/29
- Status/date: merged / 2026-05-08
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +387/-3, 650 cached patch lines.
- Motivation: make `nvidia/Kimi-K2.5-NVFP4` agentic serving a repeatable perf CI lane.
- Key implementation: adds a Kimi K2.5 agentic perf YAML using `tokenspeed_mla`, NVFP4, speculative draft, and EvalScope agentic workloads.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--attention-backend tokenspeed_mla
+--quantization nvfp4
```

- Reviewed files: PR workflow, `kimi-k2.5-nvfp4-evalscope-agentic.yaml`, CI pipeline helpers
- Risk and verification: keep the agentic perf lane separate from shared synthetic serving workloads.

### PR #126 - perf(K2.5): Optimize lm_head

- Link: https://github.com/lightseekorg/tokenspeed/pull/126
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 6 files, +1173/-3, 1,246 cached patch lines.
- Motivation: Kimi K2.5 decode spends meaningful time in the final `lm_head` GEMM.
- Key implementation: gates a fused CUDA `lm_head_gemm` path to Kimi and falls back when shapes are unsupported.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+self._use_fused_lm_head = getattr(self.config, "model_type", None) == "kimi_k2"
+logits = _lm_head_matmul(hidden_states, lm_head.weight)
```

- Reviewed files: `logits_processor.py`, `lm_head_gemm.cu`, binding, Python wrapper, setup
- Risk and verification: include `lm_head` as its own profiler bucket for Kimi-style models.

### PR #184 - perf(K2.5): optimize top_k_renorm_prob + top_p_renorm_prob

- Link: https://github.com/lightseekorg/tokenspeed/pull/184
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 8 files, +3104/-12, 3,580 cached patch lines.
- Motivation: back-to-back top-k and deterministic top-p renormalization caused repeated scans and extra launches.
- Key implementation: adds a fused TopK+TopP renormalization CUDA path and wires it into `flashinfer_full.py`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
-probs = top_k_renorm_prob(probs, top_ks)
-probs = top_p_renorm_prob(probs, top_ps, is_deterministic=True)
+probs = fused_topk_topp_renorm(probs, top_ks, top_ps)
```

- Reviewed files: fused sampling CUDA sources, `flashinfer_full.py`, `server_args.py`, tests
- Risk and verification: sampling can be the bottleneck; also track limits such as `top_k < 128`.

### PR #253 - ci(eval): add Kimi-K2.5-NVFP4 ocr_bench task

- Link: https://github.com/lightseekorg/tokenspeed/pull/253
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +48/-0, 72 cached patch lines.
- Motivation: Kimi K2.5 needs a multimodal OCR regression lane.
- Key implementation: adds an OCR EvalScope YAML using the Kimi NVFP4 server config.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--datasets ocr_bench
```

- Reviewed files: `kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- Risk and verification: text-only throughput does not cover the Kimi K2.5 multimodal path.

### PR #418 - Add InstantTensor weight loader

- Link: https://github.com/lightseekorg/tokenspeed/pull/418
- Status/date: merged / 2026-06-15
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 25 files, +468/-60, 1,373 cached patch lines.
- Motivation: Kimi-scale checkpoints need a faster loader path.
- Key implementation: adds `--load-format instanttensor`, loader utilities, Kimi model integration, and CI/doc updates.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+--load-format instanttensor
+        elif self.load_config.load_format == LoadFormat.INSTANTTENSOR:
+            weights_iterator = instanttensor_weights_iterator(hf_weights_files)
```

- Reviewed files: `model_loader/loader.py`, `weight_utils.py`, `kimi_k25.py`, `server_args.py`, docs and eval configs
- Risk and verification: separate cold-start loading evidence from steady-state throughput.

### PR #444 - feat(moe): add trtllm mxint4 MoE path for Kimi-K2.x

- Link: https://github.com/lightseekorg/tokenspeed/pull/444
- Status/date: merged / 2026-06-14
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 8 files, +469/-6, 581 cached patch lines.
- Motivation: Kimi K2.x needed an INT4 W4A16 group-32 MoE path.
- Key implementation: adds MXINT4 weight packing, quant config detection, and FlashInfer TRT-LLM MoE process/apply ops.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+from tokenspeed.runtime.layers.moe.weights.mxint4 import create_mxint4_weight_pair
+name="flashinfer_trtllm_mxint4_moe_apply"
```

- Reviewed files: `expert.py`, `weights/mxint4.py`, quantization configs, `trtllm_mxint4.py`
- Risk and verification: record weight dtype, group size, activation dtype, and MoE backend in benchmark tables.

### PR #454 - [AMD] Support Kimi K2.5 MXFP4 serving

- Link: https://github.com/lightseekorg/tokenspeed/pull/454
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 33 files, +1924/-142, 3,856 cached patch lines.
- Motivation: serve Kimi K2.5 MXFP4 on AMD.
- Key implementation: adds MXFP4 quantization/layers/dense support and updates MLA backend, Kimi model code, and tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+--quantization mxfp4
+model_type == "kimi_k25"
```

- Reviewed files: MXFP4 layers/quantization, dense paths, attention backends, `kimi_k25.py`, tests
- Risk and verification: this is hardware-specific and should not be merged with NVIDIA NVFP4 conclusions.

### PR #477 - perf(kernel): Optimize Kimi Vision FA4 QKV + RoPE

- Link: https://github.com/lightseekorg/tokenspeed/pull/477
- Status/date: merged / 2026-06-19
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 3 files, +195/-7, 304 cached patch lines.
- Motivation: the Kimi vision FA4 path had extra packed-QKV and complex-RoPE layout movement.
- Key implementation: adds `packed_qkv_complex_rotary` and wires it into multimodal encoder attention.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+        if use_packed_qkv_complex_rotary:
+            q, k, v = packed_qkv_complex_rotary(
+def packed_qkv_complex_rotary(
```

- Reviewed files: `mm_encoder_attention.py`, `kimi_k25.py`, `qkv_rotary.py`
- Risk and verification: profile QKV/RoPE layout work before blaming FA4 itself.

### PR #482 - ci: use FA4 mm attention for Kimi OCR eval

- Link: https://github.com/lightseekorg/tokenspeed/pull/482
- Status/date: merged / 2026-06-19
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +1/-0, 22 cached patch lines.
- Motivation: make OCR eval exercise the FA4 multimodal attention path.
- Key implementation: adds `--mm-attention-backend fa4` to the Kimi OCR EvalScope YAML.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+--mm-attention-backend fa4
```

- Reviewed files: `kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- Risk and verification: always record the multimodal attention backend in Kimi OCR comparisons.

### PR #476 - Add AMD Kimi MXFP4 CI job

- Link: https://github.com/lightseekorg/tokenspeed/pull/476
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 3 files, +138/-4, 181 cached patch lines.
- Motivation: keep AMD Kimi MXFP4 AIME25 and MLA metadata paths covered after #454.
- Key implementation: adds an AMD MXFP4 eval YAML and `MLAAttnBackend` metadata tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+--model amd/Kimi-K2.5-MXFP4
+--quantization mxfp4
```

- Reviewed files: `mla.py`, AMD AIME25 YAML, `test_mla_verify_metadata.py`
- Risk and verification: treat AMD MXFP4 as a separate lane from NVIDIA Kimi NVFP4.
