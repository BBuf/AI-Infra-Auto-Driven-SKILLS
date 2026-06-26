# TokenSpeed Kimi Model PR Optimization History

## 2026-06-26 Latest Source Scan

Rechecked TokenSpeed upstream `lightseekorg/tokenspeed@5aedf69d6b476baa65571011de6ea60fd5a238a8` against the tracked files listed below.
The file-level match used a GitHub mirror `git log --name-only`; PR titles, links, and merge times were batch-verified through the GitHub GraphQL Pull Request API. Previous freshness anchor: `2026-06-26`.

Result: 2 additional PR-numbered merge(s) touched tracked files and are not yet promoted into full per-PR diff audit cards below. Treat this section as a freshness index; promote any row into a full card only after manual diff review.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-26 | [#519](https://github.com/lightseekorg/tokenspeed/pull/519) | feat: distributed argmax for EAGLE greedy sampling | `logits_processor.py` |
| 2026-06-25 | [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | perf(kernel): optimize Qwen vision QKV rotary layout | `qkv_rotary.py` |

## 2026-06-26 PR Backfill Audit

Checked against TokenSpeed upstream `HEAD@5aedf69d6b476baa65571011de6ea60fd5a238a8`. This uses a SGLang-style timeline plus per-PR diff audit cards for Kimi K2.5/K2.x.

Filter used in this pass: merged PRs whose titles or files matched `Kimi`, `kimi_k25`, `K2.5`, `NVFP4`, `MXFP4`, `MXINT4`, `lm_head`, `top_k/top_p`, `InstantTensor`, `OCR`, `FA4`, `vision`, or `MLA`. Formatting-only and unrelated infrastructure changes were excluded.

## Model Implementation File Coverage

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

## PR Coverage Overview

- Reviewed PRs: 10
- Diff source: `gh pr diff` / GitHub PR patches cached under `/tmp/model_pr_diffs/tokenspeed/pr*.diff`
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

### PR #29 - Add Kimi K2.5 agentic perf CI task

- Link: https://github.com/lightseekorg/tokenspeed/pull/29
- State/time: merged / 2026-05-08
- Diff coverage: 5 files, +387/-3, 650 cached patch lines.
- Motivation: make `nvidia/Kimi-K2.5-NVFP4` agentic serving a repeatable perf CI lane.
- Key implementation: adds a Kimi K2.5 agentic perf YAML using `tokenspeed_mla`, NVFP4, speculative draft, and EvalScope agentic workloads.
- Code excerpt:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--attention-backend tokenspeed_mla
+--quantization nvfp4
```

- Reviewed files: PR workflow, `kimi-k2.5-nvfp4-evalscope-agentic.yaml`, CI pipeline helpers
- Validation/risk: keep the agentic perf lane separate from shared synthetic serving workloads.

### PR #126 - perf(K2.5): Optimize lm_head

- Link: https://github.com/lightseekorg/tokenspeed/pull/126
- State/time: merged / 2026-05-13
- Diff coverage: 6 files, +1173/-3, 1,246 cached patch lines.
- Motivation: Kimi K2.5 decode spends meaningful time in the final `lm_head` GEMM.
- Key implementation: gates a fused CUDA `lm_head_gemm` path to Kimi and falls back when shapes are unsupported.
- Code excerpt:

```diff
+self._use_fused_lm_head = getattr(self.config, "model_type", None) == "kimi_k2"
+logits = _lm_head_matmul(hidden_states, lm_head.weight)
```

- Reviewed files: `logits_processor.py`, `lm_head_gemm.cu`, binding, Python wrapper, setup
- Validation/risk: include `lm_head` as its own profiler bucket for Kimi-style models.

### PR #184 - perf(K2.5): optimize top_k_renorm_prob + top_p_renorm_prob

- Link: https://github.com/lightseekorg/tokenspeed/pull/184
- State/time: merged / 2026-05-20
- Diff coverage: 8 files, +3104/-12, 3,580 cached patch lines.
- Motivation: back-to-back top-k and deterministic top-p renormalization caused repeated scans and extra launches.
- Key implementation: adds a fused TopK+TopP renormalization CUDA path and wires it into `flashinfer_full.py`.
- Code excerpt:

```diff
-probs = top_k_renorm_prob(probs, top_ks)
-probs = top_p_renorm_prob(probs, top_ps, is_deterministic=True)
+probs = fused_topk_topp_renorm(probs, top_ks, top_ps)
```

- Reviewed files: fused sampling CUDA sources, `flashinfer_full.py`, `server_args.py`, tests
- Validation/risk: sampling can be the bottleneck; also track limits such as `top_k < 128`.

### PR #253 - ci(eval): add Kimi-K2.5-NVFP4 ocr_bench task

- Link: https://github.com/lightseekorg/tokenspeed/pull/253
- State/time: merged / 2026-05-28
- Diff coverage: 1 file, +48/-0, 72 cached patch lines.
- Motivation: Kimi K2.5 needs a multimodal OCR regression lane.
- Key implementation: adds an OCR EvalScope YAML using the Kimi NVFP4 server config.
- Code excerpt:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--datasets ocr_bench
```

- Reviewed files: `kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- Validation/risk: text-only throughput does not cover the Kimi K2.5 multimodal path.

### PR #418 - Add InstantTensor weight loader

- Link: https://github.com/lightseekorg/tokenspeed/pull/418
- State/time: merged / 2026-06-15
- Diff coverage: 25 files, +468/-60, 1,373 cached patch lines.
- Motivation: Kimi-scale checkpoints need a faster loader path.
- Key implementation: adds `--load-format instanttensor`, loader utilities, Kimi model integration, and CI/doc updates.
- Code excerpt:

```diff
+--load-format instanttensor
+        elif self.load_config.load_format == LoadFormat.INSTANTTENSOR:
+            weights_iterator = instanttensor_weights_iterator(hf_weights_files)
```

- Reviewed files: `model_loader/loader.py`, `weight_utils.py`, `kimi_k25.py`, `server_args.py`, docs and eval configs
- Validation/risk: separate cold-start loading evidence from steady-state throughput.

### PR #444 - feat(moe): add trtllm mxint4 MoE path for Kimi-K2.x

- Link: https://github.com/lightseekorg/tokenspeed/pull/444
- State/time: merged / 2026-06-14
- Diff coverage: 8 files, +469/-6, 581 cached patch lines.
- Motivation: Kimi K2.x needed an INT4 W4A16 group-32 MoE path.
- Key implementation: adds MXINT4 weight packing, quant config detection, and FlashInfer TRT-LLM MoE process/apply ops.
- Code excerpt:

```diff
+from tokenspeed.runtime.layers.moe.weights.mxint4 import create_mxint4_weight_pair
+name="flashinfer_trtllm_mxint4_moe_apply"
```

- Reviewed files: `expert.py`, `weights/mxint4.py`, quantization configs, `trtllm_mxint4.py`
- Validation/risk: record weight dtype, group size, activation dtype, and MoE backend in benchmark tables.

### PR #454 - [AMD] Support Kimi K2.5 MXFP4 serving

- Link: https://github.com/lightseekorg/tokenspeed/pull/454
- State/time: merged / 2026-06-16
- Diff coverage: 33 files, +1924/-142, 3,856 cached patch lines.
- Motivation: serve Kimi K2.5 MXFP4 on AMD.
- Key implementation: adds MXFP4 quantization/layers/dense support and updates MLA backend, Kimi model code, and tests.
- Code excerpt:

```diff
+--quantization mxfp4
+model_type == "kimi_k25"
```

- Reviewed files: MXFP4 layers/quantization, dense paths, attention backends, `kimi_k25.py`, tests
- Validation/risk: this is hardware-specific and should not be merged with NVIDIA NVFP4 conclusions.

### PR #477 - perf(kernel): Optimize Kimi Vision FA4 QKV + RoPE

- Link: https://github.com/lightseekorg/tokenspeed/pull/477
- State/time: merged / 2026-06-19
- Diff coverage: 3 files, +195/-7, 304 cached patch lines.
- Motivation: the Kimi vision FA4 path had extra packed-QKV and complex-RoPE layout movement.
- Key implementation: adds `packed_qkv_complex_rotary` and wires it into multimodal encoder attention.
- Code excerpt:

```diff
+        if use_packed_qkv_complex_rotary:
+            q, k, v = packed_qkv_complex_rotary(
+def packed_qkv_complex_rotary(
```

- Reviewed files: `mm_encoder_attention.py`, `kimi_k25.py`, `qkv_rotary.py`
- Validation/risk: profile QKV/RoPE layout work before blaming FA4 itself.

### PR #482 - ci: use FA4 mm attention for Kimi OCR eval

- Link: https://github.com/lightseekorg/tokenspeed/pull/482
- State/time: merged / 2026-06-19
- Diff coverage: 1 file, +1/-0, 22 cached patch lines.
- Motivation: make OCR eval exercise the FA4 multimodal attention path.
- Key implementation: adds `--mm-attention-backend fa4` to the Kimi OCR EvalScope YAML.
- Code excerpt:

```diff
+--mm-attention-backend fa4
```

- Reviewed files: `kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- Validation/risk: always record the multimodal attention backend in Kimi OCR comparisons.

### PR #476 - Add AMD Kimi MXFP4 CI job

- Link: https://github.com/lightseekorg/tokenspeed/pull/476
- State/time: merged / 2026-06-26
- Diff coverage: 3 files, +138/-4, 181 cached patch lines.
- Motivation: keep AMD Kimi MXFP4 AIME25 and MLA metadata paths covered after #454.
- Key implementation: adds an AMD MXFP4 eval YAML and `MLAAttnBackend` metadata tests.
- Code excerpt:

```diff
+--model amd/Kimi-K2.5-MXFP4
+--quantization mxfp4
```

- Reviewed files: `mla.py`, AMD AIME25 YAML, `test_mla_verify_metadata.py`
- Validation/risk: treat AMD MXFP4 as a separate lane from NVIDIA Kimi NVFP4.
