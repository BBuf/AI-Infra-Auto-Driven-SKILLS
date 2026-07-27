# TensorRT-LLM Kimi Model PR Optimization History

## 2026-07-28 Source Head Refresh

Rechecked TensorRT-LLM upstream main at `NVIDIA/TensorRT-LLM@9fe5853263750ade5b7dc24fb31a1215ec822d45`.
The seven-commit range from the previous recorded head
`1b4ffc0291d75a21ad20118e8f44de6e3831f786` was inspected with
`git log --name-only` and complete local source diffs.

Result: PR #16805 is promoted because it fixes disaggregated speculative
draft-token and sequence-length accounting used by Kimi-style PD flows.
PR #16763 is retained as a cross-model startup-memory card. The remaining four
commits only adjust CI, fakes, Slurm cleanup, or Docker copies and are not
presented as model optimization evidence. The seventh commit, VisualGen/Wan
Attention2D plus TP PR #16677, is also outside the Kimi LLM scope. PR #14848
remains the prior high-signal Kimi runtime merge.

### PR #16805 - Fix disaggregated draft-token accounting

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16805
- Status/date: merged / 2026-07-27
- Trace source: merge commit
  `93924532ff65e6dce000c1dcee604e585386781b`; full six-commit increment and
  five-file PR diff read locally.
- Diff scope read: 5 files, +164/-4.
- Motivation: generation-only requests received first-generation and draft
  tokens through `ContextPhaseParams`, but the decode request did not adopt the
  draft tokens and sequence-length setup counted only the first token.
- Key implementation: `GenericLlmRequest` adopts handoff draft tokens at
  construction or late context assignment; a shared helper counts first-gen
  plus draft tokens, and decoder sequence-length setup uses that total.
- Code diff details: `llmRequest.h` adds draft-token adoption and the shared
  count helper; `createNewDecoderRequests.cpp` replaces first-token-only
  arithmetic; C++ and Python tests cover constructor and late-assignment paths.
- Key code excerpts:

```diff
+        adoptContextPhaseDraftTokens();
+        auto numTokens = static_cast<SizeType32>(
+            contextPhaseParams.getFirstGenTokens().size());
+        numTokens += static_cast<SizeType32>(draftTokens->size());
```

- Reviewed files: `llmRequest.h`, `createNewDecoderRequests.cpp`, C++ request
  tests, Python executor-request tests, and binding tests.
- Risk and verification: validate context/decode handoff with and without draft
  tokens, late context assignment, and downstream sequence lengths; this is a
  correctness fix, not throughput evidence.

### PR #16763 - Unify phase-1 CUDA graph cleanup

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16763
- Status/date: merged / 2026-07-27
- Trace source: final upstream commit
  `6046f34e1ac8fc20c2c5553d00a98eb15a172555`; complete two-file diff read
  locally.
- Diff scope read: 2 files, +5/-8.
- Motivation: KV-capacity estimation already shuts down the phase-1 executor
  and releases its CUDA graphs; a second explicit graph release duplicated
  ownership and complicated final KV-cache allocation.
- Key implementation: rely on `configure_kv_cache_capacity` for graph/resource
  teardown and clear only profiling attention metadata before rebuilding the
  final KV managers.
- Code diff details: removes the second `_release_cuda_graphs()` call from
  `py_executor_creator.py`, keeps `eng.attn_metadata = None`, and unwaives the
  two configurations that exercise the corrected ownership path.
- Key code excerpts:

```diff
-            if eng.attn_metadata is not None:
-                if llm_args.cuda_graph_config is not None:
-                    eng._release_cuda_graphs()
+            if eng is not None:
                 eng.attn_metadata = None
```

- Reviewed files: `py_executor_creator.py` and the two newly unwaived
  DeepSeek-V3Lite integration cases.
- Risk and verification: compare startup memory before/after capacity
  estimation and ensure graph resources are released exactly once.

## 2026-06-27 PR Backfill Audit

The per-PR diff audit cards on this page were generated from TensorRT-LLM
upstream `HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980`. The root
TensorRT-LLM history index now tracks the 2026-06-27 runtime refresh at
`aaffa2f9fef3025e0f698d978385a73460344e0b`. This page provides model
implementation coverage, a timeline, and per-PR diff audit cards for Kimi K2
Thinking / Kimi K2.5.

Filter used in this pass: merged PRs whose titles or files matched `Kimi`, `kimi_k25`, `KimiK25`, `K2.5`, `K2 Thinking`, `NVFP4`, `multimodal`, `tool_parser`, `reasoning_parser`, `guided decoding`, `spec dec`, `rejection sampling`, or `NIXL`. Formatting-only and unrelated infrastructure PRs were excluded.

## Implementation File Coverage

| File | Related PRs |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` | [#9711](https://github.com/NVIDIA/TensorRT-LLM/pull/9711) |
| `tensorrt_llm/serve/tool_parser/kimi_k2_tool_parser.py` | [#9830](https://github.com/NVIDIA/TensorRT-LLM/pull/9830) |
| `tensorrt_llm/_torch/models/modeling_deepseekv3.py` | [#11777](https://github.com/NVIDIA/TensorRT-LLM/pull/11777), [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) |
| `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_kimi_k2.py` | [#11780](https://github.com/NVIDIA/TensorRT-LLM/pull/11780) |
| `examples/auto_deploy/model_registry/configs/kimi_k2.yaml` | [#11780](https://github.com/NVIDIA/TensorRT-LLM/pull/11780) |
| `tensorrt_llm/_torch/models/modeling_kimi_k25.py` | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788), [#14379](https://github.com/NVIDIA/TensorRT-LLM/pull/14379), [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180) |
| `tensorrt_llm/llmapi/reasoning_parser.py` | [#13801](https://github.com/NVIDIA/TensorRT-LLM/pull/13801) |
| `tensorrt_llm/_torch/modules/embedding.py` | [#15233](https://github.com/NVIDIA/TensorRT-LLM/pull/15233) |
| `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py` | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_*.yaml` | [#15443](https://github.com/NVIDIA/TensorRT-LLM/pull/15443) |

## PR Coverage Summary

- Reviewed PRs: 10
- File trace command: `git log --name-only -- <model-files>`
- Diff source: `gh pr diff` / GitHub Pull Request files API patches cached under `/tmp/model_pr_diffs/tensorrt_llm/pr*.diff`
- Reviewed patch lines: 8,414
- Main TensorRT-LLM Kimi themes: Blackwell/GB200 deployment guide, OpenAI tool parser, K2.5 text NVFP4, AutoDeploy Kimi K2.5, multimodal vision/video path, reasoning parser, speculative/guided decoding, rejection-sampling embedding mask, and NIXL disaggregated perf lanes.

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-12-05 | [#9711](https://github.com/NVIDIA/TensorRT-LLM/pull/9711) | merged | Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell | deployment guide |
| 2025-12-12 | [#9830](https://github.com/NVIDIA/TensorRT-LLM/pull/9830) | merged | Support tool parser for Kimi K2 | OpenAI server/tool parser |
| 2026-03-04 | [#11777](https://github.com/NVIDIA/TensorRT-LLM/pull/11777) | merged | Add Kimi-K2.5 text model support (NVFP4) | `modeling_deepseekv3.py`, accuracy tests |
| 2026-03-05 | [#11780](https://github.com/NVIDIA/TensorRT-LLM/pull/11780) | merged | AutoDeploy onboarding agent + Kimi K2.5 AD modeling code | AutoDeploy Kimi model/config/tests |
| 2026-05-11 | [#13801](https://github.com/NVIDIA/TensorRT-LLM/pull/13801) | merged | Add reasoning parser for kimi-k2.5 and enable auto flow | command/reasoning parser |
| 2026-05-14 | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) | merged | Add Kimi K2.5 multimodal vision support | `modeling_kimi_k25.py`, multimodal eval/tests |
| 2026-05-22 | [#14379](https://github.com/NVIDIA/TensorRT-LLM/pull/14379) | merged | Fix Kimi_k25 with spec dec | `modeling_kimi_k25.py` |
| 2026-06-17 | [#15233](https://github.com/NVIDIA/TensorRT-LLM/pull/15233) | merged | Fix embedding vocab mask for rejection sampling in Kimi-K2.5 | `embedding.py` |
| 2026-06-23 | [#15443](https://github.com/NVIDIA/TensorRT-LLM/pull/15443) | merged | Un-waive K2.5 Thinking FP4 disagg-NIXL tests | waives and perf-sanity YAML |
| 2026-06-25 | [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180) | merged | Add necessary methods for guided decoding in Kimi K2.5 | `modeling_kimi_k25.py` |

## Per-PR Diff Audit Cards

### PR #14848 - RMSNorm NVFP4 quant fusion for DeepSeek-V3.2 / Kimi-K2.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14848
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 2,487-line diff, 16 files, +1993/-101.
- Motivation: static-NVFP4 Kimi-K2.5 executed RMSNorm and activation quantization as separate kernels before each compatible linear, materializing the normalized tensor and paying an extra launch.
- Key implementation: adds Blackwell-only C++/CUDA operators for fused optional residual-add + RMSNorm + NVFP4 quantization, supports packed and row-strided input, returns the unquantized norm when another consumer needs it, and routes eligible RMSNorm-to-linear edges through the fused result.
- Code diff details: `rmsNormFp4QuantKernel` performs the reduction and emits packed E2M1 values plus E4M3 block scales; Python dispatch keeps unsupported architectures and shapes on the existing path.
- Key code excerpts:

```diff
+// Fused (optional residual-add +) RMSNorm + NVFP4 input-quantize.
+__global__ void rmsNormFp4QuantKernel(RmsNormFp4QuantParams params)
+{
+    float const denom = rsqrtf(acc / params.hidden_size + params.eps);
+    uint32_t const quant_val = cvt_warp_fp16_to_fp4<T, kSfVecSize, false>(
+        pv, sf_scale, sf_out_ptr);
+}
```

- Reviewed files: runtime: `cpp/tensorrt_llm/kernels/rmsNormFp4QuantKernels.{cu,h}`, `cpp/tensorrt_llm/thop/rmsNormFp4Quant.cpp`, `tensorrt_llm/_torch/modules/{rms_norm,linear,mla}.py`, `modeling_deepseekv3.py`; tests: `test_fused_rmsnorm_fp4_quantize.py`, `test_fp4_num_tokens_slice.py`, B200 test database.
- Risk and verification: the fused FP4 epilogue is restricted to SM10.x; validation compares packed FP4 values and unswizzled scale factors, checks strided inputs and no input mutation, and separates acceptable RMSNorm rounding drift from bit-exact re-quantization of the returned norm.

### PR #9711 - Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9711
- Status/date: merged / 2025-12-05
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +309/-0, 534 cached patch lines.
- Motivation: provide an official Blackwell/GB200 deployment guide for Kimi K2 Thinking NVFP4.
- Key implementation: documents Docker, `trtllm-serve`, 8-way EP/attention DP, SLURM wide EP, and disaggregated serving.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+trtllm-serve nvidia/Kimi-K2-Thinking-NVFP4 \
+--extra_llm_api_options
```

- Reviewed files: deployment guide markdown
- Risk and verification: record Blackwell/GB200 and disaggregation assumptions when using this as competitor evidence.

### PR #9830 - Support tool parser for Kimi K2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9830
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +374/-1, 528 cached patch lines.
- Motivation: Kimi K2 OpenAI-compatible serving needs correct tool-call parsing for agentic workloads.
- Key implementation: adds a Kimi K2 tool parser and wires it into the OpenAI server postprocess and parser factory.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+from .kimi_k2_tool_parser import KimiK2ToolParser
+class KimiK2ToolParser(BaseToolParser):
+        "kimi_k2": KimiK2ToolParser,
```

- Reviewed files: OpenAI server, postprocess handlers, `kimi_k2_tool_parser.py`, factory, tests
- Risk and verification: agentic correctness includes parser behavior, not just speed.

### PR #11777 - Add Kimi-K2.5 text model support (NVFP4)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11777
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +96/-0, 532 cached patch lines.
- Motivation: support Kimi-K2.5 text NVFP4 in the PyTorch backend.
- Key implementation: adapts the DeepSeekV3-style runtime and adds accuracy refs/tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+MODEL_NAME = "moonshotai/Kimi-K2.5"
+quant_algo: NVFP4
```

- Reviewed files: `modeling_deepseekv3.py`, accuracy refs/tests
- Risk and verification: separate text-only Kimi K2.5 from multimodal Kimi paths.

### PR #11780 - AutoDeploy onboarding agent + Kimi K2.5 AD modeling code

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11780
- Status/date: merged / 2026-03-05
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 9 files, +2190/-9, 2,807 cached patch lines.
- Motivation: add AutoDeploy modeling code for Kimi K2.5.
- Key implementation: adds `modeling_kimi_k2.py`, registry config, MLA custom ops, and AutoDeploy tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+model_factory: KimiK2ForCausalLM
+flashinfer_mla
```

- Reviewed files: agent scaffold, `kimi_k2.yaml`, MLA ops, `modeling_kimi_k2.py`, AD tests
- Risk and verification: competitor path may be AutoDeploy rather than the plain PyTorch wrapper.

### PR #13801 - Add reasoning parser for Kimi-K2.5 and enable auto flow

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13801
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +5/-1, 47 cached patch lines.
- Motivation: auto-select the right reasoning parser for Kimi-K2.5 thinking outputs.
- Key implementation: adds `kimi_k2/kimi_k25` auto-detect hints and registers `kimi_k25` with `reasoning_at_start=True`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+"kimi_k25": "kimi_k25",
+@register_reasoning_parser("kimi_k25", reasoning_at_start=True)
```

- Reviewed files: `commands/serve.py`, `reasoning_parser.py`
- Risk and verification: parser selection affects eval scores.

### PR #12788 - Add Kimi K2.5 multimodal vision support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12788
- Status/date: merged / 2026-05-14
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 12 files, +2912/-64, 3,536 cached patch lines.
- Motivation: enable text/image/video Kimi K2.5 multimodal serving.
- Key implementation: adds `KimiK25ForConditionalGeneration`, vision model, input processor, placeholders, multimodal eval, and tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+@register_auto_model("KimiK25ForConditionalGeneration")
+class KimiK25ForConditionalGeneration(PreTrainedModel):
+    "video_placeholder": "<|kimi_k25_video_placeholder|>",
```

- Reviewed files: `modeling_kimi_k25.py`, `modeling_deepseekv3.py`, eval wrappers, multimodal tests
- Risk and verification: profile vision encoder, placeholder expansion, hashing fallback, and text decode as separate stages.

### PR #14379 - Fix Kimi_k25 with spec dec

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14379
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +53/-35, 187 cached patch lines.
- Motivation: speculative decoding missed Kimi K2.5 multimodal params and `lm_head` delegation.
- Key implementation: threads `multimodal_params` through `forward` and adds an `lm_head` proxy.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+multimodal_params: Optional[List[MultimodalParams]] = None
+def lm_head(self): return self.llm.lm_head
```

- Reviewed files: `modeling_kimi_k25.py`
- Risk and verification: spec-dec comparisons need to verify context-only multimodal handling.

### PR #15233 - Fix embedding vocab mask for rejection sampling in Kimi-K2.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15233
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +15/-8, 129 cached patch lines.
- Motivation: FlashInfer rejection sampling can pad rejected tokens with non-vocab values.
- Key implementation: masks/clamps input before `F.embedding` in `pre_comm_embedding_ops`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+# flashinfer's rejection kernel pads non-accepted tokens
+        input_, input_mask = get_masked_input_and_mask(
+            input_,
+            0,
+            weight.shape[0],
+        )
```

- Reviewed files: `embedding.py`
- Risk and verification: correctness risk sits in embedding preprocessing, not a visible hot kernel.

### PR #15443 - Un-waive K2.5 Thinking FP4 disagg-NIXL tests

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15443
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +2/-3, 86 cached patch lines.
- Motivation: Kimi K2.5 Thinking FP4 disaggregated NIXL lanes became stable enough to un-waive.
- Key implementation: removes Kimi NIXL skips and raises KV transfer timeout in perf-sanity YAML.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+kv_transfer_timeout_ms: 600000
```

- Reviewed files: `waives.txt`, Kimi NIXL perf-sanity YAML
- Risk and verification: disaggregated NIXL is a separate benchmark bucket.

### PR #15180 - Add necessary methods for guided decoding in Kimi K2.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15180
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +3/-0, 28 cached patch lines.
- Motivation: Kimi K2.5 wrapper missed guided decoding delegation methods.
- Key implementation: proxies `set_guided_decoder` to the inner LLM.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+def set_guided_decoder(self, *args, **kwargs):
+    return self.llm.set_guided_decoder(*args, **kwargs)
```

- Reviewed files: `modeling_kimi_k25.py`
- Risk and verification: guided decoding changes decode control flow; record whether it is enabled.
