# TensorRT-LLM Kimi Model PR Optimization History

## 2026-06-26 Latest Source Scan

Rechecked TensorRT-LLM upstream `NVIDIA/TensorRT-LLM@0722c5f47d2cae69ac1a237da51e550dd214532c` against the tracked files listed below.
The file-level match used a GitHub mirror `git log --name-only`; PR titles, links, and merge times were batch-verified through the GitHub GraphQL Pull Request API. Previous freshness anchor: `2026-06-26`.

Result: no additional PR-numbered merges touched the tracked files beyond the existing timeline/backfill rows.

## 2026-06-26 PR Backfill Audit

The per-PR diff audit cards on this page were generated from TensorRT-LLM
upstream `HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980`. The root
TensorRT-LLM history index tracks the latest 2026-06-26 runtime refresh at
`0722c5f47d2cae69ac1a237da51e550dd214532c`. This page provides model
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
