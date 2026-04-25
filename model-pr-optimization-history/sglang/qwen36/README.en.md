# SGLang Qwen3.6 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on `2026-04-22` and sgl-cookbook `origin/main` `816bad5` on `2026-04-21`.

Scope: Qwen3.6-35B-A3B, Qwen3.6-27B dense, FP8/BF16 deployment, multimodal input, thinking preservation, MTP, Mamba scheduler, Qwen3 reasoning parser, and Qwen3-Coder tool parser.

## Summary

Qwen3.6 is currently a docs/deployment layer over shared hybrid Qwen runtime. Do not add a dedicated runtime fork until Qwen3-Next, Qwen3.5, Qwen VLM, and Qwen3-Coder parser paths are ruled out.

## Code Surfaces

- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`
- `docs_new/docs.json`
- `python/sglang/srt/layers/quantization/utils.py`
- `python/sglang/srt/utils/offloader.py`
- `test/registered/unit/utils/test_offloader_tied_params.py`

## Diff-Reviewed PR Cards

### PR #23034 - Qwen3.6 docs and deployment generator

- Link: https://github.com/sgl-project/sglang/pull/23034
- State: merged at `2026-04-17T05:33:34Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `7324` lines, `73` files; Qwen3.6, Qwen3.5 deployment, docs navigation, and GLM/Qwen warning hunks reviewed manually.
- Motivation: SGLang had no Qwen3.6 cookbook page or command generator, so users had to infer the reasoning parser, Qwen3-Coder tool parser, MTP flags, B200 attention backend, and Mamba scheduler behavior from adjacent Qwen docs. The same PR also points the Qwen cookbook card at the latest Qwen3.6 page and fixes warning rendering in adjacent Qwen3.5/GLM-5 docs.
- Key implementation: add `Qwen3.6.mdx`, register it in `docs_new/docs.json`, redirect the Qwen cookbook card from Qwen3.5 to Qwen3.6, and add `qwen36-deployment.jsx`. The generator forces Mamba V2 when MTP is enabled, emits `SGLANG_ENABLE_SPEC_V2=1`, combines `--reasoning-parser qwen3` with `--tool-call-parser qwen3_coder`, emits EAGLE MTP flags, and adds `--attention-backend trtllm_mha` on B200. The same PR updates Qwen3.5 MTP/Mamba logic, confirming Qwen3.6 should first reuse shared hybrid Qwen assumptions.
- Key code excerpts:

```diff
+                      "cookbook/autoregressive/Qwen/Qwen3.6",
...
-    href="/cookbook/autoregressive/Qwen/Qwen3.5"
+    href="/cookbook/autoregressive/Qwen/Qwen3.6"
```

```jsx
commandRule: (value) => value === 'enabled' ? '--reasoning-parser qwen3' : null,
commandRule: (value) => value === 'enabled' ? '--tool-call-parser qwen3_coder' : null,
commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4' : null,
commandRule: (value) => value === 'v2' ? '--mamba-scheduler-strategy extra_buffer' : null,
```

```jsx
const mtpEnabled = values.speculative === 'enabled';
if (mtpEnabled) {
  return [
    { id: 'v1', label: 'V1', default: false, disabled: true },
    { id: 'v2', label: 'V2', default: true },
  ];
}
```

- Reviewed files: `Qwen3.6.mdx`, `qwen36-deployment.jsx`, `qwen35-deployment.jsx`, `intro.mdx`, `docs.json`, `GLM-5.mdx`.
- Validation implications: check generated BF16/FP8 commands with and without MTP on H100/H200/B200. Reasoning and tool-call parsers must be validated together; MTP commands must keep `SGLANG_ENABLE_SPEC_V2=1` and Mamba V2.

### PR #23467 - FP8 modules_to_not_convert boundary matching

- Link: https://github.com/sgl-project/sglang/pull/23467
- State: merged at `2026-04-22T14:16:22Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `174` lines, `1` file.
- Motivation: Qwen3.6-27B-FP8 configs contain MoE-template entries like `model.language_model.layers.N.mlp.gate` in `modules_to_not_convert`. The old substring test matched that against dense fused `mlp.gate_up_proj`, causing dense MLPs to instantiate as BF16 while the checkpoint supplied FP8 scales. The observed failure was `weight_scale_inv not found` warnings and bad outputs. Qwen3.5 had a related `linear_attn.in_proj_a` versus fused `in_proj_ba` risk.
- Key implementation: `python/sglang/srt/layers/quantization/utils.py` adds `_module_path_match` for dotted module boundaries and `_FALLBACK_FUSED_SHARDS` for HF FP8 configs that omit `packed_modules_mapping`.
- Key code excerpts:

```python
def _module_path_match(ignored: str, prefix: str) -> bool:
    if ignored == prefix:
        return True
    if prefix.startswith(ignored + "."):
        return True
    return ("." + ignored + ".") in ("." + prefix + ".")
```

```python
_FALLBACK_FUSED_SHARDS: Mapping[str, List[str]] = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
}
```

```diff
-        is_skipped = any(ignored in prefix for ignored in ignored_layers)
+        is_skipped = any(
+            _module_path_match(ignored, prefix) for ignored in ignored_layers
+        )
```

- Reviewed files: `python/sglang/srt/layers/quantization/utils.py`.
- Validation implications: Qwen3.6-27B-FP8 should load without dense-MLP `weight_scale_inv not found` warnings. Qwen3.5 FP8 must be checked as a regression lane because the fallback fused-shard table preserves `in_proj_a` masking for `in_proj_ba`.

### PR #23486 - Qwen3.6-27B dense cookbook

- Link: https://github.com/sgl-project/sglang/pull/23486
- State: merged at `2026-04-22T17:22:46Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `198` lines, `2` files.
- Motivation: the first Qwen3.6 page covered only the 35B-A3B MoE checkpoint, while Qwen3.6 also released 27B dense BF16/FP8 weights. The old generator always emitted `Qwen/Qwen3.6-35B-A3B`, leaving the dense variant undocumented.
- Key implementation: the docs now describe both 35B-A3B MoE and 27B dense, and the generator adds a `modelSize` radio. `modelConfigs` is nested by size, while `generateCommand()` derives the Hugging Face repo from `sizeConfig.baseName`. The install hint changes from `uv pip install "sglang[all]"` to `uv pip install sglang`.
- Key code excerpts:

```jsx
modelSize: {
  name: 'modelSize',
  title: 'Model Size',
  items: [
    { id: '35b-a3b', label: '35B-A3B (MoE)', default: true },
    { id: '27b', label: '27B (Dense)', default: false },
  ],
},
```

```jsx
const sizeConfig = modelConfigs[modelSize];
const quantSuffix = quantization === 'fp8' ? '-FP8' : '';
const modelName = `Qwen/Qwen3.6-${sizeConfig.baseName}${quantSuffix}`;
```

```diff
-uv pip install "sglang[all]"
+uv pip install sglang
```

- Reviewed files: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`.
- Validation implications: command generation must cover `Qwen3.6-35B-A3B`, `Qwen3.6-35B-A3B-FP8`, `Qwen3.6-27B`, and `Qwen3.6-27B-FP8`. The PR body reports H200 TP=2 MMMU sanity after #23467: BF16 `55.1%`, FP8 `53.0%`, within Wilson 95% CI.

### PR #23474 - hybrid linear-attn CPU offload

- Link: https://github.com/sgl-project/sglang/pull/23474
- State: open as of `2026-04-23`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `395` lines, `2` files.
- Motivation: hybrid linear-attention models such as Qwen3-Next, Qwen3.5, and Kimi-Linear cache weight views as plain tensor attributes. After `--cpu-offload-gb` rebinds `Parameter.data` to pinned CPU memory, later checkpoint loading writes into the new CPU storage while the cached view still points at the original random GPU storage. Tied parameters also appear under multiple state-dict keys; moving each key separately creates conflicting device tensors for `functional_call(..., tie_weights=True)`.
- Key implementation: `OffloaderV1.maybe_offload_to_cpu()` records plain tensor aliases before offload, shares one device tensor per tied source parameter, recreates aliases with `as_strided(size, stride, offset)` during forward, and restores the old attributes afterward. The new test file reproduces tied-parameter and cached-view failures with minimal modules.
- Key code excerpts:

```python
view_aliases: Dict[int, List] = {}
param_data_ptr_to_param = {
    p.data.untyped_storage().data_ptr(): p for p in module.parameters()
}
```

```python
for k, v in module.state_dict(keep_vars=True).items():
    dev = src_to_dev.get(id(v))
    if dev is None:
        dev = v.to(device, non_blocking=True)
        src_to_dev[id(v)] = dev
    device_state[k] = dev
```

```python
sub.__dict__[attr_name] = dev_tensor.as_strided(size, stride, offset)
```

- Reviewed files: `python/sglang/srt/utils/offloader.py`, `test/registered/unit/utils/test_offloader_tied_params.py`.
- Validation implications: this remains a Qwen3.6 radar PR because Qwen3.6 shares the hybrid GDN/linear-attention risk surface. Regression lanes should cover `--cpu-offload-gb`, tied parameters, cached conv-weight views, and a no-offload baseline.

## Next Work

Build smoke tests for text-only, image, video, reasoning, tool calls, and MTP. Reuse Qwen3-Next/Qwen3.5 validation for CPU offload and hybrid cache issues.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3.6` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config | `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | quantization | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | tests/benchmarks | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-22 | [#23486](https://github.com/sgl-project/sglang/pull/23486) | merged | docs(cookbook): add Qwen3.6-27B dense variant | docs/config | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |

### File-level PR diff reading notes

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- Link: https://github.com/sgl-project/sglang/pull/23034
- Status/date: `merged`, created 2026-04-17, merged 2026-04-17; author `zijiexia`.
- Diff scope read: `73` files, `+2214/-215`; areas: model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config; keywords: doc, spec, attention, config, cuda, cache, moe, quant, eagle, expert.
- Code diff details:
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: +---
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: +---
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunks: +---; symbols: per_token_group_quant_8bit, add
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45 (286 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s; Please consult the documentation below and [server_args.py](https://github.com
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` added +219/-0 (219 lines); hunks: +export const Qwen36Deployment = () => {
- Optimization/support interpretation: The concrete diff surface is `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`; keywords observed in patches: doc, spec, attention, config, cuda, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- Link: https://github.com/sgl-project/sglang/pull/23467
- Status/date: `merged`, created 2026-04-22, merged 2026-04-22; author `mickqian`.
- Diff scope read: `1` files, `+31/-4`; areas: quantization; keywords: config, fp8, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: def __getattr__(self, name):; def is_layer_skipped(; symbols: __getattr__, _module_path_match, names, is_layer_skipped
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/utils.py`; keywords observed in patches: config, fp8, kv, moe, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- Link: https://github.com/sgl-project/sglang/pull/23474
- Status/date: `open`, created 2026-04-22; author `kawaruko`.
- Diff scope read: `2` files, `+284/-8`; areas: tests/benchmarks; keywords: attention, cache, cuda, spec, test.
- Code diff details:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: +"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: import logging; def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:; symbols: maybe_offload_to_cpu, maybe_offload_to_cpu, forward
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`; keywords observed in patches: attention, cache, cuda, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23486 - docs(cookbook): add Qwen3.6-27B dense variant

- Link: https://github.com/sgl-project/sglang/pull/23486
- Status/date: `merged`, created 2026-04-22, merged 2026-04-22; author `JustinTong0323`.
- Diff scope read: `2` files, `+55/-17`; areas: docs/config; keywords: config, doc, fp8, moe, quant, spec, attention, expert, vision.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunks: ---; Qwen3.6 features a Gated Delta Networks combined with sparse Mixture-of-Experts
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunks: export const Qwen36Deployment = () => {; export const Qwen36Deployment = () => {
- Optimization/support interpretation: The concrete diff surface is `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; keywords observed in patches: config, doc, fp8, moe, quant, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 4; open PRs: 1.
- Open PRs to keep tracking: [#23474](https://github.com/sgl-project/sglang/pull/23474)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
