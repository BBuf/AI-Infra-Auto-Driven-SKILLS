# Qwen3.6 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Manual diff review date: `2026-04-23`
- Searched paths: Qwen3.6 docs/snippets, Qwen3.5 adjacent snippet, FP8 quantization skip logic, and hybrid offloader path.
- Searched PR terms: `Qwen3.6`, `Qwen36`, `qwen36`, `qwen3_6`, `modules_to_not_convert`, `cpu-offload-gb`, `hybrid linear-attn`.

## Runtime and Docs Surfaces

- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`
- `docs_new/docs.json`
- `python/sglang/srt/layers/quantization/utils.py`
- `python/sglang/srt/utils/offloader.py`
- `test/registered/unit/utils/test_offloader_tied_params.py`

## Diff-Reviewed PR Cards

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- Link: https://github.com/sgl-project/sglang/pull/23034
- State: merged at `2026-04-17T05:33:34Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `7324` lines, `73` files; Qwen3.6, Qwen3.5 deployment, docs navigation, and GLM/Qwen warning hunks reviewed manually.
- Motivation: Qwen3.6 had no SGLang cookbook entry or command generator, so users had to infer the correct reasoning parser, Qwen3-Coder tool parser, MTP flags, B200 attention backend, and Mamba scheduler rules from adjacent Qwen3.5/Qwen3-Next docs. The same PR also fixed cookbook navigation so the Qwen card lands on the newest Qwen page and rendered FP8-KV warnings consistently for adjacent Qwen3.5/GLM-5 docs.
- Key implementation: the patch adds `Qwen3.6.mdx`, registers it in `docs_new/docs.json`, redirects the cookbook Qwen card to Qwen3.6, and adds `qwen36-deployment.jsx`. The command generator forces Mamba V2 when MTP is enabled, emits `SGLANG_ENABLE_SPEC_V2=1`, attaches `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`, and adds `--attention-backend trtllm_mha` on B200. The Qwen3.5 generator receives the same MTP/Mamba rule, which matters because Qwen3.6 currently reuses the shared hybrid Qwen deployment assumptions instead of a dedicated runtime class.
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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`
  - snippets: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`
  - supporting docs cleanup: server/speculative/tool parser pages and notebook formatting hunks that do not change Qwen3.6 runtime behavior
- Validation implications: any Qwen3.6 doc update must regenerate/check the command for BF16 and FP8, with and without MTP, and verify the combined `--reasoning-parser qwen3` plus `--tool-call-parser qwen3_coder` path. B200 examples must keep `--attention-backend trtllm_mha`, and MTP examples must keep `SGLANG_ENABLE_SPEC_V2=1` plus Mamba V2.

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- Link: https://github.com/sgl-project/sglang/pull/23467
- State: merged at `2026-04-22T14:16:22Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `174` lines, `1` file.
- Motivation: Qwen3.6-27B-FP8 configs contain MoE-template names such as `model.language_model.layers.N.mlp.gate` under `modules_to_not_convert`. The old loader used substring matching (`ignored in prefix`), so `mlp.gate` incorrectly matched dense fused names such as `mlp.gate_up_proj`. SGLang then instantiated those MLPs as BF16 while the checkpoint provided FP8 scales, producing `weight_scale_inv not found` warnings and broken outputs. The same failure mode could affect Qwen3.5 because `linear_attn.in_proj_a` can collide with the fused `in_proj_ba` path when the HF quant config does not ship `packed_modules_mapping`.
- Key implementation: `python/sglang/srt/layers/quantization/utils.py` now matches ignored modules only on dotted module boundaries through `_module_path_match`. It also adds `_FALLBACK_FUSED_SHARDS` so fused projections can still be checked shard-by-shard when the quant config omits `packed_modules_mapping`. `is_layer_skipped` first chooses the explicit fused mapping when present and otherwise falls back to the built-in fused-shard table.
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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/utils.py`
- Validation implications: Qwen3.6-27B-FP8 load should produce zero `weight_scale_inv not found` warnings for the dense MLP path. Regression must also include Qwen3.5 FP8 because the fallback fused-shard map deliberately preserves `in_proj_a` versus `in_proj_ba` behavior.

### PR #23486 - docs(cookbook): add Qwen3.6-27B dense variant

- Link: https://github.com/sgl-project/sglang/pull/23486
- State: merged at `2026-04-22T17:22:46Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `198` lines, `2` files.
- Motivation: the first Qwen3.6 cookbook page described only the 35B-A3B MoE checkpoint, but Qwen3.6 also released a 27B dense variant and matching FP8 checkpoint. Without an explicit model-size selector, the generated command always targeted `Qwen/Qwen3.6-35B-A3B`, leaving the dense model under-documented even though it shares the same reasoning, tool-call, multimodal, and MTP launch semantics.
- Key implementation: the cookbook page changes the model introduction, available-model table, hardware estimates, and invocation section to cover both 35B-A3B MoE and 27B dense. The deployment snippet adds a `modelSize` radio and nests `modelConfigs` under model size; `generateCommand()` derives the Hugging Face repo name from `sizeConfig.baseName`, so `--model-path` becomes the only model-size-dependent flag. The install command is narrowed from `uv pip install "sglang[all]"` to `uv pip install sglang` because diffusion/tracing/http2 extras are irrelevant to this autoregressive VLM path.
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

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
  - snippet: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- Validation implications: generated commands must be checked for all four model-path variants: `Qwen3.6-35B-A3B`, `Qwen3.6-35B-A3B-FP8`, `Qwen3.6-27B`, and `Qwen3.6-27B-FP8`. The PR body reports H200 TP=2 MMMU sanity results after the FP8 loader fix from #23467: BF16 `55.1%`, FP8 `53.0%`, within Wilson 95% CI.

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- Link: https://github.com/sgl-project/sglang/pull/23474
- State: open as of `2026-04-23`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `395` lines, `2` files.
- Motivation: hybrid linear-attention models such as Qwen3-Next, Qwen3.5, and Kimi-Linear cache tensor views of weights as plain attributes, for example a `conv1d.weight.view(...)` saved on a sibling attention module. When `--cpu-offload-gb` rebinds `Parameter.data` to pinned CPU memory, later checkpoint loading writes into the new CPU storage while the cached tensor view still points at the original random GPU storage. The same model family can register tied parameters under multiple state-dict keys; blindly moving every key with `.to(device)` creates distinct device tensors and makes `functional_call(..., tie_weights=True)` reject the call.
- Key implementation: `OffloaderV1.maybe_offload_to_cpu()` records plain tensor attributes that alias each parameter's original storage before offloading. At forward time it builds one shared device tensor per tied source parameter, reconstructs alias views with `as_strided(size, stride, offset)`, calls `functional_call`, then restores the old attributes. The test file creates minimal tied-parameter and cached-view modules to reproduce the two failure modes without needing a full Qwen checkpoint.
- Key code excerpts:

```python
view_aliases: Dict[int, List] = {}
param_data_ptr_to_param = {
    p.data.untyped_storage().data_ptr(): p for p in module.parameters()
}
```

```python
src_to_dev = {}
device_state = {}
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

- Reviewed files:
  - runtime: `python/sglang/srt/utils/offloader.py`
  - tests: `test/registered/unit/utils/test_offloader_tied_params.py`
- Validation implications: keep this PR on the Qwen3.6 radar even though it is not Qwen3.6-specific, because Qwen3.6 shares the same hybrid GDN/linear-attention risk profile. Regressions should cover `--cpu-offload-gb`, tied `A_log`/`dt_bias`-style parameters, cached conv-weight views, and a normal no-offload run to ensure alias restore does not leak across forwards.

## Cookbook Evidence

- sgl-cookbook [#245](https://github.com/sgl-project/sgl-cookbook/pull/245): Qwen cookbook refresh. This is a documentation parity pointer only; do not treat it as a runtime optimization PR unless its diff is reviewed separately.

## Validation Notes

- Qwen3.6 currently depends more on docs/config accuracy and shared hybrid runtime than on a dedicated `qwen3_6.py` model class.
- Because docs recommend both reasoning and tool-call parsers, parser validation must cover the combined mode.
- Multimodal examples should be tested separately from text-only MTP.
- Do not create a Qwen3.6-specific model fork unless shared Qwen3-Next/Qwen3.5/Qwen VLM runtime paths are proven insufficient.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Qwen3.6 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config | `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | quantization | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | tests/benchmarks | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-22 | [#23486](https://github.com/sgl-project/sglang/pull/23486) | merged | docs(cookbook): add Qwen3.6-27B dense variant | docs/config | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |

## Diff Cards

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


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
