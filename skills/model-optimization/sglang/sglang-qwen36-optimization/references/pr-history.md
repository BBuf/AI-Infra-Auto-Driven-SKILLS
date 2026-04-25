# sglang Qwen3.6 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` | [#23486](https://github.com/sgl-project/sglang/pull/23486) |
| `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` | [#23486](https://github.com/sgl-project/sglang/pull/23486) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-22 | [#23486](https://github.com/sgl-project/sglang/pull/23486) | merged | docs(cookbook): add Qwen3.6-27B dense variant | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- Link: https://github.com/sgl-project/sglang/pull/23034
- Status/date: merged / 2026-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 73 files, +2214/-215, 3198 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs"; model line: Qwen3.6; category: bug fix; main diff: `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`; PR body summary: - **Add Qwen3.6 documentation**: New full deployment guide for Qwen3.6-35B-A3B (hybrid GDN + sparse MoE architecture) with JSX deployment snippet, covering MTP, tool calling (`q....
- Key implementation: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ===="); `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that, touching `NewModelDetector, that`; `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471.
- Code diff details:
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ====")
  - `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunks: -0,0 +1,299; symbols: per_token_group_quant_8bit, add
- Key code excerpts:

```diff
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -207,7 +207,7 @@ print_highlight("==== Text ====")
-The reasoning separation is enable by default when specify .
+The reasoning separation is enable by default when specify .
@@ -226,7 +226,7 @@ print_highlight("==== Original Output ====")
-### SGLang Native API
+### SGLang Native API
@@ -315,4 +315,3 @@ llm.shutdown()
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -718,7 +718,7 @@ for tool_call in tool_calls:
-> **Note:**
+> **Note:**
@@ -738,4 +738,3 @@ terminate_process(server_process)
diff -- docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx
@@ -0,0 +1,509 @@
+---
+title: "DP, DPA and SGLang DP Router"
+metatags:
```

- Reviewed files:
  - docs: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3; `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2; `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0; `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0; `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45
- Risk and verification: This is mostly docs/examples in `docs_new/.gitignore`, `docs_new/cards/logos/google.png`, `docs_new/cards/logos/mova.png`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- Link: https://github.com/sgl-project/sglang/pull/23474
- Status/date: open / 2026-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +284/-8, 330 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models"; model line: Qwen3.6; category: bug fix; main diff: `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`; PR body summary: Fixes #23150. `--cpu-offload-gb > 0` was broken on hybrid linear-attention models (Qwen3-Next, Qwen3.5, Kimi-Linear): the first `/v1/chat/completions` request raised While fixin....
- Key implementation: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent, touching `_TiedChild, __init__, forward`; `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward, touching `maybe_offload_to_cpu, forward`.
- Code diff details:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward
- Key code excerpts:

```diff
diff -- test/registered/unit/utils/test_offloader_tied_params.py
@@ -0,0 +1,199 @@
+"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).
+Two failure modes caused the Qwen3-Next / Qwen3.5 CPU-offload regression:
+1. **Tied parameters**: a single nn.Parameter is registered under both a parent
+   and a child module (Qwen3GatedDeltaNet + RadixLinearAttention share
+   ``A_log`` / ``dt_bias``). state_dict() then lists the same tensor under
+   multiple keys, and functional_call(..., tie_weights=True) rejects it when
diff -- python/sglang/srt/utils/offloader.py
@@ -1,7 +1,7 @@
-from typing import Callable, Generator, List, Optional
+from typing import Callable, Dict, Generator, List, Optional
@@ -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
+        # Record tensor views that alias each parameter's *original* storage
+        # BEFORE we rebind .data to pinned CPU memory. Some hybrid linear-attn
+        # models (e.g. Qwen3-Next) cache such views, which would otherwise point
```

- Reviewed files:
  - tests: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0
  - runtime: `python/sglang/srt/utils/offloader.py` modified +85/-8
- Risk and verification: The diff ships test coverage in `test/registered/unit/utils/test_offloader_tied_params.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- Link: https://github.com/sgl-project/sglang/pull/23467
- Status/date: merged / 2026-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +31/-4, 63 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert"; model line: Qwen3.6; category: bug fix; main diff: `python/sglang/srt/layers/quantization/utils.py`; PR body summary: - `is_layer_skipped` uses naive substring match (`ignored in prefix`) on `modules_to_not_convert` entries, which silently fires when an entry is a prefix-substring of a fused li....
- Key implementation: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped, touching `__getattr__, _module_path_match, is_layer_skipped`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/utils.py
@@ -43,6 +43,28 @@ def __getattr__(self, name):
+def _module_path_match(ignored: str, prefix: str) -> bool:
+    # Match on dotted module-path boundaries so that `mlp.gate` does NOT
+    # match `mlp.gate_up_proj`. Needed for quant configs (e.g. Qwen3.6-FP8)
+    # whose `modules_to_not_convert` lists MoE-template names like `mlp.gate`
+    # that collide with fused dense MLP names by plain substring.
+    if ignored == prefix:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23486 - docs(cookbook): add Qwen3.6-27B dense variant

- Link: https://github.com/sgl-project/sglang/pull/23486
- Status/date: merged / 2026-04-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; associated commits `de962f327432`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +55/-17, 170 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): add Qwen3.6-27B dense variant"; model line: Qwen3.6; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; PR body summary: Qwen3.6 ships a 27B dense variant (`Qwen3.6-27B` / `Qwen3.6-27B-FP8`) alongside the existing 35B-A3B MoE. Update the cookbook page and deployment snippet to cover both. - Rewrit....
- Key implementation: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunks: -1,26 +1,29; -29,30 +32,43 @@ Qwen3.6 features a Gated Delta Networks combined with sparse...; `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunks: -10,6 +10,14 @@ export const Qwen36Deployment = () => {; -66,9 +74,18 @@ export const Qwen36Deployment = () => {.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunks: -1,26 +1,29; -29,30 +32,43 @@ Qwen3.6 features a Gated Delta Networks combined with sparse...
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunks: -10,6 +10,14 @@ export const Qwen36Deployment = () => {; -66,9 +74,18 @@ export const Qwen36Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx
@@ -1,26 +1,29 @@
-    description: "Deploy Qwen3.6 with SGLang - open-weight 35B MoE multimodal model with 3B active parameters, thinking preservation, tool calling, MTP, and long-context support."
+    description: "Deploy Qwen3.6 with SGLang - open-weight multimodal series with a 35B MoE (3B active) variant and a 27B dense variant, hybrid reasoning, tool calling, MTP, and l
-[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is the first open-weight variant of the Qwen3.6 series developed by Alibaba. Built on direct feedback from the commu
+The Qwen3.6 series is developed by Alibaba. Built on direct feedback from the community, Qwen3.6 prioritizes stability and real-world utility, delivering substantial upgrades in a
-Qwen3.6 features a Gated Delta Networks combined with sparse Mixture-of-Experts architecture (35B total parameters, 3B activated), supporting multimodal inputs (text, image, video
+- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — **Sparse MoE** (35B total, 3B active) on a Gated Delta Networks backbone.
diff -- docs_new/src/snippets/autoregressive/qwen36-deployment.jsx
@@ -10,6 +10,14 @@ export const Qwen36Deployment = () => {
+    modelSize: {
+      name: 'modelSize',
+      title: 'Model Size',
+      items: [
+        { id: '35b-a3b', label: '35B-A3B (MoE)', default: true },
+        { id: '27b', label: '27B (Dense)', default: false },
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10; `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.
