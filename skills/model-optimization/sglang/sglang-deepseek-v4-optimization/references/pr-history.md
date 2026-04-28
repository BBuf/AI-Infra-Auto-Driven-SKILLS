# sglang DeepSeek V4 PR Diff Audit Reference

- Rebuilt on: 2026-04-28
- Source baseline: `sgl-project/sglang` `origin/main` commit `6fbad22fe`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 0
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23628](https://github.com/sgl-project/sglang/pull/23628), [#23684](https://github.com/sgl-project/sglang/pull/23684), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697) |
| `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23617](https://github.com/sgl-project/sglang/pull/23617), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23634](https://github.com/sgl-project/sglang/pull/23634), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23690](https://github.com/sgl-project/sglang/pull/23690), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23698](https://github.com/sgl-project/sglang/pull/23698) |
| `python/sglang/srt/models/deepseek_v4.py` | [#23787](https://github.com/sgl-project/sglang/pull/23787), [#23832](https://github.com/sgl-project/sglang/pull/23832) |
| `python/sglang/srt/models/deepseek_v4_nextn.py` | [#23787](https://github.com/sgl-project/sglang/pull/23787) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | [#23787](https://github.com/sgl-project/sglang/pull/23787), [#23832](https://github.com/sgl-project/sglang/pull/23832) |
| `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` | [#23756](https://github.com/sgl-project/sglang/pull/23756) |
| `python/sglang/srt/models/deepseek_v2.py` | [#23776](https://github.com/sgl-project/sglang/pull/23776), [#23787](https://github.com/sgl-project/sglang/pull/23787) |
| `scripts/bench_gpqa_aime.py` | [#23810](https://github.com/sgl-project/sglang/pull/23810) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23617](https://github.com/sgl-project/sglang/pull/23617) | merged | Further update Deepseek V4 docs | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-24 | [#23628](https://github.com/sgl-project/sglang/pull/23628) | merged | [codex] docs: note H200 DeepSeek-V4 checkpoint | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23622](https://github.com/sgl-project/sglang/pull/23622) | merged | Again update DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23634](https://github.com/sgl-project/sglang/pull/23634) | merged | Update pro fp8 checkpoint in DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23684](https://github.com/sgl-project/sglang/pull/23684) | merged | docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23689](https://github.com/sgl-project/sglang/pull/23689) | merged | docs(DeepSeek-V4): mark b200\|small\|pd-disagg + h200\|small\|{cp,pd-disagg} verified | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23691](https://github.com/sgl-project/sglang/pull/23691) | merged | docs(DeepSeek-V4): mark gb300\|{small,big}\|{cp,pd-disagg} verified + GB300-specific fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23690](https://github.com/sgl-project/sglang/pull/23690) | merged | Small udpate gb300 recipe for deepseek v4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23697](https://github.com/sgl-project/sglang/pull/23697) | merged | update: b300 container for dsv4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23698](https://github.com/sgl-project/sglang/pull/23698) | merged | docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-26 | [#23725](https://github.com/sgl-project/sglang/pull/23725) | merged | docs(DeepSeek-V4): add GB200 platform to cookbook recipe | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-26 | [#23737](https://github.com/sgl-project/sglang/pull/23737) | merged | docs(DeepSeek-V4): mark gb200\|big\|low-latency verified | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-26 | [#23742](https://github.com/sgl-project/sglang/pull/23742) | merged | docs(DeepSeek-V4): add h200\|big verified recipes + tune H200 Pro parameters | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-27 | [#23756](https://github.com/sgl-project/sglang/pull/23756) | merged | feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch | `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py` |
| 2026-04-27 | [#23776](https://github.com/sgl-project/sglang/pull/23776) | merged | [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-27 | [#23787](https://github.com/sgl-project/sglang/pull/23787) | merged | amd/deepseek_v4 integration 1/N - 0426 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` |
| 2026-04-27 | [#23810](https://github.com/sgl-project/sglang/pull/23810) | merged | Add benchmarking scripts for deepseek v4 | `scripts/bench_gpqa_aime.py` |
| 2026-04-27 | [#23817](https://github.com/sgl-project/sglang/pull/23817) | merged | docs: verify GB300 Pro DeepSeek V4 recipes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-27 | [#23832](https://github.com/sgl-project/sglang/pull/23832) | merged | amd/deepseek_v4 integration 2/N - cuda graph 0426 | `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-28 | [#23883](https://github.com/sgl-project/sglang/pull/23883) | merged | Enable DeepGemm warmup in DeepSeek-V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #23605 - Add DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23605
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `492883c8ca66`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1024/-1, 1041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0 (569 lines); hunks: -0,0 +1,569
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0 (453 lines); hunks: -0,0 +1,453
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -0,0 +1,569 @@
+export const DeepSeekV4Deployment = () => {
+  // DeepSeek-V4 deployment matrix (small / real checkpoint):
+  //   Hardware × Recipe → concrete launch command.
+  //
+  //   Hardware (quantization determined by GPU generation):
+  //     B200  → FP4 weights, Flash TP=4 / Pro TP=8 single-node
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -0,0 +1,453 @@
+---
+title: DeepSeek-V4
+metatags:
+    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek. Blackwell deployments use the FP4 checkpoint; Hopper deployments use the FP8 checkpoi
+tag: NEW
+---
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` added +569/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` added +453/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23617 - Further update Deepseek V4 docs

- Link: https://github.com/sgl-project/sglang/pull/23617
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `734e1e2965cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-6, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Further update Deepseek V4 docs"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6 (11 lines); hunks: -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -137,12 +137,11 @@ export const DeepSeekV4Deployment = () => {
-    // H200 needs a separate FP8-only Instruct ckpt (Flash / Pro public repos
-    // ship FP4-mixed weights). That ckpt is still being uploaded, so we emit a
-    // placeholder that fails loudly on copy-paste instead of silently pulling
-    // the wrong weights. Replace with the real slug once Hopper ckpts are public.
-    "h200|small":  { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Flash-hopper>", tp: 4,  multinode: false },
-    "h200|big":    { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-hopper>",   tp: 16, multinode: true, nnodes: 2 },
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-6
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23628 - [codex] docs: note H200 DeepSeek-V4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/23628
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `1a37e57fb1ae`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[codex] docs: note H200 DeepSeek-V4 checkpoint"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; PR body summary: - Add a highlighted note in the DeepSeek-V4 deployment section for H200 GPU users. - Clarify that H200 deployments should use the SGLang checkpoint under `sgl-project` instead o....
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -99,6 +99,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+<Note>
+For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
+</Note>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23622 - Again update DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23622
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3a620cb761ff`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +32/-9, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Again update DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9 (28 lines); hunks: -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {; -161,7 +161,16 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0 (13 lines); hunks: -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -42,11 +42,11 @@ export const DeepSeekV4Deployment = () => {
-        { id: "low-latency",    label: "Low-Latency",      default: true,  subtitle: "MTP 3/4" },
-        { id: "balanced",       label: "Balanced",         default: false, subtitle: "MTP 1/2 + DeepEP" },
-        { id: "max-throughput", label: "Max-Throughput",   default: false, subtitle: "DP + DeepEP" },
-        { id: "cp",             label: "Context-Parallel", default: false, subtitle: "long prompts" },
-        { id: "pd-disagg",      label: "PD-Disagg",        default: false, subtitle: "1P + 1D + router" },
+        { id: "low-latency",    label: "Low-Latency",      default: true  },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -95,6 +95,19 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+For how to actually launch one of these images, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). A minimal example (substitute the
+'''bash Command
+docker run --gpus all \
+    --shm-size 32g \
+    -p 30000:30000 \
+    -v ~/.cache/huggingface:/root/.cache/huggingface \
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +19/-9; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +13/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23634 - Update pro fp8 checkpoint in DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23634
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `92bb5c6bbee9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update pro fp8 checkpoint in DeepSeek V4 cookbook"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2 (4 lines); hunks: -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -139,9 +139,9 @@ export const DeepSeekV4Deployment = () => {
-    // repackagings; Flash is public, Pro is still being uploaded.
+    // repackagings for both variants.
-    "h200|big":    { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-FP8>",     tp: 16, multinode: true, nnodes: 2 },
+    "h200|big":    { slug: "sgl-project/DeepSeek-V4-Pro-FP8",          tp: 16, multinode: true, nnodes: 2 },
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23684 - docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models

- Link: https://github.com/sgl-project/sglang/pull/23684
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `fd401c2fb451`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; no usable PR-body summary.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*....
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -147,6 +147,10 @@ The generator currently picks values on the **conservative*...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -147,6 +147,10 @@ The generator currently picks values on the **conservative** side (mirroring an
+**Base model usage**
+In order to use base models, please enable `SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1` and use latest code, before the next round of testing matrix is finished.
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23689 - docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified

- Link: https://github.com/sgl-project/sglang/pull/23689
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `d2c61acf2597`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +22/-1, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0 (14 lines); hunks: -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {; -387,6 +399,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1 (9 lines); hunks: -145,7 +145,14 @@ The generator currently picks values on the **conservative*...
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -164,14 +164,26 @@ export const DeepSeekV4Deployment = () => {
+    "b200|small|pd-disagg",
+    "h200|small|cp",
+    "h200|small|pd-disagg",
+    // h200|big|pd-disagg: pending verification (needs 4-node H200 cluster with
+    //   shared IB fabric: 2-node prefill + 2-node decode).
+  // Recipes whose command is intentionally not yet provided (e.g. blocked by an
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -145,7 +145,14 @@ The generator currently picks values on the **conservative** side (mirroring an
-The H200 image and checkpoint are currently being uploaded — public path coming shortly.
+H200 image (`lmsysorg/sglang:deepseek-v4-hopper`) and FP8 checkpoints
+(`sgl-project/DeepSeek-V4-Flash-FP8`, `sgl-project/DeepSeek-V4-Pro-FP8`) are
+publicly available.
+PD-Disagg recipes on H200 may require `docker run --privileged --ulimit memlock=-1`
+(or `--device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK`) so mooncake
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +14/-0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +8/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23691 - docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes

- Link: https://github.com/sgl-project/sglang/pull/23691
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `8a395994edcf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +56/-5, 113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5 (54 lines); hunks: -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {; -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0 (7 lines); hunks: -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpo...
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -176,6 +176,10 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|small|cp",
+    "gb300|big|cp",
+    "gb300|small|pd-disagg",
+    "gb300|big|pd-disagg",
@@ -372,7 +376,17 @@ export const DeepSeekV4Deployment = () => {
-      flags.push("  --mem-fraction-static 0.78");
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -158,6 +158,13 @@ TCP, which can lead to garbled KV transfer on large checkpoints.
+**GB300 PD-Disagg cross-pod MNNVL**
+On some GB300 clusters with cross-pod KV transfer over NVLink, mooncake may
+fail with `nvlink_transport.cpp:497 Requested address ... not found!`. If
+this happens, prepend `MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`
+to both prefill and decode `sglang serve` commands.
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +49/-5; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +7/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23690 - Small udpate gb300 recipe for deepseek v4

- Link: https://github.com/sgl-project/sglang/pull/23690
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `69485a176c87`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Small udpate gb300 recipe for deepseek v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -172,6 +172,9 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|small|low-latency",
+    "gb300|small|balanced",
+    "gb300|small|max-throughput",
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23697 - update: b300 container for dsv4

- Link: https://github.com/sgl-project/sglang/pull/23697
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `0d224e505333`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update: b300 container for dsv4"; model line: DeepSeek V4; category: model implementation change; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2 (9 lines); hunks: -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {; -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0 (4 lines); hunks: -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -26,6 +26,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "b300",  label: "B300 (FP4)",  default: false  },
@@ -222,7 +223,9 @@ export const DeepSeekV4Deployment = () => {
-    const { hardware, modelSize, recipe, reasoningParser, toolcall } = values;
+    const { hardware: rawHardware, modelSize, recipe, reasoningParser, toolcall } = values;
+    // B300 usage is identical to B200 — alias so we don't duplicate every spec entry.
+    const hardware = rawHardware === "b300" ? "b200" : rawHardware;
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -80,6 +80,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA B300</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>lmsysorg/sglang:deepseek-v4-b300</code></td>
+    </tr>
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +7/-2; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +4/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23698 - docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9

- Link: https://github.com/sgl-project/sglang/pull/23698
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `880599cd430f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-3, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; no usable PR-body summary.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3 (8 lines); hunks: -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -495,11 +495,13 @@ export const DeepSeekV4Deployment = () => {
-        // OOM during CG capture. Verified working on 2026-04-25 (journal
-        // 2026-04-25-001 Cell D, Δ10).
+        // OOM during CG capture. mem-frac sweep at 0.83 / 0.87 / 0.89 / 0.91
+        // all pass static smoke; 0.9 picked as the default — leaves
+        // ~14 GB / GPU post-CG headroom for mooncake transfer + activation
+        // peaks while giving ~1M-token KV pool.
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +5/-3
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23725 - docs(DeepSeek-V4): add GB200 platform to cookbook recipe

- Link: https://github.com/sgl-project/sglang/pull/23725
- Status/date: merged / 2026-04-26T03:54:56Z
- Trace source: current-main DeepSeek-V4 cookbook and command generator.
- Diff scope read: GitHub Pull Request files API returned 2 files, +58/-8; this card covers the complete docs diff.
- Motivation: Title: "docs(DeepSeek-V4): add GB200 platform to cookbook recipe"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; adds GB200 as an explicit Blackwell platform instead of forcing users to infer it from B200/GB300.
- Key implementation: adds GB200 hardware choices, Docker image row, and platform-specific recipe entries; keeps checkpoint dtype aligned with the Blackwell FP4 path.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` extends the hardware selector and recipe table.
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` documents the GB200 Docker image row.
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
+        { id: "gb200", label: "GB200 (FP4)", default: false },
+    "gb200|small": { slug: "deepseek-ai/DeepSeek-V4-Flash", tp: 4, multinode: false },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
+      <td>NVIDIA GB200</td>
+      <td><code>lmsysorg/sglang:deepseek-v4-gb200</code></td>
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`
- Risk and verification: Generated GB200 commands must continue to use Blackwell-compatible checkpoints, image tags, and TP/recipe defaults.

### PR #23737 - docs(DeepSeek-V4): mark gb200|big|low-latency verified

- Link: https://github.com/sgl-project/sglang/pull/23737
- Status/date: merged / 2026-04-26T18:16:00Z
- Trace source: current-main DeepSeek-V4 deployment snippet.
- Diff scope read: GitHub Pull Request files API returned 1 file, +1/-0; this card covers the complete docs diff.
- Motivation: Title: "docs(DeepSeek-V4): mark gb200|big|low-latency verified"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; promotes one GB200 Pro recipe from unverified to verified.
- Key implementation: updates the recipe metadata for `gb200|big|low-latency` so the generator emits a runnable command instead of a commented placeholder.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` flips the GB200 Pro low-latency verification bit.
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
+      verified: true,
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Risk and verification: Re-check that only the verified GB200 cell is uncommented and that adjacent GB200 cells keep their intended verification state.

### PR #23742 - docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters

- Link: https://github.com/sgl-project/sglang/pull/23742
- Status/date: merged / 2026-04-26T04:44:52Z
- Trace source: current-main DeepSeek-V4 deployment snippet.
- Diff scope read: GitHub Pull Request files API returned 1 file, +22/-8; this card covers the complete command-generator diff.
- Motivation: Title: "docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; tunes H200 Pro recipes and marks verified cells.
- Key implementation: adjusts H200 Pro TP/multinode parameters and verification metadata for low-latency, balanced, and throughput recipes.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` changes H200 Pro recipe fields and verification state.
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
+    "h200|big": { slug: "sgl-project/DeepSeek-V4-Pro-FP8", tp: 16, multinode: true, nnodes: 2 },
+      memFractionStatic: 0.7,
+      swaFullTokensRatio: 0.3,
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Risk and verification: H200 Pro commands are multi-node and checkpoint-specific; verify FP8 slug, TP=16, node count, and SWA ratio before use.

### PR #23756 - feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch

- Link: https://github.com/sgl-project/sglang/pull/23756
- Status/date: merged / 2026-04-27T23:34:35Z
- Trace source: current-main DeepGEMM compile utility and environment flag.
- Diff scope read: GitHub Pull Request files API returned 2 files, +47/-12; this card covers the complete runtime diff.
- Motivation: Title: "feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py`; adds a deployment-facing fast warmup toggle for DeepGEMM compilation.
- Key implementation: introduces the `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` environment knob and alters DeepGEMM warmup compilation behavior so DeepSeek-V4 deployments can reduce warmup latency when the flag is enabled.
- Code diff details:
  - `python/sglang/srt/environ.py` adds the environment variable.
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` branches warmup/compilation behavior on the flag.
- Key code excerpts:

```diff
diff -- python/sglang/srt/environ.py
+SGLANG_JIT_DEEPGEMM_FAST_WARMUP = bool(os.getenv("SGLANG_JIT_DEEPGEMM_FAST_WARMUP"))
diff -- python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py
+if SGLANG_JIT_DEEPGEMM_FAST_WARMUP:
+    # use fast warmup path for DeepGEMM kernels
```

- Reviewed files:
  - runtime: `python/sglang/srt/environ.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`
- Risk and verification: Warmup changes can hide compile misses; validate first-token latency, kernel compilation cache state, and correctness with the flag both enabled and disabled.

### PR #23776 - [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP

- Link: https://github.com/sgl-project/sglang/pull/23776
- Status/date: merged / 2026-04-27T02:50:46Z
- Trace source: current-main shared DeepSeek model code used by V4.
- Diff scope read: GitHub Pull Request files API returned 1 file, +10/-0; this card covers the complete model diff.
- Motivation: Title: "[DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; fixes bad chat output by honoring V4's SwiGLU clamp requirement in the shared MLP path.
- Key implementation: reads `swiglu_limit` from config and clamps the relevant activation path in `DeepseekV2MLP`, which DeepSeek-V4 reuses.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` adds config plumbing and activation clamp logic.
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
+        self.swiglu_limit = getattr(config, "swiglu_limit", None)
+        if self.swiglu_limit is not None:
+            gate = gate.clamp(min=None, max=self.swiglu_limit)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py`
- Risk and verification: Re-test DeepSeek-V4 chat generation and any DeepSeek-V2/V3 model that shares this MLP path; the clamp must be gated by config to avoid unintended behavior.

### PR #23787 - amd/deepseek_v4 integration 1/N - 0426

- Link: https://github.com/sgl-project/sglang/pull/23787
- Status/date: merged / 2026-04-27T02:13:10Z
- Trace source: current-main DeepSeek-V4 runtime bring-up files.
- Diff scope read: GitHub Pull Request files API returned 128 files, +18341/-879; this card prioritizes DeepSeek-V4 model, attention, tokenizer/parser, memory pool, JIT kernels, and docs.
- Motivation: Title: "amd/deepseek_v4 integration 1/N - 0426"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`; lands the first full DeepSeek-V4 runtime integration after the docs-only matrix.
- Key implementation: adds `deepseek_v4.py` and `deepseek_v4_nextn.py`, DSV4 tokenizer/encoding and function-call detector, compressed attention/indexer stack, DeepSeek-V4 attention backend, memory pool, JIT kernels, FlashMLA tests, and model config wiring.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` adds the main model runtime.
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` adds the V4 attention backend.
  - `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` and `python/sglang/srt/function_call/deepseekv4_detector.py` add tokenizer/parser support.
  - `python/sglang/jit_kernel/deepseek_v4.py` and `python/sglang/jit_kernel/csrc/deepseek_v4/*` add V4 JIT kernels.
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
+class DeepseekV4ForCausalLM(nn.Module):
+class DeepseekV4Model(nn.Module):
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
+class DeepseekV4AttentionBackend:
diff -- python/sglang/srt/function_call/deepseekv4_detector.py
+class DeepSeekV4Detector:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/mem_cache/deepseekv4_memory_pool.py`
  - parser/kernel/docs: `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`, `python/sglang/srt/function_call/deepseekv4_detector.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Risk and verification: This is the main runtime landing; verify model load, parser/tool-call output, compressed attention, memory pool sizing, JIT kernels, MTP/nextn, and documented launch commands.

### PR #23810 - Add benchmarking scripts for deepseek v4

- Link: https://github.com/sgl-project/sglang/pull/23810
- Status/date: merged / 2026-04-27T07:49:22Z
- Trace source: current-main DeepSeek-V4 benchmark script.
- Diff scope read: GitHub Pull Request files API returned 1 file, +243/-0; this card covers the complete benchmark diff.
- Motivation: Title: "Add benchmarking scripts for deepseek v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `scripts/bench_gpqa_aime.py`; adds a repeatable GPQA/AIME-oriented script for DeepSeek-V4 evaluation workflows.
- Key implementation: adds CLI/script scaffolding for benchmark prompts, generation calls, and result collection around GPQA/AIME tasks.
- Code diff details:
  - `scripts/bench_gpqa_aime.py` is a new benchmark helper.
- Key code excerpts:

```diff
diff -- scripts/bench_gpqa_aime.py
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--model", type=str, required=True)
+    parser.add_argument("--base-url", type=str, default="http://localhost:30000/v1")
```

- Reviewed files:
  - scripts: `scripts/bench_gpqa_aime.py`
- Risk and verification: Keep benchmark defaults aligned with current server CLI and tokenizer behavior; use it as evaluation support, not as proof of runtime correctness by itself.

### PR #23817 - docs: verify GB300 Pro DeepSeek V4 recipes

- Link: https://github.com/sgl-project/sglang/pull/23817
- Status/date: merged / 2026-04-27T07:21:27Z
- Trace source: current-main DeepSeek-V4 deployment snippet.
- Diff scope read: GitHub Pull Request files API returned 1 file, +6/-0; this card covers the complete docs diff.
- Motivation: Title: "docs: verify GB300 Pro DeepSeek V4 recipes"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; marks additional GB300 Pro recipe cells verified.
- Key implementation: flips GB300 Pro verification metadata so the generator emits active commands for the now-validated recipes.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` changes GB300 Pro recipe verification entries.
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
+      verified: true,
+      verifiedNote: "GB300 Pro recipe verified"
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Risk and verification: Check that command-generator comments match verified/unverified state and that Pro GB300 memory settings still match the latest recipe.

### PR #23832 - amd/deepseek_v4 integration 2/N - cuda graph 0426

- Link: https://github.com/sgl-project/sglang/pull/23832
- Status/date: merged / 2026-04-27T15:44:44Z
- Trace source: current-main DeepSeek-V4 CUDA-graph and attention/indexer files.
- Diff scope read: GitHub Pull Request files API returned 26 files, +534/-92; this card prioritizes CUDA-graph replay metadata, compressed indexer, DeepSeek-V4 attention backend, and shared DeepSeek model changes.
- Motivation: Title: "amd/deepseek_v4 integration 2/N - cuda graph 0426"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; adds graph-capture compatibility to the V4 attention/indexer path.
- Key implementation: extends attention backend replay metadata signatures with `out_cache_loc` and `actual_forward_mode`, vectorizes/caches compressed indexer helper tensors to avoid device allocations during graph capture, and updates DeepSeek model/runner plumbing for CUDA-graph replay.
- Code diff details:
  - `python/sglang/srt/layers/attention/base_attn_backend.py` adds replay metadata parameters.
  - `python/sglang/srt/layers/attention/compressed/indexer.py` vectorizes FP8 paged MQA logits and caches arange tensors.
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` wires V4 replay behavior.
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/base_attn_backend.py
+        out_cache_loc: Optional[torch.Tensor] = None,
+        actual_forward_mode: Optional[ForwardMode] = None,
diff -- python/sglang/srt/layers/attention/compressed/indexer.py
+_arange_cache: Dict[str, torch.Tensor] = {}
+    """Vectorized implementation that avoids .item() and Python loops,
+    making it compatible with CUDA graph capture."""
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/base_attn_backend.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py`
- Risk and verification: Graph capture bugs can be silent until replay; validate prefill/decode, long context, compressed attention, and CUDA/HIP graph replay paths.

### PR #23883 - Enable DeepGemm warmup in DeepSeek-V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23883
- Status/date: merged / 2026-04-28T01:41:12Z
- Trace source: current-main DeepSeek-V4 deployment snippet.
- Diff scope read: GitHub Pull Request files API returned 1 file, +3/-5; this card covers the complete docs diff.
- Motivation: Title: "Enable DeepGemm warmup in DeepSeek-V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; makes DeepGEMM warmup explicit in generated commands after the runtime flag landed.
- Key implementation: updates generated command environment/options so DeepGEMM warmup behavior is enabled by default for the relevant DeepSeek-V4 recipes.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` changes DeepGEMM warmup env/command lines.
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
+SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
-# DeepGEMM warmup disabled for this recipe
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Risk and verification: Keep docs in sync with the runtime flag from #23756; verify generated commands still work when warmup is enabled on the targeted hardware.
