# sglang DeepSeek V4 PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 5
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Open Optimization Items

| PR | Signal | Why it matters |
| --- | --- | --- |
| [#23882](https://github.com/sgl-project/sglang/pull/23882) | DeepSeek-V4 rebase tracking | Check before assuming local DeepSeek-V4 support is final. |
| [#24047](https://github.com/sgl-project/sglang/pull/24047) | DeepSeek-V4 / SM120 support | Affects FP4, MoE, and attention kernel eligibility on SM120 hardware. |

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `.github/workflows/release-docker-deepseek-v4.yml` | [#23728](https://github.com/sgl-project/sglang/pull/23728), [#23730](https://github.com/sgl-project/sglang/pull/23730), [#23778](https://github.com/sgl-project/sglang/pull/23778) |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23628](https://github.com/sgl-project/sglang/pull/23628), [#23684](https://github.com/sgl-project/sglang/pull/23684), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23725](https://github.com/sgl-project/sglang/pull/23725), [#23980](https://github.com/sgl-project/sglang/pull/23980), [#24035](https://github.com/sgl-project/sglang/pull/24035) |
| `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23617](https://github.com/sgl-project/sglang/pull/23617), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23634](https://github.com/sgl-project/sglang/pull/23634), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23690](https://github.com/sgl-project/sglang/pull/23690), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23698](https://github.com/sgl-project/sglang/pull/23698), [#23715](https://github.com/sgl-project/sglang/pull/23715), [#23725](https://github.com/sgl-project/sglang/pull/23725), [#23737](https://github.com/sgl-project/sglang/pull/23737), ... (17 total) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23605](https://github.com/sgl-project/sglang/pull/23605) | merged | Add DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23617](https://github.com/sgl-project/sglang/pull/23617) | merged | Further update Deepseek V4 docs | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-24 | [#23628](https://github.com/sgl-project/sglang/pull/23628) | merged | docs: note H200 DeepSeek-V4 checkpoint | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23622](https://github.com/sgl-project/sglang/pull/23622) | merged | Again update DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-24 | [#23634](https://github.com/sgl-project/sglang/pull/23634) | merged | Update pro fp8 checkpoint in DeepSeek V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23684](https://github.com/sgl-project/sglang/pull/23684) | merged | docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23689](https://github.com/sgl-project/sglang/pull/23689) | merged | docs(DeepSeek-V4): mark b200\|small\|pd-disagg + h200\|small\|{cp,pd-disagg} verified | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23691](https://github.com/sgl-project/sglang/pull/23691) | merged | docs(DeepSeek-V4): mark gb300\|{small,big}\|{cp,pd-disagg} verified + GB300-specific fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23690](https://github.com/sgl-project/sglang/pull/23690) | merged | Small udpate gb300 recipe for deepseek v4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23697](https://github.com/sgl-project/sglang/pull/23697) | merged | update: b300 container for dsv4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-25 | [#23698](https://github.com/sgl-project/sglang/pull/23698) | merged | docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23715](https://github.com/sgl-project/sglang/pull/23715) | merged | docs(DeepSeek-V4): mark h200\|big\|pd-disagg verified + recipe fixes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-25 | [#23728](https://github.com/sgl-project/sglang/pull/23728) | merged | ci: add docker release workflow for deepseek_v4 branch | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-25 | [#23730](https://github.com/sgl-project/sglang/pull/23730) | merged | [CI] release-docker-deepseek-v4: select which flavors to push | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-26 | [#23725](https://github.com/sgl-project/sglang/pull/23725) | merged | docs(DeepSeek-V4): add GB200 platform to cookbook recipe | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-26 | [#23742](https://github.com/sgl-project/sglang/pull/23742) | merged | docs(DeepSeek-V4): add h200\|big verified recipes + tune H200 Pro parameters | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-26 | [#23737](https://github.com/sgl-project/sglang/pull/23737) | merged | docs(DeepSeek-V4): mark gb200\|big\|low-latency verified | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-26 | [#23778](https://github.com/sgl-project/sglang/pull/23778) | merged | ci(deepseek-v4): add b300/grace-blackwell dev-branch build options | `.github/workflows/release-docker-deepseek-v4.yml` |
| 2026-04-27 | [#23787](https://github.com/sgl-project/sglang/pull/23787) | merged | amd/deepseek_v4 integration 1/N - 0426 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` |
| 2026-04-27 | [#23776](https://github.com/sgl-project/sglang/pull/23776) | merged | [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-27 | [#23817](https://github.com/sgl-project/sglang/pull/23817) | merged | docs: verify GB300 Pro DeepSeek V4 recipes | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-27 | [#23810](https://github.com/sgl-project/sglang/pull/23810) | merged | Add benchmarking scripts for deepseek v4 | `scripts/bench_gpqa_aime.py` |
| 2026-04-27 | [#23832](https://github.com/sgl-project/sglang/pull/23832) | merged | amd/deepseek_v4 integration 2/N - cuda graph 0426 | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py` |
| 2026-04-27 | [#23756](https://github.com/sgl-project/sglang/pull/23756) | merged | feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch | `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py` |
| 2026-04-28 | [#23883](https://github.com/sgl-project/sglang/pull/23883) | merged | Enable DeepGemm warmup in DeepSeek-V4 cookbook | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-28 | [#23943](https://github.com/sgl-project/sglang/pull/23943) | merged | [Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` |
| 2026-04-29 | [#23980](https://github.com/sgl-project/sglang/pull/23980) | merged | docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4 | `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |
| 2026-04-29 | [#24035](https://github.com/sgl-project/sglang/pull/24035) | merged | [minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4 | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` |

## Per-PR Diff Audit Cards

### PR #23605 - Add DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23605
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `492883c8ca66`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1024/-1, 1041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Add DeepSeek V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Further update Deepseek V4 docs"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Further update Deepseek V4 docs"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23628 - docs: note H200 DeepSeek-V4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/23628
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `1a37e57fb1ae`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: note H200 DeepSeek-V4 checkpoint"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs: note H200 DeepSeek-V4 checkpoint"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Again update DeepSeek V4 cookbook"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "Again update DeepSeek V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Update pro fp8 checkpoint in DeepSeek V4 cookbook"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Update pro fp8 checkpoint in DeepSeek V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): note SGLANG_FIX_DSV4_BASE_MODEL_LOAD for base models"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): mark b200|small|pd-disagg + h200|small|{cp,pd-disagg} verified"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): mark gb300|{small,big}|{cp,pd-disagg} verified + GB300-specific fixes"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Small udpate gb300 recipe for deepseek v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Small udpate gb300 recipe for deepseek v4"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "update: b300 container for dsv4"; model line: DeepSeek V4; category: model implementation change; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "update: b300 container for dsv4"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23715 - docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes

- Link: https://github.com/sgl-project/sglang/pull/23715
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `d4c16656262b`
- Diff scope read: GitHub Pull Request files API returned 1 files, +31/-4, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes"; model line: DeepSeek V4; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): mark h200|big|pd-disagg verified + recipe fixes"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4 (35 lines); hunks: -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {; -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4 (35 lines); hunks: -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {; -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -178,8 +178,7 @@ export const DeepSeekV4Deployment = () => {
-    // h200|big|pd-disagg: pending verification (needs 4-node H200 cluster with
-    //   shared IB fabric: 2-node prefill + 2-node decode).
+    "h200|big|pd-disagg",
@@ -480,6 +479,12 @@ export const DeepSeekV4Deployment = () => {
+      // H200 Pro PD: tp=16 multinode + DeepEP needs the dispatch buffer cap on
+      // BOTH prefill + decode (matches production playground LWS for the same
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +31/-4
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23728 - ci: add docker release workflow for deepseek_v4 branch

- Link: https://github.com/sgl-project/sglang/pull/23728
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `.github/workflows/release-docker-deepseek-v4.yml`; associated commits `0c826374a85a`
- Diff scope read: GitHub Pull Request files API returned 1 files, +93/-0, 94 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: add docker release workflow for deepseek_v4 branch"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `.github/workflows/release-docker-deepseek-v4.yml`; technical summary: Covers "ci: add docker release workflow for deepseek_v4 branch"; the main implementation surface is `.github/workflows/release-docker-deepseek-v4.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0 (93 lines); hunks: -0,0 +1,93.
- Code diff details:
  - `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0 (93 lines); hunks: -0,0 +1,93
- Key code excerpts:

```diff
diff -- .github/workflows/release-docker-deepseek-v4.yml
@@ -0,0 +1,93 @@
+name: Build and Push DeepSeek-V4 Docker Images
+# Builds the 4 Dockerfiles added in #23600 from the deepseek_v4 branch and
+# pushes them to Docker Hub. Each Dockerfile is single-arch and does its own
+# `git clone -b deepseek_v4` inside, so no build context source is required
+# beyond the Dockerfiles themselves and `--no-cache` is mandatory.
+on:
```

- Reviewed files:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` added +93/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23730 - [CI] release-docker-deepseek-v4: select which flavors to push

- Link: https://github.com/sgl-project/sglang/pull/23730
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `.github/workflows/release-docker-deepseek-v4.yml`; associated commits `921e14dcac53`
- Diff scope read: GitHub Pull Request files API returned 1 files, +56/-18, 92 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] release-docker-deepseek-v4: select which flavors to push"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `.github/workflows/release-docker-deepseek-v4.yml`; technical summary: Covers "[CI] release-docker-deepseek-v4: select which flavors to push"; the main implementation surface is `.github/workflows/release-docker-deepseek-v4.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18 (74 lines); hunks: -12,35 +12,73 @@ on:.
- Code diff details:
  - `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18 (74 lines); hunks: -12,35 +12,73 @@ on:
- Key code excerpts:

```diff
diff -- .github/workflows/release-docker-deepseek-v4.yml
@@ -12,35 +12,73 @@ on:
+      build_hopper:
+        description: "Build and push the Hopper (H200) image."
+        required: false
+        type: boolean
+        default: true
+      build_blackwell:
```

- Reviewed files:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` modified +56/-18
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23725 - docs(DeepSeek-V4): add GB200 platform to cookbook recipe

- Link: https://github.com/sgl-project/sglang/pull/23725
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `049f1bf6fb42`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +58/-8, 195 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): add GB200 platform to cookbook recipe"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(DeepSeek-V4): add GB200 platform to cookbook recipe"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6 (58 lines); hunks: -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {; -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../.....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6 (58 lines); hunks: -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {; -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2 (8 lines); hunks: -29,13 +29,13 @@ tag: NEW; -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../....
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -4,6 +4,7 @@ export const DeepSeekV4Deployment = () => {
+  //     GB200 → FP4 weights, Flash TP=4 / Pro TP=8 2-node
@@ -27,6 +28,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "gb200", label: "GB200 (FP4)", default: false },
@@ -138,6 +140,8 @@ export const DeepSeekV4Deployment = () => {
+    "gb200|small": { slug: "deepseek-ai/DeepSeek-V4-Flash", tp: 4,  multinode: false },
+    "gb200|big":   { slug: "deepseek-ai/DeepSeek-V4-Pro",   tp: 8,  multinode: true, nnodes: 2 },
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -29,13 +29,13 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>single-node serving: B200 / GB300 / H200 on 4 GPUs</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>single-node serving: B200 / GB200 / GB300 / H200 on 4 GPUs</td>
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
@@ -88,6 +88,10 @@ Please refer to the [official SGLang installation guide](../../../docs/get-start
+    <tr>
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +52/-6; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +6/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23742 - docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters

- Link: https://github.com/sgl-project/sglang/pull/23742
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3cfd1561df78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-8, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8 (30 lines); hunks: -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {; -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8 (30 lines); hunks: -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {; -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -184,6 +184,9 @@ export const DeepSeekV4Deployment = () => {
+    "h200|big|low-latency",
+    "h200|big|balanced",
+    "h200|big|max-throughput",
@@ -272,7 +275,9 @@ export const DeepSeekV4Deployment = () => {
-        recipeEnv.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256");
+        recipeEnv.push(isBig
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +22/-8
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23737 - docs(DeepSeek-V4): mark gb200|big|low-latency verified

- Link: https://github.com/sgl-project/sglang/pull/23737
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3d95ca7546fb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(DeepSeek-V4): mark gb200|big|low-latency verified"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs(DeepSeek-V4): mark gb200|big|low-latency verified"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0 (1 lines); hunks: -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0 (1 lines); hunks: -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -174,6 +174,7 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|big|low-latency",
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +1/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23778 - ci(deepseek-v4): add b300/grace-blackwell dev-branch build options

- Link: https://github.com/sgl-project/sglang/pull/23778
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `.github/workflows/release-docker-deepseek-v4.yml`; associated commits `977830e91e41`
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-5, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci(deepseek-v4): add b300/grace-blackwell dev-branch build options"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `.github/workflows/release-docker-deepseek-v4.yml`; technical summary: Covers "ci(deepseek-v4): add b300/grace-blackwell dev-branch build options"; the main implementation surface is `.github/workflows/release-docker-deepseek-v4.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5 (28 lines); hunks: -32,6 +32,16 @@ on:; -50,19 +60,27 @@ jobs:.
- Code diff details:
  - `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5 (28 lines); hunks: -32,6 +32,16 @@ on:; -50,19 +60,27 @@ jobs:
- Key code excerpts:

```diff
diff -- .github/workflows/release-docker-deepseek-v4.yml
@@ -32,6 +32,16 @@ on:
+      build_b300_dev:
+        description: "Build and push the B300 image from the deepseek_v4_dev branch."
+        required: false
+        type: boolean
+        default: true
+      build_grace_blackwell_dev:
```

- Reviewed files:
  - ci: `.github/workflows/release-docker-deepseek-v4.yml` modified +23/-5
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23787 - amd/deepseek_v4 integration 1/N - 0426

- Link: https://github.com/sgl-project/sglang/pull/23787
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 128 files, +18341/-879, 18279 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "amd/deepseek_v4 integration 1/N - 0426"; model line: DeepSeek V4; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`; technical summary: Covers "amd/deepseek_v4 integration 1/N - 0426"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` added +2803/-0 (2803 lines); `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0 (1330 lines); hunks: -0,0 +1,1330; symbols: _copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadataRadix, touching `_copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data`; `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0 (840 lines); hunks: -0,0 +1,840; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `python/sglang/srt/layers/mhc.py` added +686/-0 (686 lines); hunks: -0,0 +1,686; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang, touching `hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` added +2803/-0 (2803 lines)
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0 (1330 lines); hunks: -0,0 +1,1330; symbols: _copy_metadata, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadataRadix
  - `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0 (840 lines); hunks: -0,0 +1,840; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `python/sglang/srt/layers/mhc.py` added +686/-0 (686 lines); hunks: -0,0 +1,686; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang
  - `python/sglang/srt/layers/attention/compressed/indexer.py` added +616/-0 (616 lines); hunks: -0,0 +1,616; symbols: fp8_paged_mqa_logits_torch, topk_transform_512_pytorch_vectorized, _fused_scale_kernel, fused_scale
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py
@@ -0,0 +1,1330 @@
+"""
+Some comments on the common terms used in DeepSeekV4Backend:
+topk_lengths:
+    NOTE: TL;DR: topk_lengths == seq_lens
+    The FlashMLA sparse decode kernel will attend to `k` tokens for each query.
+    `topk_lengths` indicates how many tokens each query will attend to.
diff -- python/sglang/srt/entrypoints/openai/encoding_dsv4.py
@@ -0,0 +1,840 @@
+# Adapted from the DeepSeek-V4 release reference implementation.
+"""
+DeepSeek-V4 Encoding
+A self-contained implementation for encoding/decoding DeepSeek-V4 chat messages
+with tool calling, thinking mode, and quick instruction task support.
+"""
diff -- python/sglang/srt/layers/mhc.py
@@ -0,0 +1,686 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` added +2803/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` added +1330/-0; `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +840/-0; `python/sglang/srt/layers/mhc.py` added +686/-0; `python/sglang/srt/layers/attention/compressed/indexer.py` added +616/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +591/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_activation.py`, `python/sglang/srt/flashmla_tests/__init__.py`, `python/sglang/srt/flashmla_tests/kernelkit/.gitignore`, `python/sglang/srt/flashmla_tests/kernelkit/__init__.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23776 - [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP

- Link: https://github.com/sgl-project/sglang/pull/23776
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-0, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP"; model line: DeepSeek V4; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -227,9 +227,11 @@ def __init__(; -283,6 +285,12 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -227,9 +227,11 @@ def __init__(; -283,6 +285,12 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -227,9 +227,11 @@ def __init__(
+        swiglu_limit: Optional[float] = None,
+        self.swiglu_limit = swiglu_limit
@@ -283,6 +285,12 @@ def forward(
+        if self.swiglu_limit is not None:
+            _g, _u = gate_up.chunk(2, dim=-1)
+            _lim = float(self.swiglu_limit)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23817 - docs: verify GB300 Pro DeepSeek V4 recipes

- Link: https://github.com/sgl-project/sglang/pull/23817
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `c2ec64f243d4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-0, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: verify GB300 Pro DeepSeek V4 recipes"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "docs: verify GB300 Pro DeepSeek V4 recipes"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0 (6 lines); hunks: -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {; -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0 (6 lines); hunks: -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {; -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -182,7 +182,9 @@ export const DeepSeekV4Deployment = () => {
+    "gb300|big|balanced",
+    "gb300|big|max-throughput",
@@ -365,6 +367,8 @@ export const DeepSeekV4Deployment = () => {
+      } else if (isBig && hardware === "gb300") {
+        flags.push("  --mem-fraction-static 0.9");
@@ -401,6 +405,8 @@ export const DeepSeekV4Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +6/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23810 - Add benchmarking scripts for deepseek v4

- Link: https://github.com/sgl-project/sglang/pull/23810
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +243/-0, 244 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add benchmarking scripts for deepseek v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `scripts/bench_gpqa_aime.py`; technical summary: Covers "Add benchmarking scripts for deepseek v4"; the main implementation surface is `scripts/bench_gpqa_aime.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `scripts/bench_gpqa_aime.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: _venv_cmd, get_timestamp, get_random_int, setup_ns, touching `_venv_cmd, get_timestamp, get_random_int`.
- Code diff details:
  - `scripts/bench_gpqa_aime.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: _venv_cmd, get_timestamp, get_random_int, setup_ns
- Key code excerpts:

```diff
diff -- scripts/bench_gpqa_aime.py
@@ -0,0 +1,243 @@
+# This script should be used inside the container. Before testing anything, please
+# 1. install typer
+# 2. set the following environment variables:
+# - HOST: the host to connect to (default 127.0.0.1)
+# - PORT: the port to connect to (default 30010)
+# - HF_TOKEN: needed for `setup-ns`
```

- Reviewed files:
  - other: `scripts/bench_gpqa_aime.py` added +243/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #23832 - amd/deepseek_v4 integration 2/N - cuda graph 0426

- Link: https://github.com/sgl-project/sglang/pull/23832
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +534/-92, 973 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "amd/deepseek_v4 integration 2/N - cuda graph 0426"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py`; technical summary: Covers "amd/deepseek_v4 integration 2/N - cuda graph 0426"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/compressed/indexer.py`, `python/sglang/srt/layers/attention/compressed/metadata.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1 (396 lines); hunks: -1,5 +1,5; -27,6 +27,7; symbols: fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2, _padded_H, touching `fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2`; `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76 (154 lines); hunks: -1,6 +1,6; -37,6 +37,8; symbols: fp8_paged_mqa_logits_torch, touching `fp8_paged_mqa_logits_torch`; `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11 (23 lines); hunks: -169,18 +169,19 @@ def max_seq_len(self) -> int:; symbols: max_seq_len, copy_, touching `max_seq_len, copy_`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1 (10 lines); hunks: -1152,7 +1152,9 @@ def run_once():; -1162,6 +1164,9 @@ def run_once():; symbols: run_once, replay_prepare, touching `run_once, replay_prepare`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1 (396 lines); hunks: -1,5 +1,5; -27,6 +27,7; symbols: fast_log2_ceil, tilelang_sparse_fwd, _next_power_of_2, _padded_H
  - `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76 (154 lines); hunks: -1,6 +1,6; -37,6 +37,8; symbols: fp8_paged_mqa_logits_torch
  - `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11 (23 lines); hunks: -169,18 +169,19 @@ def max_seq_len(self) -> int:; symbols: max_seq_len, copy_
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1 (10 lines); hunks: -1152,7 +1152,9 @@ def run_once():; -1162,6 +1164,9 @@ def run_once():; symbols: run_once, replay_prepare
  - `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` modified +7/-0 (7 lines); hunks: -13,6 +13,10 @@ def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):; -32,6 +36,9 @@ def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):; symbols: flash_mla_with_kvcache_entrypoint
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -1,5 +1,5 @@
-from typing import Optional, Tuple
+from typing import Any, Optional, Tuple
@@ -27,6 +27,7 @@
+INT32 = "int32"
@@ -1375,3 +1376,396 @@ def tilelang_sparse_fwd(
+def _next_power_of_2(x: int) -> int:
diff -- python/sglang/srt/layers/attention/compressed/indexer.py
@@ -1,6 +1,6 @@
-from typing import TYPE_CHECKING, Any, List, Optional, Tuple
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
@@ -37,6 +37,8 @@
+_arange_cache: Dict[str, torch.Tensor] = {}
@@ -48,6 +50,8 @@ def fp8_paged_mqa_logits_torch(
+    """Vectorized implementation that avoids .item() and Python loops,
diff -- python/sglang/srt/layers/attention/compressed/metadata.py
@@ -169,18 +169,19 @@ def max_seq_len(self) -> int:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +395/-1; `python/sglang/srt/layers/attention/compressed/indexer.py` modified +78/-76; `python/sglang/srt/layers/attention/compressed/metadata.py` modified +12/-11; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-1; `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` modified +7/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/attention/base_attn_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23756 - feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch

- Link: https://github.com/sgl-project/sglang/pull/23756
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +47/-12, 90 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py`; technical summary: Covers "feat: port SGLANG_JIT_DEEPGEMM_FAST_WARMUP to deepseek_v4 branch"; the main implementation surface is `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12 (58 lines); hunks: -22,7 +22,7; -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: Server...; symbols: update_deep_gemm_config, _compile_deep_gemm_one_type_all, touching `update_deep_gemm_config, _compile_deep_gemm_one_type_all`; `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -336,6 +336,7 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12 (58 lines); hunks: -22,7 +22,7; -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: Server...; symbols: update_deep_gemm_config, _compile_deep_gemm_one_type_all
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -336,6 +336,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py
@@ -22,7 +22,7 @@
-_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
+_BUILTIN_M_LIST: List[int] = []
@@ -44,14 +44,43 @@ def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
-    # Generate m_max
-    m_max = 1024 * 16
-    if server_args.chunked_prefill_size < 1:
diff -- python/sglang/srt/environ.py
@@ -336,6 +336,7 @@ class Envs:
+    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +46/-12; `python/sglang/srt/environ.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23883 - Enable DeepGemm warmup in DeepSeek-V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23883
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `3177fa795154`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-5, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable DeepGemm warmup in DeepSeek-V4 cookbook"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "Enable DeepGemm warmup in DeepSeek-V4 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5 (8 lines); hunks: -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {; -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5 (8 lines); hunks: -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {; -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -255,7 +255,6 @@ export const DeepSeekV4Deployment = () => {
-    const COMMON_ENV = ["SGLANG_JIT_DEEPGEMM_PRECOMPILE=0"];
@@ -461,8 +460,8 @@ export const DeepSeekV4Deployment = () => {
-    // Assemble: [HW env] [recipe env] [common env] \ sglang serve \ flags...
-    const envAll = [...HW_ENV, ...recipeEnv, ...COMMON_ENV];
+    // Assemble: [HW env] [recipe env] \ sglang serve \ flags...
+    const envAll = [...HW_ENV, ...recipeEnv];
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +3/-5
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23943 - [Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe

- Link: https://github.com/sgl-project/sglang/pull/23943
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `4e1ef6b3cf9b`
- Diff scope read: GitHub Pull Request files API returned 1 files, +32/-0, 39 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; technical summary: Covers "[Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0 (32 lines); hunks: -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0 (32 lines); hunks: -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -482,6 +482,38 @@ export const DeepSeekV4Deployment = () => {
+    // H200 Pro low-latency: show BOTH a single-node (TP=8 marlin) variant
+    // and the existing multi-node (TP=16 DP-attn + DeepEP) variant.
+    if (hardware === "h200" && isBig && recipe === "low-latency") {
+      const singleFlags = [
+        "  --trust-remote-code",
+        "  --model-path deepseek-ai/DeepSeek-V4-Pro",
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +32/-0
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23980 - docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/23980
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; associated commits `4e885baa9bf1`
- Diff scope read: GitHub Pull Request files API returned 2 files, +84/-8, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "docs(cookbook): add H200 (FP4) deployment option for DeepSeek-V4"; the main implementation surface is `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3 (82 lines); hunks: -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {; -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3 (82 lines); hunks: -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {; -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5 (10 lines); hunks: -1,7 +1,7; -35,7 +35,7 @@ tag: NEW
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
@@ -31,6 +31,7 @@ export const DeepSeekV4Deployment = () => {
+        { id: "h200-fp4", label: "H200 (FP4)", default: false },
@@ -70,7 +71,19 @@ export const DeepSeekV4Deployment = () => {
-  const resolveItems = (option) => option.items;
+  // Recipes that are not supported on the H200 (FP4) Marlin path.
+  const H200_FP4_UNSUPPORTED_RECIPES = new Set(["cp", "pd-disagg"]);
+  const resolveItems = (option, vals) => {
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -1,7 +1,7 @@
-    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek. Blackwell deployments use the FP4 checkpoint; Hopper deployments use the FP8 checkpoi
+    description: "Deploy DeepSeek-V4 with SGLang — a next-generation MoE model from DeepSeek."
@@ -35,7 +35,7 @@ tag: NEW
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 16 GPU (2 nodes)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>high-capacity: B200 8 GPU / GB200 8 GPU (2 nodes) / GB300 4 GPU / H200 8 GPU(fp4)/16 GPU(fp8)</t
@@ -153,9 +153,9 @@ The generator currently picks values on the **conservative** side (mirroring an
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` modified +79/-3; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +5/-5
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #24035 - [minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4

- Link: https://github.com/sgl-project/sglang/pull/24035
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; associated commits `b3ead32d3ca2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-3, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; technical summary: Covers "[minor] Remove incorrect note after supporting w4a16 moe for DeepSeek V4"; the main implementation surface is `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3 (3 lines); hunks: -120,9 +120,6 @@ docker run --gpus all \.
- Code diff details:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3 (3 lines); hunks: -120,9 +120,6 @@ docker run --gpus all \
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -120,9 +120,6 @@ docker run --gpus all \
-<Note>
-For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
-</Note>
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +0/-3
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.
