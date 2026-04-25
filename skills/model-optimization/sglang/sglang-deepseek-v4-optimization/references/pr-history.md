# sglang DeepSeek V4 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 0
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23628](https://github.com/sgl-project/sglang/pull/23628), [#23684](https://github.com/sgl-project/sglang/pull/23684), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697) |
| `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` | [#23605](https://github.com/sgl-project/sglang/pull/23605), [#23617](https://github.com/sgl-project/sglang/pull/23617), [#23622](https://github.com/sgl-project/sglang/pull/23622), [#23634](https://github.com/sgl-project/sglang/pull/23634), [#23689](https://github.com/sgl-project/sglang/pull/23689), [#23690](https://github.com/sgl-project/sglang/pull/23690), [#23691](https://github.com/sgl-project/sglang/pull/23691), [#23697](https://github.com/sgl-project/sglang/pull/23697), [#23698](https://github.com/sgl-project/sglang/pull/23698) |

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
