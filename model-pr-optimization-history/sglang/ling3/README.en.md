# SGLang Ling 3.0 (BailingMoeV3) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` | [#38434](https://github.com/sgl-project/sglang/pull/38434), [#38527](https://github.com/sgl-project/sglang/pull/38527), [#38539](https://github.com/sgl-project/sglang/pull/38539), [#39419](https://github.com/sgl-project/sglang/pull/39419) |
| `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` | [#33556](https://github.com/sgl-project/sglang/pull/33556), [#33882](https://github.com/sgl-project/sglang/pull/33882), [#34363](https://github.com/sgl-project/sglang/pull/34363), [#35861](https://github.com/sgl-project/sglang/pull/35861), [#36204](https://github.com/sgl-project/sglang/pull/36204), [#38434](https://github.com/sgl-project/sglang/pull/38434) |
| `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` | [#34283](https://github.com/sgl-project/sglang/pull/34283), [#34395](https://github.com/sgl-project/sglang/pull/34395), [#38434](https://github.com/sgl-project/sglang/pull/38434) |
| `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` | [#33556](https://github.com/sgl-project/sglang/pull/33556), [#34363](https://github.com/sgl-project/sglang/pull/34363), [#35861](https://github.com/sgl-project/sglang/pull/35861), [#36204](https://github.com/sgl-project/sglang/pull/36204), [#36364](https://github.com/sgl-project/sglang/pull/36364) |
| `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` | [#38434](https://github.com/sgl-project/sglang/pull/38434), [#38527](https://github.com/sgl-project/sglang/pull/38527), [#39419](https://github.com/sgl-project/sglang/pull/39419) |
| `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` | [#38434](https://github.com/sgl-project/sglang/pull/38434), [#38527](https://github.com/sgl-project/sglang/pull/38527), [#39419](https://github.com/sgl-project/sglang/pull/39419) |
| `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` | [#33556](https://github.com/sgl-project/sglang/pull/33556), [#33882](https://github.com/sgl-project/sglang/pull/33882), [#34363](https://github.com/sgl-project/sglang/pull/34363), [#35861](https://github.com/sgl-project/sglang/pull/35861), [#36204](https://github.com/sgl-project/sglang/pull/36204), [#36364](https://github.com/sgl-project/sglang/pull/36364) |
| `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx` | [#34283](https://github.com/sgl-project/sglang/pull/34283), [#34395](https://github.com/sgl-project/sglang/pull/34395) |
| `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx` | [#34283](https://github.com/sgl-project/sglang/pull/34283), [#34395](https://github.com/sgl-project/sglang/pull/34395) |
| `python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py` | [#38526](https://github.com/sgl-project/sglang/pull/38526) |
| `python/sglang/srt/function_call/ling3_detector.py` | [#33561](https://github.com/sgl-project/sglang/pull/33561) |
| `python/sglang/srt/models/bailing_mm_v3.py` | [#38526](https://github.com/sgl-project/sglang/pull/38526) |
| `python/sglang/srt/models/bailing_moe_v3.py` | [#33561](https://github.com/sgl-project/sglang/pull/33561), [#36584](https://github.com/sgl-project/sglang/pull/36584), [#38526](https://github.com/sgl-project/sglang/pull/38526) |

## PR Coverage Summary

- Git-traced PRs: 15
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 15
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-08-05 | [#33556](https://github.com/sgl-project/sglang/pull/33556) | merged | Add Ling-3.0-flash cookbook | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` |
| 2026-08-07 | [#33882](https://github.com/sgl-project/sglang/pull/33882) | merged | Docs: Ling-3.0-flash cookbook — serve native 256K, drop YaRN override | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` |
| 2026-08-10 | [#34283](https://github.com/sgl-project/sglang/pull/34283) | merged | Cookbook: add Ling-3.0-tiny | `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` |
| 2026-08-11 | [#34363](https://github.com/sgl-project/sglang/pull/34363) | merged | [Docs] Add Ling-3.0-flash INT4 and MXFP4 recipes | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` |
| 2026-08-11 | [#34395](https://github.com/sgl-project/sglang/pull/34395) | merged | [Docs] Add Ling-3.0-tiny INT4 recipes | `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` |
| 2026-08-21 | [#35861](https://github.com/sgl-project/sglang/pull/35861) | merged | docs: add DSPARK speculative decoding option to Ling-3.0-flash cookbook | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` |
| 2026-08-24 | [#36204](https://github.com/sgl-project/sglang/pull/36204) | merged | docs: mark Ling-3.0-flash DSPARK verified for all four quantizations on H200 | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` |
| 2026-08-27 | [#33561](https://github.com/sgl-project/sglang/pull/33561) | merged | [Model] Support Ling-3.0-flash (BailingMoeV3) | `python/sglang/srt/models/bailing_moe_v3.py`, `python/sglang/srt/function_call/ling3_detector.py` |
| 2026-08-27 | [#36584](https://github.com/sgl-project/sglang/pull/36584) | merged | Fix BailingMoeV3 reading enable_dp_lm_head off live topology instead of config | `python/sglang/srt/models/bailing_moe_v3.py` |
| 2026-08-27 | [#36364](https://github.com/sgl-project/sglang/pull/36364) | merged | docs(cookbook): add GB10 (DGX Spark) MXFP4 cells for Ling-3.0-flash | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` |
| 2026-09-08 | [#38434](https://github.com/sgl-project/sglang/pull/38434) | merged | Add Ling-3.0-flash-VL cookbook | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` |
| 2026-09-08 | [#38539](https://github.com/sgl-project/sglang/pull/38539) | merged | Point Ling-3.0-flash-VL cookbook install section at the model image | `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` |
| 2026-09-10 | [#38527](https://github.com/sgl-project/sglang/pull/38527) | merged | Add INT4 and FP4 lanes to the Ling-3.0-flash-VL cookbook | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` |
| 2026-09-16 | [#38526](https://github.com/sgl-project/sglang/pull/38526) | merged | Add Ling-3.0-flash-VL model support | `python/sglang/srt/models/bailing_mm_v3.py`, `python/sglang/srt/models/bailing_moe_v3.py`, `python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py` |
| 2026-09-18 | [#39419](https://github.com/sgl-project/sglang/pull/39419) | merged | Verify the Ling-3.0-flash-VL FP4 lane on H200 and disable shared-expert fusion in quant recipes | `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` |

## Per-PR Diff Audit Cards

### PR #33556 - Add Ling-3.0-flash cookbook

- Link: https://github.com/sgl-project/sglang/pull/33556
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; associated commits `b3cdd016baeb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +990/-19, 1138 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` added +615/-0 (615 lines); hunks: -0,0 +1,615; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` added +97/-0 (97 lines); hunks: -0,0 +1,97; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: cards, touching `cards`.
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` added +615/-0 (615 lines); hunks: -0,0 +1,615
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` added +97/-0 (97 lines); hunks: -0,0 +1,97
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` added +206/-0 (206 lines); hunks: -0,0 +1,206; symbols: cards
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx
@@ -0,0 +1,615 @@
+export const config = {
+  modelName: "Ling-3.0-flash",
+  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
+  groupHardware: false,
+  variants: [
+    { id: "default", label: "Ling-3.0-flash" },
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx
@@ -0,0 +1,97 @@
+// Ling-3.0-flash per-cell benchmark numbers, keyed by the same `match` tuple as
+// ling-3.0-flash.jsx cells. See _deployment.jsx for the speed/accuracy schema.
+//
+// Accuracy harness (one harness for the whole GSM8K column, per
+// config.benchmarkCommands.accuracy): sgl-eval run gsm8k, full 1319 questions,
+// --num-threads 32. Every filled entry below also recorded 100% stop /
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx
@@ -0,0 +1,206 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` added +615/-0; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` added +97/-0; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` added +206/-0
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/cookbook/autoregressive/InclusionAI/Ring-2.6-1T.mdx`, `docs/cookbook/autoregressive/intro.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #33882 - Docs: Ling-3.0-flash cookbook — serve native 256K, drop YaRN override

- Link: https://github.com/sgl-project/sglang/pull/33882
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; associated commits `0da25ee6f79a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +2/-86, 426 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +0/-84 (84 lines); hunks: -125,13 +125,10 @@ sgl-eval run gsm8k \\; -140,13 +137,10 @@ sgl-eval run gsm8k \\; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +2/-2 (4 lines); hunks: -42,7 +42,7 @@ import { Playground } from "/src/snippets/_playground.jsx";; -61,7 +61,7 @@ It is a hybrid-reasoning model with thinking enabled by defaul....
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +0/-84 (84 lines); hunks: -125,13 +125,10 @@ sgl-eval run gsm8k \\; -140,13 +137,10 @@ sgl-eval run gsm8k \\
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +2/-2 (4 lines); hunks: -42,7 +42,7 @@ import { Playground } from "/src/snippets/_playground.jsx";; -61,7 +61,7 @@ It is a hybrid-reasoning model with thinking enabled by defaul...
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx
@@ -125,13 +125,10 @@ sgl-eval run gsm8k \\
-      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
-        "--context-length 262144",
-        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddin
@@ -140,13 +137,10 @@ sgl-eval run gsm8k \\
-      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
-        "--context-length 262144",
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx
@@ -42,7 +42,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
-It is a hybrid-reasoning model with thinking enabled by default, and it supports structured tool calling. Native context length is 128K, extendable to 256K with YaRN.
+It is a hybrid-reasoning model with thinking enabled by default, and it supports structured tool calling. Native context length is 256K.
@@ -61,7 +61,7 @@ It is a hybrid-reasoning model with thinking enabled by default, and it supports
-- Native context is 128K. The recipes set `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` to acknowledge the longer context explicitly, then use `--context-length 262144` and YaRN w
+- Native context is 256K; SGLang reads it from the checkpoint's `max_position_embeddings`, so no `--context-length` flag is needed.
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +0/-84; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #34283 - Cookbook: add Ling-3.0-tiny

- Link: https://github.com/sgl-project/sglang/pull/34283
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx`; associated commits `77c90e7e5493`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +386/-0, 396 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx` added +191/-0 (191 lines); hunks: -0,0 +1,191; `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx` added +28/-0 (28 lines); hunks: -0,0 +1,28; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` added +166/-0 (166 lines); hunks: -0,0 +1,166.
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx` added +191/-0 (191 lines); hunks: -0,0 +1,191
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx` added +28/-0 (28 lines); hunks: -0,0 +1,28
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` added +166/-0 (166 lines); hunks: -0,0 +1,166
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx
@@ -0,0 +1,191 @@
+export const config = {
+  modelName: "Ling-3.0-tiny",
+  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
+  groupHardware: false,
+  variants: [
+    { id: "default", label: "Ling-3.0-tiny" },
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx
@@ -0,0 +1,28 @@
+// Measured on lmsysorg/sglang:dev-Ling-3.0-tiny, 1× H200. TTFT/TPOT are P50
+// (median) from sglang.bench_serving (random ISL 8192 / OSL 1024, --flush-cache);
+// tokens_per_sec_per_gpu = output tok/s × (isl+osl)/osl. Accuracy from sgl-eval
+// full GSM8K (1319).
+export const benchmarks = [
+  {
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx
@@ -0,0 +1,166 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx` added +191/-0; `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx` added +28/-0; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` added +166/-0
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx`, `docs/docs.json`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #34363 - [Docs] Add Ling-3.0-flash INT4 and MXFP4 recipes

- Link: https://github.com/sgl-project/sglang/pull/34363
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; associated commits `1c06c160f99c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +117/-8, 196 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +60/-0 (60 lines); hunks: -10,6 +10,8 @@ export const config = {; -23,6 +25,8 @@ export const config = {; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +51/-6 (57 lines); hunks: -1,12 +1,9; -40,6 +37,30 @@ export const benchmarks = [; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +6/-2 (8 lines); hunks: -1,6 +1,6; -48,6 +48,8 @@ It is a hybrid-reasoning model with thinking enabled by defaul...; symbols: cards, touching `cards`.
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +60/-0 (60 lines); hunks: -10,6 +10,8 @@ export const config = {; -23,6 +25,8 @@ export const config = {
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +51/-6 (57 lines); hunks: -1,12 +1,9; -40,6 +37,30 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +6/-2 (8 lines); hunks: -1,6 +1,6; -48,6 +48,8 @@ It is a hybrid-reasoning model with thinking enabled by defaul...; symbols: cards
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx
@@ -10,6 +10,8 @@ export const config = {
+    { id: "int4", label: "INT4" },
+    { id: "mxfp4", label: "MXFP4" },
@@ -23,6 +25,8 @@ export const config = {
+    "default|int4": "inclusionAI/Ling-3.0-flash-int4",
+    "default|mxfp4": "inclusionAI/Ling-3.0-flash-fp4",
@@ -58,6 +62,7 @@ export const config = {
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx
@@ -1,12 +1,9 @@
-// Accuracy harness (one harness for the whole GSM8K column, per
-// config.benchmarkCommands.accuracy): sgl-eval run gsm8k, full 1319 questions,
-// --num-threads 32. Every filled entry below also recorded 100% stop /
-// 0% truncated / 0% error.
-//
-// Speed numbers are not measured yet — entries carry accuracy only.
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx
@@ -1,6 +1,6 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +60/-0; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +51/-6; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +6/-2
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #34395 - [Docs] Add Ling-3.0-tiny INT4 recipes

- Link: https://github.com/sgl-project/sglang/pull/34395
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx`; associated commits `d5d41d07edae`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +92/-8, 163 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx` modified +60/-1 (61 lines); hunks: -10,6 +10,7 @@ export const config = {; -21,6 +22,7 @@ export const config = {; `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx` modified +27/-4 (31 lines); hunks: -1,7 +1,6; -25,4 +24,28 @@ export const benchmarks = [; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` modified +5/-3 (8 lines); hunks: -1,6 +1,6; -46,16 +46,18 @@ It is a thinking model with chain-of-thought enabled by defa....
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx` modified +60/-1 (61 lines); hunks: -10,6 +10,7 @@ export const config = {; -21,6 +22,7 @@ export const config = {
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx` modified +27/-4 (31 lines); hunks: -1,7 +1,6; -25,4 +24,28 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` modified +5/-3 (8 lines); hunks: -1,6 +1,6; -46,16 +46,18 @@ It is a thinking model with chain-of-thought enabled by defa...
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx
@@ -10,6 +10,7 @@ export const config = {
+    { id: "int4", label: "INT4" },
@@ -21,6 +22,7 @@ export const config = {
+    "default|int4": "inclusionAI/Ling-3.0-tiny-int4",
@@ -51,13 +53,16 @@ export const config = {
+  --random-range-ratio 1 \\
-  --num-threads 32`,
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx
@@ -1,7 +1,6 @@
-// Measured on lmsysorg/sglang:dev-Ling-3.0-tiny, 1× H200. TTFT/TPOT are P50
-// (median) from sglang.bench_serving (random ISL 8192 / OSL 1024, --flush-cache);
-// tokens_per_sec_per_gpu = output tok/s × (isl+osl)/osl. Accuracy from sgl-eval
-// full GSM8K (1319).
+// TTFT/TPOT are P50. INT4 uses 80 exact ISL 8192 / OSL 1024 requests with
+// --flush-cache; BF16/FP8 retain their original published measurements.
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx
@@ -1,6 +1,6 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx` modified +60/-1; `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx` modified +27/-4; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` modified +5/-3
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #35861 - docs: add DSPARK speculative decoding option to Ling-3.0-flash cookbook

- Link: https://github.com/sgl-project/sglang/pull/35861
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; associated commits `05c584c44fb0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +153/-62, 586 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +108/-40 (148 lines); hunks: -4,9 +4,7 @@ export const config = {; -18,9 +16,7 @@ export const config = {; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +37/-19 (56 lines); hunks: -12,14 +12,14 @@ export const benchmarks = [; -28,17 +28,17 @@ export const benchmarks = [; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +4/-2 (6 lines); hunks: -20,7 +20,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -50,6 +50,7 @@ It is a hybrid-reasoning model with thinking enabled by defaul....
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +108/-40 (148 lines); hunks: -4,9 +4,7 @@ export const config = {; -18,9 +16,7 @@ export const config = {
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +37/-19 (56 lines); hunks: -12,14 +12,14 @@ export const benchmarks = [; -28,17 +28,17 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +4/-2 (6 lines); hunks: -20,7 +20,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -50,6 +50,7 @@ It is a hybrid-reasoning model with thinking enabled by defaul...
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx
@@ -4,9 +4,7 @@ export const config = {
-  variants: [
-    { id: "default", label: "Ling-3.0-flash" },
-  ],
+  variants: [{ id: "default", label: "Ling-3.0-flash" }],
@@ -18,9 +16,7 @@ export const config = {
-  nodesOptions: [
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx
@@ -12,14 +12,14 @@ export const benchmarks = [
-    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
+    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", spec: "nextn", nodes: "single" },
-    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
+    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", spec: "off", nodes: "single" },
@@ -28,17 +28,17 @@ export const benchmarks = [
-    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx
@@ -20,7 +20,7 @@ For how to launch the image, see [Install → Method 3: Using Docker](../../../d
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +108/-40; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +37/-19; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +4/-2
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/_playground.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #36204 - docs: mark Ling-3.0-flash DSPARK verified for all four quantizations on H200

- Link: https://github.com/sgl-project/sglang/pull/36204
- Status/date: merged / 2026-08-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; associated commits `6e2f87d58941`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +56/-2, 94 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +31/-1 (32 lines); hunks: -187,6 +187,9 @@ sgl-eval run gsm8k \\; -342,8 +345,35 @@ sgl-eval run gsm8k \\; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +24/-0 (24 lines); hunks: -16,6 +16,12 @@ export const benchmarks = [; -37,6 +43,24 @@ export const benchmarks = [; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +1/-1 (2 lines); hunks: -66,7 +66,7 @@ It is a hybrid-reasoning model with thinking enabled by defaul....
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +31/-1 (32 lines); hunks: -187,6 +187,9 @@ sgl-eval run gsm8k \\; -342,8 +345,35 @@ sgl-eval run gsm8k \\
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +24/-0 (24 lines); hunks: -16,6 +16,12 @@ export const benchmarks = [; -37,6 +43,24 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +1/-1 (2 lines); hunks: -66,7 +66,7 @@ It is a hybrid-reasoning model with thinking enabled by defaul...
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx
@@ -187,6 +187,9 @@ sgl-eval run gsm8k \\
+    // hw|quant pairs with a measured full-GSM8K DSPARK run; see the mdx
+    // DSPARK tip for scores and stop rates.
+    const DSPARK_VERIFIED = new Set(["b200|bf16", "h200|bf16", "h200|fp8"]);
@@ -342,8 +345,35 @@ sgl-eval run gsm8k \\
-        dsparkTwin(c, c.match.hw === "b200" && c.match.quant === "bf16"),
+        dsparkTwin(c, DSPARK_VERIFIED.has(`${c.match.hw}|${c.match.quant}`)),
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx
@@ -16,6 +16,12 @@ export const benchmarks = [
+  {
+    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", spec: "dspark", nodes: "single" },
+    sglang_version: "PR #33561 @ 76a3e673",
+    accuracy: { gsm8k_pct: 96.36 },
+    notes: "Full GSM8K stop rate 99.62%; accept length ~4.5-5.1.",
+  },
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx
@@ -66,7 +66,7 @@ It is a hybrid-reasoning model with thinking enabled by default, and it supports
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +31/-1; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +24/-0; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #33561 - [Model] Support Ling-3.0-flash (BailingMoeV3)

- Link: https://github.com/sgl-project/sglang/pull/33561
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/ling3_detector.py`, `python/sglang/srt/models/bailing_moe_v3.py`; associated commits `20621aa14bda`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 76 files, +5188/-319, 7367 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/bailing_moe_v3.py` added +1982/-0 (1982 lines); hunks: -0,0 +1,1982; symbols: DsV3MLA, __init__, forward, _forward_gated, touching `DsV3MLA, __init__, forward`; `python/sglang/srt/function_call/ling3_detector.py` added +28/-0 (28 lines); hunks: -0,0 +1,28; symbols: Ling3Detector, __init__, touching `Ling3Detector, __init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_v3.py` added +1982/-0 (1982 lines); hunks: -0,0 +1,1982; symbols: DsV3MLA, __init__, forward, _forward_gated
  - `python/sglang/srt/function_call/ling3_detector.py` added +28/-0 (28 lines); hunks: -0,0 +1,28; symbols: Ling3Detector, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_v3.py
@@ -0,0 +1,1982 @@
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
+from __future__ import annotations
+import copy
+import logging
+from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
+import torch
diff -- python/sglang/srt/function_call/ling3_detector.py
@@ -0,0 +1,28 @@
+import re
+from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
+class Ling3Detector(Glm4MoeDetector):
+    """
+    Detector for Ling3 tool calls.
+    Ling3 uses the GLM-4.5 XML format, but model outputs may either put a newline
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/bailing_moe_v3.py` added +1982/-0; `python/sglang/srt/function_call/ling3_detector.py` added +28/-0
- Risk and verification: The diff ships test coverage in `test/registered/attention/test_kda_kernels.py`, `test/registered/attention/test_kda_prefill_flashkda.py`, `test/registered/kernels/test_fused_kda_conv_recurrent_verify.py`, `test/registered/unit/configs/test_model_config_shapes.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36584 - Fix BailingMoeV3 reading enable_dp_lm_head off live topology instead of config

- Link: https://github.com/sgl-project/sglang/pull/36584
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/bailing_moe_v3.py`; associated commits `15688fea7d8b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +2/-2, 18 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/bailing_moe_v3.py` modified +1/-1 (2 lines); hunks: -1333,7 +1333,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_v3.py` modified +1/-1 (2 lines); hunks: -1333,7 +1333,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_v3.py
@@ -1333,7 +1333,7 @@ def __init__(
-                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
+                    use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/bailing_moe_v3.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/models/test_shared_experts_fusion_gates.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36364 - docs(cookbook): add GB10 (DGX Spark) MXFP4 cells for Ling-3.0-flash

- Link: https://github.com/sgl-project/sglang/pull/36364
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; associated commits `11de5e228187`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +57/-1, 93 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +29/-1 (30 lines); hunks: -1,7 +1,7; -46,6 +46,7 @@ export const config = {; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +28/-0 (28 lines); hunks: -165,6 +165,34 @@ export const benchmarks = [.
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +29/-1 (30 lines); hunks: -1,7 +1,7; -46,6 +46,7 @@ export const config = {
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +28/-0 (28 lines); hunks: -165,6 +165,34 @@ export const benchmarks = [
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx
@@ -1,7 +1,7 @@
-  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
+  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300", "dgx-spark"],
@@ -46,6 +46,7 @@ export const config = {
+    "dgx-spark": "lmsysorg/sglang:dev-Ling-3.0-flash",
@@ -374,6 +375,19 @@ sgl-eval run gsm8k \\
+    {
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx
@@ -165,6 +165,34 @@ export const benchmarks = [
+  // ====================================================================
+  // GB10 + MXFP4 (TP1, DGX Spark sm121)
+  // ====================================================================
+  {
+    match: { hw: "dgx-spark", variant: "default", quant: "mxfp4", strategy: "low-latency", spec: "dspark", nodes: "single" },
+    sglang_version: "PR #33561 @ 2f85329efe",
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx` modified +29/-1; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx` modified +28/-0
- Risk and verification: This is mostly docs/examples in `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38434 - Add Ling-3.0-flash-VL cookbook

- Link: https://github.com/sgl-project/sglang/pull/38434
- Status/date: merged / 2026-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`; associated commits `482e9f257bb9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +628/-3, 669 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` added +256/-0 (256 lines); hunks: -0,0 +1,256; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` added +55/-0 (55 lines); hunks: -0,0 +1,55; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` added +310/-0 (310 lines); hunks: -0,0 +1,310; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6.
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` added +256/-0 (256 lines); hunks: -0,0 +1,256
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` added +55/-0 (55 lines); hunks: -0,0 +1,55
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` added +310/-0 (310 lines); hunks: -0,0 +1,310
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` modified +0/-1 (1 lines); hunks: -1,7 +1,6
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx
@@ -0,0 +1,256 @@
+export const config = {
+  modelName: "Ling-3.0-flash-VL",
+  supportedHardware: ["gb300", "b300", "b200", "h200", "h100"],
+  groupHardware: false,
+  variants: [{ id: "default", label: "Ling-3.0-flash-VL" }],
+  quantizations: [
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx
@@ -0,0 +1,55 @@
+// Ling-3.0-flash-VL per-cell benchmark numbers, keyed by the same `match` tuple as
+// ling-3.0-flash-vl.jsx cells. See _deployment.jsx for the speed/accuracy schema.
+//
+// Speed: bench_serving --flush-cache --random-range-ratio 1, temperature 0. Speed cards
+// use the `random` dataset (text-only, isl 8192 / osl 1024) across LL (conc 1/16) and HT
+// (conc 1024/4096); per-cell notes carry the separate `image` workload (one random 720p
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx
@@ -0,0 +1,310 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` added +256/-0; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` added +55/-0; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` added +310/-0; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx` modified +0/-1; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx` modified +0/-1
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx`, `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38539 - Point Ling-3.0-flash-VL cookbook install section at the model image

- Link: https://github.com/sgl-project/sglang/pull/38539
- Status/date: merged / 2026-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`; associated commits `afe90a8bc908`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-2, 26 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +11/-2 (13 lines); hunks: -22,14 +22,23 @@ pip install uv.
- Code diff details:
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +11/-2 (13 lines); hunks: -22,14 +22,23 @@ pip install uv
- Key code excerpts:

```diff
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx
@@ -22,14 +22,23 @@ pip install uv
-Then run the **Python** output of the command panel below in that environment.
+Then run the **Python** output of the command panel below in that environment. Ling-3.0-flash-VL support requires sglang with sgl-project/sglang#38526 (or newer); until it merges,
+'''bash Command
+pip install --upgrade pip
+pip install uv
+uv pip install "sglang @ git+https://github.com/sgl-project/sglang.git@refs/pull/38526/head#subdirectory=python"
```

- Extracted files (not manually reviewed):
  - docs: `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +11/-2
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38527 - Add INT4 and FP4 lanes to the Ling-3.0-flash-VL cookbook

- Link: https://github.com/sgl-project/sglang/pull/38527
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`; associated commits `9a2f17f41d18`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +250/-28, 436 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` modified +188/-23 (211 lines); hunks: -1,26 +1,29; -44,6 +47,7 @@ export const config = {; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` modified +55/-3 (58 lines); hunks: -28,8 +28,8 @@ export const benchmarks = [; -50,6 +50,58 @@ export const benchmarks = [; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +7/-2 (9 lines); hunks: -49,7 +49,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -74,6 +74,9 @@ It is a thinking model: the chat template turns chain-of-thoug...; symbols: and, touching `and`.
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` modified +188/-23 (211 lines); hunks: -1,26 +1,29; -44,6 +47,7 @@ export const config = {
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` modified +55/-3 (58 lines); hunks: -28,8 +28,8 @@ export const benchmarks = [; -50,6 +50,58 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +7/-2 (9 lines); hunks: -49,7 +49,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -74,6 +74,9 @@ It is a thinking model: the chat template turns chain-of-thoug...; symbols: and
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx
@@ -1,26 +1,29 @@
-  supportedHardware: ["gb300", "b300", "b200", "h200", "h100"],
+  supportedHardware: ["gb300", "b300", "b200", "h200", "h100", "dgx-spark"],
+    { id: "int4", label: "INT4 (GPTQ)" },
+    { id: "fp4", label: "FP4 (MXFP4)" },
+    "default|int4": "inclusionAI/Ling-3.0-flash-VL-int4",
+    "default|fp4": "inclusionAI/Ling-3.0-flash-VL-fp4",
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx
@@ -28,8 +28,8 @@ export const benchmarks = [
-    accuracy: { mmmu_pro_pct: 76.01, gsm8k_pct: 97.19 },
-    notes: "4×GB300, TP=4. Measured with online dynamic FP8 (--quantization fp8 on the BF16 checkpoint), the same serving path the FP8 variant uses. Accuracy vs BF16 on the same b
+    accuracy: { mmmu_pro_pct: 77.34, gsm8k_pct: 97.19 },
+    notes: "GB300 TP=1, official FP8 checkpoint. MMMU-Pro 77.34% (stop 99.77%, truncated 0.23%, error 0) measured at TP=4 --ep 4 on 4×GB300; TP=1 serving smoke also verified on on
@@ -50,6 +50,58 @@ export const benchmarks = [
-  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx
@@ -49,7 +49,7 @@ For how to launch the image, see [Install → Method 3: Using Docker](../../../d
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` modified +188/-23; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` modified +55/-3; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +7/-2
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #38526 - Add Ling-3.0-flash-VL model support

- Link: https://github.com/sgl-project/sglang/pull/38526
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py`, `python/sglang/srt/models/bailing_mm_v3.py`, `python/sglang/srt/models/bailing_moe_v3.py`; associated commits `f0bf652534c5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 37 files, +3332/-123, 4428 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/bailing_mm_v3.py` added +325/-0 (325 lines); hunks: -0,0 +1,325; symbols: BailingMoeV3VLForConditionalGeneration, shared_experts_fusion_disable_reason, __init__, get_input_embeddings, touching `BailingMoeV3VLForConditionalGeneration, shared_experts_fusion_disable_reason, __init__`; `python/sglang/srt/models/bailing_moe_v3.py` modified +160/-29 (189 lines); hunks: -18,6 +18,7; -50,6 +51,10; symbols: _get_awq_dequantize, __init__, forward, _get_bailing_num_shared_experts, touching `_get_awq_dequantize, __init__, forward`; `python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py` added +48/-0 (48 lines); hunks: -0,0 +1,48; symbols: _bailing_moe_v3_overrides, touching `_bailing_moe_v3_overrides`.
- Code diff details:
  - `python/sglang/srt/models/bailing_mm_v3.py` added +325/-0 (325 lines); hunks: -0,0 +1,325; symbols: BailingMoeV3VLForConditionalGeneration, shared_experts_fusion_disable_reason, __init__, get_input_embeddings
  - `python/sglang/srt/models/bailing_moe_v3.py` modified +160/-29 (189 lines); hunks: -18,6 +18,7; -50,6 +51,10; symbols: _get_awq_dequantize, __init__, forward, _get_bailing_num_shared_experts
  - `python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py` added +48/-0 (48 lines); hunks: -0,0 +1,48; symbols: _bailing_moe_v3_overrides
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_mm_v3.py
@@ -0,0 +1,325 @@
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/bailing_moe_v3.py
@@ -18,6 +18,7 @@
+from sglang.srt.configs.bailing_hybrid import is_bailing_multi_gate_enabled
@@ -50,6 +51,10 @@
+from sglang.srt.layers.multi_gate import (
+    create_multi_gate_mm_indices,
+    multi_gate_triton_kernel,
+)
diff -- python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py
@@ -0,0 +1,48 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/bailing_mm_v3.py` added +325/-0; `python/sglang/srt/models/bailing_moe_v3.py` modified +160/-29; `python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py` added +48/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/eplb/test_waterfill_eplb.py`, `test/registered/unit/layers/test_bailing_mrope_shape.py`, `test/registered/unit/managers/test_bailing_modality_metadata.py`, `test/registered/unit/models/test_bailing_vl_loader.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39419 - Verify the Ling-3.0-flash-VL FP4 lane on H200 and disable shared-expert fusion in quant recipes

- Link: https://github.com/sgl-project/sglang/pull/39419
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`; associated commits `4dbba37965aa`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +24/-2, 47 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` modified +16/-0 (16 lines); hunks: -385,6 +385,22 @@ sgl-eval run gsm8k \\; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` modified +6/-0 (6 lines); hunks: -80,6 +80,12 @@ export const benchmarks = [; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +2/-2 (4 lines); hunks: -95,8 +95,8 @@ It is a thinking model: the chat template turns chain-of-thoug...; symbols: and, touching `and`.
- Code diff details:
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` modified +16/-0 (16 lines); hunks: -385,6 +385,22 @@ sgl-eval run gsm8k \\
  - `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` modified +6/-0 (6 lines); hunks: -80,6 +80,12 @@ export const benchmarks = [
  - `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +2/-2 (4 lines); hunks: -95,8 +95,8 @@ It is a thinking model: the chat template turns chain-of-thoug...; symbols: and
- Key code excerpts:

```diff
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx
@@ -385,6 +385,22 @@ sgl-eval run gsm8k \\
+    {
+      match: { hw: "h200", variant: "default", quant: "fp4", strategy: "balanced", nodes: "single" },
+      verified: true,
+      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
+      flags: [
+        "--trust-remote-code",
diff -- docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx
@@ -80,6 +80,12 @@ export const benchmarks = [
+  {
+    match: { hw: "h200", variant: "default", quant: "fp4", strategy: "balanced", nodes: "single" },
+    sglang_version: "dev @ bf254483a1",
+    accuracy: { mmmu_pro_pct: 76.24, gsm8k_pct: 96.66 },
+    notes: "1×H200 (141 GB), TP=1, flashinfer_mxfp4 MoE backend (SM90 CUTLASS W4A16), auto-selected — verified without an explicit --moe-runner-backend flag (healthy in 330 s). Se
+  },
diff -- docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx
@@ -95,8 +95,8 @@ It is a thinking model: the chat template turns chain-of-thought on by default a
```

- Extracted files (not manually reviewed):
  - docs: `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx` modified +16/-0; `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx` modified +6/-0; `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL.mdx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl-benchmarks.jsx`, `docs/src/snippets/configs/inclusionAI/ling-3.0-flash-vl.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
