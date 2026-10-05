# TokenSpeed GPT-OSS Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/models/gpt_oss.py` | [#154](https://github.com/lightseekorg/tokenspeed/pull/154) |
| `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` | [#139](https://github.com/lightseekorg/tokenspeed/pull/139), [#154](https://github.com/lightseekorg/tokenspeed/pull/154), [#278](https://github.com/lightseekorg/tokenspeed/pull/278), [#479](https://github.com/lightseekorg/tokenspeed/pull/479), [#801](https://github.com/lightseekorg/tokenspeed/pull/801) |
| `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` | [#487](https://github.com/lightseekorg/tokenspeed/pull/487), [#489](https://github.com/lightseekorg/tokenspeed/pull/489) |
| `test/runtime/models/test_gpt_oss.py` | no direct PR-number commit |
| `test/runtime/test_gpt_oss_mxfp4_streaming.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 7
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 7
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-14 | [#139](https://github.com/lightseekorg/tokenspeed/pull/139) | merged | chore(ci): drop `--stream` and generation-config from gpt-oss gpqa eval | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-05-15 | [#154](https://github.com/lightseekorg/tokenspeed/pull/154) | merged | [AMD]Support a-fp8-w-mxfp4 gpt-oss-120b model | `python/tokenspeed/runtime/models/gpt_oss.py`, `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-05-27 | [#278](https://github.com/lightseekorg/tokenspeed/pull/278) | merged | ci(eval): use --repeats 3 on gpt-oss-120b gpqa-diamond to suppress noise | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-06-19 | [#479](https://github.com/lightseekorg/tokenspeed/pull/479) | merged | ci: use default mha and moe backend for gpt-oss ci | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-06-20 | [#487](https://github.com/lightseekorg/tokenspeed/pull/487) | merged | ci: add mi350 1gpu gpt-oss perf bench | `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` |
| 2026-06-20 | [#489](https://github.com/lightseekorg/tokenspeed/pull/489) | merged | ci: deduplicate mi350/mi355 gpt-oss bench yaml | `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml`, `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` |
| 2026-07-27 | [#801](https://github.com/lightseekorg/tokenspeed/pull/801) | merged | ci: enable flat KV cache build for GPT-OSS tasks | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |

## Per-PR Diff Audit Cards

### PR #139 - chore(ci): drop `--stream` and generation-config from gpt-oss gpqa eval

- Link: https://github.com/lightseekorg/tokenspeed/pull/139
- Status/date: merged / 2026-05-14
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; associated commits `61537d7c15c4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +0/-2, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-2 (2 lines); hunks: -49,8 +49,6 @@ eval:.
- Code diff details:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-2 (2 lines); hunks: -49,8 +49,6 @@ eval:
- Key code excerpts:

```diff
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -49,8 +49,6 @@ eval:
-    --stream
-    --generation-config '{"do_sample":false,"temperature":0.0,"max_tokens":65536}'
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-2
- Risk and verification: The diff ships test coverage in `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #154 - [AMD]Support a-fp8-w-mxfp4 gpt-oss-120b model

- Link: https://github.com/lightseekorg/tokenspeed/pull/154
- Status/date: merged / 2026-05-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/gpt_oss.py`, `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; associated commits `ce376dad3281`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +427/-14, 643 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/gpt_oss.py` modified +142/-0 (142 lines); hunks: -25,6 +25,7; -773,6 +774,26 @@ def _copy_into_param(param, narrow_weight):; symbols: _copy_into_param, _load_mxfp4_per_expert_weights, touching `_copy_into_param, _load_mxfp4_per_expert_weights`; `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +4/-2 (6 lines); hunks: -12,17 +12,19 @@ runner:; -44,7 +46,7 @@ eval:.
- Code diff details:
  - `python/tokenspeed/runtime/models/gpt_oss.py` modified +142/-0 (142 lines); hunks: -25,6 +25,7; -773,6 +774,26 @@ def _copy_into_param(param, narrow_weight):; symbols: _copy_into_param, _load_mxfp4_per_expert_weights
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +4/-2 (6 lines); hunks: -12,17 +12,19 @@ runner:; -44,7 +46,7 @@ eval:
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/gpt_oss.py
@@ -25,6 +25,7 @@
+import re
@@ -773,6 +774,26 @@ def _copy_into_param(param, narrow_weight):
+        # Detect AMD-Quark per-expert checkpoints (e.g.
+        # ``amd/gpt-oss-120b-w-mxfp4-a-fp8``). These store one set of tensors
+        # per expert (``...experts.{e}.gate_up_proj.{weight,...}``) plus a
+        # scalar ``input_scale`` for static FP8 activation quantization.
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -12,17 +12,19 @@ runner:
+      GPT_OSS_EVAL_MODEL: openai/gpt-oss-120b
+      GPT_OSS_EVAL_MODEL: amd/gpt-oss-120b-w-mxfp4-a-fp8
-    --model openai/gpt-oss-120b
+    --model ${GPT_OSS_EVAL_MODEL}
@@ -44,7 +46,7 @@ eval:
-    --model openai/gpt-oss-120b
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/gpt_oss.py` modified +142/-0
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +4/-2
- Risk and verification: The diff ships test coverage in `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`, `test/ci_system/pipeline.py`, `test/ci_system/test_pipeline.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #278 - ci(eval): use --repeats 3 on gpt-oss-120b gpqa-diamond to suppress noise

- Link: https://github.com/lightseekorg/tokenspeed/pull/278
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; associated commits `f475bca39e47`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-1, 25 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +11/-1 (12 lines); hunks: -45,13 +45,23 @@ eval:.
- Code diff details:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +11/-1 (12 lines); hunks: -45,13 +45,23 @@ eval:
- Key code excerpts:

```diff
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -45,13 +45,23 @@ eval:
+  # --repeats 3 samples each of the 198 questions 3x and averages, shrinking the
+  # per-run stddev by ~sqrt(3). At the 0.7 threshold this drops the noise-only
+  # failure rate from ~11% to ~2%.
+  # --eval-batch-size 64 lifts evalscope's client-side concurrency from 16 to 64
+  # (server's max_num_seqs=160, so still 2.5x headroom). At batch=16 the server
+  # was pinned at 13-16 in-flight requests with page_ratio=0.00-0.01, leaving
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +11/-1
- Risk and verification: The diff ships test coverage in `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #479 - ci: use default mha and moe backend for gpt-oss ci

- Link: https://github.com/lightseekorg/tokenspeed/pull/479
- Status/date: merged / 2026-06-19
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; associated commits `6a43a1183f8a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +0/-4, 18 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-4 (4 lines); hunks: -10,8 +10,6 @@ runner:; -30,8 +28,6 @@ server:.
- Code diff details:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-4 (4 lines); hunks: -10,8 +10,6 @@ runner:; -30,8 +28,6 @@ server:
- Key code excerpts:

```diff
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -10,8 +10,6 @@ runner:
-      GPT_OSS_EVAL_ATTENTION_BACKEND: trtllm
-      GPT_OSS_EVAL_MOE_BACKEND: flashinfer_trtllm
@@ -30,8 +28,6 @@ server:
-    ${GPT_OSS_EVAL_ATTENTION_BACKEND:+--attention-backend ${GPT_OSS_EVAL_ATTENTION_BACKEND}}
-    ${GPT_OSS_EVAL_MOE_BACKEND:+--moe-backend ${GPT_OSS_EVAL_MOE_BACKEND}}
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-4
- Risk and verification: The diff ships test coverage in `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #487 - ci: add mi350 1gpu gpt-oss perf bench

- Link: https://github.com/lightseekorg/tokenspeed/pull/487
- Status/date: merged / 2026-06-20
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml`; associated commits `c1c9e4e0499a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +77/-0, 92 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` added +75/-0 (75 lines); hunks: -0,0 +1,75.
- Code diff details:
  - `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` added +75/-0 (75 lines); hunks: -0,0 +1,75
- Key code excerpts:

```diff
diff -- test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml
@@ -0,0 +1,75 @@
+api_version: ci.tokenspeed.io/v1
+name: perf-gpt-oss-120b-mxfp4-mi350
+type: perf
+triggers:
+  - per-commit
+  - manual
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` added +75/-0
- Risk and verification: The diff ships test coverage in `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml`, `test/ci_system/test_pipeline.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #489 - ci: deduplicate mi350/mi355 gpt-oss bench yaml

- Link: https://github.com/lightseekorg/tokenspeed/pull/489
- Status/date: merged / 2026-06-20
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml`; associated commits `2b54faaf5b83`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +5/-77, 102 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml` removed +0/-75 (75 lines); hunks: -1,75 +0,0; `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` renamed +5/-2 (7 lines); hunks: -1,12 +1,13; -33,7 +34,9 @@ perf:.
- Code diff details:
  - `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml` removed +0/-75 (75 lines); hunks: -1,75 +0,0
  - `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` renamed +5/-2 (7 lines); hunks: -1,12 +1,13; -33,7 +34,9 @@ perf:
- Key code excerpts:

```diff
diff -- test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml
@@ -1,75 +0,0 @@
-api_version: ci.tokenspeed.io/v1
-name: perf-gpt-oss-120b-mxfp4-mi355
-type: perf
-triggers:
-  - per-commit
-  - manual
diff -- test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml
@@ -1,12 +1,13 @@
-name: perf-gpt-oss-120b-mxfp4-mi350
+name: perf-gpt-oss-120b-mxfp4-mi35x
+    - amd-mi355-1gpu-bench
@@ -33,7 +34,9 @@ perf:
-    OUTPUTS_DIR=/tmp/tokenspeed-gpt-oss-mi350-perf &&
+    GPU_LABEL=${CI_RUNNER_LABEL#amd-} &&
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml` removed +0/-75; `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` renamed +5/-2
- Risk and verification: The diff ships test coverage in `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml`, `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #801 - ci: enable flat KV cache build for GPT-OSS tasks

- Link: https://github.com/lightseekorg/tokenspeed/pull/801
- Status/date: merged / 2026-07-27
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`; associated commits `f92f69abafee`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +2/-0, 16 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +1/-0 (1 lines); hunks: -16,6 +16,7 @@ runner:.
- Code diff details:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +1/-0 (1 lines); hunks: -16,6 +16,7 @@ runner:
- Key code excerpts:

```diff
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -16,6 +16,7 @@ runner:
+  TOKENSPEED_FLAT_KV: "ON"
```

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`, `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
