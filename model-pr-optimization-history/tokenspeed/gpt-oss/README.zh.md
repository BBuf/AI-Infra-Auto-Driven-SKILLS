# TokenSpeed GPT-OSS 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/models/gpt_oss.py` | [#154](https://github.com/lightseekorg/tokenspeed/pull/154) |
| `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` | [#139](https://github.com/lightseekorg/tokenspeed/pull/139), [#154](https://github.com/lightseekorg/tokenspeed/pull/154), [#278](https://github.com/lightseekorg/tokenspeed/pull/278), [#479](https://github.com/lightseekorg/tokenspeed/pull/479), [#801](https://github.com/lightseekorg/tokenspeed/pull/801) |
| `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` | [#487](https://github.com/lightseekorg/tokenspeed/pull/487), [#489](https://github.com/lightseekorg/tokenspeed/pull/489) |
| `test/runtime/models/test_gpt_oss.py` | 无直接 PR 号提交 |
| `test/runtime/test_gpt_oss_mxfp4_streaming.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 7
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 7
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-14 | [#139](https://github.com/lightseekorg/tokenspeed/pull/139) | merged | chore(ci): drop `--stream` and generation-config from gpt-oss gpqa eval | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-05-15 | [#154](https://github.com/lightseekorg/tokenspeed/pull/154) | merged | [AMD]Support a-fp8-w-mxfp4 gpt-oss-120b model | `python/tokenspeed/runtime/models/gpt_oss.py`, `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-05-27 | [#278](https://github.com/lightseekorg/tokenspeed/pull/278) | merged | ci(eval): use --repeats 3 on gpt-oss-120b gpqa-diamond to suppress noise | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-06-19 | [#479](https://github.com/lightseekorg/tokenspeed/pull/479) | merged | ci: use default mha and moe backend for gpt-oss ci | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |
| 2026-06-20 | [#487](https://github.com/lightseekorg/tokenspeed/pull/487) | merged | ci: add mi350 1gpu gpt-oss perf bench | `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` |
| 2026-06-20 | [#489](https://github.com/lightseekorg/tokenspeed/pull/489) | merged | ci: deduplicate mi350/mi355 gpt-oss bench yaml | `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml`, `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` |
| 2026-07-27 | [#801](https://github.com/lightseekorg/tokenspeed/pull/801) | merged | ci: enable flat KV cache build for GPT-OSS tasks | `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` |

## 逐 PR diff 审计卡

### PR #139 - chore(ci): drop `--stream` and generation-config from gpt-oss gpqa eval

- 链接: https://github.com/lightseekorg/tokenspeed/pull/139
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；关联提交 `61537d7c15c4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+0/-2，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-2 (2 lines); hunks: -49,8 +49,6 @@ eval:。
- 代码 diff 细节:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-2 (2 lines); hunks: -49,8 +49,6 @@ eval:
- 关键代码摘录:

```diff
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -49,8 +49,6 @@ eval:
-    --stream
-    --generation-config '{"do_sample":false,"temperature":0.0,"max_tokens":65536}'
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-2
- 验证与风险: diff 自带测试面 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #154 - [AMD]Support a-fp8-w-mxfp4 gpt-oss-120b model

- 链接: https://github.com/lightseekorg/tokenspeed/pull/154
- 状态/时间: merged / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/gpt_oss.py`, `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；关联提交 `ce376dad3281`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+427/-14，可读 patch 643 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/gpt_oss.py` modified +142/-0 (142 lines); hunks: -25,6 +25,7; -773,6 +774,26 @@ def _copy_into_param(param, narrow_weight):; symbols: _copy_into_param, _load_mxfp4_per_expert_weights，涉及 `_copy_into_param, _load_mxfp4_per_expert_weights`；`test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +4/-2 (6 lines); hunks: -12,17 +12,19 @@ runner:; -44,7 +46,7 @@ eval:。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/gpt_oss.py` modified +142/-0 (142 lines); hunks: -25,6 +25,7; -773,6 +774,26 @@ def _copy_into_param(param, narrow_weight):; symbols: _copy_into_param, _load_mxfp4_per_expert_weights
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +4/-2 (6 lines); hunks: -12,17 +12,19 @@ runner:; -44,7 +46,7 @@ eval:
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/gpt_oss.py` modified +142/-0
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +4/-2
- 验证与风险: diff 自带测试面 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`, `test/ci_system/pipeline.py`, `test/ci_system/test_pipeline.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #278 - ci(eval): use --repeats 3 on gpt-oss-120b gpqa-diamond to suppress noise

- 链接: https://github.com/lightseekorg/tokenspeed/pull/278
- 状态/时间: merged / 2026-05-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；关联提交 `f475bca39e47`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+11/-1，可读 patch 25 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +11/-1 (12 lines); hunks: -45,13 +45,23 @@ eval:。
- 代码 diff 细节:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +11/-1 (12 lines); hunks: -45,13 +45,23 @@ eval:
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +11/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #479 - ci: use default mha and moe backend for gpt-oss ci

- 链接: https://github.com/lightseekorg/tokenspeed/pull/479
- 状态/时间: merged / 2026-06-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；关联提交 `6a43a1183f8a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+0/-4，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-4 (4 lines); hunks: -10,8 +10,6 @@ runner:; -30,8 +28,6 @@ server:。
- 代码 diff 细节:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-4 (4 lines); hunks: -10,8 +10,6 @@ runner:; -30,8 +28,6 @@ server:
- 关键代码摘录:

```diff
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -10,8 +10,6 @@ runner:
-      GPT_OSS_EVAL_ATTENTION_BACKEND: trtllm
-      GPT_OSS_EVAL_MOE_BACKEND: flashinfer_trtllm
@@ -30,8 +28,6 @@ server:
-    ${GPT_OSS_EVAL_ATTENTION_BACKEND:+--attention-backend ${GPT_OSS_EVAL_ATTENTION_BACKEND}}
-    ${GPT_OSS_EVAL_MOE_BACKEND:+--moe-backend ${GPT_OSS_EVAL_MOE_BACKEND}}
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +0/-4
- 验证与风险: diff 自带测试面 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #487 - ci: add mi350 1gpu gpt-oss perf bench

- 链接: https://github.com/lightseekorg/tokenspeed/pull/487
- 状态/时间: merged / 2026-06-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml`；关联提交 `c1c9e4e0499a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+77/-0，可读 patch 92 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` added +75/-0 (75 lines); hunks: -0,0 +1,75。
- 代码 diff 细节:
  - `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` added +75/-0 (75 lines); hunks: -0,0 +1,75
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml` added +75/-0
- 验证与风险: diff 自带测试面 `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml`, `test/ci_system/test_pipeline.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #489 - ci: deduplicate mi350/mi355 gpt-oss bench yaml

- 链接: https://github.com/lightseekorg/tokenspeed/pull/489
- 状态/时间: merged / 2026-06-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi350.yaml`；关联提交 `2b54faaf5b83`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+5/-77，可读 patch 102 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml` removed +0/-75 (75 lines); hunks: -1,75 +0,0；`test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` renamed +5/-2 (7 lines); hunks: -1,12 +1,13; -33,7 +34,9 @@ perf:。
- 代码 diff 细节:
  - `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml` removed +0/-75 (75 lines); hunks: -1,75 +0,0
  - `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` renamed +5/-2 (7 lines); hunks: -1,12 +1,13; -33,7 +34,9 @@ perf:
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml` removed +0/-75; `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml` renamed +5/-2
- 验证与风险: diff 自带测试面 `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi355.yaml`, `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #801 - ci: enable flat KV cache build for GPT-OSS tasks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/801
- 状态/时间: merged / 2026-07-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；关联提交 `f92f69abafee`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+2/-0，可读 patch 16 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +1/-0 (1 lines); hunks: -16,6 +16,7 @@ runner:。
- 代码 diff 细节:
  - `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +1/-0 (1 lines); hunks: -16,6 +16,7 @@ runner:
- 关键代码摘录:

```diff
diff -- test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml
@@ -16,6 +16,7 @@ runner:
+  TOKENSPEED_FLAT_KV: "ON"
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml` modified +1/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`, `test/ci/perf/gpt-oss-120b-mxfp4-evalscope-mi35x.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
