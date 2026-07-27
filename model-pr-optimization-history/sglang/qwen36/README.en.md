# sglang Qwen3.6 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` | [#23486](https://github.com/sgl-project/sglang/pull/23486), [#29905](https://github.com/sgl-project/sglang/pull/29905), [#31413](https://github.com/sgl-project/sglang/pull/31413) |
| `docs_new/docs/hardware-platforms/ascend-npus/best_practice/qwen3_6_27b.mdx` | no direct PR-number commit |
| `docs_new/docs/hardware-platforms/ascend-npus/best_practice/qwen3_6_35b_a3b.mdx` | no direct PR-number commit |
| `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_6_27b.mdx` | no direct PR-number commit |
| `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_6_35b_a3b.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` | [#23486](https://github.com/sgl-project/sglang/pull/23486), [#29905](https://github.com/sgl-project/sglang/pull/29905), [#29964](https://github.com/sgl-project/sglang/pull/29964), [#31413](https://github.com/sgl-project/sglang/pull/31413) |
| `test/registered/ascend/accuracy/qwen3_6_27b/test_npu_qwen3_6_27b_1p_gpqa.py` | no direct PR-number commit |
| `test/registered/ascend/accuracy/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_aime26.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_1p_in1024x1024_30_out1024_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_1p_in1080p_30_out256_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_2p_in64k_out1k_prefix90_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in3k5_out1k5_50ms_gpqa.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_2p_in128k_out1k_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_2p_in16k_out1k_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_2p_in64k_out1k_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in128k_out1k_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in128k_out1k_prefix90_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_50ms.py` | no direct PR-number commit |
| `test/registered/ascend/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms_aime26.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 4
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-22 | [#23486](https://github.com/sgl-project/sglang/pull/23486) | merged | docs(cookbook): add Qwen3.6-27B dense variant | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |
| 2026-07-02 | [#29905](https://github.com/sgl-project/sglang/pull/29905) | merged | docs: add Qwen3.6-27B-NVFP4 variant to cookbook | `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` |
| 2026-07-07 | [#29964](https://github.com/sgl-project/sglang/pull/29964) | merged | [Docs] Use trtllm_mha for Qwen3.6 B300 | `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |
| 2026-07-25 | [#31413](https://github.com/sgl-project/sglang/pull/31413) | merged | [Docs] Add Qwen3.6 35B NVFP4 to cookbook | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #23486 - docs(cookbook): add Qwen3.6-27B dense variant

- Link: https://github.com/sgl-project/sglang/pull/23486
- Status/date: merged / 2026-04-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; associated commits `de962f327432`
- Diff scope read: GitHub Pull Request files API returned 2 files, +55/-17, 170 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): add Qwen3.6-27B dense variant"; model line: Qwen3.6; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; technical summary: Covers "docs(cookbook): add Qwen3.6-27B dense variant"; the main implementation surface is `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #29905 - docs: add Qwen3.6-27B-NVFP4 variant to cookbook

- Link: https://github.com/sgl-project/sglang/pull/29905
- Status/date: merged / 2026-07-02
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; associated commits `1c75243f5eda`
- Diff scope read: GitHub Pull Request files API returned 2 files, +50/-12, 111 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: add Qwen3.6-27B-NVFP4 variant to cookbook"; model line: Qwen3.6; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`; technical summary: Covers "docs: add Qwen3.6-27B-NVFP4 variant to cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +42/-12 (54 lines); hunks: -23,10 +23,19 @@ export const Qwen36Deployment = () => {; -93,8 +102,8 @@ export const Qwen36Deployment = () => {; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +8/-0 (8 lines); hunks: -57,6 +57,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -75,6 +80,9 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +42/-12 (54 lines); hunks: -23,10 +23,19 @@ export const Qwen36Deployment = () => {; -93,8 +102,8 @@ export const Qwen36Deployment = () => {
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +8/-0 (8 lines); hunks: -57,6 +57,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -75,6 +80,9 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/qwen36-deployment.jsx
@@ -23,10 +23,19 @@ export const Qwen36Deployment = () => {
-      items: [
-        { id: 'fp8', label: 'FP8', default: true },
-        { id: 'bf16', label: 'BF16', default: false },
-      ],
+      // NVFP4 is a Blackwell-only, 27B-only checkpoint (nvidia/Qwen3.6-27B-NVFP4);
+      // only surface it when both conditions hold so we never emit an unrunnable command.
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx
@@ -57,6 +57,11 @@ Both variants share the same hybrid reasoning, tool-calling, and multimodal inte
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Qwen3.6-27B (NVFP4)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Dense 27B (Blackwell)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>[nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)</td>
+    </tr>
@@ -75,6 +80,9 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=pytho
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +42/-12; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +8/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #29964 - [Docs] Use trtllm_mha for Qwen3.6 B300

- Link: https://github.com/sgl-project/sglang/pull/29964
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; associated commits `f32b4ecd26ff`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-4, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Use trtllm_mha for Qwen3.6 B300"; model line: Qwen3.6; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; technical summary: Covers "[Docs] Use trtllm_mha for Qwen3.6 B300"; the main implementation surface is `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +1/-4 (5 lines); hunks: -222,12 +222,9 @@ export const Qwen36Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +1/-4 (5 lines); hunks: -222,12 +222,9 @@ export const Qwen36Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/qwen36-deployment.jsx
@@ -222,12 +222,9 @@ export const Qwen36Deployment = () => {
-    if (hardware === 'b200') {
+    if (hardware === 'b200' || hardware === 'b300') {
-    if (hardware === 'b300') {
-      cmd += ` \\\n  --attention-backend flashinfer`;
-    }
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +1/-4
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #31413 - [Docs] Add Qwen3.6 35B NVFP4 to cookbook

- Link: https://github.com/sgl-project/sglang/pull/31413
- Status/date: merged / 2026-07-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; associated commits `953c587adf16`
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-9, 115 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add Qwen3.6 35B NVFP4 to cookbook"; model line: Qwen3.6; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; technical summary: Covers "[Docs] Add Qwen3.6 35B NVFP4 to cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +21/-3 (24 lines); hunks: -47,6 +47,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -78,7 +83,7 @@ uv pip install sglang; `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +5/-6 (11 lines); hunks: -23,14 +23,13 @@ export const Qwen36Deployment = () => {; -94,8 +93,8 @@ export const Qwen36Deployment = () => {.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +21/-3 (24 lines); hunks: -47,6 +47,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -78,7 +83,7 @@ uv pip install sglang
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +5/-6 (11 lines); hunks: -23,14 +23,13 @@ export const Qwen36Deployment = () => {; -94,8 +93,8 @@ export const Qwen36Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx
@@ -47,6 +47,11 @@ Both variants share the same hybrid reasoning, tool-calling, and multimodal inte
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Qwen3.6-35B-A3B (NVFP4)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>MoE 35B / 3B active (Blackwell)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>[nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)</td>
+    </tr>
@@ -78,7 +83,7 @@ uv pip install sglang
diff -- docs_new/src/snippets/autoregressive/qwen36-deployment.jsx
@@ -23,14 +23,13 @@ export const Qwen36Deployment = () => {
-      // NVFP4 is a Blackwell-only, 27B-only checkpoint (nvidia/Qwen3.6-27B-NVFP4);
-      // only surface it when both conditions hold so we never emit an unrunnable command.
+      // NVFP4 checkpoints are available for both model sizes on Blackwell (B200/B300).
-        const nvfp4Supported = values.modelSize === '27b' && (values.hardware === 'b200' || values.hardware === 'b300');
+        const nvfp4Supported = values.hardware === 'b200' || values.hardware === 'b300';
@@ -94,8 +93,8 @@ export const Qwen36Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +21/-3; `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +5/-6
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
