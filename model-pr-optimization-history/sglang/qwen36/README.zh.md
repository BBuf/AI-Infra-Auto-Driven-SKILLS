# sglang Qwen3.6 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/Qwen/Qwen3.6.mdx` | 无直接 PR 号提交 |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_6_27b.mdx` | 无直接 PR 号提交 |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_6_35b_a3b.mdx` | 无直接 PR 号提交 |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_6_27b.mdx` | 无直接 PR 号提交 |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_6_35b_a3b.mdx` | 无直接 PR 号提交 |
| `docs/src/snippets/autoregressive/qwen36-deployment.jsx` | 无直接 PR 号提交 |
| `test/registered/npu/accuracy/qwen3_6_27b/test_npu_qwen3_6_27b_1p_gpqa.py` | 无直接 PR 号提交 |
| `test/registered/npu/accuracy/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in3k5_out1k5_50ms_gpqa.py` | 无直接 PR 号提交 |
| `test/registered/npu/accuracy/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_aime26.py` | 无直接 PR 号提交 |
| `test/registered/npu/accuracy/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms_aime26.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_1p_in1024x1024_30_out1024_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_1p_in1080p_30_out256_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_2p_in64k_out1k_prefix90_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in3k5_out1k5_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_2p_in128k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_2p_in16k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_2p_in64k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in128k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in128k_out1k_prefix90_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_50ms.py` | 无直接 PR 号提交 |
| `test/registered/npu/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 4
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-22 | [#23486](https://github.com/sgl-project/sglang/pull/23486) | merged | docs(cookbook): add Qwen3.6-27B dense variant | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |
| 2026-07-02 | [#29905](https://github.com/sgl-project/sglang/pull/29905) | merged | docs: add Qwen3.6-27B-NVFP4 variant to cookbook | `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` |
| 2026-07-07 | [#29964](https://github.com/sgl-project/sglang/pull/29964) | merged | [Docs] Use trtllm_mha for Qwen3.6 B300 | `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |
| 2026-07-25 | [#31413](https://github.com/sgl-project/sglang/pull/31413) | merged | [Docs] Add Qwen3.6 35B NVFP4 to cookbook | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |

## 逐 PR diff 审计卡

### PR #23486 - docs(cookbook): add Qwen3.6-27B dense variant

- 链接: https://github.com/sgl-project/sglang/pull/23486
- 状态/时间: merged / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+55/-17，可读 patch 170 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): add Qwen3.6-27B dense variant」；模型线: Qwen3.6；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；技术摘要: 覆盖「docs(cookbook): add Qwen3.6-27B dense variant」；主要实现面是 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunks: -1,26 +1,29; -29,30 +32,43 @@ Qwen3.6 features a Gated Delta Networks combined with sparse...；`docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunks: -10,6 +10,14 @@ export const Qwen36Deployment = () => {; -66,9 +74,18 @@ export const Qwen36Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunks: -1,26 +1,29; -29,30 +32,43 @@ Qwen3.6 features a Gated Delta Networks combined with sparse...
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunks: -10,6 +10,14 @@ export const Qwen36Deployment = () => {; -66,9 +74,18 @@ export const Qwen36Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10; `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29905 - docs: add Qwen3.6-27B-NVFP4 variant to cookbook

- 链接: https://github.com/sgl-project/sglang/pull/29905
- 状态/时间: merged / 2026-07-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+50/-12，可读 patch 111 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add Qwen3.6-27B-NVFP4 variant to cookbook」；模型线: Qwen3.6；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`；技术摘要: 覆盖「docs: add Qwen3.6-27B-NVFP4 variant to cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +42/-12 (54 lines); hunks: -23,10 +23,19 @@ export const Qwen36Deployment = () => {; -93,8 +102,8 @@ export const Qwen36Deployment = () => {；`docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +8/-0 (8 lines); hunks: -57,6 +57,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -75,6 +80,9 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +42/-12 (54 lines); hunks: -23,10 +23,19 @@ export const Qwen36Deployment = () => {; -93,8 +102,8 @@ export const Qwen36Deployment = () => {
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +8/-0 (8 lines); hunks: -57,6 +57,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -75,6 +80,9 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +42/-12; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +8/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29964 - [Docs] Use trtllm_mha for Qwen3.6 B300

- 链接: https://github.com/sgl-project/sglang/pull/29964
- 状态/时间: merged / 2026-07-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-4，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Use trtllm_mha for Qwen3.6 B300」；模型线: Qwen3.6；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；技术摘要: 覆盖「[Docs] Use trtllm_mha for Qwen3.6 B300」；主要实现面是 `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +1/-4 (5 lines); hunks: -222,12 +222,9 @@ export const Qwen36Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +1/-4 (5 lines); hunks: -222,12 +222,9 @@ export const Qwen36Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/qwen36-deployment.jsx
@@ -222,12 +222,9 @@ export const Qwen36Deployment = () => {
-    if (hardware === 'b200') {
+    if (hardware === 'b200' || hardware === 'b300') {
-    if (hardware === 'b300') {
-      cmd += ` \\\n  --attention-backend flashinfer`;
-    }
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +1/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #31413 - [Docs] Add Qwen3.6 35B NVFP4 to cookbook

- 链接: https://github.com/sgl-project/sglang/pull/31413
- 状态/时间: merged / 2026-07-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+26/-9，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add Qwen3.6 35B NVFP4 to cookbook」；模型线: Qwen3.6；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；技术摘要: 覆盖「[Docs] Add Qwen3.6 35B NVFP4 to cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +21/-3 (24 lines); hunks: -47,6 +47,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -78,7 +83,7 @@ uv pip install sglang；`docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +5/-6 (11 lines); hunks: -23,14 +23,13 @@ export const Qwen36Deployment = () => {; -94,8 +93,8 @@ export const Qwen36Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +21/-3 (24 lines); hunks: -47,6 +47,11 @@ Both variants share the same hybrid reasoning, tool-calling,...; -78,7 +83,7 @@ uv pip install sglang
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +5/-6 (11 lines); hunks: -23,14 +23,13 @@ export const Qwen36Deployment = () => {; -94,8 +93,8 @@ export const Qwen36Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +21/-3; `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +5/-6
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
