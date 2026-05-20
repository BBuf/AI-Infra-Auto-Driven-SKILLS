# sglang LLaDA 2.1 模型 PR 优化历史

## 2026-05-19 新增覆盖

按 sglang 上游 `origin/main@5073c82a37`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `LLaDA 2.1` 缺失的历史 PR 优化文档。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `skills/llm-serving-auto-benchmark/configs/cookbook-llm/llada2-1-mini.yaml` | [#24250](https://github.com/sgl-project/sglang/pull/24250) |
| `docs_new/cookbook/autoregressive/InclusionAI/LLaDA-2.1.mdx` | [#23337](https://github.com/sgl-project/sglang/pull/23337), [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `docs_new/src/snippets/autoregressive/llada-21-deployment.jsx` | [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `python/sglang/srt/models/llada2.py` | [#23748](https://github.com/sgl-project/sglang/pull/23748), [#23732](https://github.com/sgl-project/sglang/pull/23732), [#21135](https://github.com/sgl-project/sglang/pull/21135), [#17784](https://github.com/sgl-project/sglang/pull/17784), [#18485](https://github.com/sgl-project/sglang/pull/18485), [#18844](https://github.com/sgl-project/sglang/pull/18844), [#18860](https://github.com/sgl-project/sglang/pull/18860), [#17570](https://github.com/sgl-project/sglang/pull/17570), [#15835](https://github.com/sgl-project/sglang/pull/15835), [#13730](https://github.com/sgl-project/sglang/pull/13730), [#14337](https://github.com/sgl-project/sglang/pull/14337), [#12588](https://github.com/sgl-project/sglang/pull/12588) |
| `test/registered/dllm/test_llada2_mini.py` | [#25420](https://github.com/sgl-project/sglang/pull/25420), [#25197](https://github.com/sgl-project/sglang/pull/25197), [#23785](https://github.com/sgl-project/sglang/pull/23785), [#22565](https://github.com/sgl-project/sglang/pull/22565), [#22305](https://github.com/sgl-project/sglang/pull/22305), [#21667](https://github.com/sgl-project/sglang/pull/21667), [#21187](https://github.com/sgl-project/sglang/pull/21187), [#18724](https://github.com/sgl-project/sglang/pull/18724), [#17484](https://github.com/sgl-project/sglang/pull/17484), [#16826](https://github.com/sgl-project/sglang/pull/16826), [#16949](https://github.com/sgl-project/sglang/pull/16949), [#16835](https://github.com/sgl-project/sglang/pull/16835), ... (13 total) |
| `test/registered/dllm/test_llada2_mini_amd.py` | [#21667](https://github.com/sgl-project/sglang/pull/21667), [#21187](https://github.com/sgl-project/sglang/pull/21187), [#18423](https://github.com/sgl-project/sglang/pull/18423), [#16826](https://github.com/sgl-project/sglang/pull/16826), [#16675](https://github.com/sgl-project/sglang/pull/16675), [#16420](https://github.com/sgl-project/sglang/pull/16420) |
| `test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py` | [#23835](https://github.com/sgl-project/sglang/pull/23835), [#20751](https://github.com/sgl-project/sglang/pull/20751) |

## PR 覆盖总览

- git 追溯 PR 数: 32
- 关键词/补充 PR 数: 0
- 当前文档总 PR 数: 32
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-11-26 | [#12588](https://github.com/sgl-project/sglang/pull/12588) | merged | [Feature] Initial block diffusion language model support | `python/sglang/srt/dllm/algorithm/__init__.py`, `python/sglang/srt/dllm/algorithm/base.py`, `python/sglang/srt/dllm/algorithm/low_confidence.py` |
| 2025-12-07 | [#14337](https://github.com/sgl-project/sglang/pull/14337) | merged | remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.) | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py` |
| 2025-12-12 | [#13730](https://github.com/sgl-project/sglang/pull/13730) | merged | [bugfix] fix TBO crashes when attn_tp_size > 1 | `python/sglang/srt/batch_overlap/operations.py`, `python/sglang/srt/batch_overlap/two_batch_overlap.py`, `python/sglang/srt/layers/communicator.py` |
| 2025-12-28 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | `python/sglang/jit_kernel/benchmark/bench_qknorm.py`, `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` |
| 2026-01-06 | [#16420](https://github.com/sgl-project/sglang/pull/16420) | merged | ci: migrate DLLM tests to test/registered/dllm/ | `test/registered/dllm/test_llada2_mini.py`, `test/registered/dllm/test_llada2_mini_amd.py`, `test/srt/run_suite.py` |
| 2026-01-08 | [#16675](https://github.com/sgl-project/sglang/pull/16675) | merged | [AMD] Fix CI - unit-test-backend-1-gpu-amd-mi35x and unit-test-backend-2-gpu-amd, stage-b-test-small-1-gpu-amd | `.github/workflows/pr-test-amd.yml`, `scripts/ci/slash_command_handler.py`, `test/registered/attention/test_create_kvindices.py` |
| 2026-01-11 | [#16835](https://github.com/sgl-project/sglang/pull/16835) | merged | Update est_time for stage-b-test-small-1-gpu tests | `test/registered/attention/test_torch_native_attention_backend.py`, `test/registered/backends/test_torch_compile.py`, `test/registered/core/test_deterministic.py` |
| 2026-01-15 | [#16826](https://github.com/sgl-project/sglang/pull/16826) | merged | [CI] Reorganize stage-b 1-GPU tests for 5090 compatibility | `.github/workflows/pr-test.yml`, `scripts/ci/slash_command_handler.py`, `test/registered/attention/test_create_kvindices.py` |
| 2026-01-15 | [#16949](https://github.com/sgl-project/sglang/pull/16949) | merged | [AMD CI] migrate and re-enable CI tests to new CI registry | `.github/workflows/pr-test-amd.yml`, `scripts/ci/slash_command_handler.py`, `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py` |
| 2026-01-24 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-02-09 | [#18423](https://github.com/sgl-project/sglang/pull/18423) | merged | [AMD] Update aiter to v0.1.10.post2 | `.github/workflows/pr-test-amd.yml`, `docker/rocm.Dockerfile`, `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-02-10 | [#17484](https://github.com/sgl-project/sglang/pull/17484) | merged | [DLLM] Basic dLLM scheduling strategy and implementation | `python/sglang/srt/dllm/mixin/req.py`, `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-02-15 | [#18860](https://github.com/sgl-project/sglang/pull/18860) | merged | update pre-commit config | `.github/workflows/lint.yml`, `.pre-commit-config.yaml`, `3rdparty/amd/tuning/benchmark_moe_rocm.py` |
| 2026-02-21 | [#18844](https://github.com/sgl-project/sglang/pull/18844) | merged | [Feature] rewrite rope kernel; remove flashinfer dependencies | `python/sglang/jit_kernel/benchmark/bench_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/rope.cuh`, `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` |
| 2026-03-05 | [#18724](https://github.com/sgl-project/sglang/pull/18724) | merged | [DLLM] Add initial radix cache support | `python/sglang/srt/dllm/mixin/req.py`, `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-03-09 | [#18485](https://github.com/sgl-project/sglang/pull/18485) | merged | [NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`, `python/sglang/srt/managers/scheduler.py` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `docs/advanced_features/vlm_query.ipynb`, `python/pyproject.toml`, `python/pyproject_cpu.toml` |
| 2026-03-23 | [#21187](https://github.com/sgl-project/sglang/pull/21187) | merged | ci: unify PR test suite naming | `agent-skills/write-sglang-test/SKILL.md`, `.github/actions/wait-for-jobs/action.yml`, `.github/workflows/pr-test-amd-rocm720.yml` |
| 2026-03-26 | [#21135](https://github.com/sgl-project/sglang/pull/21135) | merged | fix: use get_rope_config() to support models without rope_parameters | `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/ernie4.py` |
| 2026-04-01 | [#20751](https://github.com/sgl-project/sglang/pull/20751) | merged | [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture | `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml` |
| 2026-04-02 | [#21667](https://github.com/sgl-project/sglang/pull/21667) | merged | Unify GSM8K eval path to Chat API for regression CI readiness | `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/few_shot_gsm8k.py`, `python/sglang/test/few_shot_gsm8k_engine.py` |
| 2026-04-10 | [#22305](https://github.com/sgl-project/sglang/pull/22305) | merged | [CI] Update est_time for 64 tests based on actual elapsed times | `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen3_30b.py` |
| 2026-04-11 | [#22565](https://github.com/sgl-project/sglang/pull/22565) | merged | chore: update CI test est_time values | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_hicache.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` |
| 2026-04-26 | [#23732](https://github.com/sgl-project/sglang/pull/23732) | merged | Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-27 | [#23748](https://github.com/sgl-project/sglang/pull/23748) | merged | refactor(moe): centralize post-experts all-reduce skip predicate | `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2026-04-27 | [#23785](https://github.com/sgl-project/sglang/pull/23785) | merged | chore: update CI test est_time values | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` |
| 2026-05-02 | [#23835](https://github.com/sgl-project/sglang/pull/23835) | merged | [NPU] Add GitHub test summary and deduplicate test code. Part 1 | `python/sglang/test/ascend/gsm8k_ascend_mixin.py`, `python/sglang/test/ascend/test_ascend_utils.py`, `python/sglang/test/ascend/test_mmlu.py` |
| 2026-05-02 | [#24250](https://github.com/sgl-project/sglang/pull/24250) | merged | [SKILL] Upgrade sglang profile and auto_benchmark skills | `agent-skills/llm-serving-auto-benchmark/SKILL.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` |
| 2026-05-14 | [#25197](https://github.com/sgl-project/sglang/pull/25197) | merged | ci: decouple stage and runner for cuda registry | `python/sglang/test/ci/ci_register.py`, `scripts/ci/utils/ci_coverage_report.py`, `scripts/ci/utils/compute_partitions.py` |
| 2026-05-16 | [#25420](https://github.com/sgl-project/sglang/pull/25420) | merged | [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI | `agent-skills/add-jit-kernel/SKILL.md`, `agent-skills/ci-workflow-guide/SKILL.md`, `agent-skills/write-sglang-test/SKILL.md` |

## 逐 PR diff 审计卡

### PR #12588 - [Feature] Initial block diffusion language model support

- 链接: https://github.com/sgl-project/sglang/pull/12588
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+1286/-6，可读 patch 1544 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Initial block diffusion language model support」；模型线: LLaDA 2.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/dllm/algorithm/__init__.py`, `python/sglang/srt/dllm/algorithm/base.py`, `python/sglang/srt/dllm/algorithm/low_confidence.py`。
- 实现要点:
  - `python/sglang/srt/dllm/algorithm/__init__.py` added +39/-0；symbols: import_algorithms, get_algorithm
  - `python/sglang/srt/dllm/algorithm/base.py` added +18/-0；symbols: DllmAlgorithm, __init__, from_server_args
  - `python/sglang/srt/dllm/algorithm/low_confidence.py` added +59/-0；symbols: LowConfidence, run
  - `python/sglang/srt/dllm/config.py` added +40/-0；symbols: DllmConfig, __init__, from_server_args
- 代码 diff 细节:
  - `python/sglang/srt/dllm/algorithm/__init__.py` added +39/-0
  - `python/sglang/srt/dllm/algorithm/base.py` added +18/-0
  - `python/sglang/srt/dllm/algorithm/low_confidence.py` added +59/-0
  - `python/sglang/srt/dllm/config.py` added +40/-0
- 关键代码摘录:

```diff
diff -- python/sglang/srt/dllm/algorithm/__init__.py
@@ -0,0 +1,39 @@
+import importlib
+import logging
+import pkgutil
+
+from sglang.srt.dllm.config import DllmConfig
+
+logger = logging.getLogger(__name__)
+
+
+def import_algorithms():
+    mapping = {}
+    package_name = "sglang.srt.dllm.algorithm"
+    package = importlib.import_module(package_name)
diff -- python/sglang/srt/dllm/algorithm/base.py
@@ -0,0 +1,18 @@
+from sglang.srt.dllm.algorithm import get_algorithm
+from sglang.srt.dllm.config import DllmConfig
+from sglang.srt.server_args import ServerArgs
+
+
+class DllmAlgorithm:
+
+    def __init__(
+        self,
+        config: DllmConfig,
+    ):
+        self.block_size = config.block_size
+        self.mask_id = config.mask_id
```
- 已读文件:
  - runtime: `python/sglang/srt/dllm/algorithm/__init__.py` added +39/-0; `python/sglang/srt/dllm/algorithm/base.py` added +18/-0; `python/sglang/srt/dllm/algorithm/low_confidence.py` added +59/-0; `python/sglang/srt/dllm/config.py` added +40/-0; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +8/-1; `python/sglang/srt/layers/logits_processor.py` modified +18/-1; `python/sglang/srt/managers/schedule_batch.py` modified +42/-1; `python/sglang/srt/managers/scheduler.py` modified +18/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #14337 - remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)

- 链接: https://github.com/sgl-project/sglang/pull/14337
- 状态/时间: merged / 2025-12-07
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+0/-8，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)」；模型线: LLaDA 2.1；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`。
- 实现要点:
  - `python/sglang/srt/models/bailing_moe.py` modified +0/-2
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2
  - `python/sglang/srt/models/llada2.py` modified +0/-2
  - `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +0/-2
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2
  - `python/sglang/srt/models/llada2.py` modified +0/-2
  - `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -349,11 +349,9 @@ def forward_normal(
         num_tokens, hidden_size = hidden_states.shape
         hidden_states = hidden_states.view(-1, hidden_size)

-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
         if (
             self.alt_stream is not None
             and hidden_states.shape[0] > 0
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
             and get_is_capture_mode()
         ):
             final_hidden_states, shared_output = self.forward_normal_dual_stream(
diff -- python/sglang/srt/models/kimi_linear.py
@@ -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
         hidden_states = hidden_states.view(-1, hidden_size)

         shared_output = None
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024

         if (
             self.alt_stream is not None
             and self.num_shared_experts is not None
             and hidden_states.shape[0] > 0
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
             and get_is_capture_mode()
         ):
             current_stream = torch.cuda.current_stream()
```
- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +0/-2; `python/sglang/srt/models/kimi_linear.py` modified +0/-2; `python/sglang/srt/models/llada2.py` modified +0/-2; `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #13730 - [bugfix] fix TBO crashes when attn_tp_size > 1

- 链接: https://github.com/sgl-project/sglang/pull/13730
- 状态/时间: merged / 2025-12-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+285/-16，可读 patch 617 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] fix TBO crashes when attn_tp_size > 1」；模型线: LLaDA 2.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/batch_overlap/operations.py`, `python/sglang/srt/batch_overlap/two_batch_overlap.py`, `python/sglang/srt/layers/communicator.py`。
- 实现要点:
  - `python/sglang/srt/batch_overlap/operations.py` modified +10/-8
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +39/-6；symbols: _pad, _model_forward_tbo_merge_outputs
  - `python/sglang/srt/layers/communicator.py` modified +14/-1；symbols: _should_gather_for_tbo
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0
- 代码 diff 细节:
  - `python/sglang/srt/batch_overlap/operations.py` modified +10/-8
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +39/-6
  - `python/sglang/srt/layers/communicator.py` modified +14/-1
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0
- 关键代码摘录:

```diff
diff -- python/sglang/srt/batch_overlap/operations.py
@@ -83,7 +83,7 @@ def __init__(self, debug_name: str, stages: List[Stage], inputs: dict):
         # handling DP attention
         forward_batch: ForwardBatch = inputs["forward_batch"]
         self._global_dp_buffer_len = forward_batch.global_dp_buffer_len
-        self._local_dp_buffer_len = forward_batch.input_ids.shape[0]
+        self._local_dp_buffer_len = forward_batch.tbo_padded_len
         self._global_num_tokens = forward_batch.global_num_tokens_cpu
         self._is_dp_max_padding = forward_batch.dp_padding_mode.is_max_len()

@@ -92,13 +92,15 @@ def next(self):

         stage = self._stages[self._index]

-        if self._global_dp_buffer_len is not None:
diff -- python/sglang/srt/batch_overlap/two_batch_overlap.py
@@ -20,6 +20,7 @@
     CommunicateSummableTensorPairFn,
     ScatterMode,
 )
+from sglang.srt.layers.dp_attention import get_attention_tp_size
 from sglang.srt.layers.moe import (
     get_deepep_mode,
     get_moe_a2a_backend,
@@ -630,6 +631,11 @@ def filter_batch(
             ), f"{key=} {old_value=} {num_tokens=} {batch=}"
             output_dict[key] = old_value[start_token_index:end_token_index]

+        attention_tp_size = get_attention_tp_size()
+        output_dict["tbo_padded_len"] = (
```
- 已读文件:
  - runtime: `python/sglang/srt/batch_overlap/operations.py` modified +10/-8; `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +39/-6; `python/sglang/srt/layers/communicator.py` modified +14/-1; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/deepseek_v2.py` modified +2/-0; `python/sglang/srt/models/falcon_h1.py` modified +3/-1; `python/sglang/srt/models/glm4_moe.py` modified +2/-0
  - tests: `test/srt/ep/test_deepep_small.py` modified +178/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- 链接: https://github.com/sgl-project/sglang/pull/15835
- 状态/时间: merged / 2025-12-28
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+827/-127，可读 patch 1151 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] JIT Fused QK norm + qk norm clean up」；模型线: LLaDA 2.1；类别: 性能/后端优化；主要 diff: `python/sglang/jit_kernel/benchmark/bench_qknorm.py`, `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh`。
- 实现要点:
  - `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0；symbols: sglang_aot_qknorm, sglang_jit_qknorm, flashinfer_qknorm, torch_impl_qknorm
  - `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0；symbols: void, auto, int64_t, tvm
  - `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` added +26/-0
  - `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` modified +10/-0
- 代码 diff 细节:
  - `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0
  - `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0
  - `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` added +26/-0
  - `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` modified +10/-0
- 关键代码摘录:

```diff
diff -- python/sglang/jit_kernel/benchmark/bench_qknorm.py
@@ -0,0 +1,130 @@
+import itertools
+import os
+from typing import Tuple
+
+import torch
+import triton
+import triton.testing
+
+IS_CI = (
+    os.getenv("CI", "false").lower() == "true"
+    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
+)
+
diff -- python/sglang/jit_kernel/csrc/norm.cuh
@@ -0,0 +1,202 @@
+#include <sgl_kernel/runtime.cuh>
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/warp.cuh>
+
+#include <cuda_bf16.h>
+#include <cuda_fp16.h>
+#include <dlpack/dlpack.h>
+#include <tvm/ffi/container/tensor.h>
+
+#include <cstdint>
+#include <type_traits>
```
- 已读文件:
  - runtime: `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0; `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0; `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` added +26/-0; `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` modified +10/-0; `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` modified +31/-1; `python/sglang/jit_kernel/include/sgl_kernel/warp.cuh` added +14/-0; `python/sglang/jit_kernel/norm.py` added +55/-0; `python/sglang/jit_kernel/tests/test_qknorm.py` added +85/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #16420 - ci: migrate DLLM tests to test/registered/dllm/

- 链接: https://github.com/sgl-project/sglang/pull/16420
- 状态/时间: merged / 2026-01-06
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+8/-2，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: migrate DLLM tests to test/registered/dllm/」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `test/registered/dllm/test_llada2_mini.py`, `test/registered/dllm/test_llada2_mini_amd.py`, `test/srt/run_suite.py`。
- 实现要点:
  - `test/registered/dllm/test_llada2_mini.py` renamed +4/-0
  - `test/registered/dllm/test_llada2_mini_amd.py` renamed +4/-0
  - `test/srt/run_suite.py` modified +0/-2
- 代码 diff 细节:
  - `test/registered/dllm/test_llada2_mini.py` renamed +4/-0
  - `test/registered/dllm/test_llada2_mini_amd.py` renamed +4/-0
  - `test/srt/run_suite.py` modified +0/-2
- 关键代码摘录:

```diff
diff -- test/registered/dllm/test_llada2_mini.py
@@ -1,3 +1,7 @@
+from sglang.test.ci.ci_register import register_cuda_ci
+
+register_cuda_ci(est_time=520, suite="stage-b-test-small-1-gpu")
+
 import unittest
 from types import SimpleNamespace

diff -- test/registered/dllm/test_llada2_mini_amd.py
@@ -1,3 +1,7 @@
+from sglang.test.ci.ci_register import register_amd_ci
+
+register_amd_ci(est_time=520, suite="stage-b-test-small-1-gpu")
+
 """
 Test LLaDA2 (Diffusion Language Model) on AMD GPUs.

```
- 已读文件:
  - tests: `test/registered/dllm/test_llada2_mini.py` renamed +4/-0; `test/registered/dllm/test_llada2_mini_amd.py` renamed +4/-0; `test/srt/run_suite.py` modified +0/-2
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #16675 - [AMD] Fix CI - unit-test-backend-1-gpu-amd-mi35x and unit-test-backend-2-gpu-amd, stage-b-test-small-1-gpu-amd

- 链接: https://github.com/sgl-project/sglang/pull/16675
- 状态/时间: merged / 2026-01-08
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 60 个文件，+106/-143，可读 patch 699 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix CI - unit-test-backend-1-gpu-amd-mi35x and unit-test-backend-2-gpu-amd, stage-b-test-small-1-gpu-amd」；模型线: LLaDA 2.1；类别: 缺陷修复；主要 diff: `.github/workflows/pr-test-amd.yml`, `scripts/ci/slash_command_handler.py`, `test/registered/attention/test_create_kvindices.py`。
- 实现要点:
  - `.github/workflows/pr-test-amd.yml` modified +44/-85
  - `scripts/ci/slash_command_handler.py` modified +1/-0
  - `test/registered/attention/test_create_kvindices.py` modified +1/-1
  - `test/registered/attention/test_radix_attention.py` modified +1/-1
- 代码 diff 细节:
  - `.github/workflows/pr-test-amd.yml` modified +44/-85
  - `scripts/ci/slash_command_handler.py` modified +1/-0
  - `test/registered/attention/test_create_kvindices.py` modified +1/-1
  - `test/registered/attention/test_radix_attention.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test-amd.yml
@@ -224,7 +224,46 @@ jobs:
       - name: Run test
         timeout-minutes: 30
         run: |
-          bash scripts/ci/amd_ci_exec.sh -w "/sglang-checkout/test" python3 run_suite.py --hw amd --suite stage-b-test-small-1-gpu --auto-partition-id ${{ matrix.part }} --auto-partition-size 12
+          bash scripts/ci/amd_ci_exec.sh -w "/sglang-checkout/test" python3 run_suite.py --hw amd --suite stage-b-test-small-1-gpu-amd --auto-partition-id ${{ matrix.part }} --auto-partition-size 12
+
+  stage-b-test-small-1-gpu-amd-mi35x:
+    needs: [check-changes, stage-a-test-1-amd]
+    if: |
+      always() &&
+      (
+        (inputs.target_stage == 'stage-b-test-small-1-gpu-amd-mi35x') ||
+        (
diff -- scripts/ci/slash_command_handler.py
@@ -177,6 +177,7 @@ def handle_rerun_stage(
         "sgl-kernel-unit-test-amd",
         "stage-a-test-1-amd",
         "stage-b-test-small-1-gpu-amd",
+        "stage-b-test-small-1-gpu-amd-mi35x",
         "stage-b-test-large-2-gpu-amd",
         "unit-test-backend-1-gpu-amd",
         "unit-test-backend-2-gpu-amd",
```
- 已读文件:
  - tests: `test/registered/attention/test_create_kvindices.py` modified +1/-1; `test/registered/attention/test_radix_attention.py` modified +1/-1; `test/registered/attention/test_swa_unittest.py` modified +1/-1; `test/registered/attention/test_torch_native_attention_backend.py` modified +1/-1; `test/registered/attention/test_triton_attention_backend.py` modified +1/-1; `test/registered/attention/test_triton_attention_kernels.py` modified +1/-1; `test/registered/attention/test_triton_sliding_window.py` modified +1/-1; `test/registered/backends/test_torch_compile.py` modified +1/-1
  - other: `.github/workflows/pr-test-amd.yml` modified +44/-85; `scripts/ci/slash_command_handler.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #16835 - Update est_time for stage-b-test-small-1-gpu tests

- 链接: https://github.com/sgl-project/sglang/pull/16835
- 状态/时间: merged / 2026-01-11
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+24/-24，可读 patch 211 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update est_time for stage-b-test-small-1-gpu tests」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `test/registered/attention/test_torch_native_attention_backend.py`, `test/registered/backends/test_torch_compile.py`, `test/registered/core/test_deterministic.py`。
- 实现要点:
  - `test/registered/attention/test_torch_native_attention_backend.py` modified +1/-1
  - `test/registered/backends/test_torch_compile.py` modified +1/-1
  - `test/registered/core/test_deterministic.py` modified +1/-1
  - `test/registered/core/test_gpt_oss_1gpu.py` modified +1/-1
- 代码 diff 细节:
  - `test/registered/attention/test_torch_native_attention_backend.py` modified +1/-1
  - `test/registered/backends/test_torch_compile.py` modified +1/-1
  - `test/registered/core/test_deterministic.py` modified +1/-1
  - `test/registered/core/test_gpt_oss_1gpu.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- test/registered/attention/test_torch_native_attention_backend.py
@@ -18,7 +18,7 @@
 )

 # Torch native attention backend integration test with MMLU eval
-register_cuda_ci(est_time=150, suite="stage-b-test-small-1-gpu")
+register_cuda_ci(est_time=169, suite="stage-b-test-small-1-gpu")
 register_amd_ci(est_time=150, suite="stage-b-test-small-1-gpu-amd")


diff -- test/registered/backends/test_torch_compile.py
@@ -16,7 +16,7 @@
     popen_launch_server,
 )

-register_cuda_ci(est_time=190, suite="stage-b-test-small-1-gpu")
+register_cuda_ci(est_time=144, suite="stage-b-test-small-1-gpu")
 register_amd_ci(est_time=1100, suite="stage-b-test-small-1-gpu-amd")


```
- 已读文件:
  - tests: `test/registered/attention/test_torch_native_attention_backend.py` modified +1/-1; `test/registered/backends/test_torch_compile.py` modified +1/-1; `test/registered/core/test_deterministic.py` modified +1/-1; `test/registered/core/test_gpt_oss_1gpu.py` modified +1/-1; `test/registered/cuda_graph/test_piecewise_cuda_graph_small_1_gpu.py` modified +1/-1; `test/registered/dllm/test_llada2_mini.py` modified +1/-1; `test/registered/hicache/test_hicache_variants.py` modified +1/-1; `test/registered/lora/test_lora_update.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #16826 - [CI] Reorganize stage-b 1-GPU tests for 5090 compatibility

- 链接: https://github.com/sgl-project/sglang/pull/16826
- 状态/时间: merged / 2026-01-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+179/-280，可读 patch 1402 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Reorganize stage-b 1-GPU tests for 5090 compatibility」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test.yml`, `scripts/ci/slash_command_handler.py`, `test/registered/attention/test_create_kvindices.py`。
- 实现要点:
  - `.github/workflows/pr-test.yml` modified +15/-58
  - `scripts/ci/slash_command_handler.py` modified +0/-1
  - `test/registered/attention/test_create_kvindices.py` modified +0/-1
  - `test/registered/attention/test_mamba_unittest.py` modified +0/-1
- 代码 diff 细节:
  - `.github/workflows/pr-test.yml` modified +15/-58
  - `scripts/ci/slash_command_handler.py` modified +0/-1
  - `test/registered/attention/test_create_kvindices.py` modified +0/-1
  - `test/registered/attention/test_mamba_unittest.py` modified +0/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test.yml
@@ -595,14 +595,15 @@ jobs:
           ((needs.check-changes.outputs.main_package == 'true') || (needs.check-changes.outputs.sgl_kernel == 'true'))
         )
       )
-    runs-on: 1-gpu-runner
+    runs-on: 1-gpu-5090
     env:
-      RUNNER_LABELS: 1-gpu-runner
+      RUNNER_LABELS: 1-gpu-5090
+      IS_BLACKWELL: "1"
     strategy:
       fail-fast: false
-      max-parallel: ${{ fromJson(needs.check-changes.outputs.max_parallel) }}
+      max-parallel: 4
diff -- scripts/ci/slash_command_handler.py
@@ -227,7 +227,6 @@ def handle_rerun_stage(
         "stage-a-cpu-only",
         "stage-b-test-small-1-gpu",
         "stage-b-test-large-1-gpu",
-        "stage-b-test-small-1-gpu-5090",
         "stage-b-test-large-2-gpu",
         "stage-c-test-large-4-gpu",
         "stage-c-test-large-4-gpu-b200",
```
- 已读文件:
  - tests: `test/registered/attention/test_create_kvindices.py` modified +0/-1; `test/registered/attention/test_mamba_unittest.py` modified +0/-1; `test/registered/attention/test_radix_attention.py` modified +0/-1; `test/registered/attention/test_radix_cache_unit.py` modified +0/-1; `test/registered/attention/test_swa_unittest.py` modified +1/-1; `test/registered/attention/test_torch_native_attention_backend.py` modified +0/-1; `test/registered/attention/test_triton_attention_backend.py` modified +1/-1; `test/registered/attention/test_triton_attention_kernels.py` modified +1/-1
  - other: `.github/workflows/pr-test.yml` modified +15/-58; `scripts/ci/slash_command_handler.py` modified +0/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #16949 - [AMD CI] migrate and re-enable CI tests to new CI registry

- 链接: https://github.com/sgl-project/sglang/pull/16949
- 状态/时间: merged / 2026-01-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+86/-40，可读 patch 420 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD CI] migrate and re-enable CI tests to new CI registry」；模型线: LLaDA 2.1；类别: 模型支持/运行时入口；主要 diff: `.github/workflows/pr-test-amd.yml`, `scripts/ci/slash_command_handler.py`, `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`。
- 实现要点:
  - `.github/workflows/pr-test-amd.yml` modified +16/-6
  - `scripts/ci/slash_command_handler.py` modified +1/-0
  - `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py` renamed +3/-0
  - `test/registered/attention/test_mamba_unittest.py` modified +2/-1
- 代码 diff 细节:
  - `.github/workflows/pr-test-amd.yml` modified +16/-6
  - `scripts/ci/slash_command_handler.py` modified +1/-0
  - `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py` renamed +3/-0
  - `test/registered/attention/test_mamba_unittest.py` modified +2/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test-amd.yml
@@ -634,10 +634,18 @@ jobs:
         run: |
           bash scripts/ci/amd_ci_exec.sh python3 run_suite.py --suite per-commit-8-gpu-amd --auto-partition-id ${{ matrix.part }} --auto-partition-size 2 --timeout-per-file 3600

-  unit-test-backend-8-gpu-amd-mi35x:
-    needs: [check-changes, stage-a-test-1-amd]
-    if: always() && !failure() && !cancelled() &&
-      ((needs.check-changes.outputs.main_package == 'true') || (needs.check-changes.outputs.sgl_kernel == 'true'))
+  stage-c-test-large-8-gpu-amd-mi35x:
+    needs: [check-changes, call-gate, stage-b-test-small-1-gpu-amd, stage-b-test-large-2-gpu-amd]
+    if: |
+      always() &&
+      (
+        (inputs.target_stage == 'stage-c-test-large-8-gpu-amd-mi35x') ||
diff -- scripts/ci/slash_command_handler.py
@@ -258,6 +258,7 @@ def handle_rerun_stage(
         "stage-b-test-small-1-gpu-amd",
         "stage-b-test-small-1-gpu-amd-mi35x",
         "stage-b-test-large-2-gpu-amd",
+        "stage-c-test-large-8-gpu-amd-mi35x",
         "unit-test-backend-1-gpu-amd",
         "unit-test-backend-2-gpu-amd",
         "unit-test-backend-8-gpu-amd",
```
- 已读文件:
  - tests: `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py` renamed +3/-0; `test/registered/attention/test_mamba_unittest.py` modified +2/-1; `test/registered/attention/test_radix_cache_unit.py` modified +2/-1; `test/registered/core/test_hidden_states.py` modified +9/-1; `test/registered/dllm/test_llada2_mini.py` modified +6/-2; `test/registered/hicache/test_hicache_storage.py` modified +2/-1; `test/registered/hicache/test_hicache_variants.py` modified +3/-2; `test/registered/layers/mamba/test_causal_conv1d.py` modified +2/-1
  - other: `.github/workflows/pr-test-amd.yml` modified +16/-6; `scripts/ci/slash_command_handler.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #17570 - Use attn tp group in embedding for more models

- 链接: https://github.com/sgl-project/sglang/pull/17570
- 状态/时间: merged / 2026-01-24
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+19/-19，可读 patch 171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use attn tp group in embedding for more models」；模型线: LLaDA 2.1；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`。
- 实现要点:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1
  - `python/sglang/srt/models/glm4.py` modified +1/-1
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1
  - `python/sglang/srt/models/glm4.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -717,7 +717,7 @@ def __init__(
                 self.embed_dim,
                 quant_config=quant_config,
                 prefix=add_prefix("word_embeddings", prefix),
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
             )
         else:
             self.word_embeddings = PPMissingLayer()
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -62,7 +62,7 @@ def __init__(
         self.word_embeddings = VocabParallelEmbedding(
             config.vocab_size,
             config.hidden_size,
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
             prefix=add_prefix("word_embeddings", prefix),
         )

```
- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +1/-1; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/falcon_h1.py` modified +1/-1; `python/sglang/srt/models/glm4.py` modified +1/-1; `python/sglang/srt/models/glm4_moe.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #18423 - [AMD] Update aiter to v0.1.10.post2

- 链接: https://github.com/sgl-project/sglang/pull/18423
- 状态/时间: merged / 2026-02-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+79/-41，可读 patch 391 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Update aiter to v0.1.10.post2」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test-amd.yml`, `docker/rocm.Dockerfile`, `python/sglang/srt/layers/attention/aiter_backend.py`。
- 实现要点:
  - `.github/workflows/pr-test-amd.yml` modified +3/-3
  - `docker/rocm.Dockerfile` modified +2/-2
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +19/-8
  - `scripts/ci/amd/amd_ci_install_dependency.sh` modified +1/-1
- 代码 diff 细节:
  - `.github/workflows/pr-test-amd.yml` modified +3/-3
  - `docker/rocm.Dockerfile` modified +2/-2
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +19/-8
  - `scripts/ci/amd/amd_ci_install_dependency.sh` modified +1/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test-amd.yml
@@ -251,7 +251,7 @@ jobs:
       fail-fast: false
       matrix:
         runner: [linux-mi325-gpu-1]
-        part: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
+        part: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
     runs-on: ${{matrix.runner}}
     steps:
       - name: Checkout code
@@ -273,7 +273,7 @@ jobs:
       - name: Run test
         timeout-minutes: 30
         run: |
-          bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout/test" python3 run_suite.py --hw amd --suite stage-b-test-small-1-gpu-amd --auto-partition-id ${{ matrix.part }} --auto-partition-size 13 --timeout-per-file 1800
diff -- docker/rocm.Dockerfile
@@ -21,7 +21,7 @@ ENV BUILD_TRITON="0"
 ENV BUILD_LLVM="0"
 ENV BUILD_AITER_ALL="1"
 ENV BUILD_MOONCAKE="1"
-ENV AITER_COMMIT="v0.1.9.post1"
+ENV AITER_COMMIT="v0.1.10.post2"

 # ===============================
 # Base image 950 and args
@@ -31,7 +31,7 @@ ENV BUILD_TRITON="0"
 ENV BUILD_LLVM="0"
 ENV BUILD_AITER_ALL="0"
 ENV BUILD_MOONCAKE="1"
-ENV AITER_COMMIT="v0.1.9.post1"
```
- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +19/-8
  - tests: `test/registered/attention/test_triton_attention_backend.py` modified +1/-1; `test/registered/dllm/test_llada2_mini_amd.py` modified +1/-1; `test/registered/eval/test_eval_accuracy_large.py` modified +1/-1; `test/registered/mla/test_mla_fp8.py` modified +1/-1; `test/registered/models/test_vlm_models.py` modified +1/-1; `test/registered/perf/test_bench_serving_2gpu.py` modified +1/-1; `test/registered/quant/test_eval_fp8_accuracy.py` modified +1/-1; `test/registered/quant/test_torchao.py` modified +1/-1
  - other: `.github/workflows/pr-test-amd.yml` modified +3/-3; `docker/rocm.Dockerfile` modified +2/-2; `scripts/ci/amd/amd_ci_install_dependency.sh` modified +1/-1; `scripts/ci/amd/amd_ci_warmup_aiter.py` modified +44/-17
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #17484 - [DLLM] Basic dLLM scheduling strategy and implementation

- 链接: https://github.com/sgl-project/sglang/pull/17484
- 状态/时间: merged / 2026-02-10
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+461/-210，可读 patch 911 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DLLM] Basic dLLM scheduling strategy and implementation」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/dllm/mixin/req.py`, `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/schedule_batch.py`。
- 实现要点:
  - `python/sglang/srt/dllm/mixin/req.py` added +67/-0；symbols: DllmReqPhase, ReqDllmMixin, init_diffusion_llm, is_dllm
  - `python/sglang/srt/dllm/mixin/scheduler.py` added +313/-0；symbols: SchedulerDllmMixin, init_diffusion_llm, get_new_batch_dllm, _fetch_waiting_reqs
  - `python/sglang/srt/managers/schedule_batch.py` modified +4/-77；symbols: Req
  - `python/sglang/srt/managers/schedule_policy.py` modified +29/-3；symbols: add_dllm_staging_req
- 代码 diff 细节:
  - `python/sglang/srt/dllm/mixin/req.py` added +67/-0
  - `python/sglang/srt/dllm/mixin/scheduler.py` added +313/-0
  - `python/sglang/srt/managers/schedule_batch.py` modified +4/-77
  - `python/sglang/srt/managers/schedule_policy.py` modified +29/-3
- 关键代码摘录:

```diff
diff -- python/sglang/srt/dllm/mixin/req.py
@@ -0,0 +1,67 @@
+from __future__ import annotations
+
+import enum
+from typing import TYPE_CHECKING, Optional
+
+from sglang.srt.dllm.config import DllmConfig
+
+if TYPE_CHECKING:
+    from sglang.srt.managers.schedule_batch import Req
+
+
+class DllmReqPhase(str, enum.Enum):
+    STAGING_PREFILL = "staging_prefill"
diff -- python/sglang/srt/dllm/mixin/scheduler.py
@@ -0,0 +1,313 @@
+from __future__ import annotations
+
+import logging
+import time
+from typing import TYPE_CHECKING, List, Optional, Set, Union
+
+from sglang.srt.dllm.config import DllmConfig
+from sglang.srt.dllm.mixin.req import DllmReqPhase
+from sglang.srt.managers.schedule_batch import Req, RequestStage, ScheduleBatch
+from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
+from sglang.srt.model_executor.forward_batch_info import ForwardMode
+
+logger = logging.getLogger(__name__)
```
- 已读文件:
  - runtime: `python/sglang/srt/dllm/mixin/req.py` added +67/-0; `python/sglang/srt/dllm/mixin/scheduler.py` added +313/-0; `python/sglang/srt/managers/schedule_batch.py` modified +4/-77; `python/sglang/srt/managers/schedule_policy.py` modified +29/-3; `python/sglang/srt/managers/scheduler.py` modified +19/-55; `python/sglang/srt/model_executor/forward_batch_info.py` modified +1/-1; `python/sglang/srt/server_args.py` modified +18/-0
  - tests: `test/registered/dllm/test_dllm_batching.py` removed +0/-71; `test/registered/dllm/test_llada2_mini.py` modified +10/-3
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #18860 - update pre-commit config

- 链接: https://github.com/sgl-project/sglang/pull/18860
- 状态/时间: merged / 2026-02-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+170/-159，可读 patch 1254 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update pre-commit config」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `.github/workflows/lint.yml`, `.pre-commit-config.yaml`, `3rdparty/amd/tuning/benchmark_moe_rocm.py`。
- 实现要点:
  - `.github/workflows/lint.yml` modified +2/-2
  - `.pre-commit-config.yaml` modified +6/-6
  - `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
  - `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1
- 代码 diff 细节:
  - `.github/workflows/lint.yml` modified +2/-2
  - `.pre-commit-config.yaml` modified +6/-6
  - `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
  - `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/lint.yml
@@ -26,9 +26,9 @@ jobs:
         run: SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure

       - name: Run sgl-kernel clang-format checks
-        uses: DoozyX/clang-format-lint-action@v0.18.1
+        uses: DoozyX/clang-format-lint-action@v0.20
         with:
           source: sgl-kernel
           extensions: h,c,cpp,hpp,cu,cuh,cc
-          clangFormatVersion: 18
+          clangFormatVersion: 20
           style: file
diff -- .pre-commit-config.yaml
@@ -3,7 +3,7 @@ exclude: ^(python/sglang/multimodal_gen/csrc|python/sglang/jit_kernel/flash_atte

 repos:
   - repo: https://github.com/pre-commit/pre-commit-hooks
-    rev: v5.0.0
+    rev: v6.0.0
     hooks:
       - id: check-symlinks
       - id: destroyed-symlinks
@@ -21,12 +21,12 @@ repos:
       - id: debug-statements
       - id: no-commit-to-branch
   - repo: https://github.com/PyCQA/isort
-    rev: 5.13.2
```
- 已读文件:
  - runtime: `python/sglang/jit_kernel/include/sgl_kernel/type.cuh` modified +16/-15; `python/sglang/multimodal_gen/apps/webui/main.py` modified +2/-4; `python/sglang/multimodal_gen/configs/models/encoders/qwen3.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/cache/__init__.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/parallel_state.py` modified +2/-1; `python/sglang/multimodal_gen/runtime/layers/activation.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/layers/layernorm.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/layers/rotary_embedding.py` modified +1/-0
  - docs/bench: `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1; `benchmark/tip_suggestion/bench_other.py` modified +2/-8; `benchmark/tip_suggestion/bench_sglang.py` modified +2/-8; `benchmark/tip_suggestion/lmql_funcs.py` modified +2/-8; `docs/advanced_features/lora.ipynb` modified +10/-20; `docs/advanced_features/structured_outputs.ipynb` modified +0/-1; `docs/advanced_features/structured_outputs_for_reasoning_models.ipynb` modified +0/-1; `docs/advanced_features/vlm_query.ipynb` modified +0/-1
  - other: `.github/workflows/lint.yml` modified +2/-2; `.pre-commit-config.yaml` modified +6/-6; `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #18844 - [Feature] rewrite rope kernel; remove flashinfer dependencies

- 链接: https://github.com/sgl-project/sglang/pull/18844
- 状态/时间: merged / 2026-02-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1147/-1099，可读 patch 2459 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] rewrite rope kernel; remove flashinfer dependencies」；模型线: LLaDA 2.1；类别: 性能/后端优化；主要 diff: `python/sglang/jit_kernel/benchmark/bench_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/rope.cuh`, `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh`。
- 实现要点:
  - `python/sglang/jit_kernel/benchmark/bench_rope.py` added +350/-0；symbols: create_cos_sin_cache, flashinfer_rope, sglang_rope_v0, sglang_rope_v1
  - `python/sglang/jit_kernel/csrc/elementwise/rope.cuh` modified +424/-616；symbols: void, auto, int64_t, tvm
  - `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` modified +26/-0
  - `python/sglang/jit_kernel/rope.py` modified +112/-186；symbols: _jit_fused_rope_module, apply_rope_inplace, apply_rope_inplace_with_kvcache
- 代码 diff 细节:
  - `python/sglang/jit_kernel/benchmark/bench_rope.py` added +350/-0
  - `python/sglang/jit_kernel/csrc/elementwise/rope.cuh` modified +424/-616
  - `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` modified +26/-0
  - `python/sglang/jit_kernel/rope.py` modified +112/-186
- 关键代码摘录:

```diff
diff -- python/sglang/jit_kernel/benchmark/bench_rope.py
@@ -0,0 +1,350 @@
+import itertools
+
+import torch
+import triton
+import triton.testing
+
+from sglang.jit_kernel.benchmark.utils import (
+    DEFAULT_DEVICE,
+    DEFAULT_DTYPE,
+    get_benchmark_range,
+    run_benchmark,
+)
+
diff -- python/sglang/jit_kernel/csrc/elementwise/rope.cuh
@@ -1,655 +1,463 @@
-/*
- * Copyright (c) 2024 by FlashInfer team.
- *
- * Licensed under the Apache License, Version 2.0 (the "License");
- * you may not use this file except in compliance with the License.
- * You may obtain a copy of the License at
- *
- *   http://www.apache.org/licenses/LICENSE-2.0
- *
- * Unless required by applicable law or agreed to in writing, software
- * distributed under the License is distributed on an "AS IS" BASIS,
- * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
- * See the License for the specific language governing permissions and
```
- 已读文件:
  - runtime: `python/sglang/jit_kernel/benchmark/bench_rope.py` added +350/-0; `python/sglang/jit_kernel/csrc/elementwise/rope.cuh` modified +424/-616; `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` modified +26/-0; `python/sglang/jit_kernel/rope.py` modified +112/-186; `python/sglang/jit_kernel/tests/test_rope.py` modified +212/-269; `python/sglang/srt/layers/rotary_embedding.py` modified +14/-16; `python/sglang/srt/models/gpt_oss.py` modified +1/-6; `python/sglang/srt/models/llada2.py` modified +6/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #18724 - [DLLM] Add initial radix cache support

- 链接: https://github.com/sgl-project/sglang/pull/18724
- 状态/时间: merged / 2026-03-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+84/-57，可读 patch 201 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DLLM] Add initial radix cache support」；模型线: LLaDA 2.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/dllm/mixin/req.py`, `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/schedule_batch.py`。
- 实现要点:
  - `python/sglang/srt/dllm/mixin/req.py` modified +18/-11；symbols: _update_block_offset_for_dllm
  - `python/sglang/srt/dllm/mixin/scheduler.py` modified +42/-1；symbols: process_batch_result_dllm
  - `python/sglang/srt/managers/schedule_batch.py` modified +3/-0
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +0/-40
- 代码 diff 细节:
  - `python/sglang/srt/dllm/mixin/req.py` modified +18/-11
  - `python/sglang/srt/dllm/mixin/scheduler.py` modified +42/-1
  - `python/sglang/srt/managers/schedule_batch.py` modified +3/-0
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +0/-40
- 关键代码摘录:

```diff
diff -- python/sglang/srt/dllm/mixin/req.py
@@ -19,7 +19,6 @@ class DllmReqPhase(str, enum.Enum):
 class ReqDllmMixin:
     def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
         self.dllm_phase: Optional[DllmReqPhase] = None
-        self.dllm_ids = []
         self.dllm_block_offset = 0
         self.dllm_config = dllm_config

@@ -55,13 +54,21 @@ def determine_dllm_phase(self: Req):
             self.dllm_phase = DllmReqPhase.STAGING_DECODE

     def _init_fill_ids_for_dllm(self: Req):
-        if not self.dllm_ids:
-            self.dllm_ids = (
diff -- python/sglang/srt/dllm/mixin/scheduler.py
@@ -7,13 +7,14 @@
 from sglang.srt.dllm.mixin.req import DllmReqPhase
 from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
 from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
+from sglang.srt.mem_cache.common import release_kv_cache
 from sglang.srt.model_executor.forward_batch_info import ForwardMode
 from sglang.srt.observability.req_time_stats import set_time_batch

 logger = logging.getLogger(__name__)

 if TYPE_CHECKING:
-    from sglang.srt.managers.scheduler import Scheduler
+    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

```
- 已读文件:
  - runtime: `python/sglang/srt/dllm/mixin/req.py` modified +18/-11; `python/sglang/srt/dllm/mixin/scheduler.py` modified +42/-1; `python/sglang/srt/managers/schedule_batch.py` modified +3/-0; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +0/-40; `python/sglang/srt/server_args.py` modified +20/-4
  - tests: `test/registered/dllm/test_llada2_mini.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #18485 - [NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications

- 链接: https://github.com/sgl-project/sglang/pull/18485
- 状态/时间: merged / 2026-03-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+250/-9，可读 patch 400 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications」；模型线: LLaDA 2.1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`, `python/sglang/srt/managers/scheduler.py`。
- 实现要点:
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +103/-2；symbols: forward_dllm
  - `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` modified +18/-5
  - `python/sglang/srt/managers/scheduler.py` modified +21/-0
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +1/-1
- 代码 diff 细节:
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +103/-2
  - `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` modified +18/-5
  - `python/sglang/srt/managers/scheduler.py` modified +21/-0
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py
@@ -11,6 +11,7 @@
 )

 from sglang.srt.configs.model_config import AttentionArch
+from sglang.srt.dllm.config import DllmConfig
 from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
     AscendTorchNativeAttnBackend,
 )
@@ -252,6 +253,13 @@ def __init__(self, model_runner: ModelRunner):
         if self.use_mla:
             self.ringmla_mask = self.ascend_attn_mask_builder.ringmla_mask

+        # dllm model config
+        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
diff -- python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py
@@ -83,10 +83,16 @@ def __init__(self, model_runner: ModelRunner):
         self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")

     def _init_arch_map(self):
-        self.attr_name: Dict[str, str] = {
-            AttentionArch.MLA: "actual_seq_lengths_kv",
-            AttentionArch.MHA: "context_lens",
-        }
+        if self.is_dllm:
+            self.attr_name: Dict[str, str] = {
+                AttentionArch.MLA: "actual_seq_lengths_kv",
+                AttentionArch.MHA: "actual_seq_lengths_kv",
+            }
+        else:
```
- 已读文件:
  - runtime: `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +103/-2; `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` modified +18/-5; `python/sglang/srt/managers/scheduler.py` modified +21/-0; `python/sglang/srt/model_executor/forward_batch_info.py` modified +1/-1; `python/sglang/srt/models/llada2.py` modified +13/-1; `python/sglang/srt/server_args.py` modified +6/-0
  - tests: `test/srt/ascend/test_llada2_mini_ascend.py` added +87/-0; `test/srt/run_suite.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #17784 - Upgrade transformers==5.3.0

- 链接: https://github.com/sgl-project/sglang/pull/17784
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 95 个文件，+1136/-343，可读 patch 2752 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade transformers==5.3.0」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `docs/advanced_features/vlm_query.ipynb`, `python/pyproject.toml`, `python/pyproject_cpu.toml`。
- 实现要点:
  - `docs/advanced_features/vlm_query.ipynb` modified +3/-3
  - `python/pyproject.toml` modified +4/-4
  - `python/pyproject_cpu.toml` modified +3/-4
  - `python/pyproject_npu.toml` modified +3/-4
- 代码 diff 细节:
  - `docs/advanced_features/vlm_query.ipynb` modified +3/-3
  - `python/pyproject.toml` modified +4/-4
  - `python/pyproject_cpu.toml` modified +3/-4
  - `python/pyproject_npu.toml` modified +3/-4
- 关键代码摘录:

```diff
diff -- docs/advanced_features/vlm_query.ipynb
@@ -182,9 +182,8 @@
     "from transformers import Qwen2_5_VLForConditionalGeneration\n",
     "\n",
     "processor = AutoProcessor.from_pretrained(model_path, use_fast=True)\n",
-    "vision = (\n",
-    "    Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval().visual.cuda()\n",
-    ")"
+    "model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval()\n",
+    "vision = model.model.visual.cuda()"
    ]
   },
   {
@@ -203,6 +202,7 @@
     "precomputed_embeddings = vision(\n",
diff -- python/pyproject.toml
@@ -30,8 +30,6 @@ dependencies = [
   "flashinfer_python==0.6.6", # keep it aligned with jit-cache version in Dockerfile
   "flashinfer_cubin==0.6.6",
   "gguf",
-  "hf_transfer",
-  "huggingface_hub",
   "interegular",
   "llguidance>=0.7.11,<0.8.0",
   "modelscope",
@@ -72,7 +70,8 @@ dependencies = [
   "av ; sys_platform == 'linux' and (platform_machine == 'aarch64' or platform_machine == 'arm64' or platform_machine == 'armv7l')",
   "torchvision",
   "tqdm",
-  "transformers==4.57.1",
```
- 已读文件:
  - runtime: `python/pyproject.toml` modified +4/-4; `python/pyproject_cpu.toml` modified +3/-4; `python/pyproject_npu.toml` modified +3/-4; `python/pyproject_other.toml` modified +3/-4; `python/pyproject_xpu.toml` modified +3/-4; `python/sglang/check_env.py` modified +0/-1; `python/sglang/multimodal_gen/runtime/loader/weight_utils.py` modified +0/-16; `python/sglang/multimodal_gen/runtime/models/encoders/llama.py` modified +2/-2
  - docs/bench: `docs/advanced_features/vlm_query.ipynb` modified +3/-3
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21187 - ci: unify PR test suite naming

- 链接: https://github.com/sgl-project/sglang/pull/21187
- 状态/时间: merged / 2026-03-23
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+267/-267，可读 patch 1675 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: unify PR test suite naming」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `agent-skills/write-sglang-test/SKILL.md`, `.github/actions/wait-for-jobs/action.yml`, `.github/workflows/pr-test-amd-rocm720.yml`。
- 实现要点:
  - `agent-skills/write-sglang-test/SKILL.md` modified +12/-12
  - `.github/actions/wait-for-jobs/action.yml` modified +1/-1
  - `.github/workflows/pr-test-amd-rocm720.yml` modified +30/-30
  - `.github/workflows/pr-test-amd.yml` modified +38/-38
- 代码 diff 细节:
  - `agent-skills/write-sglang-test/SKILL.md` modified +12/-12
  - `.github/actions/wait-for-jobs/action.yml` modified +1/-1
  - `.github/workflows/pr-test-amd-rocm720.yml` modified +30/-30
  - `.github/workflows/pr-test-amd.yml` modified +38/-38
- 关键代码摘录:

```diff
diff -- agent-skills/write-sglang-test/SKILL.md
@@ -18,10 +18,10 @@ description: Guide for writing SGLang CI/UT tests following project conventions.

 | Scenario | Model | CI Registration | Suite |
 |----------|-------|-----------------|-------|
-| **Unit tests** (no server / engine launch) | None | `register_cpu_ci` (prefer) or `register_cuda_ci` | `stage-a-cpu-only` or `stage-b-test-small-1-gpu` |
-| **Common / backend-independent** (middleware, abort, routing, config, arg parsing) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` only | `stage-b-test-small-1-gpu` |
-| **Model-agnostic functionality** (sampling, session, OpenAI API features) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` (+ AMD if relevant) | `stage-b-test-small-1-gpu` |
-| **General performance** (single node, no spec/DP/parallelism) | `DEFAULT_MODEL_NAME_FOR_TEST` (8B) | `register_cuda_ci` | `stage-b-test-large-1-gpu` |
+| **Unit tests** (no server / engine launch) | None | `register_cpu_ci` (prefer) or `register_cuda_ci` | `stage-a-test-cpu` or `stage-b-test-1-gpu-small` |
+| **Common / backend-independent** (middleware, abort, routing, config, arg parsing) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` only | `stage-b-test-1-gpu-small` |
+| **Model-agnostic functionality** (sampling, session, OpenAI API features) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` (+ AMD if relevant) | `stage-b-test-1-gpu-small` |
+| **General performance** (single node, no spec/DP/parallelism) | `DEFAULT_MODEL_NAME_FOR_TEST` (8B) | `register_cuda_ci` | `stage-b-test-1-gpu-large` |
 | **Bigger features** (spec, DP, TP, disaggregation) | Case by case | Case by case | See suite table below |

diff -- .github/actions/wait-for-jobs/action.yml
@@ -8,7 +8,7 @@ inputs:
   jobs:
     description: |
       JSON array of job specs to wait for. Each element is either:
-        - a string: exact job name (e.g. "stage-a-test-small-1-gpu")
+        - a string: exact job name (e.g. "stage-a-test-1-gpu-small")
         - an object { "prefix": "...", "expected_count": N }: for matrix jobs
     required: true
   max-wait-minutes:
```
- 已读文件:
  - runtime: `python/sglang/jit_kernel/tests/test_moe_lora_align_block_size.py` modified +1/-1
  - tests: `test/README.md` modified +19/-19; `test/registered/amd/disaggregation/test_disaggregation_basic.py` modified +2/-2; `test/registered/attention/test_chunk_gated_delta_rule.py` modified +1/-1; `test/registered/attention/test_create_kvindices.py` modified +2/-2; `test/registered/attention/test_fa3.py` modified +1/-1; `test/registered/attention/test_hybrid_attn_backend.py` modified +1/-1; `test/registered/attention/test_kda_kernels.py` modified +1/-1; `test/registered/attention/test_normal_decode_set_metadata.py` modified +1/-1
  - other: `agent-skills/write-sglang-test/SKILL.md` modified +12/-12; `.github/actions/wait-for-jobs/action.yml` modified +1/-1; `.github/workflows/pr-test-amd-rocm720.yml` modified +30/-30; `.github/workflows/pr-test-amd.yml` modified +38/-38; `.github/workflows/pr-test.yml` modified +28/-28; `scripts/ci/utils/slash_command_handler.py` modified +19/-19
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21135 - fix: use get_rope_config() to support models without rope_parameters

- 链接: https://github.com/sgl-project/sglang/pull/21135
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+44/-42，可读 patch 342 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: use get_rope_config() to support models without rope_parameters」；模型线: LLaDA 2.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/ernie4.py`。
- 实现要点:
  - `python/sglang/srt/models/baichuan.py` modified +2/-1
  - `python/sglang/srt/models/deepseek.py` modified +2/-2
  - `python/sglang/srt/models/ernie4.py` modified +2/-2
  - `python/sglang/srt/models/exaone.py` modified +2/-2
- 代码 diff 细节:
  - `python/sglang/srt/models/baichuan.py` modified +2/-1
  - `python/sglang/srt/models/deepseek.py` modified +2/-2
  - `python/sglang/srt/models/ernie4.py` modified +2/-2
  - `python/sglang/srt/models/exaone.py` modified +2/-2
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/baichuan.py
@@ -48,6 +48,7 @@
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 from sglang.srt.model_loader.weight_utils import default_weight_loader
 from sglang.srt.utils import add_prefix, is_npu
+from sglang.srt.utils.hf_transformers_utils import get_rope_config

 _is_npu = is_npu()

@@ -229,7 +230,7 @@ def __init__(
     ):
         super().__init__()
         self.hidden_size = config.hidden_size
-        rope_theta = config.rope_parameters["rope_theta"]
+        rope_theta, _ = get_rope_config(config)
diff -- python/sglang/srt/models/deepseek.py
@@ -49,6 +49,7 @@
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 from sglang.srt.model_loader.weight_utils import default_weight_loader
 from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu
+from sglang.srt.utils.hf_transformers_utils import get_rope_config

 _is_cpu_amx_available = cpu_has_amx_support()
 _is_cpu = is_cpu()
@@ -310,8 +311,7 @@ def __init__(
     ) -> None:
         super().__init__()
         self.hidden_size = config.hidden_size
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
```
- 已读文件:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +2/-1; `python/sglang/srt/models/deepseek.py` modified +2/-2; `python/sglang/srt/models/ernie4.py` modified +2/-2; `python/sglang/srt/models/exaone.py` modified +2/-2; `python/sglang/srt/models/glm4.py` modified +5/-3; `python/sglang/srt/models/glm4_moe.py` modified +5/-5; `python/sglang/srt/models/grok.py` modified +2/-5; `python/sglang/srt/models/hunyuan.py` modified +2/-2
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #20751 - [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture

- 链接: https://github.com/sgl-project/sglang/pull/20751
- 状态/时间: merged / 2026-04-01
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 43 个文件，+673/-106，可读 patch 1465 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture」；模型线: LLaDA 2.1；类别: 模型支持/运行时入口；主要 diff: `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml`。
- 实现要点:
  - `.github/workflows/full-test-npu.yml` added +355/-0
  - `.github/workflows/nightly-test-npu.yml` modified +124/-36
  - `.github/workflows/pr-test-npu.yml` modified +70/-40
  - `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1
- 代码 diff 细节:
  - `.github/workflows/full-test-npu.yml` added +355/-0
  - `.github/workflows/nightly-test-npu.yml` modified +124/-36
  - `.github/workflows/pr-test-npu.yml` modified +70/-40
  - `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/full-test-npu.yml
@@ -0,0 +1,355 @@
+name: Full Test (NPU)
+
+on:
+#  pull_request:
+#    branches:
+#      - main
+#    paths:
+#      - ".github/workflows/full-test-npu.yml"
+  workflow_dispatch:
+    inputs:
+      ref:
+        description: 'Git ref (branch, tag, or SHA) to test. If not provided, uses the default branch.'
+        required: false
diff -- .github/workflows/nightly-test-npu.yml
@@ -2,7 +2,7 @@ name: Nightly Test (NPU)

 on:
   schedule:
-    - cron: '0 17 * * *'  # Execute at 1:00 a.m. Beijing Time every day
+    - cron: '0 18 * * *'  # Execute at 2:00 a.m. Beijing Time every day
   pull_request:
     branches:
       - main
@@ -21,40 +21,95 @@ on:
         required: false
         type: string
         default: 'all'
+      image_a3:
```
- 已读文件:
  - runtime: `python/pyproject_npu.toml` modified +2/-0; `python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9
  - tests: `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mha.py` renamed +4/-0; `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mla.py` renamed +4/-0; `test/registered/ascend/basic_function/backends/test_npu_sampling_backend.py` renamed +4/-0; `test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_compile_graph_tp1_bf16.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_graph_tp1_bf16.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_graph_tp2_bf16.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py` renamed +4/-0
  - other: `.github/workflows/full-test-npu.yml` added +355/-0; `.github/workflows/nightly-test-npu.yml` modified +124/-36; `.github/workflows/pr-test-npu.yml` modified +70/-40; `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21667 - Unify GSM8K eval path to Chat API for regression CI readiness

- 链接: https://github.com/sgl-project/sglang/pull/21667
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 79 个文件，+1349/-1359，可读 patch 5014 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Unify GSM8K eval path to Chat API for regression CI readiness」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/few_shot_gsm8k.py`, `python/sglang/test/few_shot_gsm8k_engine.py`。
- 实现要点:
  - `python/sglang/test/accuracy_test_runner.py` modified +1/-63
  - `python/sglang/test/few_shot_gsm8k.py` modified +12/-0
  - `python/sglang/test/few_shot_gsm8k_engine.py` modified +14/-0
  - `python/sglang/test/kits/eval_accuracy_kit.py` modified +18/-12
- 代码 diff 细节:
  - `python/sglang/test/accuracy_test_runner.py` modified +1/-63
  - `python/sglang/test/few_shot_gsm8k.py` modified +12/-0
  - `python/sglang/test/few_shot_gsm8k_engine.py` modified +14/-0
  - `python/sglang/test/kits/eval_accuracy_kit.py` modified +18/-12
- 关键代码摘录:

```diff
diff -- python/sglang/test/accuracy_test_runner.py
@@ -432,56 +432,6 @@ def _run_nemo_skills_eval(
             kill_process_tree(process.pid)


-def _run_few_shot_eval(
-    model: ModelLaunchSettings,
-    base_url: str,
-    num_questions: Optional[int] = None,
-    num_shots: int = 8,
-    max_tokens: int = 512,
-) -> Tuple[bool, Optional[str], Optional[dict]]:
-    """Run evaluation using few_shot backend (few_shot_gsm8k.py).
-
-    Returns:
diff -- python/sglang/test/few_shot_gsm8k.py
@@ -1,6 +1,11 @@
 """
 Run few-shot GSM-8K evaluation.

+.. deprecated::
+    This module is deprecated. Use ``sglang.test.run_eval`` with
+    ``eval_name="gsm8k"`` instead, which routes through the unified
+    Chat API evaluation framework with dump_metric support.
+
 Usage:
 python3 -m sglang.test.few_shot_gsm8k --num-questions 200
 """
@@ -9,6 +14,7 @@
 import ast
```
- 已读文件:
  - runtime: `python/sglang/test/accuracy_test_runner.py` modified +1/-63; `python/sglang/test/few_shot_gsm8k.py` modified +12/-0; `python/sglang/test/few_shot_gsm8k_engine.py` modified +14/-0; `python/sglang/test/kits/eval_accuracy_kit.py` modified +18/-12; `python/sglang/test/run_eval.py` modified +8/-3; `python/sglang/test/server_fixtures/disaggregation_fixture.py` modified +1/-0; `python/sglang/test/simple_eval_common.py` modified +4/-1
  - tests: `test/manual/ep/test_moe_deepep_eval_accuracy_large.py` modified +9/-9; `test/manual/ep/test_mooncake_expert_backup.py` modified +11/-10; `test/manual/ep/test_nixl_ep.py` modified +11/-11; `test/manual/hicache/test_pp_with_hicache.py` modified +13/-13; `test/manual/models/test_falcon_h1_models.py` modified +33/-33; `test/manual/models/test_grok_models.py` modified +8/-8; `test/manual/models/test_kimi_k2_models.py` modified +11/-11; `test/manual/models/test_llama4_models.py` modified +8/-9
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #22305 - [CI] Update est_time for 64 tests based on actual elapsed times

- 链接: https://github.com/sgl-project/sglang/pull/22305
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 61 个文件，+61/-61，可读 patch 546 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Update est_time for 64 tests based on actual elapsed times」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`。
- 实现要点:
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen3_next_models.py` modified +1/-1
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen3_next_models.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -11,7 +11,7 @@
     popen_launch_server,
 )

-register_cuda_ci(est_time=600, suite="stage-c-test-4-gpu-b200")
+register_cuda_ci(est_time=290, suite="stage-c-test-4-gpu-b200")

 NEMOTRON_3_SUPER_NVFP4_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

diff -- test/registered/4-gpu-models/test_qwen35_hicache.py
@@ -18,7 +18,7 @@
     popen_launch_server,
 )

-register_cuda_ci(est_time=600, suite="stage-c-test-4-gpu-h100")
+register_cuda_ci(est_time=620, suite="stage-c-test-4-gpu-h100")

 QWEN35_27B_MODEL = "Qwen/Qwen3.5-27B"
 ACC_THRESHOLDS = {QWEN35_27B_MODEL: {"gsm8k": 0.8}}
```
- 已读文件:
  - tests: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_next_models.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_dsa_models_basic.py` modified +1/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #22565 - chore: update CI test est_time values

- 链接: https://github.com/sgl-project/sglang/pull/22565
- 状态/时间: merged / 2026-04-11
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+101/-101，可读 patch 896 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「chore: update CI test est_time values」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_hicache.py`。
- 实现要点:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_models.py` modified +1/-1
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_models.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -3,8 +3,8 @@
 from sglang.test.ci.ci_register import register_cuda_ci
 from sglang.test.gpt_oss_common import BaseTestGptOss

-register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-h100")
-register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-b200")
+register_cuda_ci(est_time=328, suite="stage-c-test-4-gpu-h100")
+register_cuda_ci(est_time=312, suite="stage-c-test-4-gpu-b200")


 class TestGptOss4Gpu(BaseTestGptOss):
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -11,7 +11,7 @@
     popen_launch_server,
 )

-register_cuda_ci(est_time=290, suite="stage-c-test-4-gpu-b200")
+register_cuda_ci(est_time=294, suite="stage-c-test-4-gpu-b200")

 NEMOTRON_3_SUPER_NVFP4_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

```
- 已读文件:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_models.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_next_models.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- 链接: https://github.com/sgl-project/sglang/pull/23001
- 状态/时间: merged / 2026-04-20
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+19458/-0，可读 patch 19508 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add new Mintlify documentation site (docs_new/)」；模型线: LLaDA 2.1；类别: 模型支持/运行时入口；主要 diff: `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`。
- 实现要点:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- 代码 diff 细节:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- 关键代码摘录:

```diff
diff -- .gitignore
@@ -192,6 +192,7 @@ work_dirs/
 *.csv

 !logo.png
+!docs_new/images/*.png

 # Prerequisites
 *.d
diff -- docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml
@@ -0,0 +1,39 @@
+name: Sync LMSYS SGLang blogs
+
+on:
+  workflow_dispatch:
+  schedule:
+    - cron: "0 */12 * * *"
+
+permissions:
+  contents: write
+
+jobs:
+  sync:
+    runs-on: ubuntu-latest
```
- 已读文件:
  - docs/bench: `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0; `docs_new/.gitignore` added +30/-0; `docs_new/.mintignore` added +7/-0; `docs_new/AGENTS.md` added +381/-0; `docs_new/CONTRIBUTING.md` added +34/-0; `docs_new/LICENSE` added +201/-0; `docs_new/README.md` added +126/-0; `docs_new/cards/Autoregressive-benchmark-card.png` added +0/-0
  - other: `.gitignore` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- 链接: https://github.com/sgl-project/sglang/pull/23337
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+3881/-1454，可读 patch 7195 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Sync docs_new with legacy docs and update migration redirects」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`。
- 实现要点:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- 代码 diff 细节:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- 关键代码摘录:

```diff
diff -- .pre-commit-config.yaml
@@ -93,6 +93,13 @@ repos:
         language: system
         files: ^test/registered/.*\.py$
         pass_filenames: false
+      - id: check-no-docs-changes
+        name: reject changes under legacy docs/
+        entry: python3 scripts/ci/check_no_docs_changes.py
+        language: system
+        pass_filenames: false
+        always_run: true
+        stages: [pre-commit]
   - repo: https://github.com/lycheeverse/lychee.git
     rev: lychee-v0.22.0
     hooks:
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx
@@ -26,7 +26,7 @@ To use DeepSeek-Math-V2, you must agree to DeepSeek's Community License. See [LI

 ## 2. SGLang Installation

-Please refer to the [official SGLang installation guide](../../../docs/get-started/installation) for installation instructions.
+Please refer to the [official SGLang installation guide](../../../docs/get-started/install) for installation instructions.

 ## 3. Model Deployment

```
- 已读文件:
  - docs/bench: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/Ernie/Ernie4.5.mdx` modified +1/-1
  - other: `.pre-commit-config.yaml` modified +7/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23732 - Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)

- 链接: https://github.com/sgl-project/sglang/pull/23732
- 状态/时间: merged / 2026-04-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+59/-12，可读 patch 290 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)」；模型线: LLaDA 2.1；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py`。
- 实现要点:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-0
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-0
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -58,6 +58,7 @@
 from sglang.srt.layers.moe import (
     get_deepep_mode,
     get_moe_a2a_backend,
+    should_use_dp_reduce_scatterv,
     should_use_flashinfer_cutlass_moe_fp4_allgather,
 )
 from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
@@ -386,6 +387,7 @@ def forward_normal(
             and not should_allreduce_fusion
             and not use_reduce_scatter
             and not should_use_flashinfer_cutlass_moe_fp4_allgather()
+            and not should_use_dp_reduce_scatterv()
         ):
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -34,6 +34,7 @@
     RowParallelLinear,
 )
 from sglang.srt.layers.logits_processor import LogitsProcessor
+from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
 from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
 from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
 from sglang.srt.layers.moe.topk import TopK
@@ -347,7 +348,12 @@ def forward(
         if self.num_shared_experts > 0:
             final_hidden_states = final_hidden_states + shared_output

-        if self.tp_size > 1 and not use_reduce_scatter and not should_allreduce_fusion:
+        if (
```
- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-0; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1; `python/sglang/srt/models/deepseek_v2.py` modified +3/-0; `python/sglang/srt/models/exaone_moe.py` modified +6/-2; `python/sglang/srt/models/glm4_moe.py` modified +3/-0; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4; `python/sglang/srt/models/llada2.py` modified +10/-2; `python/sglang/srt/models/llama4.py` modified +6/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23748 - refactor(moe): centralize post-experts all-reduce skip predicate

- 链接: https://github.com/sgl-project/sglang/pull/23748
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+134/-132，可读 patch 532 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor(moe): centralize post-experts all-reduce skip predicate」；模型线: LLaDA 2.1；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`。
- 实现要点:
  - `python/sglang/srt/layers/moe/__init__.py` modified +2/-0
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0；symbols: should_skip_post_experts_all_reduce
  - `python/sglang/srt/models/bailing_moe.py` modified +5/-8
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +5/-6
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/__init__.py` modified +2/-0
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0
  - `python/sglang/srt/models/bailing_moe.py` modified +5/-8
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +5/-6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/__init__.py
@@ -10,6 +10,7 @@
     get_tbo_token_distribution_threshold,
     initialize_moe_config,
     is_tbo_enabled,
+    should_skip_post_experts_all_reduce,
     should_use_dp_reduce_scatterv,
     should_use_flashinfer_cutlass_moe_fp4_allgather,
 )
@@ -24,6 +25,7 @@
     "get_moe_a2a_backend",
     "get_moe_runner_backend",
     "get_deepep_mode",
+    "should_skip_post_experts_all_reduce",
     "should_use_dp_reduce_scatterv",
diff -- python/sglang/srt/layers/moe/utils.py
@@ -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():
     )


+def should_skip_post_experts_all_reduce(
+    *,
+    is_tp_path: bool,
+    use_reduce_scatter: bool = False,
+    should_allreduce_fusion: bool = False,
+) -> bool:
+    """Whether to skip the post-experts all-reduce (EP or TP) because a
+    downstream component will fuse, replace, or absorb it.
+
+    Skip reasons, in order:
```
- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/__init__.py` modified +2/-0; `python/sglang/srt/layers/moe/utils.py` modified +33/-0; `python/sglang/srt/models/bailing_moe.py` modified +5/-8; `python/sglang/srt/models/bailing_moe_linear.py` modified +5/-6; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13; `python/sglang/srt/models/exaone_moe.py` modified +7/-5; `python/sglang/srt/models/glm4_moe.py` modified +9/-13; `python/sglang/srt/models/hunyuan_v3.py` modified +13/-7
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23785 - chore: update CI test est_time values

- 链接: https://github.com/sgl-project/sglang/pull/23785
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+101/-101，可读 patch 898 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「chore: update CI test est_time values」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`。
- 实现要点:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` modified +1/-1
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -3,8 +3,8 @@
 from sglang.test.ci.ci_register import register_cuda_ci
 from sglang.test.gpt_oss_common import BaseTestGptOss

-register_cuda_ci(est_time=328, suite="stage-c-test-4-gpu-h100")
-register_cuda_ci(est_time=740, suite="stage-c-test-4-gpu-b200-small")
+register_cuda_ci(est_time=392, suite="stage-c-test-4-gpu-h100")
+register_cuda_ci(est_time=584, suite="stage-c-test-4-gpu-b200-small")


 class TestGptOss4Gpu(BaseTestGptOss):
diff -- test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py
@@ -15,7 +15,7 @@
     popen_launch_server,
 )

-register_cuda_ci(est_time=540, suite="stage-c-test-4-gpu-b200-small")
+register_cuda_ci(est_time=422, suite="stage-c-test-4-gpu-b200-small")

 QWEN35_FP4_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"
 ACC_THRESHOLDS = {QWEN35_FP4_MODEL: {"gsm8k": 0.95}}
```
- 已读文件:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2; `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_next_models.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v3_basic.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #23835 - [NPU] Add GitHub test summary and deduplicate test code. Part 1

- 链接: https://github.com/sgl-project/sglang/pull/23835
- 状态/时间: merged / 2026-05-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+333/-332，可读 patch 851 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Add GitHub test summary and deduplicate test code. Part 1」；模型线: LLaDA 2.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/test/ascend/gsm8k_ascend_mixin.py`, `python/sglang/test/ascend/test_ascend_utils.py`, `python/sglang/test/ascend/test_mmlu.py`。
- 实现要点:
  - `python/sglang/test/ascend/gsm8k_ascend_mixin.py` modified +70/-35
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +48/-0；symbols: write_results_to_github_step_summary, write_github_step_summary_once
  - `python/sglang/test/ascend/test_mmlu.py` added +37/-0；symbols: TestMMLU, test_mmlu
  - `python/sglang/test/ascend/vlm_utils.py` modified +18/-2
- 代码 diff 细节:
  - `python/sglang/test/ascend/gsm8k_ascend_mixin.py` modified +70/-35
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +48/-0
  - `python/sglang/test/ascend/test_mmlu.py` added +37/-0
  - `python/sglang/test/ascend/vlm_utils.py` modified +18/-2
- 关键代码摘录:

```diff
diff -- python/sglang/test/ascend/gsm8k_ascend_mixin.py
@@ -1,19 +1,22 @@
 import os
+import subprocess
 from abc import ABC
 from types import SimpleNamespace

 from sglang.srt.utils import kill_process_tree
+from sglang.test.ascend.test_ascend_utils import write_results_to_github_step_summary
 from sglang.test.few_shot_gsm8k import run_eval
 from sglang.test.test_utils import (
     DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
     DEFAULT_URL_FOR_TEST,
     popen_launch_server,
+    write_github_step_summary,
diff -- python/sglang/test/ascend/test_ascend_utils.py
@@ -24,7 +24,9 @@
     DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
     DEFAULT_URL_FOR_TEST,
     auto_config_device,
+    is_in_ci,
     popen_launch_server,
+    write_github_step_summary,
 )

 # Model weights storage directory
@@ -90,6 +92,9 @@
 LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH = os.path.join(
     MODEL_WEIGHTS_DIR, "meta-llama/Llama-4-Scout-17B-16E-Instruct"
 )
```
- 已读文件:
  - runtime: `python/sglang/test/ascend/gsm8k_ascend_mixin.py` modified +70/-35; `python/sglang/test/ascend/test_ascend_utils.py` modified +48/-0; `python/sglang/test/ascend/test_mmlu.py` added +37/-0; `python/sglang/test/ascend/vlm_utils.py` modified +18/-2
  - tests: `test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py` modified +23/-59; `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py` modified +54/-70; `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py` modified +43/-88; `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py` modified +40/-78
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #24250 - [SKILL] Upgrade sglang profile and auto_benchmark skills

- 链接: https://github.com/sgl-project/sglang/pull/24250
- 状态/时间: merged / 2026-05-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+9334/-3813，可读 patch 13573 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SKILL] Upgrade sglang profile and auto_benchmark skills」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `agent-skills/llm-serving-auto-benchmark/SKILL.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml`。
- 实现要点:
  - `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0
- 代码 diff 细节:
  - `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0
- 关键代码摘录:

```diff
diff -- agent-skills/llm-serving-auto-benchmark/SKILL.md
@@ -0,0 +1,527 @@
+---
+name: llm-serving-auto-benchmark
+description: Framework-independent LLM serving benchmark skill for comparing SGLang, vLLM, TensorRT-LLM, or another serving framework. Use when a user wants to find the best deployment command for one model across multiple serving frameworks under the same workload, GPU budget, and latency SLA.
+---
+
+# LLM Serving Auto Benchmark
+
+## Overview
+
+Use this skill to compare LLM serving frameworks such as SGLang, vLLM, and
+TensorRT-LLM for the same model and workload.
+
+Use a config-driven workflow:
diff -- skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md
@@ -0,0 +1,17 @@
+# Cookbook LLM Configs
+
+These configs define a framework-neutral LLM serving cookbook model set and translate each model into a three-framework run plan for SGLang, vLLM, and TensorRT-LLM.
+
+Scope:
+- SGLang can preserve source-recipe `base_flags` and `search_space` where applicable; if a sequence limit is smaller than the default synthetic scenario, the config raises that limit so the shipped workload can run.
+- vLLM uses framework-native `vllm serve` flags. The translation keeps the same model, tokenizer, dataset shape, GPU count, and high-impact batching/prefix-cache knobs; it does not copy SGLang-only parser or scheduler flags.
+- TensorRT-LLM uses `trtllm-serve serve` with `backend: pytorch` fixed in `base_server_flags`. Backend choice is never searched.
+- The two default random scenarios remain aligned pairs: `chat` uses `1000 -> 1000`, and `summarization` uses `8000 -> 1000`.
+
+Before a real run, capture the target framework `--help` output and validate the configs:
+
+```bash
```
- 已读文件:
  - docs/bench: `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.1.yaml` added +132/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.2.yaml` added +132/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.yaml` added +133/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/devstral-small-2-24b-instruct-2512.yaml` added +123/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/ernie-4.5-21b-a3b-pt.yaml` added +117/-0
  - other: `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #25197 - ci: decouple stage and runner for cuda registry

- 链接: https://github.com/sgl-project/sglang/pull/25197
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+207/-129，可读 patch 1149 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: decouple stage and runner for cuda registry」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `python/sglang/test/ci/ci_register.py`, `scripts/ci/utils/ci_coverage_report.py`, `scripts/ci/utils/compute_partitions.py`。
- 实现要点:
  - `python/sglang/test/ci/ci_register.py` modified +81/-24；symbols: effective_suite, _parse_call_args
  - `scripts/ci/utils/ci_coverage_report.py` modified +8/-6
  - `scripts/ci/utils/compute_partitions.py` modified +1/-1
  - `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py` modified +1/-1
- 代码 diff 细节:
  - `python/sglang/test/ci/ci_register.py` modified +81/-24
  - `scripts/ci/utils/ci_coverage_report.py` modified +8/-6
  - `scripts/ci/utils/compute_partitions.py` modified +1/-1
  - `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- python/sglang/test/ci/ci_register.py
@@ -16,7 +16,12 @@
     "ut_parse_one_file",
 ]

+# `suite` stays in positional slot 2 for backward compat with existing
+# `register_cpu_ci(5, "stage-a-test-cpu")` style positional calls. New fields
+# (`stage`, `runner_config`) are kwarg-only.
 _PARAM_ORDER = ("est_time", "suite", "nightly", "disabled")
+_KWARG_ONLY = ("stage", "runner_config")
+_ALL_PARAMS = _PARAM_ORDER + _KWARG_ONLY
 _UNSET = object()


@@ -31,45 +36,69 @@ class HWBackend(Enum):
diff -- scripts/ci/utils/ci_coverage_report.py
@@ -155,7 +155,9 @@ def generate_summary_section(data: dict) -> str:
         for t in sorted(disabled_tests, key=lambda x: (x.backend.name, x.filename)):
             test_name = get_test_basename(t.filename)
             reason = t.disabled[:50] + "..." if len(t.disabled) > 50 else t.disabled
-            lines.append(f"| `{test_name}` | {t.backend.name} | {t.suite} | {reason} |")
+            lines.append(
+                f"| `{test_name}` | {t.backend.name} | {t.effective_suite} | {reason} |"
+            )
         lines.append("\n</details>\n")

     return "\n".join(lines)
@@ -197,7 +199,7 @@ def generate_by_folder_section(data: dict) -> str:
                     else ("Nightly" if t.nightly else "Per-Commit")
                 )
```
- 已读文件:
  - runtime: `python/sglang/test/ci/ci_register.py` modified +81/-24
  - tests: `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py` modified +1/-1; `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_models.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_30b.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py` modified +1/-1
  - other: `scripts/ci/utils/ci_coverage_report.py` modified +8/-6; `scripts/ci/utils/compute_partitions.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #25420 - [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI

- 链接: https://github.com/sgl-project/sglang/pull/25420
- 状态/时间: merged / 2026-05-16
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+327/-329，可读 patch 2091 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI」；模型线: LLaDA 2.1；类别: 文档/测试/CI；主要 diff: `agent-skills/add-jit-kernel/SKILL.md`, `agent-skills/ci-workflow-guide/SKILL.md`, `agent-skills/write-sglang-test/SKILL.md`。
- 实现要点:
  - `agent-skills/add-jit-kernel/SKILL.md` modified +9/-9
  - `agent-skills/ci-workflow-guide/SKILL.md` modified +63/-64
  - `agent-skills/write-sglang-test/SKILL.md` modified +40/-41
  - `.github/actions/check-pr-test-health/action.yml` renamed +7/-7
- 代码 diff 细节:
  - `agent-skills/add-jit-kernel/SKILL.md` modified +9/-9
  - `agent-skills/ci-workflow-guide/SKILL.md` modified +63/-64
  - `agent-skills/write-sglang-test/SKILL.md` modified +40/-41
  - `.github/actions/check-pr-test-health/action.yml` renamed +7/-7
- 关键代码摘录:

```diff
diff -- agent-skills/add-jit-kernel/SKILL.md
@@ -435,17 +435,17 @@ if torch.cuda.get_device_capability()[0] < 9:

 JIT kernel tests live under `python/sglang/jit_kernel/tests/`. **CI does not run `pytest` in that directory directly.** The unified runner `test/run_suite.py` discovers every `test_*.py` there (and every `bench_*.py` under `benchmark/`), collects `register_*_ci(...)` calls by **statically parsing each file's AST**, and executes the selected suite. Every test file must register at least one CUDA entry or the collector fails its sanity check.

-- **PR / per-commit CUDA suites** (see `test/run_suite.py` → `PER_COMMIT_SUITES`): JIT unit tests use `stage-b-kernel-unit-1-gpu-large` on H100 and `stage-b-kernel-unit-1-gpu-b200` on B200/SM100 paths (see `.github/workflows/pr-test-jit-kernel.yml`). Multi-GPU JIT tests use `stage-b-kernel-unit-8-gpu-h200`.
+- **PR / per-commit CUDA suites** (see `test/run_suite.py` → `PER_COMMIT_SUITES`): JIT unit tests use `base-b-kernel-unit-1-gpu-large` on H100 and `base-b-kernel-unit-1-gpu-b200` on B200/SM100 paths (see `.github/workflows/pr-test-jit-kernel.yml`). Multi-GPU JIT tests use `base-b-kernel-unit-8-gpu-h200`.
 - **Nightly kernel suite**: `nightly-kernel-1-gpu` with `--nightly` — typically used with `SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1` in CI for expanded parameter grids (see `python/sglang/jit_kernel/utils.py` → `should_run_full_tests` / `get_ci_test_range`). Wired in `.github/workflows/nightly-test-nvidia.yml` (e.g. `python3 run_suite.py --hw cuda --suite nightly-kernel-1-gpu --nightly --continue-on-error`).

 Registration pattern (module level, **literal** `est_time` and `suite` strings — required for AST parsing):

 ```python
 from sglang.test.ci.ci_register import register_cuda_ci

-register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
diff -- agent-skills/ci-workflow-guide/SKILL.md
@@ -11,7 +11,7 @@ This skill covers the CI **infrastructure** layer — how tests are dispatched,

 ## Naming Conventions

-- **Suite**: `stage-{a,b,c}-test-{gpu_count}-gpu-{hardware}` (e.g., `stage-b-test-1-gpu-small`)
+- **Suite**: `base-{a,b,c}-test-{gpu_count}-gpu-{hardware}` (e.g., `base-b-test-1-gpu-small`)
 - **Test group**: Directory-level registered test group under `test/registered/` (e.g., `hicache` maps to `test/registered/hicache/test_*.py`)
 - **CI runner**: `{gpu_count}-gpu-{hardware}` (e.g., `1-gpu-5090`, `4-gpu-h100`, `8-gpu-h200`)

@@ -23,7 +23,7 @@ This skill covers the CI **infrastructure** layer — how tests are dispatched,
 |------|------|
 | `.github/workflows/pr-test.yml` | Main workflow — all stages, jobs, conditions, matrix definitions |
 | `.github/workflows/pr-gate.yml` | PR gating: draft check, `run-ci` label, per-user rate limiting |
-| `.github/actions/check-stage-health/action.yml` | Cross-job fast-fail: queries API for any failed job |
```
- 已读文件:
  - runtime: `python/sglang/jit_kernel/benchmark/bench_activation.py` modified +1/-1; `python/sglang/jit_kernel/benchmark/bench_awq_dequantize.py` modified +1/-1; `python/sglang/jit_kernel/benchmark/bench_cast.py` modified +1/-1; `python/sglang/jit_kernel/benchmark/bench_clamp_position.py` modified +1/-1; `python/sglang/jit_kernel/benchmark/bench_concat_mla.py` modified +1/-1; `python/sglang/jit_kernel/benchmark/bench_custom_all_reduce.py` modified +1/-1; `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +1/-1; `python/sglang/jit_kernel/benchmark/bench_hadamard.py` modified +1/-1
  - other: `agent-skills/add-jit-kernel/SKILL.md` modified +9/-9; `agent-skills/ci-workflow-guide/SKILL.md` modified +63/-64; `agent-skills/write-sglang-test/SKILL.md` modified +40/-41; `.github/actions/check-pr-test-health/action.yml` renamed +7/-7; `.github/actions/wait-for-jobs/action.yml` modified +2/-2; `.github/workflows/_pr-test-sgl-kernel-build.yml` modified +2/-2; `.github/workflows/_pr-test-stage.yml` modified +4/-4; `.github/workflows/ci-auto-bisect.yml` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。
