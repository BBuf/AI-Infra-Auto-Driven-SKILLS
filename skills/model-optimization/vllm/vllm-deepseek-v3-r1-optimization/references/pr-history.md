# vllm DeepSeek V3/R1 PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 4
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/online_serving/elastic_ep/serve_deepseek_v2.sh` | no direct PR-number commit |
| `examples/tool_chat_template_deepseekv3.jinja` | [#17784](https://github.com/vllm-project/vllm/pull/17784) |
| `examples/tool_chat_template_deepseekv31.jinja` | [#23454](https://github.com/vllm-project/vllm/pull/23454) |
| `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` | [#30356](https://github.com/vllm-project/vllm/pull/30356) |
| `tests/evals/gsm8k/configs/DeepSeek-R1-DP_MI325.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` | [#30356](https://github.com/vllm-project/vllm/pull/30356) |
| `tests/evals/gsm8k/configs/DeepSeek-R1-TP_MI325.yaml` | no direct PR-number commit |
| `tests/reasoning/test_deepseekv3_reasoning_parser.py` | [#24972](https://github.com/vllm-project/vllm/pull/24972), [#25589](https://github.com/vllm-project/vllm/pull/25589) |
| `tests/tool_parsers/test_deepseekv31_tool_parser.py` | no direct PR-number commit |
| `tests/tool_parsers/test_deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#36056](https://github.com/vllm-project/vllm/pull/36056), [#41198](https://github.com/vllm-project/vllm/pull/41198) |
| `tests/tool_parsers/test_deepseekv3_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/models/deepseek_mtp.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#29545](https://github.com/vllm-project/vllm/pull/29545), [#38684](https://github.com/vllm-project/vllm/pull/38684), [#38870](https://github.com/vllm-project/vllm/pull/38870) |
| `vllm/model_executor/models/deepseek_v2.py` | [#13833](https://github.com/vllm-project/vllm/pull/13833), [#23971](https://github.com/vllm-project/vllm/pull/23971), [#24119](https://github.com/vllm-project/vllm/pull/24119), [#25896](https://github.com/vllm-project/vllm/pull/25896), [#25999](https://github.com/vllm-project/vllm/pull/25999), [#26456](https://github.com/vllm-project/vllm/pull/26456), [#26465](https://github.com/vllm-project/vllm/pull/26465), [#26670](https://github.com/vllm-project/vllm/pull/26670), [#26763](https://github.com/vllm-project/vllm/pull/26763), [#27532](https://github.com/vllm-project/vllm/pull/27532), [#27568](https://github.com/vllm-project/vllm/pull/27568), [#28968](https://github.com/vllm-project/vllm/pull/28968), ... (25 total) |
| `vllm/reasoning/deepseek_r1_reasoning_parser.py` | no direct PR-number commit |
| `vllm/tool_parsers/deepseekv31_tool_parser.py` | no direct PR-number commit |
| `vllm/tool_parsers/deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#33964](https://github.com/vllm-project/vllm/pull/33964), [#36056](https://github.com/vllm-project/vllm/pull/36056), [#41198](https://github.com/vllm-project/vllm/pull/41198) |
| `vllm/tool_parsers/deepseekv3_tool_parser.py` | no direct PR-number commit |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-02-26 | [#13833](https://github.com/vllm-project/vllm/pull/13833) | merged | DeepSeek V2/V3/R1 only place `lm_head` on last pp rank | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-05-12 | [#17784](https://github.com/vllm-project/vllm/pull/17784) | merged | [Feature] Support DeepSeekV3 Function Call | `examples/tool_chat_template_deepseekv3.jinja` |
| 2025-07-22 | [#21116](https://github.com/vllm-project/vllm/pull/21116) | merged | [perf] Add fused MLA QKV + strided layernorm | `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/quantization/fp8.py` |
| 2025-08-07 | [#22352](https://github.com/vllm-project/vllm/pull/22352) | merged | [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM` | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-08-23 | [#23454](https://github.com/vllm-project/vllm/pull/23454) | merged | Support DeepSeek-V3.1 tool call | `examples/tool_chat_template_deepseekv31.jinja` |
| 2025-08-30 | [#23123](https://github.com/vllm-project/vllm/pull/23123) | merged | Add routed_scaling_factor to MoE grouped topk | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-08-30 | [#23971](https://github.com/vllm-project/vllm/pull/23971) | merged | Add LoRA support for DeepSeek models (V2, V3, R1-0528) | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-09-02 | [#24119](https://github.com/vllm-project/vllm/pull/24119) | merged | [Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-09-30 | [#25896](https://github.com/vllm-project/vllm/pull/25896) | merged | [New Model] DeepSeek-V3.2 (Rebased to Main) | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2025-10-02 | [#25999](https://github.com/vllm-project/vllm/pull/25999) | merged | [Deepseek v3.2] Support indexer prefill chunking | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-15 | [#25589](https://github.com/vllm-project/vllm/pull/25589) | merged | [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972) | `tests/reasoning/test_deepseekv3_reasoning_parser.py` |
| 2025-10-15 | [#26456](https://github.com/vllm-project/vllm/pull/26456) | merged | [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-21 | [#26763](https://github.com/vllm-project/vllm/pull/26763) | merged | [Deepseek v3.2] Optimize top_k_per_row | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-21 | [#26465](https://github.com/vllm-project/vllm/pull/26465) | merged | [Deepseek v3.2] Remove extra logics in indexer | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-19 | [#28968](https://github.com/vllm-project/vllm/pull/28968) | merged | [DeepSeek] Fix DeepSeek V3.2 Rope Embedding | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-20 | [#26670](https://github.com/vllm-project/vllm/pull/26670) | merged | [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-02 | [#29545](https://github.com/vllm-project/vllm/pull/29545) | merged | [Bugfix] Fix DeepSeek R1 MTP weight loading | `vllm/model_executor/models/deepseek_mtp.py` |
| 2025-12-08 | [#27568](https://github.com/vllm-project/vllm/pull/27568) | merged | [DeepSeek v3.2] Make top-k work for any logit values. | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-12 | [#27532](https://github.com/vllm-project/vllm/pull/27532) | merged | [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-17 | [#30841](https://github.com/vllm-project/vllm/pull/30841) | merged | [Bugfix] deepseek-V3.2 self.weights_proj has no bias | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-19 | [#31046](https://github.com/vllm-project/vllm/pull/31046) | merged | [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-24 | [#31160](https://github.com/vllm-project/vllm/pull/31160) | merged | [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-05 | [#30356](https://github.com/vllm-project/vllm/pull/30356) | merged | [CI][DeepSeek] Add nightly DeepSeek R1 `lm_eval` tests on H200 | `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` |
| 2026-01-16 | [#32175](https://github.com/vllm-project/vllm/pull/32175) | merged | [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-21 | [#29287](https://github.com/vllm-project/vllm/pull/29287) | merged | [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-02-06 | [#33964](https://github.com/vllm-project/vllm/pull/33964) | merged | [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32 | `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-02-07 | [#24972](https://github.com/vllm-project/vllm/pull/24972) | closed | [Model] Deepseek-V3.1 reasoning parser | `tests/reasoning/test_deepseekv3_reasoning_parser.py` |
| 2026-02-18 | [#34758](https://github.com/vllm-project/vllm/pull/34758) | merged | [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup) | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-02-19 | [#34876](https://github.com/vllm-project/vllm/pull/34876) | merged | [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-02-23 | [#34302](https://github.com/vllm-project/vllm/pull/34302) | merged | [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup) | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-03-02 | [#35751](https://github.com/vllm-project/vllm/pull/35751) | merged | [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-03-07 | [#36247](https://github.com/vllm-project/vllm/pull/36247) | merged | [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-03-19 | [#36056](https://github.com/vllm-project/vllm/pull/36056) | merged | [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1 | `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py` |
| 2026-03-30 | [#33703](https://github.com/vllm-project/vllm/pull/33703) | merged | [Bugfix] Support multi-type params parsing for DeepSeek v3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-02 | [#38684](https://github.com/vllm-project/vllm/pull/38684) | merged | [Perf] DSV3.2 Indexer Fused Weights Projection | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-04 | [#38870](https://github.com/vllm-project/vllm/pull/38870) | merged | [Bugfix] Fix DSV32 weight loading | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-08 | [#37421](https://github.com/vllm-project/vllm/pull/37421) | merged | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-04-27 | [#35968](https://github.com/vllm-project/vllm/pull/35968) | closed | [Performance] DeepSeek V3.2 multi-stream indexer overlap | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py` |
| 2026-04-29 | [#41198](https://github.com/vllm-project/vllm/pull/41198) | merged | [Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |

## Per-PR Diff Audit Cards

### PR #13833 - DeepSeek V2/V3/R1 only place `lm_head` on last pp rank

- Link: https://github.com/vllm-project/vllm/pull/13833
- Status/date: merged / 2025-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `24679788ed38`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek V2/V3/R1 only place `lm_head` on last pp rank"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "DeepSeek V2/V3/R1 only place `lm_head` on last pp rank"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -636,9 +636,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -636,9 +636,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -636,9 +636,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.lm_head = ParallelLMHead(config.vocab_size,
-                                      config.hidden_size,
-                                      quant_config=quant_config)
+        if get_pp_group().is_last_rank:
+            self.lm_head = ParallelLMHead(config.vocab_size,
+                                          config.hidden_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17784 - [Feature] Support DeepSeekV3 Function Call

- Link: https://github.com/vllm-project/vllm/pull/17784
- Status/date: merged / 2025-05-12
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_deepseekv3.jinja`; associated commits `3a5ea7512926`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +473/-1, 495 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support DeepSeekV3 Function Call"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `examples/tool_chat_template_deepseekv3.jinja`; technical summary: Covers "[Feature] Support DeepSeekV3 Function Call"; the main implementation surface is `examples/tool_chat_template_deepseekv3.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_deepseekv3.jinja` added +96/-0 (96 lines); hunks: -0,0 +1,96.
- Code diff details:
  - `examples/tool_chat_template_deepseekv3.jinja` added +96/-0 (96 lines); hunks: -0,0 +1,96
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_deepseekv3.jinja
@@ -0,0 +1,96 @@
+{% if not add_generation_prompt is defined %}
+    {% set add_generation_prompt = false %}
+{% endif %}
+{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_last_user=false) %}
+{%- for message in messages %}
+    {%- if message['role'] == 'system' %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_deepseekv3.jinja` added +96/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv3_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21116 - [perf] Add fused MLA QKV + strided layernorm

- Link: https://github.com/vllm-project/vllm/pull/21116
- Status/date: merged / 2025-07-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +214/-66, 648 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] Add fused MLA QKV + strided layernorm"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/quantization/fp8.py`; technical summary: Covers "[perf] Add fused MLA QKV + strided layernorm"; the main implementation surface is `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/quantization/fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/linear.py` modified +77/-1 (78 lines); hunks: -259,6 +259,8 @@ def __init__(; -300,6 +302,12 @@ def __init__(; symbols: __init__, extra_repr, MergedReplicatedLinear, touching `__init__, extra_repr, MergedReplicatedLinear`; `vllm/model_executor/models/deepseek_v2.py` modified +39/-18 (57 lines); hunks: -42,6 +42,7; -336,7 +337,7 @@ def forward(; symbols: forward, __init__, load_weights, touching `forward, __init__, load_weights`; `vllm/model_executor/layers/quantization/fp8.py` modified +10/-3 (13 lines); hunks: -257,9 +257,16 @@ def create_weights(; symbols: create_weights, touching `create_weights`; `csrc/layernorm_kernels.cu` modified +40/-23 (63 lines); hunks: -15,15 +15,16 @@ namespace vllm {; -37,7 +38,7 @@ __global__ void rms_norm_kernel(.
- Code diff details:
  - `vllm/model_executor/layers/linear.py` modified +77/-1 (78 lines); hunks: -259,6 +259,8 @@ def __init__(; -300,6 +302,12 @@ def __init__(; symbols: __init__, extra_repr, MergedReplicatedLinear
  - `vllm/model_executor/models/deepseek_v2.py` modified +39/-18 (57 lines); hunks: -42,6 +42,7; -336,7 +337,7 @@ def forward(; symbols: forward, __init__, load_weights
  - `vllm/model_executor/layers/quantization/fp8.py` modified +10/-3 (13 lines); hunks: -257,9 +257,16 @@ def create_weights(; symbols: create_weights
  - `csrc/layernorm_kernels.cu` modified +40/-23 (63 lines); hunks: -15,15 +15,16 @@ namespace vllm {; -37,7 +38,7 @@ __global__ void rms_norm_kernel(
  - `csrc/layernorm_quant_kernels.cu` modified +25/-14 (39 lines); hunks: -23,16 +23,17 @@ namespace vllm {; -49,7 +50,7 @@ __global__ void rms_norm_static_fp8_quant_kernel(
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/linear.py
@@ -259,6 +259,8 @@ def __init__(
+        self.quant_config = quant_config
+        self.prefix = prefix
@@ -300,6 +302,12 @@ def __init__(
+        # If MergedReplicatedLinear, use output size of each partition.
+        if hasattr(self, "output_sizes"):
+            self.output_partition_sizes = self.output_sizes
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -42,6 +42,7 @@
+                                               MergedReplicatedLinear,
@@ -336,7 +337,7 @@ def forward(
-        kv_a = self.kv_a_layernorm(kv_a.contiguous())
+        kv_a = self.kv_a_layernorm(kv_a)
@@ -407,14 +408,24 @@ def __init__(
-            self.q_a_proj = ReplicatedLinear(self.hidden_size,
diff -- vllm/model_executor/layers/quantization/fp8.py
@@ -257,9 +257,16 @@ def create_weights(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/linear.py` modified +77/-1; `vllm/model_executor/models/deepseek_v2.py` modified +39/-18; `vllm/model_executor/layers/quantization/fp8.py` modified +10/-3
  - other: `csrc/layernorm_kernels.cu` modified +40/-23; `csrc/layernorm_quant_kernels.cu` modified +25/-14; `csrc/quantization/fp8/common.cu` modified +4/-0
  - tests: `tests/kernels/core/test_layernorm.py` modified +19/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_layernorm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22352 - [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`

- Link: https://github.com/vllm-project/vllm/pull/22352
- Status/date: merged / 2025-08-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-0, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +16/-0 (16 lines); hunks: -726,13 +726,29 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM, __init__, touching `forward, DeepseekV2ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +16/-0 (16 lines); hunks: -726,13 +726,29 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -726,13 +726,29 @@ def forward(
+    packed_modules_mapping = {
+        "gate_up_proj": ["gate_proj", "up_proj"],
+    }
+        # `packed_modules_mapping` needs to be modified before
+        # initializing DeepseekV2Model, as it is passed inplace to
+        # quantization config init and may be used to select the
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +16/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23454 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/vllm-project/vllm/pull/23454
- Status/date: merged / 2025-08-23
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_deepseekv31.jinja`; associated commits `b8f17f5d980e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +468/-0, 491 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek-V3.1 tool call"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `examples/tool_chat_template_deepseekv31.jinja`; technical summary: Covers "Support DeepSeek-V3.1 tool call"; the main implementation surface is `examples/tool_chat_template_deepseekv31.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_deepseekv31.jinja
@@ -0,0 +1,91 @@
+{% if not add_generation_prompt is defined %}
+  {% set add_generation_prompt = false %}
+{% endif %}
+{% if not thinking is defined %}
+  {% set thinking = false %}
+{% endif %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23123 - Add routed_scaling_factor to MoE grouped topk

- Link: https://github.com/vllm-project/vllm/pull/23123
- Status/date: merged / 2025-08-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +77/-4, 570 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add routed_scaling_factor to MoE grouped topk"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; technical summary: Covers "Add routed_scaling_factor to MoE grouped topk"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-0 (18 lines); hunks: -244,6 +244,7 @@ def apply(; -400,6 +401,7 @@ def apply(; symbols: apply, forward_cuda, touching `apply, forward_cuda`; `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +12/-0 (12 lines); hunks: -21,6 +21,7 @@ def grouped_topk(; -65,6 +66,8 @@ def grouped_topk(; symbols: grouped_topk, select_experts, __call__, touching `grouped_topk, select_experts, __call__`; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +10/-0 (10 lines); hunks: -350,6 +350,7 @@ def apply(; -375,6 +376,7 @@ def apply(; symbols: apply, touching `apply`; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +4/-3 (7 lines); hunks: -1011,7 +1011,8 @@ def grouped_topk(; -1790,8 +1791,8 @@ def fused_moe(; symbols: grouped_topk, fused_moe, touching `grouped_topk, fused_moe`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-0 (18 lines); hunks: -244,6 +244,7 @@ def apply(; -400,6 +401,7 @@ def apply(; symbols: apply, forward_cuda
  - `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +12/-0 (12 lines); hunks: -21,6 +21,7 @@ def grouped_topk(; -65,6 +66,8 @@ def grouped_topk(; symbols: grouped_topk, select_experts, __call__
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +10/-0 (10 lines); hunks: -350,6 +350,7 @@ def apply(; -375,6 +376,7 @@ def apply(; symbols: apply
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +4/-3 (7 lines); hunks: -1011,7 +1011,8 @@ def grouped_topk(; -1790,8 +1791,8 @@ def fused_moe(; symbols: grouped_topk, fused_moe
  - `vllm/model_executor/layers/quantization/fp8.py` modified +3/-1 (4 lines); hunks: -955,6 +955,7 @@ def apply(; -994,7 +995,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -244,6 +244,7 @@ def apply(
+        routed_scaling_factor: float = 1.0,
@@ -400,6 +401,7 @@ def apply(
+        routed_scaling_factor: float = 1.0,
@@ -427,6 +429,7 @@ def apply(
+            routed_scaling_factor=routed_scaling_factor,
@@ -450,6 +453,7 @@ def forward_cuda(
diff -- vllm/model_executor/layers/fused_moe/cpu_fused_moe.py
@@ -21,6 +21,7 @@ def grouped_topk(
+    routed_scaling_factor: float = 1.0,
@@ -65,6 +66,8 @@ def grouped_topk(
+    if routed_scaling_factor != 1.0:
+        topk_weights = topk_weights * routed_scaling_factor
@@ -78,6 +81,7 @@ def select_experts(
+    routed_scaling_factor: float = 1.0,
diff -- vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -350,6 +350,7 @@ def apply(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-0; `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` modified +12/-0; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +10/-0; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +4/-3; `vllm/model_executor/layers/quantization/fp8.py` modified +3/-1; `vllm/model_executor/layers/quantization/modelopt.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23971 - Add LoRA support for DeepSeek models (V2, V3, R1-0528)

- Link: https://github.com/vllm-project/vllm/pull/23971
- Status/date: merged / 2025-08-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `379ea2823a75`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +12/-7, 54 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add LoRA support for DeepSeek models (V2, V3, R1-0528)"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "Add LoRA support for DeepSeek models (V2, V3, R1-0528)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -56,7 +56,7; -727,7 +727,8 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM, touching `forward, DeepseekV2ForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -56,7 +56,7; -727,7 +727,8 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -56,7 +56,7 @@
-from .interfaces import MixtureOfExperts, SupportsPP
+from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
@@ -727,7 +727,8 @@ def forward(
-class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
+class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts,
+                            SupportsLoRA):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24119 - [Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue

- Link: https://github.com/vllm-project/vllm/pull/24119
- Status/date: merged / 2025-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `930a24144c07`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -160,7 +160,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -160,7 +160,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -160,7 +160,8 @@ def __init__(
-            routed_scaling_factor=self.routed_scaling_factor,
+            # we do scaling outside, set factor to 1.0 to avoid double mul
+            routed_scaling_factor=1.0,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25896 - [New Model] DeepSeek-V3.2 (Rebased to Main)

- Link: https://github.com/vllm-project/vllm/pull/25896
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `fa7e254a7f3e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 71 files, +3918/-221, 5400 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model] DeepSeek-V3.2 (Rebased to Main)"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[New Model] DeepSeek-V3.2 (Rebased to Main)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__, touching `DeepseekV2MLP, DeepseekV2Attention, __init__`; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -33,36 +33,57 @@
+from vllm.attention.backends.abstract import AttentionBackend
+from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
-from vllm.config import CacheConfig, ParallelConfig, VllmConfig
+from vllm.config import (CacheConfig, ParallelConfig, VllmConfig,
+                         get_current_vllm_config)
+from vllm.forward_context import get_forward_context
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
+        self.is_v32 = hasattr(config, "index_topk")
+        if self.is_v32:
+            topk_tokens = config.index_topk
+            topk_indices_buffer = torch.empty(
+                vllm_config.scheduler_config.max_num_batched_tokens,
+                topk_tokens,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1
- Risk and verification: The diff ships test coverage in `tests/compile/test_fusion_attn.py`, `tests/kernels/attention/test_cache.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/attention/test_flashmla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25999 - [Deepseek v3.2] Support indexer prefill chunking

- Link: https://github.com/vllm-project/vllm/pull/25999
- Status/date: merged / 2025-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `1e50f1be7058`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +149/-79, 324 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek v3.2] Support indexer prefill chunking"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek v3.2] Support indexer prefill chunking"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -583,44 +583,43 @@ def sparse_attn_indexer(
-        num_prefills = attn_metadata.num_prefills
-        k_fp8 = torch.empty([prefill_metadata.total_seq_lens, head_dim],
-                            device=k.device,
-                            dtype=torch.float8_e4m3fn)
-        k_scale = torch.empty([prefill_metadata.total_seq_lens, 1],
-                              device=k.device,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_sparse_mla_backends.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- Link: https://github.com/vllm-project/vllm/pull/25589
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_deepseekv3_reasoning_parser.py`; associated commits `85a65e7f51ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +215/-3, 269 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `tests/reasoning/test_deepseekv3_reasoning_parser.py`; technical summary: Covers "[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)"; the main implementation surface is `tests/reasoning/test_deepseekv3_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic, touching `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`.
- Code diff details:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
- Key code excerpts:

```diff
diff -- tests/reasoning/test_deepseekv3_reasoning_parser.py
@@ -0,0 +1,76 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.reasoning import (
```

- Reviewed files:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_deepseekv3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26456 - [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather

- Link: https://github.com/vllm-project/vllm/pull/26456
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `f5ed68ef63d0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-68, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer, touching `get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -75,7 +75,7 @@
-from vllm.utils import cdiv, direct_register_custom_op
+from vllm.utils import direct_register_custom_op
@@ -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:
-@torch.inference_mode()
-def cp_gather_indexer_k_quant_cache(
-    kv_cache,  # [num_blocks, block_size, head_dim + 1]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26763 - [Deepseek v3.2] Optimize top_k_per_row

- Link: https://github.com/vllm-project/vllm/pull/26763
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `80e94529845d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +13/-49, 203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek v3.2] Optimize top_k_per_row"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek v3.2] Optimize top_k_per_row"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -577,15 +577,11 @@ def sparse_attn_indexer(
-            topk_values = torch.empty(
-                num_rows, topk_tokens, dtype=logits.dtype, device=logits.device
-            )
-                topk_values,
@@ -642,15 +638,11 @@ def sparse_attn_indexer(
-        topk_values = torch.empty(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26465 - [Deepseek v3.2] Remove extra logics in indexer

- Link: https://github.com/vllm-project/vllm/pull/26465
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `09a7e6f6179b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +141/-40, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek v3.2] Remove extra logics in indexer"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek v3.2] Remove extra logics in indexer"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -574,9 +574,9 @@ def sparse_attn_indexer(
-            topk_indices = torch.empty(
-                num_rows, topk_tokens, dtype=torch.int32, device=logits.device
-            )
+            topk_indices = topk_indices_buffer[
+                chunk.token_start : chunk.token_end, :topk_tokens
+            ]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28968 - [DeepSeek] Fix DeepSeek V3.2 Rope Embedding

- Link: https://github.com/vllm-project/vllm/pull/28968
- Status/date: merged / 2025-11-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `88f5b19f0bc6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-3, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek] Fix DeepSeek V3.2 Rope Embedding"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[DeepSeek] Fix DeepSeek V3.2 Rope Embedding"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -846,8 +846,8 @@ def forward(
-        q = torch.cat([q_pe, q_nope], dim=-1)
-        k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)
+        q = torch.cat([q_pe.squeeze(0), q_nope], dim=-1)
+        k = torch.cat([k_pe.squeeze((0, 2)), k_nope], dim=-1)
@@ -1000,6 +1000,14 @@ def __init__(
+            self.indexer_rope_emb = get_rope(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26670 - [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA

- Link: https://github.com/vllm-project/vllm/pull/26670
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `06c20c990464`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +583/-15, 700 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, touching `sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -591,6 +591,7 @@ def sparse_attn_indexer(
+    fp8_dtype = current_platform.fp8_dtype()
@@ -630,7 +631,7 @@ def sparse_attn_indexer(
-                dtype=torch.float8_e4m3fn,
+                dtype=fp8_dtype,
@@ -644,7 +645,12 @@ def sparse_attn_indexer(
-            logits = fp8_mqa_logits(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4
- Risk and verification: Runtime changes concentrate in `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/platforms/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29545 - [Bugfix] Fix DeepSeek R1 MTP weight loading

- Link: https://github.com/vllm-project/vllm/pull/29545
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`; associated commits `51c57b51dd51`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-0, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek R1 MTP weight loading"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Bugfix] Fix DeepSeek R1 MTP weight loading"; the main implementation surface is `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0 (11 lines); hunks: -346,11 +346,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -377,6 +382,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0 (11 lines); hunks: -346,11 +346,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -377,6 +382,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -346,11 +346,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                    is_expert_weight = False
+                        # Anyway, this is an expert weight and should not be
+                        # attempted to load as other weights later
+                        is_expert_weight = True
@@ -377,6 +382,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                        if is_expert_weight:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27568 - [DeepSeek v3.2] Make top-k work for any logit values.

- Link: https://github.com/vllm-project/vllm/pull/27568
- Status/date: merged / 2025-12-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `184076c3fecf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +629/-210, 1067 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek v3.2] Make top-k work for any logit values."; model line: DeepSeek V3/R1; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[DeepSeek v3.2] Make top-k work for any logit values."; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -684,18 +684,18 @@ def sparse_attn_indexer(
-            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
-            torch.ops._C.top_k_per_row(
+            torch.ops._C.top_k_per_row_prefill(
+                topk_tokens,
@@ -738,7 +738,6 @@ def sparse_attn_indexer(
-        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27532 - [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2

- Link: https://github.com/vllm-project/vllm/pull/27532
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `3e41992fecdc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 30 files, +1372/-256, 2323 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, touching `sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -83,6 +83,7 @@
+from vllm.v1.worker.workspace import current_workspace_manager
@@ -618,8 +619,15 @@ def sparse_attn_indexer(
+        # Reserve workspace for indexer during profiling run
+        current_workspace_manager().get_simultaneous(
+            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
+            ((total_seq_lens, 4), torch.uint8),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19
- Risk and verification: The diff ships test coverage in `tests/conftest.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30841 - [Bugfix] deepseek-V3.2 self.weights_proj has no bias

- Link: https://github.com/vllm-project/vllm/pull/30841
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `84896fda22d3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] deepseek-V3.2 self.weights_proj has no bias"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] deepseek-V3.2 self.weights_proj has no bias"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -835,7 +835,11 @@ def __init__(
-            hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
+            hidden_size,
+            self.n_head,
+            bias=False,
+            quant_config=None,
+            prefix=f"{prefix}.weights_proj",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31046 - [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2

- Link: https://github.com/vllm-project/vllm/pull/31046
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `4cf9429897c1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-2, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -878,8 +878,11 @@ def forward(
-        q = torch.cat([q_pe.squeeze(0), q_nope], dim=-1)
-        k = torch.cat([k_pe.squeeze((0, 2)), k_nope], dim=-1)
+        # `rotary_emb` is shape-preserving; `q_pe` is already
+        # [num_tokens, n_head, rope_dim].
+        q = torch.cat([q_pe, q_nope], dim=-1)
+        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31160 - [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2

- Link: https://github.com/vllm-project/vllm/pull/31160
- Status/date: merged / 2025-12-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `76e6a951925b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -878,11 +878,14 @@ def forward(
-        # `rotary_emb` is shape-preserving; `q_pe` is already
-        # [num_tokens, n_head, rope_dim].
+        # Note: RoPE (NeoX) can introduce extra leading dimensions during compilation
+        # so we need to reshape back to token-flattened shapes
+        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
+        k_pe = k_pe.reshape(-1, 1, self.rope_dim)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30356 - [CI][DeepSeek] Add nightly DeepSeek R1 `lm_eval` tests on H200

- Link: https://github.com/vllm-project/vllm/pull/30356
- Status/date: merged / 2026-01-05
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml`; associated commits `276e03b92c16`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +33/-1, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI][DeepSeek] Add nightly DeepSeek R1 `lm_eval` tests on H200"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml`; technical summary: Covers "[CI][DeepSeek] Add nightly DeepSeek R1 `lm_eval` tests on H200"; the main implementation surface is `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11; `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11.
- Code diff details:
  - `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml
@@ -0,0 +1,11 @@
+model_name: "deepseek-ai/DeepSeek-R1"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
diff -- tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml
@@ -0,0 +1,11 @@
+model_name: "deepseek-ai/DeepSeek-R1"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` added +11/-0; `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` added +11/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml`, `tests/evals/gsm8k/configs/models-h200.txt`, `tests/evals/gsm8k/test_gsm8k_correctness.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32175 - [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding

- Link: https://github.com/vllm-project/vllm/pull/32175
- Status/date: merged / 2026-01-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `5de6dd0662da`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -717,13 +717,20 @@ def sparse_attn_indexer(
+            # [num_decode_tokens, n_head, head_dim] -> [bs, 1+next_n, n_head, head_dim]
+            # [num_decode_tokens, n_head] -> [bs, 1+next_n, n_head]
+            padded_weights = pack_seq_triton(weights[:num_decode_tokens], decode_lens)
+            # [bs, 1+next_n, n_head] -> [bs * next_n, n_head]
+            padded_weights = padded_weights.flatten(0, 1)
+            padded_weights = weights
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29287 - [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp

- Link: https://github.com/vllm-project/vllm/pull/29287
- Status/date: merged / 2026-01-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `6c20e89c0209`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +982/-323, 1521 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer, touching `get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -43,7 +43,6 @@
-from vllm.forward_context import get_forward_context
@@ -63,6 +62,7 @@
+from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
@@ -74,16 +74,11 @@
-from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
-from vllm.utils.torch_utils import direct_register_custom_op
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/config/compilation.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33964 - [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32

- Link: https://github.com/vllm-project/vllm/pull/33964
- Status/date: merged / 2026-02-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/deepseekv32_tool_parser.py`; associated commits `7bec4351305f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32"; the main implementation surface is `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state, touching `_generate_tool_call_id, adjust_request, _reset_streaming_state`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:
+    def adjust_request(self, request):
+        request = super().adjust_request(request)
+        if request.tools and request.tool_choice != "none":
+            # Ensure tool call tokens
+            # (<｜DSML｜function_calls>, </｜DSML｜function_calls>)
+            # are not skippedduring decoding.
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/deepseekv32_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24972 - [Model] Deepseek-V3.1 reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/24972
- Status/date: closed / 2026-02-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_deepseekv3_reasoning_parser.py`; associated commits `85a65e7f51ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +214/-11, 330 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Deepseek-V3.1 reasoning parser"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `tests/reasoning/test_deepseekv3_reasoning_parser.py`; technical summary: Covers "[Model] Deepseek-V3.1 reasoning parser"; the main implementation surface is `tests/reasoning/test_deepseekv3_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic, touching `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`.
- Code diff details:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
- Key code excerpts:

```diff
diff -- tests/reasoning/test_deepseekv3_reasoning_parser.py
@@ -0,0 +1,73 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
+                                              DeltaMessage)
```

- Reviewed files:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +73/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_deepseekv3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34758 - [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)

- Link: https://github.com/vllm-project/vllm/pull/34758
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `6874638bc443`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +855/-3, 917 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention, touching `forward, DeepSeekV2FusedQkvAProj, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -32,6 +32,7 @@
+import vllm._custom_ops as ops
@@ -711,6 +712,64 @@ def forward(
+class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
+    def __init__(
+        self,
+        input_size: int,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34876 - [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix

- Link: https://github.com/vllm-project/vllm/pull/34876
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `7f51e9386470`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__, touching `DeepSeekV2FusedQkvAProj, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
-        output_size: int,
+        output_size: list[int],
@@ -726,7 +726,7 @@ def __init__(
-            prefix=f"{prefix}.kv_a_proj_with_mqa",
+            prefix=prefix,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34302 - [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)

- Link: https://github.com/vllm-project/vllm/pull/34302
- Status/date: merged / 2026-02-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `8435b2e04925`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +915/-3, 971 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype, touching `forward, DeepSeekV2Gate, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -221,6 +221,73 @@ def forward(self, x):
+class DeepSeekV2Gate(ReplicatedLinear):
+    def __init__(
+        self,
+        hidden_size: int,
+        n_experts: int,
+        quant_config: QuantizationConfig | None = None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35751 - [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile

- Link: https://github.com/vllm-project/vllm/pull/35751
- Status/date: merged / 2026-03-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `9319044ee9a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-13, 75 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, touching `forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -75,6 +75,7 @@
+from vllm.utils.torch_utils import direct_register_custom_op
@@ -717,6 +718,44 @@ def forward(
+def _min_latency_fused_qkv_a_proj_impl(
+    input_: torch.Tensor,
+    weight: torch.Tensor,
+) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36247 - [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x

- Link: https://github.com/vllm-project/vllm/pull/36247
- Status/date: merged / 2026-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `ee8a29511fc6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__, touching `_min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(
-class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
+class DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear):
@@ -848,7 +848,7 @@ def __init__(
-            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProj(
+            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36056 - [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1

- Link: https://github.com/vllm-project/vllm/pull/36056
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; associated commits `be12afd284f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +622/-437, 1113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1"; the main implementation surface is `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request, touching `__init__, type, _generate_tool_call_id`; `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call, touching `make_parser, make_tool_param, make_request`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):
-        # Sentinel tokens
-        self.dsml_token: str = "｜DSML｜"
-        self.dsml_start_check: str = "<" + self.dsml_token
+        # Sentinel token
-        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
-        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -0,0 +1,476 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit tests for DeepSeekV32ToolParser.
+These tests use a minimal mock tokenizer so no real model weights are required.
+"""
+import json
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33703 - [Bugfix] Support multi-type params parsing for DeepSeek v3.2

- Link: https://github.com/vllm-project/vllm/pull/33703
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; associated commits `a6db99ba02ec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +201/-18, 250 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Support multi-type params parsing for DeepSeek v3.2"; model line: DeepSeek V3/R1; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Support multi-type params parsing for DeepSeek v3.2"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types, touching `test_no_emission_while_incomplete, deepseekv32_tokenizer, parser`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked, touching `_parse_invoke_params, _convert_param_value, _convert_param_value_checked`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -11,6 +11,7 @@
+from vllm.tokenizers import get_tokenizer
@@ -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):
+@pytest.fixture(scope="module")
+def deepseekv32_tokenizer():
+    return get_tokenizer(tokenizer_name="deepseek-ai/DeepSeek-V3.2")
+@pytest.fixture
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:
-    def _convert_param_value(self, value: str, param_type: str) -> Any:
+    def _convert_param_value_checked(self, value: str, param_type: str) -> Any:
@@ -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str) -> Any:
-            try:
-                return int(value)
-            except (ValueError, TypeError):
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38684 - [Perf] DSV3.2 Indexer Fused Weights Projection

- Link: https://github.com/vllm-project/vllm/pull/38684
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `5f96f9aff10f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-14, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] DSV3.2 Indexer Fused Weights Projection"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Perf] DSV3.2 Indexer Fused Weights Projection"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -639,21 +639,19 @@ def __init__(
-        self.wk = ReplicatedLinear(
+        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
+        # weights_proj does not get quantized, so we run both with quant_config=None
+        # wk may be upcasted from the default quant; experiments show fusion is always
+        # faster unless WK proj is in FP4, which is not the case for all known quants.
+        self.wk_weights_proj = MergedColumnParallelLinear(
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            # Fused indexer wk + weights_proj
+            ("wk_weights_proj", "wk", 0),
+            ("wk_weights_proj", "weights_proj", 1),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38870 - [Bugfix] Fix DSV32 weight loading

- Link: https://github.com/vllm-project/vllm/pull/38870
- Status/date: merged / 2026-04-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `8617f8676b5a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-27, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DSV32 weight loading"; model line: DeepSeek V3/R1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Bugfix] Fix DSV32 weight loading"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights, touching `DeepSeekMTP, __init__, set_moe_parameters`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -625,6 +625,11 @@ def __init__(
+        self.quant_config = quant_config
+        self.is_fp4_ckpt = (
+            self.quant_config is not None
+            and self.quant_config.get_name() == "modelopt_fp4"
+        )
@@ -639,18 +644,36 @@ def __init__(
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):
+        self.quant_config = vllm_config.quant_config
+        self.is_fp4_ckpt = (
+            self.quant_config is not None
+            and self.quant_config.get_name() == "modelopt_fp4"
+        )
@@ -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- Link: https://github.com/vllm-project/vllm/pull/37421
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `b55d830ec782`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +2039/-483, 2698 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -67,7 +67,9 @@
-from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
+from vllm.model_executor.layers.sparse_attn_indexer import (
+    SparseAttnIndexer,
+)
@@ -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35968 - [Performance] DeepSeek V3.2 multi-stream indexer overlap

- Link: https://github.com/vllm-project/vllm/pull/35968
- Status/date: closed / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +187/-11, 255 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance] DeepSeek V3.2 multi-stream indexer overlap"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py`; technical summary: Covers "[Performance] DeepSeek V3.2 multi-stream indexer overlap"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +84/-8 (92 lines); hunks: -79,7 +79,8; -625,6 +626,11 @@ def __init__(; symbols: __init__, _compute_k, forward, touching `__init__, _compute_k, forward`; `vllm/model_executor/layers/layernorm.py` modified +20/-3 (23 lines); hunks: -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):; symbols: __init__, _forward_static, forward, touching `__init__, _forward_static, forward`; `tests/utils_/test_indexer_dual_stream.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides, test_fake_output_shapes_parametrized, touching `_indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +84/-8 (92 lines); hunks: -79,7 +79,8; -625,6 +626,11 @@ def __init__(; symbols: __init__, _compute_k, forward
  - `vllm/model_executor/layers/layernorm.py` modified +20/-3 (23 lines); hunks: -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):; symbols: __init__, _forward_static, forward
  - `tests/utils_/test_indexer_dual_stream.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides, test_fake_output_shapes_parametrized
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -79,7 +79,8 @@
-from vllm.utils.torch_utils import direct_register_custom_op
+from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
+from vllm.utils.torch_utils import aux_stream, direct_register_custom_op
@@ -625,6 +626,11 @@ def __init__(
+        self.events = (
+            [torch.cuda.Event(), torch.cuda.Event()]
diff -- vllm/model_executor/layers/layernorm.py
@@ -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):
+    @staticmethod
+    def _forward_static(
+        weight: torch.Tensor,
+        bias: torch.Tensor,
+        dim: int,
+        eps: float,
diff -- tests/utils_/test_indexer_dual_stream.py
@@ -0,0 +1,83 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +84/-8; `vllm/model_executor/layers/layernorm.py` modified +20/-3
  - tests: `tests/utils_/test_indexer_dual_stream.py` added +83/-0
- Risk and verification: The diff ships test coverage in `tests/utils_/test_indexer_dual_stream.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41198 - [Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls

- Link: https://github.com/vllm-project/vllm/pull/41198
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; associated commits `762022cafb1a`
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-1, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls"; model line: DeepSeek V3/R1; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0 (24 lines); hunks: -188,6 +188,30 @@ def test_multiple_tools(self, parser):; symbols: test_multiple_tools, test_type_conversion_in_non_streaming, touching `test_multiple_tools, test_type_conversion_in_non_streaming`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1 (3 lines); hunks: -191,12 +191,13 @@ def extract_tool_calls(; symbols: extract_tool_calls, touching `extract_tool_calls`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0 (24 lines); hunks: -188,6 +188,30 @@ def test_multiple_tools(self, parser):; symbols: test_multiple_tools, test_type_conversion_in_non_streaming
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1 (3 lines); hunks: -191,12 +191,13 @@ def extract_tool_calls(; symbols: extract_tool_calls
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -188,6 +188,30 @@ def test_multiple_tools(self, parser):
+    def test_type_conversion_in_non_streaming(self):
+        """Non-streaming extraction must convert params using the tool schema."""
+        tool = ChatCompletionToolsParam(
+            function=FunctionDefinition(
+                name="toggle",
+                parameters={
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -191,12 +191,13 @@ def extract_tool_calls(
+                    params = self._convert_params_with_schema(invoke_name, param_dict)
-                                arguments=json.dumps(param_dict, ensure_ascii=False),
+                                arguments=json.dumps(params, ensure_ascii=False),
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
