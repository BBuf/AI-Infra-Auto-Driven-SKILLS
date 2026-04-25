# vllm DeepSeek V3/R1 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/online_serving/elastic_ep/serve_deepseek_v2.sh` | 无直接 PR 号提交 |
| `examples/tool_chat_template_deepseekv3.jinja` | [#17784](https://github.com/vllm-project/vllm/pull/17784) |
| `examples/tool_chat_template_deepseekv31.jinja` | [#23454](https://github.com/vllm-project/vllm/pull/23454) |
| `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` | [#30356](https://github.com/vllm-project/vllm/pull/30356) |
| `tests/evals/gsm8k/configs/DeepSeek-R1-DP_MI325.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` | [#30356](https://github.com/vllm-project/vllm/pull/30356) |
| `tests/evals/gsm8k/configs/DeepSeek-R1-TP_MI325.yaml` | 无直接 PR 号提交 |
| `tests/reasoning/test_deepseekv3_reasoning_parser.py` | [#24972](https://github.com/vllm-project/vllm/pull/24972), [#25589](https://github.com/vllm-project/vllm/pull/25589) |
| `tests/tool_parsers/test_deepseekv31_tool_parser.py` | 无直接 PR 号提交 |
| `tests/tool_parsers/test_deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#36056](https://github.com/vllm-project/vllm/pull/36056) |
| `tests/tool_parsers/test_deepseekv3_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/deepseek_mtp.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#29545](https://github.com/vllm-project/vllm/pull/29545), [#38684](https://github.com/vllm-project/vllm/pull/38684), [#38870](https://github.com/vllm-project/vllm/pull/38870) |
| `vllm/model_executor/models/deepseek_v2.py` | [#13833](https://github.com/vllm-project/vllm/pull/13833), [#23971](https://github.com/vllm-project/vllm/pull/23971), [#24119](https://github.com/vllm-project/vllm/pull/24119), [#25896](https://github.com/vllm-project/vllm/pull/25896), [#25999](https://github.com/vllm-project/vllm/pull/25999), [#26456](https://github.com/vllm-project/vllm/pull/26456), [#26465](https://github.com/vllm-project/vllm/pull/26465), [#26670](https://github.com/vllm-project/vllm/pull/26670), [#26763](https://github.com/vllm-project/vllm/pull/26763), [#27532](https://github.com/vllm-project/vllm/pull/27532), [#27568](https://github.com/vllm-project/vllm/pull/27568), [#28968](https://github.com/vllm-project/vllm/pull/28968), ... (25 total) |
| `vllm/reasoning/deepseek_r1_reasoning_parser.py` | 无直接 PR 号提交 |
| `vllm/tool_parsers/deepseekv31_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/tool_parsers/deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#33964](https://github.com/vllm-project/vllm/pull/33964), [#36056](https://github.com/vllm-project/vllm/pull/36056) |
| `vllm/tool_parsers/deepseekv3_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 34
- 原文档显式引用补充 PR 数: 1
- 当前文档总 PR 数: 35
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-02-26 | [#13833](https://github.com/vllm-project/vllm/pull/13833) | merged | DeepSeek V2/V3/R1 only place `lm_head` on last pp rank | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-05-12 | [#17784](https://github.com/vllm-project/vllm/pull/17784) | merged | [Feature] Support DeepSeekV3 Function Call | `examples/tool_chat_template_deepseekv3.jinja` |
| 2025-08-07 | [#22352](https://github.com/vllm-project/vllm/pull/22352) | merged | [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM` | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-08-23 | [#23454](https://github.com/vllm-project/vllm/pull/23454) | merged | Support DeepSeek-V3.1 tool call | `examples/tool_chat_template_deepseekv31.jinja` |
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

## 逐 PR diff 审计卡

### PR #13833 - DeepSeek V2/V3/R1 only place `lm_head` on last pp rank

- 链接: https://github.com/vllm-project/vllm/pull/13833
- 状态/时间: merged / 2025-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `24679788ed38`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-3，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补强部署文档、测试或 CI 验证面，标题为「DeepSeek V2/V3/R1 only place `lm_head` on last pp rank」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：Balances memory usage across pipeline ranks by reducing the amount used on all ranks that are not the last rank.
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -636,9 +636,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -636,9 +636,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17784 - [Feature] Support DeepSeekV3 Function Call

- 链接: https://github.com/vllm-project/vllm/pull/17784
- 状态/时间: merged / 2025-05-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_deepseekv3.jinja`；关联提交 `3a5ea7512926`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+473/-1，可读 patch 495 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Feature] Support DeepSeekV3 Function Call」，变更集中在 `examples/tool_chat_template_deepseekv3.jinja`。PR 描述补充为：Support DeepSeek-V3-0324 function call. usage: test script (no streaming): Output: test script (streaming): Output:
- 实现要点: `examples/tool_chat_template_deepseekv3.jinja` added +96/-0 (96 lines); hunks: -0,0 +1,96。
- 代码 diff 细节:
  - `examples/tool_chat_template_deepseekv3.jinja` added +96/-0 (96 lines); hunks: -0,0 +1,96
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/tool_chat_template_deepseekv3.jinja` added +96/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv3_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22352 - [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`

- 链接: https://github.com/vllm-project/vllm/pull/22352
- 状态/时间: merged / 2025-08-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：This PR adds `packed_modules_mapping` class attribute to `DeepseekV2ForCausalLM`. https://github.com/vllm-project/vllm/pull/21116 added `fused_qkv_a_proj` for DeepseekV2ForCausa...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +16/-0 (16 lines); hunks: -726,13 +726,29 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM, __init__，涉及 `forward, DeepseekV2ForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +16/-0 (16 lines); hunks: -726,13 +726,29 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +16/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23454 - Support DeepSeek-V3.1 tool call

- 链接: https://github.com/vllm-project/vllm/pull/23454
- 状态/时间: merged / 2025-08-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_deepseekv31.jinja`；关联提交 `b8f17f5d980e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+468/-0，可读 patch 491 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「Support DeepSeek-V3.1 tool call」，变更集中在 `examples/tool_chat_template_deepseekv31.jinja`。PR 描述补充为：## Purpose Support DeepSeek-V3.1 tool call. The tool call format of DeepSeek-V3.1 is different from DeepSeek-V3/R1: DeepSeek-V3.1: tool_call_name tool_call_arguments DeepSeek-R1...
- 实现要点: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91。
- 代码 diff 细节:
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23971 - Add LoRA support for DeepSeek models (V2, V3, R1-0528)

- 链接: https://github.com/vllm-project/vllm/pull/23971
- 状态/时间: merged / 2025-08-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `379ea2823a75`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+12/-7，可读 patch 54 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「Add LoRA support for DeepSeek models (V2, V3, R1-0528)」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：- Add SupportsLoRA interface to DeepseekV2ForCausalLM - Enable LoRA fine-tuning for DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1-0528 - MoE detection already handled by existing is...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -56,7 +56,7; -727,7 +727,8 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM，涉及 `forward, DeepseekV2ForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -56,7 +56,7; -727,7 +727,8 @@ def forward(; symbols: forward, DeepseekV2ForCausalLM
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24119 - [Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue

- 链接: https://github.com/vllm-project/vllm/pull/24119
- 状态/时间: merged / 2025-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `930a24144c07`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose Fixes https://github.com/vllm-project/vllm/issues/24118 that is introduced by https://github.com/vllm-project/vllm/pull/23123 ## Test `lm_eval --model local-completio...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -160,7 +160,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -160,7 +160,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -160,7 +160,8 @@ def __init__(
-            routed_scaling_factor=self.routed_scaling_factor,
+            # we do scaling outside, set factor to 1.0 to avoid double mul
+            routed_scaling_factor=1.0,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25896 - [New Model] DeepSeek-V3.2 (Rebased to Main)

- 链接: https://github.com/vllm-project/vllm/pull/25896
- 状态/时间: merged / 2025-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；关联提交 `fa7e254a7f3e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 71 个文件，+3918/-221，可读 patch 5400 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 调整模型相关实现，标题为「[New Model] DeepSeek-V3.2 (Rebased to Main)」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。PR 描述补充为：Rebased dsv32, based on #25869 Run command gsm8k gsm8k, 20-shot
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__，涉及 `DeepseekV2MLP, DeepseekV2Attention, __init__`；`vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion_attn.py`, `tests/kernels/attention/test_cache.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/attention/test_flashmla.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25999 - [Deepseek v3.2] Support indexer prefill chunking

- 链接: https://github.com/vllm-project/vllm/pull/25999
- 状态/时间: merged / 2025-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `1e50f1be7058`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+149/-79，可读 patch 324 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Deepseek v3.2] Support indexer prefill chunking」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose Split the prefill to multiple steps, with each step contains a subset of prefill requests. With this approach, we can avoid the large output caused by gather kv cache...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38
- 验证与风险: diff 自带测试面 `tests/v1/attention/test_sparse_mla_backends.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- 链接: https://github.com/vllm-project/vllm/pull/25589
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_deepseekv3_reasoning_parser.py`；关联提交 `85a65e7f51ad`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+215/-3，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)」，变更集中在 `tests/reasoning/test_deepseekv3_reasoning_parser.py`。PR 描述补充为：## Purpose This PR adds a new reasoning parser for the DeepSeek-V3.1 model, named deepseek_v3. Unlike previous models such as deepseek_r1, the reasoning parser for DeepSeek-V3.1...
- 实现要点: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic，涉及 `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`。
- 代码 diff 细节:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_deepseekv3_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26456 - [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather

- 链接: https://github.com/vllm-project/vllm/pull/26456
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `f5ed68ef63d0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-68，可读 patch 104 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 优化关键推理路径或后端选择，标题为「[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：Replace torch `cp_gather_indexer_k_quant_cache` to cuda op. Follow up for #25931 gsm8k 20 shots
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer，涉及 `get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26763 - [Deepseek v3.2] Optimize top_k_per_row

- 链接: https://github.com/vllm-project/vllm/pull/26763
- 状态/时间: merged / 2025-10-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `80e94529845d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+13/-49，可读 patch 203 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 优化关键推理路径或后端选择，标题为「[Deepseek v3.2] Optimize top_k_per_row」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose This PR optimizes kernel top_k_per_row. Local testing shows it is about 2.5x from its previous version. ## Test Plan ## Test Result --- Essential Elements of an Effec...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26465 - [Deepseek v3.2] Remove extra logics in indexer

- 链接: https://github.com/vllm-project/vllm/pull/26465
- 状态/时间: merged / 2025-10-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `09a7e6f6179b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+141/-40，可读 patch 272 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补强部署文档、测试或 CI 验证面，标题为「[Deepseek v3.2] Remove extra logics in indexer」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose Remove extra logics around `index_end_pos` in the indexer. CC @dcampora ## Test Plan `lm-eval --model local-completions --tasks gsm8k --model_args model=DeepSeek-V3.2...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28968 - [DeepSeek] Fix DeepSeek V3.2 Rope Embedding

- 链接: https://github.com/vllm-project/vllm/pull/28968
- 状态/时间: merged / 2025-11-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `88f5b19f0bc6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+17/-3，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 修复已暴露的启动、加载、解析或数值问题，标题为「[DeepSeek] Fix DeepSeek V3.2 Rope Embedding」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose Deepseek recently find error in their official implementation that ROPE in indexer shouldn't be interleaved. ## Test Plan gsm8k 20-shots ## Test Result --- Essential...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__，涉及 `forward, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26670 - [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA

- 链接: https://github.com/vllm-project/vllm/pull/26670
- 状态/时间: merged / 2025-11-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `06c20c990464`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+583/-15，可读 patch 700 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose The PR add Deepseek v3.2 support on ROCm platforms. The main change in this PR include: - Replace all hardcode float8_e4m3fn to platform supported fp8 dtype, and add...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake，涉及 `sparse_attn_indexer, sparse_attn_indexer_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4
- 验证与风险: runtime 路径改动集中在 `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/platforms/rocm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29545 - [Bugfix] Fix DeepSeek R1 MTP weight loading

- 链接: https://github.com/vllm-project/vllm/pull/29545
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_mtp.py`；关联提交 `51c57b51dd51`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix DeepSeek R1 MTP weight loading」，变更集中在 `vllm/model_executor/models/deepseek_mtp.py`。PR 描述补充为：## Purpose Fixes #29448 by adding some logic from `deepseek_v2.py` that was missing in `deepseek_mtp.py` ## Test Plan with ## Test Result Previously crashed during weight loadin...
- 实现要点: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0 (11 lines); hunks: -346,11 +346,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -377,6 +382,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0 (11 lines); hunks: -346,11 +346,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; -377,6 +382,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27568 - [DeepSeek v3.2] Make top-k work for any logit values.

- 链接: https://github.com/vllm-project/vllm/pull/27568
- 状态/时间: merged / 2025-12-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `184076c3fecf`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+629/-210，可读 patch 1067 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 调整模型相关实现，标题为「[DeepSeek v3.2] Make top-k work for any logit values.」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose This PR allows top_k_per_row work for any values in logits. Even if the logits differ only in the least significant bytes, top-k is now guaranteed to always give a co...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27532 - [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2

- 链接: https://github.com/vllm-project/vllm/pull/27532
- 状态/时间: merged / 2025-12-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `3e41992fecdc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 30 个文件，+1372/-256，可读 patch 2323 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：When doing prefill up-convert the kv-cache from fp8 to bf16 and call the bf16 prefill kernel instead of the decode kernel. This PR introduce global workspace management to have...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake，涉及 `sparse_attn_indexer, sparse_attn_indexer_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19
- 验证与风险: diff 自带测试面 `tests/conftest.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30841 - [Bugfix] deepseek-V3.2 self.weights_proj has no bias

- 链接: https://github.com/vllm-project/vllm/pull/30841
- 状态/时间: merged / 2025-12-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `84896fda22d3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Bugfix] deepseek-V3.2 self.weights_proj has no bias」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose self.weights_proj has no bias,some other hardware bias maybe not initial with 0 maybe not correct H20 bias initial with 0 !image kunlun bias not initial with 0 !image...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31046 - [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2

- 链接: https://github.com/vllm-project/vllm/pull/31046
- 状态/时间: merged / 2025-12-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `4cf9429897c1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-2，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose `export MODEL="deepseek-ai/DeepSeek-V3.2"` `vllm serve "$MODEL" -tp 8 --port 9256 --enable-expert-parallel -cc '{"mode":3,"pass_config":{"fuse_norm_quant":true,"elimi...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31160 - [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2

- 链接: https://github.com/vllm-project/vllm/pull/31160
- 状态/时间: merged / 2025-12-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `76e6a951925b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-3，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose `export MODEL="deepseek-ai/DeepSeek-V3.2"` `vllm serve "$MODEL" -tp 8 --port 9256 --enable-expert-parallel` Will trigger error The root cause: Number of dimensions of...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30356 - [CI][DeepSeek] Add nightly DeepSeek R1 `lm_eval` tests on H200

- 链接: https://github.com/vllm-project/vllm/pull/30356
- 状态/时间: merged / 2026-01-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml`；关联提交 `276e03b92c16`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+33/-1，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[CI][DeepSeek] Add nightly DeepSeek R1 `lm_eval` tests on H200」，变更集中在 `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml`。PR 描述补充为：## Purpose DeepSeek R1 is frequently broken on main. This PR adds an `lm_eval` test for DeepSeek R1 on 8xH200 to catch this earlier. Currently set to be run nightly (no `source_...
- 实现要点: `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11；`tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml` added +11/-0; `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml` added +11/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-R1-TP.yaml`, `tests/evals/gsm8k/configs/models-h200.txt`, `tests/evals/gsm8k/test_gsm8k_correctness.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32175 - [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding

- 链接: https://github.com/vllm-project/vllm/pull/32175
- 状态/时间: merged / 2026-01-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `5de6dd0662da`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-2，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose Fix https://github.com/vllm-project/vllm/issues/32172 ## Test Plan ## Test Result vllm bench ok. lm_eval: Main: This PR: --- Essential Elements of an Effective PR Des...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29287 - [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp

- 链接: https://github.com/vllm-project/vllm/pull/29287
- 状态/时间: merged / 2026-01-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `6c20e89c0209`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+982/-323，可读 patch 1521 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 优化关键推理路径或后端选择，标题为「[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose This PR optimize the deepseekv3.2's performance on AMD's device, and separate `SparseAttnIndexer` out as a `CustomOp` as it contains lots of heavy kernels like `fp8_m...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer，涉及 `get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233
- 验证与风险: runtime 路径改动集中在 `vllm/_aiter_ops.py`, `vllm/config/compilation.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33964 - [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32

- 链接: https://github.com/vllm-project/vllm/pull/33964
- 状态/时间: merged / 2026-02-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/deepseekv32_tool_parser.py`；关联提交 `7bec4351305f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32」，变更集中在 `vllm/tool_parsers/deepseekv32_tool_parser.py`。PR 描述补充为：## Purpose Fix the issue where tool calling does not work when using fast detokenization with dsv32 dsv32 uses special tokens like `｜DSML｜function_calls>`, which are skipped by...
- 实现要点: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state，涉及 `_generate_tool_call_id, adjust_request, _reset_streaming_state`。
- 代码 diff 细节:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/deepseekv32_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24972 - [Model] Deepseek-V3.1 reasoning parser

- 链接: https://github.com/vllm-project/vllm/pull/24972
- 状态/时间: closed / 2026-02-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_deepseekv3_reasoning_parser.py`；关联提交 `85a65e7f51ad`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+214/-11，可读 patch 330 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 调整模型相关实现，标题为「[Model] Deepseek-V3.1 reasoning parser」，变更集中在 `tests/reasoning/test_deepseekv3_reasoning_parser.py`。PR 描述补充为：## Purpose This PR adds a new reasoning parser for the DeepSeek-V3.1 model, named deepseek_v3. Unlike previous models such as deepseek_r1, the reasoning parser for DeepSeek-V3.1...
- 实现要点: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic，涉及 `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`。
- 代码 diff 细节:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +73/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_deepseekv3_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34758 - [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)

- 链接: https://github.com/vllm-project/vllm/pull/34758
- 状态/时间: merged / 2026-02-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `6874638bc443`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+855/-3，可读 patch 917 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose - add min latency bf16 qkv_a_gemm - adapted from sgl: * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu, which adapted from...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention，涉及 `forward, DeepSeekV2FusedQkvAProj, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34876 - [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix

- 链接: https://github.com/vllm-project/vllm/pull/34876
- 状态/时间: merged / 2026-02-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `7f51e9386470`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 修复已暴露的启动、加载、解析或数值问题，标题为「[Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose Fix #34869 ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link exis...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__，涉及 `DeepSeekV2FusedQkvAProj, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
-        output_size: int,
+        output_size: list[int],
@@ -726,7 +726,7 @@ def __init__(
-            prefix=f"{prefix}.kv_a_proj_with_mqa",
+            prefix=prefix,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34302 - [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)

- 链接: https://github.com/vllm-project/vllm/pull/34302
- 状态/时间: merged / 2026-02-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `8435b2e04925`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+915/-3，可读 patch 971 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：Port the optimized router GEMM kernel from sglang's sgl-kernel for DeepSeek V3 MoE models. This kernel is specifically optimized for small batch sizes (1-16 tokens) common in de...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype，涉及 `forward, DeepSeekV2Gate, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35751 - [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile

- 链接: https://github.com/vllm-project/vllm/pull/35751
- 状态/时间: merged / 2026-03-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `9319044ee9a1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+41/-13，可读 patch 75 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 优化关键推理路径或后端选择，标题为「[MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Summary - `DeepSeekV2FusedQkvAProj.forward` had a data-dependent conditional `0 🤖 Generated with Claude Code
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj，涉及 `forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36247 - [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x

- 链接: https://github.com/vllm-project/vllm/pull/36247
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `ee8a29511fc6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Purpose When running DeepSeek-R1 FP8 on MI300x, vLLM crashed during model initialization with: The compressed-tensors quantization config matches layers by checking if a targ...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__，涉及 `_min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(
-class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
+class DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear):
@@ -848,7 +848,7 @@ def __init__(
-            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProj(
+            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36056 - [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1

- 链接: https://github.com/vllm-project/vllm/pull/36056
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；关联提交 `be12afd284f3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+622/-437，可读 patch 1113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1」，变更集中在 `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`。PR 描述补充为：## Purpose The deepseek 3.2 tool parser used an incremental state machine (~20 instance variables) that parsed DSML tags character-by-character as tokens streamed in. With strea...
- 实现要点: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request，涉及 `__init__, type, _generate_tool_call_id`；`tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call，涉及 `make_parser, make_tool_param, make_request`。
- 代码 diff 细节:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33703 - [Bugfix] Support multi-type params parsing for DeepSeek v3.2

- 链接: https://github.com/vllm-project/vllm/pull/33703
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；关联提交 `a6db99ba02ec`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+201/-18，可读 patch 250 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 补齐模型支持入口或运行时能力，标题为「[Bugfix] Support multi-type params parsing for DeepSeek v3.2」，变更集中在 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`。PR 描述补充为：## Purpose Kilo code uses multi typed params for some reason, and such calls fails to render with exception `'list' object has no attribute 'lowercase'` when Kilo code passes `t...
- 实现要点: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types，涉及 `test_no_emission_while_incomplete, deepseekv32_tokenizer, parser`；`vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked，涉及 `_parse_invoke_params, _convert_param_value, _convert_param_value_checked`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38684 - [Perf] DSV3.2 Indexer Fused Weights Projection

- 链接: https://github.com/vllm-project/vllm/pull/38684
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；关联提交 `5f96f9aff10f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-14，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 优化关键推理路径或后端选择，标题为「[Perf] DSV3.2 Indexer Fused Weights Projection」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。PR 描述补充为：## Purpose Fuse the WK and Weights_Proj projections in the DSV3.2 Indexer. This is an alternative optimization to https://github.com/vllm-project/vllm/pull/35968, which overlaps...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38870 - [Bugfix] Fix DSV32 weight loading

- 链接: https://github.com/vllm-project/vllm/pull/38870
- 状态/时间: merged / 2026-04-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；关联提交 `8617f8676b5a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-27，可读 patch 158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix DSV32 weight loading」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。PR 描述补充为：## Purpose #38684 intorude bug on the fp8 checkpoint ## Test Plan gsm8k score ## Test Result --- Essential Elements of an Effective PR Description Checklist - [x] The purpose of...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights，涉及 `DeepSeekMTP, __init__, set_moe_parameters`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- 链接: https://github.com/vllm-project/vllm/pull/37421
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `b55d830ec782`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+2039/-483，可读 patch 2698 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 DeepSeek V3/R1 优化关键推理路径或后端选择，标题为「[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode」，变更集中在 `vllm/model_executor/models/deepseek_v2.py`。PR 描述补充为：## Summary Redesigns the persistent TopK kernel used by DSA as a true persistent scheduler with dynamic per-row path selection. This supersedes and closes #34265, which took a C...
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
