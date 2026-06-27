# TensorRT-LLM Kimi 模型 PR 优化历史

## 2026-06-27 源码 head 刷新

已用 `git ls-remote` 复核 TensorRT-LLM 上游 main head：
`NVIDIA/TensorRT-LLM@aaffa2f9fef3025e0f698d978385a73460344e0b`。
下方文件级 source-scan 行仍是上一轮 tracked-file 审计结果；引用当前 open PR 状态前，先看 `model-pr-optimization-history/open-pr-watch.md`。

结果：除了本文已有 timeline/backfill 行之外，没有额外 PR-numbered merge 命中 tracked files。

## 2026-06-27 PR 补漏复核

本文的逐 PR diff 审计卡基于 TensorRT-LLM 上游
`HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980` 生成；根目录 TensorRT-LLM
history index 已跟踪 2026-06-27 runtime refresh
`aaffa2f9fef3025e0f698d978385a73460344e0b`。本文覆盖 Kimi K2 Thinking /
Kimi K2.5 相关 merged PR，并采用 SGLang 风格的模型实现覆盖、时间线和逐 PR
diff 审计卡。

本轮筛选规则：标题/文件命中 `Kimi`、`kimi_k25`、`KimiK25`、`K2.5`、`K2 Thinking`、`NVFP4`、`multimodal`、`tool_parser`、`reasoning_parser`、`guided decoding`、`spec dec`、`rejection sampling`、`NIXL` 的 merged PR；过滤纯格式化和不影响 Kimi runtime/serve/eval lane 的基础设施 PR。

## 模型实现文件覆盖

| 文件 | 关联 PR |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` | [#9711](https://github.com/NVIDIA/TensorRT-LLM/pull/9711) |
| `tensorrt_llm/serve/tool_parser/kimi_k2_tool_parser.py` | [#9830](https://github.com/NVIDIA/TensorRT-LLM/pull/9830) |
| `tensorrt_llm/_torch/models/modeling_deepseekv3.py` | [#11777](https://github.com/NVIDIA/TensorRT-LLM/pull/11777), [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) |
| `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_kimi_k2.py` | [#11780](https://github.com/NVIDIA/TensorRT-LLM/pull/11780) |
| `examples/auto_deploy/model_registry/configs/kimi_k2.yaml` | [#11780](https://github.com/NVIDIA/TensorRT-LLM/pull/11780) |
| `tensorrt_llm/_torch/models/modeling_kimi_k25.py` | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788), [#14379](https://github.com/NVIDIA/TensorRT-LLM/pull/14379), [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180) |
| `tensorrt_llm/llmapi/reasoning_parser.py` | [#13801](https://github.com/NVIDIA/TensorRT-LLM/pull/13801) |
| `tensorrt_llm/_torch/modules/embedding.py` | [#15233](https://github.com/NVIDIA/TensorRT-LLM/pull/15233) |
| `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py` | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_*.yaml` | [#15443](https://github.com/NVIDIA/TensorRT-LLM/pull/15443) |

## PR 覆盖总览

- 本轮审计 PR 数: 10
- 文件反查命令: `git log --name-only -- <model-files>`
- diff 来源: `gh pr diff` / GitHub Pull Request files API patch，本地缓存 `/tmp/model_pr_diffs/tensorrt_llm/pr*.diff`
- 已读 patch 行数: 8,414
- TensorRT-LLM Kimi 关键形态: Blackwell/GB200 deployment guide、OpenAI tool parser、K2.5 text NVFP4、AutoDeploy Kimi K2.5、multimodal vision/video path、reasoning parser、speculative/guided decoding、rejection sampling embedding mask、NIXL disaggregated perf lane。

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-12-05 | [#9711](https://github.com/NVIDIA/TensorRT-LLM/pull/9711) | merged | Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell | deployment guide |
| 2025-12-12 | [#9830](https://github.com/NVIDIA/TensorRT-LLM/pull/9830) | merged | Support tool parser for Kimi K2 | OpenAI server/tool parser |
| 2026-03-04 | [#11777](https://github.com/NVIDIA/TensorRT-LLM/pull/11777) | merged | Add Kimi-K2.5 text model support (NVFP4) | `modeling_deepseekv3.py`, accuracy tests |
| 2026-03-05 | [#11780](https://github.com/NVIDIA/TensorRT-LLM/pull/11780) | merged | AutoDeploy onboarding agent + Kimi K2.5 AD modeling code | AutoDeploy Kimi model/config/tests |
| 2026-05-11 | [#13801](https://github.com/NVIDIA/TensorRT-LLM/pull/13801) | merged | Add reasoning parser for kimi-k2.5 and enable auto flow | command/reasoning parser |
| 2026-05-14 | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) | merged | Add Kimi K2.5 multimodal vision support | `modeling_kimi_k25.py`, multimodal eval/tests |
| 2026-05-22 | [#14379](https://github.com/NVIDIA/TensorRT-LLM/pull/14379) | merged | Fix Kimi_k25 with spec dec | `modeling_kimi_k25.py` |
| 2026-06-17 | [#15233](https://github.com/NVIDIA/TensorRT-LLM/pull/15233) | merged | Fix embedding vocab mask for rejection sampling in Kimi-K2.5 | `embedding.py` |
| 2026-06-23 | [#15443](https://github.com/NVIDIA/TensorRT-LLM/pull/15443) | merged | Un-waive K2.5 Thinking FP4 disagg-NIXL tests | waives and perf-sanity YAML |
| 2026-06-25 | [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180) | merged | Add necessary methods for guided decoding in Kimi K2.5 | `modeling_kimi_k25.py` |

## 逐 PR diff 审计卡

### PR #9711 - Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/9711
- 状态/时间: merged / 2025-12-05
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+309/-0，本地 patch 534 行。
- 动机: 给 Kimi K2 Thinking NVFP4 在 DGX B200 与 GB200 NVL72 上的部署提供官方路径。
- 实现要点: 文档加入 Docker build/run、`trtllm-serve nvidia/Kimi-K2-Thinking-NVFP4`、8-way EP/attention DP、SLURM wide EP 和 disaggregated serving 示例。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+trtllm-serve nvidia/Kimi-K2-Thinking-NVFP4 \
+--extra_llm_api_options
```

- 已读文件: `deployment-guide-for-kimi-k2-thinking-on-trtllm.md`
- 验证与风险: SOTA loop 里 TensorRT-LLM Kimi lane 要标注 Blackwell/GB200 和 disagg 形态；文档命令不是 engine 后端的等价物。

### PR #9830 - Support tool parser for Kimi K2

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/9830
- 状态/时间: merged / 2025-12-12
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+374/-1，本地 patch 528 行。
- 动机: Kimi K2 OpenAI-compatible serving 需要正确解析 tool call，否则 agentic workload 的功能/评分会偏离。
- 实现要点: 新增 `kimi_k2_tool_parser.py`，接入 OpenAI server postprocess handler 和 parser factory，并补工具调用测试。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+from .kimi_k2_tool_parser import KimiK2ToolParser
+class KimiK2ToolParser(BaseToolParser):
+        "kimi_k2": KimiK2ToolParser,
```

- 已读文件: `openai_server.py`, `postprocess_handlers.py`, `kimi_k2_tool_parser.py`, `tool_parser_factory.py`, tests
- 验证与风险: 对 agentic benchmark，tool parser 是 correctness surface；不能只比较 tokens/s。

### PR #11777 - Add Kimi-K2.5 text model support (NVFP4)

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11777
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+96/-0，本地 patch 532 行。
- 动机: TensorRT-LLM PyTorch backend 需要把 Kimi-K2.5 text NVFP4 接入已有 DeepSeekV3-style runtime。
- 实现要点: 在 `modeling_deepseekv3.py` 中适配 Kimi K2.5 text model，并新增 accuracy refs/tests。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+MODEL_NAME = "moonshotai/Kimi-K2.5"
+quant_algo: NVFP4
```

- 已读文件: `modeling_deepseekv3.py`, accuracy tests/refs
- 验证与风险: Kimi K2.5 text 与 Kimi multimodal 不是同一验证面；benchmark 记录要区分 text-only 和 VLM。

### PR #11780 - AutoDeploy onboarding agent + Kimi K2.5 AD modeling code

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11780
- 状态/时间: merged / 2026-03-05
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 9 个文件，+2190/-9，本地 patch 2,807 行。
- 动机: Kimi K2.5 需要 AutoDeploy modeling code，而不是仅依赖通用 PyTorch model wrapper。
- 实现要点: 新增 `modeling_kimi_k2.py`、registry config、MLA custom ops 和 AutoDeploy tests，并加入 agent/scaffolding 文件。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+model_factory: KimiK2ForCausalLM
+flashinfer_mla
```

- 已读文件: agent scaffold files, `configs/kimi_k2.yaml`, `flashinfer_mla.py`, `torch_backend_mla.py`, `modeling_kimi_k2.py`, AD tests
- 验证与风险: TensorRT-LLM Kimi 竞品路径可能是 AutoDeploy，而不是普通 model wrapper；SGLang loop 要记录 backend/API 入口。

### PR #13801 - Add reasoning parser for Kimi-K2.5 and enable auto flow

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13801
- 状态/时间: merged / 2026-05-11
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+5/-1，本地 patch 47 行。
- 动机: Kimi-K2.5 thinking output 需要自动选择正确 reasoning parser。
- 实现要点: `commands/serve.py` 的 auto-detect hint 加入 `kimi_k2/kimi_k25`，`reasoning_parser.py` 注册 `kimi_k25` 且 `reasoning_at_start=True`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+"kimi_k25": "kimi_k25",
+@register_reasoning_parser("kimi_k25", reasoning_at_start=True)
```

- 已读文件: `commands/serve.py`, `llmapi/reasoning_parser.py`
- 验证与风险: AIME/agentic 输出解析不同会改变分数；公平对比必须记录 reasoning parser。

### PR #12788 - Add Kimi K2.5 multimodal vision support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12788
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 12 个文件，+2912/-64，本地 patch 3,536 行。
- 动机: Kimi K2.5 需要 text/image/video multimodal path，包括 vision encoder、processor、hashing fallback 和 multimodal eval。
- 实现要点: 新增 `KimiK25ForConditionalGeneration`、`KimiK25VisionModel`、input processor、vision attention/projector 结构，接入 multimodal eval/test。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+@register_auto_model("KimiK25ForConditionalGeneration")
+class KimiK25ForConditionalGeneration(PreTrainedModel):
+    "video_placeholder": "<|kimi_k25_video_placeholder|>",
```

- 已读文件: `modeling_kimi_k25.py`, `modeling_deepseekv3.py`, eval wrappers, multimodal tests, `test_modeling_kimi_k25.py`
- 验证与风险: 多模态 Kimi 比 text-only 多了 vision encoder、placeholder expansion、hashing fallback；profile 要分 stage。

### PR #14379 - Fix Kimi_k25 with spec dec

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14379
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+53/-35，本地 patch 187 行。
- 动机: Kimi K2.5 在 speculative decoding 下 multimodal params 和 `lm_head` 访问不完整。
- 实现要点: `forward` 显式接收 `multimodal_params`，按 `attn_metadata.num_contexts` 切 context params，并增加 `lm_head` 代理方法。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+multimodal_params: Optional[List[MultimodalParams]] = None
+def lm_head(self): return self.llm.lm_head
```

- 已读文件: `tensorrt_llm/_torch/models/modeling_kimi_k25.py`
- 验证与风险: SGLang 对标 speculative decoding 时，要确认 multimodal context-only 路径是否正确处理。

### PR #15233 - Fix embedding vocab mask for rejection sampling in Kimi-K2.5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15233
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+15/-8，本地 patch 129 行。
- 动机: FlashInfer rejection kernel 会为未接受 token pad 非 vocab 值，如果 embedding mask 不 clamp，会在 Kimi-K2.5 rejection sampling 中越界。
- 实现要点: 在 `pre_comm_embedding_ops` 中先 mask/clamp 输入，再做 `F.embedding`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+# flashinfer's rejection kernel pads non-accepted tokens
+        input_, input_mask = get_masked_input_and_mask(
+            input_,
+            0,
+            weight.shape[0],
+        )
```

- 已读文件: `tensorrt_llm/_torch/modules/embedding.py`
- 验证与风险: speculative/rejection sampling 的 correctness 风险在 embedding 前处理，profile 表很难直接暴露。

### PR #15443 - Un-waive K2.5 Thinking FP4 disagg-NIXL tests

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15443
- 状态/时间: merged / 2026-06-23
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+2/-3，本地 patch 86 行。
- 动机: Kimi K2.5 Thinking FP4 disaggregated NIXL e2e/gen_only lane 已足够稳定，可以从 waive 列表移除并拉长 KV transfer timeout。
- 实现要点: 删除 waives 中三条 Kimi NIXL skip，并在 perf-sanity YAML 设置 `kv_transfer_timeout_ms: 600000`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+kv_transfer_timeout_ms: 600000
```

- 已读文件: `waives.txt`, `gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml`
- 验证与风险: disagg/NIXL lane 与单机 serving 完全不同；SOTA loop 要单独建 workload bucket。

### PR #15180 - Add necessary methods for guided decoding in Kimi K2.5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15180
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+3/-0，本地 patch 28 行。
- 动机: Kimi K2.5 wrapper 缺少 guided decoding 需要的代理方法。
- 实现要点: 在 `KimiK25ForConditionalGeneration` 上透传 `set_guided_decoder` 到内部 LLM。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+def set_guided_decoder(self, *args, **kwargs):
+    return self.llm.set_guided_decoder(*args, **kwargs)
```

- 已读文件: `tensorrt_llm/_torch/models/modeling_kimi_k25.py`
- 验证与风险: guided decoding 会改变 decode control flow；benchmark 需要标注是否启用。
