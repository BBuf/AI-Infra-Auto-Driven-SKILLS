# TokenSpeed Kimi 模型 PR 优化历史

## 2026-07-28 源码 head 刷新

已复核 TokenSpeed 上游 main：
`lightseekorg/tokenspeed@e41aa8b1609a9412d7ed26aa56d910828607950f`。
已读完上一 head `d73bf0454422092f306d5575e803a08fd35ac41c`
之后的 2-commit 完整增量。

结果：PR #821 新增 Kimi K3 FlatKV/KDA/MLA 部署契约，并在保留硬件与验证限制的
前提下提升为源码指南；PR #823 只增加 README 新闻链接。本文继续覆盖 DP + EAGLE3
collective-size hang、Kimi incremental DFlash capture，以及 Kimi-K2.7 EAGLE3.1
模型语义。

| 合并日期 | PR | Runtime 信号 |
| --- | --- | --- |
| 2026-07-27 | [#821](https://github.com/lightseekorg/tokenspeed/pull/821) | Kimi K3 部署契约 |
| 2026-07-07 | [#596](https://github.com/lightseekorg/tokenspeed/pull/596) | DP + EAGLE3 mixed-step hang |
| 2026-07-25 | [#795](https://github.com/lightseekorg/tokenspeed/pull/795) | Kimi-K2.7 EAGLE3.1 |
| 2026-07-26 | [#797](https://github.com/lightseekorg/tokenspeed/pull/797) | incremental DFlash capture |

## 2026-06-27 PR 补漏复核

已按 TokenSpeed 上游 `HEAD@d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8` 复核。本文覆盖 Kimi K2.5/K2.x 相关的 merged PR，并采用 SGLang 风格的时间线和逐 PR diff 审计卡。

本轮筛选规则：标题/文件命中 `Kimi`、`kimi_k25`、`K2.5`、`NVFP4`、`MXFP4`、`MXINT4`、`lm_head`、`top_k/top_p`、`InstantTensor`、`OCR`、`FA4`、`vision`、`MLA` 的 merged PR；过滤纯格式化和不影响模型/runtime/CI lane 的基础设施 PR。

## 模型实现文件覆盖

| 文件 | 关联 PR |
| --- | --- |
| `python/tokenspeed/runtime/models/kimi_k25.py` | [#354](https://github.com/lightseekorg/tokenspeed/pull/354), [#418](https://github.com/lightseekorg/tokenspeed/pull/418), [#454](https://github.com/lightseekorg/tokenspeed/pull/454), [#477](https://github.com/lightseekorg/tokenspeed/pull/477) |
| `python/tokenspeed/runtime/layers/logits_processor.py` | [#126](https://github.com/lightseekorg/tokenspeed/pull/126) |
| `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/lm_head_gemm.py` | [#126](https://github.com/lightseekorg/tokenspeed/pull/126) |
| `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/fused_topk_topp/*` | [#184](https://github.com/lightseekorg/tokenspeed/pull/184) |
| `python/tokenspeed/runtime/layers/moe/weights/mxint4.py` | [#444](https://github.com/lightseekorg/tokenspeed/pull/444) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer/trtllm_mxint4.py` | [#444](https://github.com/lightseekorg/tokenspeed/pull/444) |
| `python/tokenspeed/runtime/layers/quantization/*mxfp4*` | [#454](https://github.com/lightseekorg/tokenspeed/pull/454) |
| `python/tokenspeed/runtime/model_loader/*` | [#418](https://github.com/lightseekorg/tokenspeed/pull/418) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/qkv_rotary.py` | [#477](https://github.com/lightseekorg/tokenspeed/pull/477), [#482](https://github.com/lightseekorg/tokenspeed/pull/482) |
| `test/ci/eval/kimi-k2.5-*.yaml` | [#29](https://github.com/lightseekorg/tokenspeed/pull/29), [#253](https://github.com/lightseekorg/tokenspeed/pull/253), [#476](https://github.com/lightseekorg/tokenspeed/pull/476), [#482](https://github.com/lightseekorg/tokenspeed/pull/482) |

## PR 覆盖总览

- 本轮审计 PR 数: 10
- 文件反查命令: `git log --name-only -- <model-files>`
- diff 来源: `gh pr diff` / GitHub Pull Request files API patch，本地缓存 `/tmp/model_pr_diffs/tokenspeed/pr*.diff`
- 已读 patch 行数: 11,975
- TokenSpeed Kimi 关键形态: K2.5 agentic/OCR eval lane、fused lm_head GEMM、TopK+TopP renorm、InstantTensor loader、MXINT4/MXFP4 MoE/quantization、FA4 multimodal attention。

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-08 | [#29](https://github.com/lightseekorg/tokenspeed/pull/29) | merged | Add Kimi K2.5 agentic perf CI task | perf YAML, CI pipeline |
| 2026-05-13 | [#126](https://github.com/lightseekorg/tokenspeed/pull/126) | merged | perf(K2.5): Optimize lm_head | `logits_processor.py`, CUDA `lm_head_gemm` |
| 2026-05-20 | [#184](https://github.com/lightseekorg/tokenspeed/pull/184) | merged | perf(K2.5): optimize top_k_renorm_prob + top_p_renorm_prob | fused sampling CUDA, backend/server args |
| 2026-05-28 | [#253](https://github.com/lightseekorg/tokenspeed/pull/253) | merged | ci(eval): add Kimi-K2.5-NVFP4 ocr_bench task | OCR eval YAML |
| 2026-06-15 | [#418](https://github.com/lightseekorg/tokenspeed/pull/418) | merged | Add InstantTensor weight loader | loader, weight utils, `kimi_k25.py`, docs/CI |
| 2026-06-14 | [#444](https://github.com/lightseekorg/tokenspeed/pull/444) | merged | feat(moe): add trtllm mxint4 MoE path for Kimi-K2.x | MXINT4 weights and FlashInfer TRT-LLM MoE op |
| 2026-06-16 | [#454](https://github.com/lightseekorg/tokenspeed/pull/454) | merged | [AMD] Support Kimi K2.5 MXFP4 serving | MXFP4 layers, dense path, MLA backend, Kimi model |
| 2026-06-19 | [#477](https://github.com/lightseekorg/tokenspeed/pull/477) | merged | perf(kernel): Optimize Kimi Vision FA4 QKV + RoPE | Kimi model, mm attention, packed complex rotary |
| 2026-06-19 | [#482](https://github.com/lightseekorg/tokenspeed/pull/482) | merged | ci: use FA4 mm attention for Kimi OCR eval | OCR eval YAML |
| 2026-06-26 | [#476](https://github.com/lightseekorg/tokenspeed/pull/476) | merged | Add AMD Kimi MXFP4 CI job | AMD eval YAML, MLA metadata unit test |

## 逐 PR diff 审计卡

### PR #821 - 新增 Kimi K3 部署 recipe

- 链接: https://github.com/lightseekorg/tokenspeed/pull/821
- 状态/时间: merged / 2026-07-27
- 反查来源: 最终上游提交
  `55a8390007e5ace17919d76e5cfaef0c68c79e25`；已读完 2-commit 增量和
  recipe 完整 diff。
- 代码 diff 已读范围: 1 file，+117/-0。
- 动机: 明确 Kimi K3 的 runtime/packaging 约束，避免把 K2.5 flags 当成可互换配置。
- 实现要点: 要求 FlatKV，记录 vendor-neutral KDA dispatch 与 NVIDIA FLA-derived /
  AMD native state layout、MLA backend 选择，以及 NVIDIA B300、AMD gfx950 分离命令。
- 代码 diff 细节: recipe 新增 FlatKV build/preflight、flattened checkpoint、
  writable module cache、NVIDIA `tokenspeed-situ` + Triton fallback 和 AMD Gluon 路径。
- 关键代码摘录:

```diff
+- K3 is FlatKV-only. Build the `tokenspeed_scheduler` extension with
+  `-DTOKENSPEED_FLAT_KVCACHE=ON`
+tokenspeed serve moonshotai/Kimi-K3 \
+  --kv-cache-dtype fp8 \
+  --tensor-parallel-size 8
```

- 已读文件: `docs/recipes/models.md`；后续 README commit 只链接 K3 公告。
- 验证与风险: NVIDIA 依赖 B300/CUDA 13 `tokenspeed-situ` sidecar，其他平台回退
  Triton；checkpoint 需 flattened、remote-code cache 需可写，默认 FP8 KV scale
  可能影响精度。这不是跨框架性能实测。

### PR #596 - 修复 Kimi DP EAGLE3 mixed-step hang

- 链接: https://github.com/lightseekorg/tokenspeed/pull/596
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 184 行 diff，6 个文件，+27/-30。
- 动机: 同一 scheduler step 中 DP rank 混合 EXTEND 与 DECODE 时，active 与 idle rank 可能为 EAGLE3 首次 catch-up collective 计算出不同 row 数，最终 hang。
- 实现要点: 把首步激活缩减定义为 active EAGLE 与 idle replay 共享的 draft-model capability，为 Kimi/DeepSeek、Llama、Qwen3.5 draft model 标记该能力，并禁止 fused `lm_head_gemm` 对 zero-token 发起 kernel。
- 代码 diff 细节: 用 `draft_first_step_reduce_for_catchup` 替换硬编码 class check，让 collective sizing 跟随模型行为而不是模型列表。
- 关键代码摘录:

```diff
+def draft_model_reduces_first_step_catchup(draft_model) -> bool:
+    return bool(getattr(draft_model, "draft_first_step_reduce_for_catchup", False))
+draft_first_step_reduce = step_idx == 0 and (
+    all_decode_or_idle or draft_reduces_first_step_catchup)
```

- 已读文件: runtime：`execution/drafter/eagle.py`、`execution/model_executor.py`、`models/{deepseek_v3,llama_eagle3,qwen3_5_nextn}.py`、`lm_head_gemm.py`；没有新增独立测试文件。
- 验证与风险: mixed forward mode 下所有 rank 必须得到相同 collective row count；PR 报告修复后 DP8 + EAGLE3 AIME25 以 28/30 完成，zero-row fused lm-head routing 必须保持 no-op。

### PR #795 - 为 Kimi-K2.7 Code 支持 EAGLE3.1

- 链接: https://github.com/lightseekorg/tokenspeed/pull/795
- 状态/时间: merged / 2026-07-25
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 49 行 diff，1 个文件，+24/-0。
- 动机: Kimi-K2.7 EAGLE3.1 MLA speculator 定义了逐 FC 输入的归一化与可选 normalized auxiliary output，而共享 DeepSeek-style drafter 尚未实现。
- 实现要点: `fc_norm` 开启时为每个拼接 FC 输入 chunk 建立 RMSNorm，在 projection 前分别归一化，并用 `norm_output` 控制 auxiliary hidden-state 输出。
- 代码 diff 细节: 所有变化都在 `Eagle3MlaModel` 内受 config gate 控制，不改变旧 EAGLE checkpoint。
- 关键代码摘录:

```diff
+if self.fc_norm is not None:
+    chunks = hidden_states.chunk(self.num_fc_input_dim, dim=-1)
+    hidden_states = torch.cat(
+        [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)], dim=-1)
```

- 已读文件: runtime：`python/tokenspeed/runtime/models/deepseek_v3.py`；验证证据：`nvidia/Kimi-K2.7-Code-NVFP4` 的 PR benchmark 与 launch recipe。
- 验证与风险: `fc_norm`/`norm_output` 必须与 checkpoint config 绑定；PR 报告 4xGB200 1-3-4 配置各类别 1.36x-1.91x，但不能外推到不同 acceptance length 或 serving shape。

### PR #797 - 支持 Kimi incremental DFlash capture

- 链接: https://github.com/lightseekorg/tokenspeed/pull/797
- 状态/时间: merged / 2026-07-26
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 166 行 diff，4 个文件，+61/-7。
- 动机: Kimi 把 DFlash capture 委托给 DeepSeek-style language model 时丢失了 executor 所需的 incremental projection callback 与 slot buffer，导致开启 incremental projection 后无法启动。
- 实现要点: 让 `KimiK25ForConditionalGeneration.set_dflash_layers_to_capture` 透传 callback 与 slot buffer，保存 layer-to-slot map，每层完成时把 hidden state 拷入相应 slot 并立即调用 incremental projection callback。
- 代码 diff 细节: 模型用 `_dflash_incr_active` 控制生命周期；CI 还让 Slurm server startup timeout 与 readiness 对齐，避免长时间 Kimi 启动被另一套超时提前终止。
- 关键代码摘录:

```diff
+self.model._dflash_capture_idx_map = {
+    layer_idx: i for i, layer_idx in enumerate(sorted(self.model.layers_to_capture))
+}
+self.model._dflash_incremental_callback(capture_idx, num_tokens)
```

- 已读文件: runtime：`models/deepseek_v3.py`、`models/kimi_k25.py`；测试/CI：`test/ci_system/{pipeline,test_pipeline}.py`。
- 验证与风险: callback 顺序、slot 容量、CUDA stream 生命周期和 `_dflash_incr_active` reset 必须与 executor 一致；PR 记录了 syntax/pre-commit 检查和 live B200 验证 lane。

### PR #29 - Add Kimi K2.5 agentic perf CI task

- 链接: https://github.com/lightseekorg/tokenspeed/pull/29
- 状态/时间: merged / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+387/-3，本地 patch 650 行。
- 动机: 把 `nvidia/Kimi-K2.5-NVFP4` 的 agentic workload 变成 perf CI，而不是只靠手工压测。
- 实现要点: 新增 Kimi K2.5 agentic perf YAML，服务端使用 `python3 -m tokenspeed.api_server`，配套 `tokenspeed_mla`、NVFP4、speculative draft 和 EvalScope agentic workload。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--attention-backend tokenspeed_mla
+--quantization nvfp4
```

- 已读文件: `.github/workflows/pr-test.yml`, `test/ci/perf/kimi-k2.5-nvfp4-evalscope-agentic.yaml`, CI pipeline helper
- 验证与风险: SOTA loop 里要把 agentic perf lane 和公共 synthetic workload 分开记录；TokenSpeed 的领先可能来自 workload/state 管理，而不是单个 kernel。

### PR #126 - perf(K2.5): Optimize lm_head

- 链接: https://github.com/lightseekorg/tokenspeed/pull/126
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 6 个文件，+1173/-3，本地 patch 1,246 行。
- 动机: Kimi K2.5 decode 末端 `lm_head` 大 GEMM 显著影响 TPOT；PR 用专用 CUDA kernel 替换一般 `torch.matmul` 路径。
- 实现要点: `LogitsProcessor` 里按 `model_type == "kimi_k2"` gate fused path；新增 `lm_head_gemm.cu`、binding 和 Python wrapper，并保留 shape 不匹配时 fallback。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+self._use_fused_lm_head = getattr(self.config, "model_type", None) == "kimi_k2"
+logits = _lm_head_matmul(hidden_states, lm_head.weight)
```

- 已读文件: `logits_processor.py`, `lm_head_gemm.cu`, `lm_head_gemm_binding.cu`, `lm_head_gemm.py`, setup files
- 验证与风险: 对 SGLang Kimi/Qwen 类模型，`lm_head` 需要单独进 profiler 表；不能只优化 attention/MoE 后就宣布收敛。

### PR #184 - perf(K2.5): optimize top_k_renorm_prob + top_p_renorm_prob

- 链接: https://github.com/lightseekorg/tokenspeed/pull/184
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 8 个文件，+3104/-12，本地 patch 3,580 行。
- 动机: sampling backend 连续调用 `top_k_renorm_prob` 与 deterministic `top_p_renorm_prob`，在高并发 decode 尾部形成重复扫描和多 launch。
- 实现要点: 新增 fused TopK+TopP renorm CUDA 路径，按 `top_k` sentinel 切分分支，接入 `flashinfer_full.py` 和 `server_args.py` 的参数限制。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
-probs = top_k_renorm_prob(probs, top_ks)
-probs = top_p_renorm_prob(probs, top_ps, is_deterministic=True)
+probs = fused_topk_topp_renorm(probs, top_ks, top_ps)
```

- 已读文件: `fused_topk_topp/*`, `flashinfer_full.py`, `server_args.py`, sampling tests
- 验证与风险: 对 SGLang profiler 的采样阶段，若 top-k/top-p 占比高，应把 sampling kernel 当成主优化面；同时检查 `top_k < 128` 这类 kernel contract。

### PR #253 - ci(eval): add Kimi-K2.5-NVFP4 ocr_bench task

- 链接: https://github.com/lightseekorg/tokenspeed/pull/253
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+48/-0，本地 patch 72 行。
- 动机: Kimi K2.5 是强多模态模型，OCR benchmark 需要进入常规 eval lane。
- 实现要点: 新增 `kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`，复用 Kimi NVFP4 server config，只把 EvalScope dataset 切到 `ocr_bench`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--datasets ocr_bench
```

- 已读文件: `test/ci/eval/kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- 验证与风险: 多模态 SOTA loop 要保留 OCR lane；text-only benchmark 不能代表 Kimi K2.5 全部优化收益。

### PR #418 - Add InstantTensor weight loader

- 链接: https://github.com/lightseekorg/tokenspeed/pull/418
- 状态/时间: merged / 2026-06-15
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 25 个文件，+468/-60，本地 patch 1,373 行。
- 动机: Kimi K2.5 大模型启动和权重加载成本高，需要 `--load-format instanttensor` 这样的专用 loader。
- 实现要点: 新增 loader/weight utils 分支，接入 server args、Kimi model 权重路径和 CI eval 文档。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--load-format instanttensor
+        elif self.load_config.load_format == LoadFormat.INSTANTTENSOR:
+            weights_iterator = instanttensor_weights_iterator(hf_weights_files)
```

- 已读文件: `runtime/model_loader/loader.py`, `weight_utils.py`, `runtime/models/kimi_k25.py`, `runtime/utils/server_args.py`, docs and eval configs
- 验证与风险: 公平 benchmark 要记录冷启动/热启动边界；如果比较 steady-state TPOT，InstantTensor 不应混入吞吐结论。

### PR #444 - feat(moe): add trtllm mxint4 MoE path for Kimi-K2.x

- 链接: https://github.com/lightseekorg/tokenspeed/pull/444
- 状态/时间: merged / 2026-06-14
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 8 个文件，+469/-6，本地 patch 581 行。
- 动机: Kimi K2.x 需要 INT4 W4A16 group-32 MoE path，现有 NVFP4/MXFP4 path 不能覆盖 MXINT4 checkpoint。
- 实现要点: 新增 `create_mxint4_weight_pair`、quant config 识别和 `flashinfer_trtllm_mxint4` MoE process/apply op。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+from tokenspeed.runtime.layers.moe.weights.mxint4 import create_mxint4_weight_pair
+name="flashinfer_trtllm_mxint4_moe_apply"
```

- 已读文件: `expert.py`, `weights/mxint4.py`, quantization configs, `trtllm_mxint4.py`
- 验证与风险: SGLang 做 Kimi K2.x 量化路径对比时，要把 weight dtype、group size、activation dtype 和 FlashInfer TRT-LLM op 名称全部记录进结果表。

### PR #454 - [AMD] Support Kimi K2.5 MXFP4 serving

- 链接: https://github.com/lightseekorg/tokenspeed/pull/454
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 33 个文件，+1924/-142，本地 patch 3,856 行。
- 动机: AMD 平台需要 Kimi K2.5 MXFP4 serving，包括 attention、dense、MoE、quantization 和模型加载路径。
- 实现要点: 新增 MXFP4 quantization/layer/dense 支持，调整 MLA backend、Kimi model 和 AMD 相关测试。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--quantization mxfp4
+model_type == "kimi_k25"
```

- 已读文件: MXFP4 layers/quantization, dense kernels, attention backends, `runtime/models/kimi_k25.py`, tests
- 验证与风险: 这是跨硬件路径，不应把 AMD MXFP4 结论直接外推到 NVIDIA NVFP4；SOTA loop 应按 GPU/backend 分 lane。

### PR #477 - perf(kernel): Optimize Kimi Vision FA4 QKV + RoPE

- 链接: https://github.com/lightseekorg/tokenspeed/pull/477
- 状态/时间: merged / 2026-06-19
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 3 个文件，+195/-7，本地 patch 304 行。
- 动机: Kimi vision FA4 path 的 complex RoPE 和 packed QKV 拆分存在额外 layout 搬运。
- 实现要点: 新增 `packed_qkv_complex_rotary` Triton kernel，在 multimodal encoder attention 中走 packed complex-RoPE fast path。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+        if use_packed_qkv_complex_rotary:
+            q, k, v = packed_qkv_complex_rotary(
+def packed_qkv_complex_rotary(
```

- 已读文件: `mm_encoder_attention.py`, `runtime/models/kimi_k25.py`, `qkv_rotary.py`
- 验证与风险: SGLang Kimi/OCR VLM 优化需要关注 FA4 attention 前的 QKV/RoPE layout，而不是只看 FA4 主 kernel。

### PR #482 - ci: use FA4 mm attention for Kimi OCR eval

- 链接: https://github.com/lightseekorg/tokenspeed/pull/482
- 状态/时间: merged / 2026-06-19
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+1/-0，本地 patch 22 行。
- 动机: OCR eval 应覆盖新的 FA4 multimodal attention path，否则 #477 的 kernel 优化没有固定回归 lane。
- 实现要点: 在 Kimi OCR eval YAML 增加 `--mm-attention-backend fa4`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--mm-attention-backend fa4
```

- 已读文件: `test/ci/eval/kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- 验证与风险: 多模态 benchmark 需要显式记录 `mm-attention-backend`，否则同一个 Kimi checkpoint 会跑出不同 kernel 形态。

### PR #476 - Add AMD Kimi MXFP4 CI job

- 链接: https://github.com/lightseekorg/tokenspeed/pull/476
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 3 个文件，+138/-4，本地 patch 181 行。
- 动机: #454 接入 AMD MXFP4 后，需要持续验证 AIME25 eval 和 MLA metadata 兼容性。
- 实现要点: 新增 `kimi-k2.5-mxfp4-evalscope-aime25-amd.yaml`，并增加 `MLAAttnBackend` metadata unit test。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--model amd/Kimi-K2.5-MXFP4
+--quantization mxfp4
```

- 已读文件: `mla.py`, `kimi-k2.5-mxfp4-evalscope-aime25-amd.yaml`, `test_mla_verify_metadata.py`
- 验证与风险: 这条 lane 是 AMD 特化；SGLang 竞品比较时要在结果里标出硬件和 quantization，不要把它当成通用 Kimi lane。
