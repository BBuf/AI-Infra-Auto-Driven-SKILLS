# TokenSpeed Kimi 模型 PR 优化历史

## 2026-06-26 最新源码扫描

已按 TokenSpeed 上游 `lightseekorg/tokenspeed@5aedf69d6b476baa65571011de6ea60fd5a238a8` 重新扫描本文下方列出的 tracked files。
文件级匹配使用 GitHub mirror 的 `git log --name-only`；PR 标题、链接和合并时间通过 GitHub GraphQL Pull Request API 批量复核。上一时效锚点：`2026-06-26`。

结果：发现 2 个额外 PR-numbered merge 触及 tracked files，但尚未提升为下方完整逐 PR diff audit card。此节只作为 freshness index；需要引用实现细节时，仍应先人工阅读 PR diff 再补完整卡片。

| 合并日期 | PR | 标题 | 命中的 tracked files |
| --- | --- | --- | --- |
| 2026-06-26 | [#519](https://github.com/lightseekorg/tokenspeed/pull/519) | feat: distributed argmax for EAGLE greedy sampling | `logits_processor.py` |
| 2026-06-25 | [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | perf(kernel): optimize Qwen vision QKV rotary layout | `qkv_rotary.py` |

## 2026-06-26 PR 补漏复核

已按 TokenSpeed 上游 `HEAD@5aedf69d6b476baa65571011de6ea60fd5a238a8` 复核。本文覆盖 Kimi K2.5/K2.x 相关的 merged PR，并采用 SGLang 风格的时间线和逐 PR diff 审计卡。

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
