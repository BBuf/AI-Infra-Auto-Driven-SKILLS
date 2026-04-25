# SGLang Mixtral Quark INT4-FP8 MoE 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `bca3dd958`（2026-04-24）整理。

范围：`mistralai/Mixtral-8x7B-Instruct-v0.1` 在 AMD-only `quark_int4fp8_moe` online MoE quantization 下的支持和验证。

## 结论

Mixtral 路线验证的是 SGLang `quark_int4fp8_moe` 量化方法：高精度 MoE expert 权重加载后在线量化成 packed INT4，并在 ROCm 上用 FP8 风格 MoE math 执行。当前 CI 覆盖是 `test/registered/quant/test_int4fp8_moe.py`。

## 已阅读 diff 的 PR 卡片

### #7392 - 在 ROCm 上加入 quark_int4fp8_moe online quantization

- 链接：https://github.com/sgl-project/sglang/pull/7392
- 状态：已合入，`2026-01-14T09:44:41Z`
- Diff 覆盖：`4055` 行，`12` 个文件。
- 新增 quantization config、INT4/FP8 工具、server flag 注册、文档和最初的 Mixtral GSM8K 回归测试。

### #17116 - 把 AMD int4fp8 MoE 测试迁移到 registered CI

- 链接：https://github.com/sgl-project/sglang/pull/17116
- 状态：已合入，`2026-01-19T16:07:39Z`
- Diff 覆盖：`902` 行，`19` 个文件。
- 将测试迁移到 registered AMD CI，并把 suite registration 纳入支持面。

### #23455 - 恢复 int4fp8 MoE 可直接执行测试

- 链接：https://github.com/sgl-project/sglang/pull/23455
- 状态：已合入，`2026-04-23T05:28:21Z`
- Diff 覆盖：`291` 行，`2` 个文件。
- 恢复 `test/registered/quant/test_int4fp8_moe.py`，补 `unittest.main()` 入口，并保留 GSM8K 分数大于 `0.56` 的门槛。

完整 PR dossier：`skills/model-optimization/sglang/sglang-mixtral-quark-int4fp8-moe-optimization/references/pr-history.md`。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Mixtral Quark INT4-FP8 MoE`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-06-20 | [#7392](https://github.com/sgl-project/sglang/pull/7392) | merged | [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm | MoE/router, quantization, tests/benchmarks, docs/config | `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`, `python/sglang/srt/layers/int4fp8_utils.py`, `test/srt/test_int4fp8_moe.py` |
| 2026-01-15 | [#17116](https://github.com/sgl-project/sglang/pull/17116) | merged | [AMD CI] Migrate and Add More Testcases | attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks | `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py` |
| 2026-04-22 | [#23455](https://github.com/sgl-project/sglang/pull/23455) | merged | [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry | MoE/router, quantization, multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py` |

### 逐 PR 代码 diff 阅读记录

### PR #7392 - [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm

- 链接：https://github.com/sgl-project/sglang/pull/7392
- 状态/时间：`merged`，created 2025-06-20, merged 2026-01-14；作者 `fxmarty-amd`。
- 代码 diff 已读范围：`12` 个文件，`+615/-15`；代码面：MoE/router, quantization, tests/benchmarks, docs/config；关键词：fp8, quant, moe, config, cache, spec, triton, attention, awq, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0 (443 lines); hunk: +import logging; 符号: tqdm_reset_no_print, QuarkInt4Fp8Config, for, __init__
  - `python/sglang/srt/layers/int4fp8_utils.py` added +73/-0 (73 lines); hunk: +"""; 符号: quantize_fp8_scale_tensorwise, quantize_int4_scale_columnwise, pack_int4_to_int32
  - `test/srt/test_int4fp8_moe.py` added +55/-0 (55 lines); hunk: +from types import SimpleNamespace; 符号: TestMixtralAccuracy, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/model_loader/weight_utils.py` modified +16/-14 (30 lines); hunk: ci_download_with_validation_and_retry,; def filter_files_not_needed_for_inference(hf_weights_files: List[str]) -> List[s; 符号: filter_files_not_needed_for_inference, np_cache_weights_iterator, np_cache_weights_iterator, safetensors_weights_iterator
  - `docs/advanced_features/quantization.md` modified +8/-0 (8 lines); hunk: python3 -m sglang.launch_server \; python3 -m sglang.launch_server \
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`, `python/sglang/srt/layers/int4fp8_utils.py`, `test/srt/test_int4fp8_moe.py`；patch 关键词为 fp8, quant, moe, config, cache, spec。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`, `python/sglang/srt/layers/int4fp8_utils.py`, `test/srt/test_int4fp8_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17116 - [AMD CI] Migrate and Add More Testcases

- 链接：https://github.com/sgl-project/sglang/pull/17116
- 状态/时间：`merged`，created 2026-01-15, merged 2026-01-19；作者 `bingxche`。
- 代码 diff 已读范围：`19` 个文件，`+310/-66`；代码面：attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks；关键词：test, cache, cuda, fp8, kv, moe, attention, config, lora, topk。
- 代码 diff 细节：
  - `.github/workflows/pr-test-amd.yml` modified +81/-47 (128 lines); hunk: jobs:; jobs:
  - `test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0 (116 lines); hunk: +import unittest; 符号: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/test_deepseek_v3_basic.py` added +84/-0 (84 lines); hunk: +import unittest; 符号: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +0/-8 (8 lines); hunk: # TestFile("lora/test_lora_backend.py", 99), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107; # TestFile("test_vision_chunked_pre
  - `test/registered/core/test_deterministic.py` modified +5/-1 (6 lines); hunk: import unittest; def get_server_args(cls):; 符号: TestFlashinferDeterministic, get_server_args, TestFa3Deterministic
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`；patch 关键词为 test, cache, cuda, fp8, kv, moe。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23455 - [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry

- 链接：https://github.com/sgl-project/sglang/pull/23455
- 状态/时间：`merged`，created 2026-04-22, merged 2026-04-23；作者 `bingxche`。
- 代码 diff 已读范围：`2` 个文件，`+220/-0`；代码面：MoE/router, quantization, multimodal/processor, tests/benchmarks；关键词：test, attention, config, fp8, moe, processor, quant, triton。
- 代码 diff 细节：
  - `test/registered/amd/test_zimage_turbo.py` added +156/-0 (156 lines); hunk: +"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""; 符号: _save_image_and_write_summary, _compute_clip_score, TestZImageTurboAMD, teardown_class
  - `test/registered/quant/test_int4fp8_moe.py` added +64/-0 (64 lines); hunk: +from types import SimpleNamespace; 符号: TestMixtralAccuracy, setUpClass, tearDownClass, test_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py`；patch 关键词为 test, attention, config, fp8, moe, processor。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
