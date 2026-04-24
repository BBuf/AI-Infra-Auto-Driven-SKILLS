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
