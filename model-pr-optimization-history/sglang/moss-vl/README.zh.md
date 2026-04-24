# SGLang Moss-VL 支持与优化时间线

范围：Moss-VL 原生 SGLang runtime、image/video processor、conversation template、多模态 scheduler metadata、cross-attention custom mask、flashinfer prefill 要求。

证据快照：SGLang `origin/main` `bca3dd958`（`2026-04-24`）。完整卡片见：`skills/model-optimization/sglang/sglang-moss-vl-optimization/references/pr-history.md`。

## 已阅读 Diff 的 PR

#23454 新增 Moss-VL runtime 支持。已完整阅读 `3397` 行 diff、`10` 个文件。该 PR 新增 `moss_vl.py`、`multimodal/processors/moss_vl.py`，在 `schedule_batch.py` 中加入 Moss-VL 多模态字段，注册 `moss-vl` conversation template，并在 `server_args.py` 中要求 flashinfer prefill。

核心契约：Moss-VL vision token 会按 frame 插入 separator；processor 的 frame visibility 会转换成 packed cross-attention custom mask；encoder-prefix placeholder token 在 text extend 前会被剥离。
