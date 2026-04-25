# SGLang Ernie4.5 / Ernie4.5-VL 支持与 PR 历史

本文记录 SGLang 在提交 `c122d343adb969cd9bbd1af2ca86727a11be3845`
附近的 Ernie4.5 多模态支持。

- 状态: 当前 mainline 已支持

## 核心结论

- Ernie4.5-VL 不是薄别名，而是带独立 runtime 和 processor 的原生落地。
- 后续最重要的优化工作基本都集中在 vision rotary 路径:
  先是 fused Triton kernel，再是 cos/sin cache 复用。

## 主要代码面

- `sglang/python/sglang/srt/models/ernie45_vl.py`
- `sglang/python/sglang/srt/models/ernie45_moe_vl.py`
- `sglang/python/sglang/srt/multimodal/processors/ernie45_vl.py`
- `sglang/python/sglang/srt/layers/rotary_embedding.py`

## 已合入 PR

- [#15679](https://github.com/sgl-project/sglang/pull/15679)
  `Add Ernie4.5 VL model support`
  已审 diff: `6` 个文件，`2072` 行新增。
  它加入原生 Ernie4.5-VL / MoE-VL runtime、processor 和多模态注册。
- [#18856](https://github.com/sgl-project/sglang/pull/18856)
  `Optimize Ernie4.5-VL rotary embedding with fused triton kernel`
  已审 diff: `1` 个文件，`268` 行新增，`3` 行删除。
  它为 Ernie4.5 的 `(h, w, t)` 布局加入 fused Triton Q/K rotary kernel。
- [#19743](https://github.com/sgl-project/sglang/pull/19743)
  `Support cos sin cache for Ernie4.5-VL`
  已审 diff: `1` 个文件，`34` 行新增，`12` 行删除。
  它把 vision tower 改成复用 `get_rope(...).get_cos_sin(...)`。

## 当前结论

如果 Ernie4.5-VL 回归了，先确认当前走的是哪条 rotary 路径，再去改整套模型。
这个家族很多 correctness / performance 变化都集中在 vision rotary 处理。
