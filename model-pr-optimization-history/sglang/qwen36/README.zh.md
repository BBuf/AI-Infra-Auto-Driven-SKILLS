# SGLang Qwen3.6 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理。覆盖 Qwen3.6-35B-A3B、Qwen3.6-27B dense、FP8/BF16 部署、multimodal、thinking preservation、MTP、Mamba scheduler、Qwen3 reasoning parser 与 Qwen3-Coder tool parser。

结论：Qwen3.6 目前主要是文档/部署层和共享 hybrid Qwen runtime 的组合，不应先新增专属 runtime fork。排查要先看 Qwen3-Next/Qwen3.5/Qwen VLM/Qwen3-Coder parser 这些共享路径。

## 代码面

- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`
- `docs_new/docs.json`
- `python/sglang/srt/layers/quantization/utils.py`
- `python/sglang/srt/utils/offloader.py`
- `test/registered/unit/utils/test_offloader_tied_params.py`

## 手工 diff 审阅 PR 卡片

### PR #23034 - Qwen3.6 文档与部署生成器

- 链接：https://github.com/sgl-project/sglang/pull/23034
- 状态：已合入，`2026-04-17T05:33:34Z`
- Diff 覆盖：已用 `gh pr diff --patch` 拉取完整 diff，`7324` 行，`73` 个文件；手工重点阅读了 Qwen3.6、Qwen3.5 deployment、docs navigation、GLM/Qwen warning 相关 hunk。
- Motivation：当时 SGLang 没有 Qwen3.6 cookbook 页面和命令生成器，用户无法直接得到 reasoning parser、Qwen3-Coder tool parser、MTP、B200 attention backend、Mamba scheduler 的组合命令。这个 PR 同时把 Qwen cookbook 卡片跳转到最新 Qwen3.6 页面，并修正相邻 Qwen3.5/GLM-5 文档中的 FP8-KV warning 渲染。
- 关键实现：新增 `Qwen3.6.mdx` 和 `qwen36-deployment.jsx`，在 `docs_new/docs.json` 注册页面，并把 cookbook intro 的 Qwen 卡片从 Qwen3.5 改到 Qwen3.6。命令生成器在 MTP 打开时强制 Mamba V2，自动加 `SGLANG_ENABLE_SPEC_V2=1`，并组合 `--reasoning-parser qwen3`、`--tool-call-parser qwen3_coder`、EAGLE MTP 参数；B200 自动加 `--attention-backend trtllm_mha`。同 PR 也修了 Qwen3.5 的 MTP/Mamba 规则，说明 Qwen3.6 仍应复用 shared hybrid Qwen 逻辑。
- 关键代码片段：

```diff
+                      "cookbook/autoregressive/Qwen/Qwen3.6",
...
-    href="/cookbook/autoregressive/Qwen/Qwen3.5"
+    href="/cookbook/autoregressive/Qwen/Qwen3.6"
```

```jsx
commandRule: (value) => value === 'enabled' ? '--reasoning-parser qwen3' : null,
commandRule: (value) => value === 'enabled' ? '--tool-call-parser qwen3_coder' : null,
commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4' : null,
commandRule: (value) => value === 'v2' ? '--mamba-scheduler-strategy extra_buffer' : null,
```

```jsx
const mtpEnabled = values.speculative === 'enabled';
if (mtpEnabled) {
  return [
    { id: 'v1', label: 'V1', default: false, disabled: true },
    { id: 'v2', label: 'V2', default: true },
  ];
}
```

- 已读文件：`Qwen3.6.mdx`、`qwen36-deployment.jsx`、`qwen35-deployment.jsx`、`intro.mdx`、`docs.json`、`GLM-5.mdx`。
- 验证影响：Qwen3.6 文档改动后必须检查 BF16/FP8、有无 MTP、B200/H100/H200 组合命令；reasoning 和 tool-call parser 要一起测，MTP 命令必须保留 `SGLANG_ENABLE_SPEC_V2=1` 与 Mamba V2。

### PR #23467 - FP8 modules_to_not_convert 边界匹配

- 链接：https://github.com/sgl-project/sglang/pull/23467
- 状态：已合入，`2026-04-22T14:16:22Z`
- Diff 覆盖：已拉取完整 diff，`174` 行，`1` 个文件。
- Motivation：Qwen3.6-27B-FP8 的 `modules_to_not_convert` 里带有 MoE 模板残留名，例如 `model.language_model.layers.N.mlp.gate`。旧逻辑用 `ignored in prefix`，会把 `mlp.gate` 误匹配到 dense fused MLP 的 `mlp.gate_up_proj`，导致这些 MLP 被当成 BF16 初始化，但 checkpoint 又带 FP8 scale，最终出现 `weight_scale_inv not found` warning 和错误输出。Qwen3.5 的 `linear_attn.in_proj_a` 与 fused `in_proj_ba` 也有同类风险。
- 关键实现：`python/sglang/srt/layers/quantization/utils.py` 新增 `_module_path_match`，只在 dotted module path 边界上匹配；同时增加 `_FALLBACK_FUSED_SHARDS`，当 HF FP8 config 没有 `packed_modules_mapping` 时，仍能把 fused projection 展开成 shard 再判断是否跳过量化。
- 关键代码片段：

```python
def _module_path_match(ignored: str, prefix: str) -> bool:
    if ignored == prefix:
        return True
    if prefix.startswith(ignored + "."):
        return True
    return ("." + ignored + ".") in ("." + prefix + ".")
```

```python
_FALLBACK_FUSED_SHARDS: Mapping[str, List[str]] = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
}
```

```diff
-        is_skipped = any(ignored in prefix for ignored in ignored_layers)
+        is_skipped = any(
+            _module_path_match(ignored, prefix) for ignored in ignored_layers
+        )
```

- 已读文件：`python/sglang/srt/layers/quantization/utils.py`。
- 验证影响：Qwen3.6-27B-FP8 加载时应不再出现 dense MLP 的 `weight_scale_inv not found`；还要回归 Qwen3.5 FP8，确认 `in_proj_a` 对 `in_proj_ba` 的跳过规则没有被破坏。

### PR #23486 - Qwen3.6-27B dense cookbook

- 链接：https://github.com/sgl-project/sglang/pull/23486
- 状态：已合入，`2026-04-22T17:22:46Z`
- Diff 覆盖：已拉取完整 diff，`198` 行，`2` 个文件。
- Motivation：第一版 Qwen3.6 页面只覆盖 35B-A3B MoE，但 Qwen3.6 同时有 27B dense 和 FP8 权重。旧命令生成器固定生成 `Qwen/Qwen3.6-35B-A3B`，dense 版本没有独立入口。
- 关键实现：文档的模型介绍、available models 表格、硬件内存估算、调用段落都扩展为 35B-A3B MoE 与 27B Dense 两条线。`qwen36-deployment.jsx` 增加 `modelSize` radio，把 `modelConfigs` 按模型大小嵌套，`generateCommand()` 通过 `sizeConfig.baseName` 生成 `--model-path`。安装提示从 `uv pip install "sglang[all]"` 改为 `uv pip install sglang`，避免 autoregressive VLM 文档误拉 diffusion/tracing/http2 extras。
- 关键代码片段：

```jsx
modelSize: {
  name: 'modelSize',
  title: 'Model Size',
  items: [
    { id: '35b-a3b', label: '35B-A3B (MoE)', default: true },
    { id: '27b', label: '27B (Dense)', default: false },
  ],
},
```

```jsx
const sizeConfig = modelConfigs[modelSize];
const quantSuffix = quantization === 'fp8' ? '-FP8' : '';
const modelName = `Qwen/Qwen3.6-${sizeConfig.baseName}${quantSuffix}`;
```

```diff
-uv pip install "sglang[all]"
+uv pip install sglang
```

- 已读文件：`docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`、`docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`。
- 验证影响：命令生成器要覆盖 `Qwen3.6-35B-A3B`、`Qwen3.6-35B-A3B-FP8`、`Qwen3.6-27B`、`Qwen3.6-27B-FP8` 四种 model path。PR body 记录了基于 #23467 FP8 修复后的 H200 TP=2 MMMU sanity：BF16 `55.1%`，FP8 `53.0%`，在 Wilson 95% CI 内。

### PR #23474 - hybrid linear-attn CPU offload

- 链接：https://github.com/sgl-project/sglang/pull/23474
- 状态：open，`2026-04-23` 时未合入
- Diff 覆盖：已拉取完整 diff，`395` 行，`2` 个文件。
- Motivation：Qwen3-Next、Qwen3.5、Kimi-Linear 这类 hybrid linear-attention 模型会把某些权重的 view/squeeze 缓存成普通 tensor attribute，例如 sibling attention module 上的 `conv_weights`。`--cpu-offload-gb` 把 `Parameter.data` 重新绑定到 pinned CPU memory 后，checkpoint load 写入的是新 CPU storage，而旧 view 仍指向初始化时的随机 GPU storage。另一个问题是 tied parameter 会在 state_dict 中出现多个 key，逐 key `.to(device)` 会生成多个 device tensor，`functional_call(..., tie_weights=True)` 会拒绝这些冲突值。
- 关键实现：`OffloaderV1.maybe_offload_to_cpu()` 在 offload 前记录普通 tensor attribute 到原始 parameter storage 的 alias；forward 时为 tied state_dict key 复用同一个 device tensor，再用 `as_strided(size, stride, offset)` 重建 alias view，`functional_call` 后恢复旧 attribute。新增的 unit test 用最小 tied-parameter module 和 cached-view module 复现这两个失败模式。
- 关键代码片段：

```python
view_aliases: Dict[int, List] = {}
param_data_ptr_to_param = {
    p.data.untyped_storage().data_ptr(): p for p in module.parameters()
}
```

```python
for k, v in module.state_dict(keep_vars=True).items():
    dev = src_to_dev.get(id(v))
    if dev is None:
        dev = v.to(device, non_blocking=True)
        src_to_dev[id(v)] = dev
    device_state[k] = dev
```

```python
sub.__dict__[attr_name] = dev_tensor.as_strided(size, stride, offset)
```

- 已读文件：`python/sglang/srt/utils/offloader.py`、`test/registered/unit/utils/test_offloader_tied_params.py`。
- 验证影响：虽然不是 Qwen3.6 专属 PR，但 Qwen3.6 共享 hybrid GDN/linear-attention 风险面，后续要跟踪它。回归要覆盖 `--cpu-offload-gb`、tied `A_log`/`dt_bias` 类参数、cached conv weight view，以及无 offload 的普通路径，确认 alias restore 不跨 forward 泄漏。

## 下一步优化建议

1. 不要先写 Qwen3.6 专属模型类；先证明 shared runtime 不能覆盖。
2. 把 text-only、image、video、reasoning、tool-call、MTP 六类请求做成最小 smoke 集。
3. CPU offload 和 hybrid cache 问题优先复用 Qwen3-Next/Qwen3.5 验证路径。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3.6`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config | `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | quantization | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | tests/benchmarks | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-22 | [#23486](https://github.com/sgl-project/sglang/pull/23486) | merged | docs(cookbook): add Qwen3.6-27B dense variant | docs/config | `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` |

### 逐 PR 代码 diff 阅读记录

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- 链接：https://github.com/sgl-project/sglang/pull/23034
- 状态/时间：`merged`，created 2026-04-17, merged 2026-04-17；作者 `zijiexia`。
- 代码 diff 已读范围：`73` 个文件，`+2214/-215`；代码面：model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config；关键词：doc, spec, attention, config, cuda, cache, moe, quant, eagle, expert。
- 代码 diff 细节：
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunk: +---
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunk: +---
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunk: +---; 符号: per_token_group_quant_8bit, add
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45 (286 lines); hunk: Please consult the documentation below and [server_args.py](https://github.com/s; Please consult the documentation below and [server_args.py](https://github.com
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` added +219/-0 (219 lines); hunk: +export const Qwen36Deployment = () => {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`；patch 关键词为 doc, spec, attention, config, cuda, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- 链接：https://github.com/sgl-project/sglang/pull/23467
- 状态/时间：`merged`，created 2026-04-22, merged 2026-04-22；作者 `mickqian`。
- 代码 diff 已读范围：`1` 个文件，`+31/-4`；代码面：quantization；关键词：config, fp8, kv, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunk: def __getattr__(self, name):; def is_layer_skipped(; 符号: __getattr__, _module_path_match, names, is_layer_skipped
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/utils.py`；patch 关键词为 config, fp8, kv, moe, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- 链接：https://github.com/sgl-project/sglang/pull/23474
- 状态/时间：`open`，created 2026-04-22；作者 `kawaruko`。
- 代码 diff 已读范围：`2` 个文件，`+284/-8`；代码面：tests/benchmarks；关键词：attention, cache, cuda, spec, test。
- 代码 diff 细节：
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunk: +"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).; 符号: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunk: import logging; def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:; 符号: maybe_offload_to_cpu, maybe_offload_to_cpu, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`；patch 关键词为 attention, cache, cuda, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23486 - docs(cookbook): add Qwen3.6-27B dense variant

- 链接：https://github.com/sgl-project/sglang/pull/23486
- 状态/时间：`merged`，created 2026-04-22, merged 2026-04-22；作者 `JustinTong0323`。
- 代码 diff 已读范围：`2` 个文件，`+55/-17`；代码面：docs/config；关键词：config, doc, fp8, moe, quant, spec, attention, expert, vision。
- 代码 diff 细节：
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` modified +30/-10 (40 lines); hunk: ---; Qwen3.6 features a Gated Delta Networks combined with sparse Mixture-of-Experts
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` modified +25/-7 (32 lines); hunk: export const Qwen36Deployment = () => {; export const Qwen36Deployment = () => {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`；patch 关键词为 config, doc, fp8, moe, quant, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：4；open PR 数：1。
- 仍需跟进的 open PR：[#23474](https://github.com/sgl-project/sglang/pull/23474)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
