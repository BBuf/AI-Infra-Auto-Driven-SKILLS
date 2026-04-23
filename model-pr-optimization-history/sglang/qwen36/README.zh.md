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
