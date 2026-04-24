# SGLang GLM-5/5.1 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `bca3dd958`（2026-04-24）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 GLM-5、GLM-5.1、GlmMoeDsa、NSA/DSA、FP8/MXFP4/NVFP4、NextN/MTP、tool template、AMD/GB300/NPU 和 dynamic chunking/profiling。

结论：GLM-5/5.1 是 shared DSA/NSA lane。任何触碰 `deepseek_v2.py`、`deepseek_nextn.py`、`nsa_backend.py`、`nsa_indexer.py` 的改动都可能同时影响 DeepSeek V3.2 和 GLM。示例命令必须保留 `--tool-call-parser glm47` 和 `--reasoning-parser glm45`。

## 代码面

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- `python/sglang/srt/layers/attention/nsa/`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `test/registered/8-gpu-models/test_dsa_models_basic.py`
- `test/registered/8-gpu-models/test_dsa_models_mtp.py`
- `test/registered/8-gpu-models/test_glm_51_fp8.py`
- `test/registered/gb300/test_glm5_fp8.py`
- `test/registered/gb300/test_glm5_nvfp4.py`
- `test/registered/amd/accuracy/`
- `test/registered/amd/perf/`

## 手工 diff 审阅 PR 卡片

### PR #18521 - Support GlmMoeDsaForCausalLM

- 链接：https://github.com/sgl-project/sglang/pull/18521
- 状态：已合入，`2026-02-10T07:20:10Z`
- Diff 覆盖：完整 diff `462` 行，`3` 个文件。
- Motivation：GLM-5 的 DSA/NSA 架构可以复用 DeepSeek V3.2 的 `DeepseekV2ForCausalLM` 和 NSA 后端，不应复制一套 GLM 专属栈；同时要兼容 RoPE 参数、draft model 架构重写和 speculative/NextN。
- 关键实现：`is_deepseek_nsa()` 识别 `GlmMoeDsaForCausalLM`；`ModelConfig._config_draft_model()` 把 GLM DSA draft 映射到 `DeepseekV3ForCausalLMNextN`；`glm4_moe.py` 新增 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`；`server_args.py` 把 GLM DSA 加入 NSA backend、deterministic inference、speculative decoding、auto speculative 参数选择。
- 关键代码片段：

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass

EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
```

```diff
+            "GlmMoeDsaForCausalLM",
         ]
         and getattr(config, "index_topk", None) is not None
```

- 验证影响：GLM-5 启动应默认进入 NSA attention；MTP 要走 DeepSeek NextN adapter；Blackwell 上的 sparse-MLA 行为需要结合 #20062 再验证。

### PR #18804 - GLM-5 fused shared expert 修复

- 链接：https://github.com/sgl-project/sglang/pull/18804
- 状态：已合入，`2026-02-16T19:50:39Z`
- Diff 覆盖：完整 diff `131` 行，`1` 个文件。
- Motivation：#18521 后 GLM-5 继承 DeepSeek DSA 路径，但没有覆写 fused shared expert 数量识别，MoE shared-expert fusion 可能按错误 architecture 读取。
- 关键实现：在 `GlmMoeDsaForCausalLM` 中实现 `determine_num_fused_shared_experts()`，显式传入 `"GlmMoeDsaForCausalLM"`。
- 关键代码片段：

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

- 验证影响：GLM-5 MoE 不能只测 server boot，还要测 shared expert fusion/routing。

### PR #18911 - AMD GLM-5 day-0 nightly

- 链接：https://github.com/sgl-project/sglang/pull/18911
- 状态：已合入，`2026-02-25T03:39:17Z`
- Diff 覆盖：完整 diff `1274` 行，`5` 个文件。
- Motivation：GLM-5 需要 ROCm day-0 覆盖；同时 HIP RoPE 不能走 CUDA-only JIT/tvm 路径，否则会在 AMD 上调用 `nvidia-smi` 类检测失败。
- 关键实现：`RotaryEmbedding.forward_hip()` 改为 `return self.forward_native(*args, **kwargs)`，兼容不同 subclass signature；AMD/ROCm nightly 增加 GLM-5 accuracy job 和 MI30x/MI35x 测试文件。
- 关键代码片段：

```python
def forward_hip(self, *args, **kwargs):
    return self.forward_native(*args, **kwargs)
```

```python
GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
```

- 验证影响：AMD GLM-5 回归要包含 HIP RoPE 和 8-GPU GSM8K accuracy。

### PR #20062 - DSA dense-attention threshold

- 链接：https://github.com/sgl-project/sglang/pull/20062
- 状态：已合入，`2026-03-09T21:36:10Z`
- Diff 覆盖：完整 diff `588` 行，`6` 个文件。
- Motivation：#18521 的 `SGLANG_NSA_FORCE_MLA` 过于粗糙。DSA 短 prefill 可以用 dense MHA 提速，但长 KV len 需要切回 sparse MLA；GLM-5 Blackwell 则需要强制 sparse MLA。
- 关键实现：新增 `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD = EnvInt(2048)`；GLM DSA + Blackwell 时设为 `0`；否则未手动设置时设为模型 `index_topk`；`nsa_backend.py` 用这个阈值决定是否走 MHA。
- 关键代码片段：

```python
if model_arch == "GlmMoeDsaForCausalLM" and is_blackwell_supported():
    envs.SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD.set(0)
```

```python
and max_kv_len <= envs.SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD.get()
```

- 验证影响：Blackwell GLM-5/5.1 要确认 sparse MLA 被启用；Hopper/AMD 要确认阈值默认跟 `index_topk` 一致。

### PR #21710 - AMD GLM-5-FP8 perf nightly

- 链接：https://github.com/sgl-project/sglang/pull/21710
- 状态：已合入，`2026-04-08T05:43:14Z`
- Diff 覆盖：完整 diff `537` 行，`6` 个文件。
- Motivation：GLM-5-FP8 已有 AMD accuracy，但缺少 MI30x/MI35x throughput/latency nightly。
- 关键实现：AMD workflows 在 accuracy 后增加非阻塞 perf step；accuracy config 切到 `zai-org/GLM-5-FP8` 并加 `--reasoning-parser glm45 --tool-call-parser glm47`；新增 perf 测试使用 `bench_one_batch`、`--kv-cache-dtype fp8_e4m3` 和 AMD tuning env。
- 关键代码片段：

```yaml
continue-on-error: true
python3 run_suite.py --hw amd --suite nightly-perf-8-gpu-glm5 --nightly
```

```python
model_path="zai-org/GLM-5-FP8",
other_args=["--reasoning-parser", "glm45", "--tool-call-parser", "glm47"]
```

- 验证影响：命令文档要和 AMD CI 中的 parser/FP8 KV 参数保持一致；perf failure 不应掩盖 accuracy failure。

### PR #21773 - AMD GLM-5-MXFP4 MI35x

- 链接：https://github.com/sgl-project/sglang/pull/21773
- 状态：已合入，`2026-04-15T01:55:36Z`
- Diff 覆盖：完整 diff `863` 行，`4` 个文件。
- Motivation：GLM-5 MXFP4/Quark checkpoint 需要独立 MI35x accuracy/perf lane，不能和 FP8 GLM-5 或 GLM-5.1 混在一起。
- 关键实现：新增 `nightly-8-gpu-mi35x-glm5-mxfp4` workflow entry，以及 `test_glm5_mxfp4_eval_mi35x.py`、`test_glm5_mxfp4_perf_mi35x.py`；accuracy/perf 都带 `SGLANG_USE_AITER=1`。
- 关键代码片段：

```yaml
nightly-8-gpu-mi35x-glm5-mxfp4-rocm720:
  runs-on: linux-mi35x-gpu-8
```

- 验证影响：GLM-5 MXFP4 要独立跟踪；#22543 loader fix 和 #23219 MTP fix 都应回归这条 lane。

### PR #22179 - DeepSeek V3.2/GLM-5 文档修正

- 链接：https://github.com/sgl-project/sglang/pull/22179
- 状态：已合入，`2026-04-06T06:26:43Z`
- Diff 覆盖：完整 diff `127` 行，`1` 个文件。
- Motivation：GLM-5 与 DeepSeek V3.2 共享 DSA/NSA 用法，但 parser 不同；旧文档没有清楚说明 GLM-5 替换 model path 后仍需保留 GLM parser。
- 关键实现：`docs/basic_usage/deepseek_v32.md` 明确 GLM-5 可以替换为 `zai-org/GLM-5-FP8`，并补充 short-sequence MHA、NSA backend choices、GLM-5 IndexCache pattern。注意该文档 hunk 写的是 `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`，而 #20062 代码里是 `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD`，后续引用需核对。
- 关键代码片段：

```diff
-To server GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
+To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
```

- 验证影响：文档要保留 `glm47` tool parser、`glm45` reasoning parser、NSA flags 和 IndexCache caveat。

### PR #22285 - GLM-5 H200 8-GPU CI

- 链接：https://github.com/sgl-project/sglang/pull/22285
- 状态：已合入，`2026-04-08T08:05:36Z`
- Diff 覆盖：完整 diff `8911` 行，`2` 个文件；重点读了重命名后的 DSA shared tests 和新增 GLM class。
- Motivation：GLM-5 需要和 DeepSeek V3.2 同等级的 H200 8-GPU DSA 回归，不应只依赖文档和 AMD job。
- 关键实现：`test_deepseek_v32_basic.py` / `test_deepseek_v32_mtp.py` 重命名为 DSA model tests；新增 GLM-5 DP/TP/MTP class，启动 `zai-org/GLM-5-FP8`，检查 GSM8K、speed 和 `avg_spec_accept_length`。
- 关键代码片段：

```python
GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
self.assertGreater(metrics["score"], 0.94)
self.assertGreater(avg_spec_accept_length, 2.7)
```

- 验证影响：GLM-5 MTP 回归要看 speculative accept length，不只是 accuracy。

### PR #22314 - AMD GLM-5 FP8 KV dispatch

- 链接：https://github.com/sgl-project/sglang/pull/22314
- 状态：已合入，`2026-04-08T04:16:02Z`
- Diff 覆盖：完整 diff `121` 行，`1` 个文件。
- Motivation：MI300/ROCm 上 GLM-5 FP8 KV 应使用 HIP raw MLA KV layout 和 fused BF16/FP16 -> FP8 paged KV write，不能误走 NVIDIA byte/scales layout。
- 关键实现：`memory_pool.py` 中 `set_mla_kv_buffer()` 先判断 `_is_hip and self.use_nsa and self.dtype == fp8_dtype`，直接调用 `set_mla_kv_buffer_triton_fp8_quant()`；非 HIP 才走 `quantize_k_cache_separate()`。
- 关键代码片段：

```python
if _is_hip and self.use_nsa and self.dtype == fp8_dtype:
    set_mla_kv_buffer_triton_fp8_quant(...)
elif self.nsa_kv_cache_store_fp8:
    cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(...)
```

- 验证影响：MI300/MI35x GLM-5 FP8 KV 要和 BF16/no-FP8-KV baseline 对比。

### PR #22336 - AMD GLM-5.1-FP8 nightly

- 链接：https://github.com/sgl-project/sglang/pull/22336
- 状态：已合入，`2026-04-09T05:57:43Z`
- Diff 覆盖：完整 diff `1485` 行，`6` 个文件。
- Motivation：GLM-5.1-FP8 是更大的 MoE DSA 模型，需要独立于 GLM-5-FP8 的 MI30x/MI35x coverage，并使用 TP=8 + EP=8。
- 关键实现：AMD workflows 增加 `nightly-8-gpu-glm51` 和 `nightly-8-gpu-mi35x-glm51`；新增 accuracy/perf 测试启动 `zai-org/GLM-5.1-FP8`，带 `--tp 8 --ep-size 8`、TileLang NSA backend、`glm45`/`glm47` parser、FP8 KV perf 参数。
- 关键代码片段：

```python
model_path="zai-org/GLM-5.1-FP8"
other_args=["--tp", "8", "--ep-size", "8", "--reasoning-parser=glm45", "--tool-call-parser=glm47"]
```

- 验证影响：GLM-5.1 文档应明确 EP=8；MI30x 和 MI35x perf env 不完全相同，问题排查要分开。

### PR #22399 - GLM-5.1 H200/B200/GB300 tests

- 链接：https://github.com/sgl-project/sglang/pull/22399
- 状态：已合入，`2026-04-09T00:04:57Z`
- Diff 覆盖：完整 diff `225` 行，`3` 个文件。
- Motivation：NVIDIA H200/B200 与 GB300 需要 GLM-5.1-FP8 coverage；同时不能把不存在的 GLM-5.1 NVFP4 checkpoint 写进测试。
- 关键实现：新增 `test_glm_51_fp8.py`，覆盖 TP8、TP8+DP8、TP8+DP8+MTP，并用 `SGLANG_ENABLE_SPEC_V2=1`；GB300 FP8 test 更新为 `zai-org/GLM-5.1-FP8`，第二个 commit 把 NVFP4 test 名称回退为 GLM-5。
- 关键代码片段：

```python
GLM_51_FP8_MODEL_PATH = "zai-org/GLM-5.1-FP8"
variant="TP8+DP8+MTP"
env={"SGLANG_ENABLE_SPEC_V2": "1"}
```

- 验证影响：GLM-5.1 FP8 是 H200/B200/GB300 路径；GLM-5 NVFP4 仍然是 GLM-5，不要误写成 GLM-5.1。

### PR #22543 - GLM-5/5.1 MXFP4 checkpoint compatibility

- 链接：https://github.com/sgl-project/sglang/pull/22543
- 状态：已合入，`2026-04-14T06:56:49Z`
- Diff 覆盖：完整 diff `122` 行，`3` 个文件。
- Motivation：GLM MXFP4/Quark checkpoint 复用 DeepSeek loader，但不应走 DeepSeek-V3 专属 Quark post-load transform；同时 Quark fused MLP 需要 `gate_up_proj` packed mapping。
- 关键实现：`deepseek_weight_loader.py` 只在 architecture 是 `DeepseekV3ForCausalLM` 时执行 `quark_post_load_weights(..., "mxfp4")`；`loader.py` 在 `model_config.quantization == "quark"` 时补 `{"gate_up_proj": ["gate_proj", "up_proj"]}`。
- 关键代码片段：

```python
if model_config.quantization == "quark":
    packed_modules_mapping.update({"gate_up_proj": ["gate_proj", "up_proj"]})
```

- 验证影响：GLM-5/5.1 MXFP4 要检查 gate/up fused weight loading，并确认 DeepSeek-only post-load 没有修改 GLM 权重。

### PR #22595 - GLM5.1 tool message content normalization

- 链接：https://github.com/sgl-project/sglang/pull/22595
- 状态：已合入，`2026-04-16T08:48:38Z`
- Diff 覆盖：完整 diff `191` 行，`2` 个文件。
- Motivation：OpenAI clients 会把 tool role content 发成 content-part list，但 GLM-5/5.1 chat template 期待 string，导致 tool result 对模型不可见并反复 tool call。
- 关键实现：`serving_chat.py` 新增 `normalize_tool_content()`，只 flatten tool role 且所有 part 都是 string 或 `{type:"text"}` 的列表，其他带语义字段的 list 保留；测试覆盖多 text part、mixed str/dict、empty list、非 tool role。
- 关键代码片段：

```python
def normalize_tool_content(role: str, content):
    if role != "tool" or not isinstance(content, list):
        return content
    ...
    return " ".join(text_parts)
```

- 验证影响：GLM-5.1 tool-calling 测试要包含 OpenAI text-part array 的 tool result，并确认模型最终回答而不是重复工具调用。

### PR #22712 - NPU GLM-5 guide

- 链接：https://github.com/sgl-project/sglang/pull/22712
- 状态：已合入，`2026-04-13T14:53:24Z`
- Diff 覆盖：完整 diff `33` 行，`1` 个文件。
- Motivation：Ascend GLM-5 文档使用 transformers main 分支会引入不可控变化，需要固定 best-practice 版本。
- 关键实现：把 transformers 安装建议改为 `transformers==5.3.0` 或 GitHub `v5.3.0` tag。
- 关键代码片段：

```diff
+pip install transformers==5.3.0
+pip install git+https://github.com/huggingface/transformers.git@v5.3.0
```

- 验证影响：NPU smoke 和文档应统一固定 transformers 5.3.0。

### PR #22850 - AMD NSA indexer kernel reduction

- 链接：https://github.com/sgl-project/sglang/pull/22850
- 状态：已合入，`2026-04-19T07:18:12Z`
- Diff 覆盖：完整 diff `141` 行，`1` 个文件。
- Motivation：AMD DSA/GLM-5 的 NSA indexer 在 `weights_proj` 和 index-K cache store 上还有额外 kernel/dtype 转换开销。
- 关键实现：`weights_proj` 参数统一 BF16，HIP 直接返回 BF16；`SGLANG_USE_AITER=1` 时 `_store_index_k_cache()` 调用 `aiter.ops.cache.indexer_k_quant_and_cache` 融合 quant 与 cache write。
- 关键代码片段：

```python
if _use_aiter:
    kv_cache = buf.unsqueeze(1).view(fp8_dtype)
    indexer_k_quant_and_cache(key, kv_cache, out_loc, self.block_size, self.scale_fmt)
    return
```

- 验证影响：AMD GLM-5/5.1 perf 要有 AITER 和非 AITER 对照，避免 fused path 引入精度漂移。

### PR #23219 - GLM-5-MXFP4 MTP

- 链接：https://github.com/sgl-project/sglang/pull/23219
- 状态：已合入，`2026-04-20T23:09:08Z`
- Diff 覆盖：完整 diff `121` 行，`1` 个文件。
- Motivation：GLM-5-MXFP4 的 Quark quant 与 shared DeepSeek NextN 结合时，draft `eh_proj` 和 MTP layer quantization 要尊重 Quark checkpoint layout 与 `exclude_layers`。
- 关键实现：Quark 下 `eh_proj` 改用 `ReplicatedLinear` 并处理 `(output, bias)` 返回；构造 `DeepseekModelNextN` 前检查 MTP layer mapped prefix 是否在 `exclude_layers` 中，若是则 `nextn_quant_config = None`。
- 关键代码片段：

```python
if quant_config is not None and quant_config.get_name() == "quark":
    self.eh_proj = ReplicatedLinear(..., quant_config=quant_config)
```

```python
if should_ignore_layer(mapped_prefix, nextn_quant_config.exclude_layers):
    nextn_quant_config = None
```

- 验证影响：GLM-5-MXFP4 MTP 要独立于 FP8 MTP 测；重点看 Quark `exclude_layers`、`eh_proj` loading 和 EAGLE 输出质量。

### PR #23060 - GLM-5 dynamic chunking profiling crash 修复

- 链接：https://github.com/sgl-project/sglang/pull/23060
- 状态：已合入，`2026-04-23T11:30:57Z`
- Diff 覆盖：完整 diff `30` 行，`1` 个文件。
- Motivation：pipeline-parallel profiling 会构造 synthetic `ForwardBatch`，GLM-5 的 DSA/DP-attention 路径依赖 `is_extend_in_batch`，未设置时 profiling 可能 crash 或走错 attention 分支。
- 关键实现：

```diff
+set_is_extend_in_batch(batch.forward_mode.is_extend())
 _ = model_runner.forward(
     forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
 )
```

- 验证影响：GLM-5 dynamic chunking / profiling 要覆盖 extend batch，不能只看普通 serving 启动。

### PR #23540 - GLM-5.1 generator 拆分 MI300X / MI325X

- 链接：https://github.com/sgl-project/sglang/pull/23540
- 状态：已合入，`2026-04-23T19:01:59Z`
- Diff 覆盖：完整 diff `154` 行，`3` 个文件。
- Motivation：GLM-5.1 命令生成器把 MI300X/MI325X 合并成一个选项，容易掩盖两条 AMD 验证线。
- 关键实现：

```diff
+{ id: 'mi300x', label: 'MI300X', default: false },
+{ id: 'mi325x', label: 'MI325X', default: false },
+mi325x: { bf16: { tp: 8, mem: 0.80 } },
```

- 验证影响：AMD GLM-5.1 文档、命令和 perf/accuracy 记录需要分别标 MI300X、MI325X、MI355X。

## 下一步优化建议

1. GLM-5/5.1 改动先标明是否影响 DeepSeek V3.2 shared DSA/NSA 文件。
2. FP8、MXFP4、NVFP4、GLM-5.1 FP8 分开跑，不要把 checkpoint 名称混用。
3. tool template 与 tool-result normalization 需要 OpenAI chat completion 测试，不能只跑模型启动。
