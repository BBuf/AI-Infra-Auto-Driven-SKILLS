# SGLang GLM-4.5 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理。覆盖 GLM-4.5、GLM-4.5-Air、GLM4-MoE、NextN/MTP、A2A/DeepEP/Mooncake/FlashInfer、reduce-scatter、shared experts fusion、FP8/NVFP4/compressed-tensors、GLM45 reasoning parser、GLM45 tool parser 与 GLM4-MoE 共享路径。

结论：GLM-4.5 是 GLM MoE 的基线。GLM-4.6、GLM-4.7、GLM-5.x 的模型、parser、量化和平台行为有大量逻辑继承自 `glm4_moe.py` / `glm4_moe_detector.py` / `reasoning_parser.py`。GLM-4.5V 属于 VLM/OCR 线，但如果 PR 修改 shared text MoE、fused-MoE、quant 或 parser，也必须在本时间线保留。

## 代码面

- `python/sglang/srt/models/glm4.py`
- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `docs/basic_usage/glm45.md`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx`
- `docs_new/src/snippets/autoregressive/glm-45-deployment.jsx`

## 手工 diff 审阅 PR 卡片

### PR #8224 - GLM-4.5 模型首版支持

- 链接：https://github.com/sgl-project/sglang/pull/8224
- 状态：已合入，`2025-07-28`，merge commit `6d6a8bc278eac424214e73544ae010bde3fb99cb`
- Diff 覆盖：`14` 个文件，`+1673/-7`；重点读了 `glm4_moe.py`、`glm4_moe_nextn.py`、`glm4_moe_detector.py`、model config、server args 和 parser 注册 hunk。
- Motivation：SGLang 需要把 GLM-4.5 作为一个完整 MoE 模型接入，包括 text MoE、NextN/MTP draft、GLM XML tool-call detector、reasoning/tool parser alias，以及 draft model architecture 改写。
- 关键实现：新增 `Glm4MoeForCausalLM`、`Glm4MoeForCausalLMNextN`、GLM4-MoE gate/sparse block/model、`Glm4MoeDetector`，并把 `glm45` parser 注册到 GLM tool format。
- 关键代码片段：

```python
if is_draft_model and self.hf_config.architectures[0] == "Glm4MoeForCausalLM":
    self.hf_config.architectures[0] = "Glm4MoeForCausalLMNextN"
```

```python
class Glm4MoeForCausalLM(DeepseekV2ForCausalLM):
    ...
EntryClass = [Glm4MoeForCausalLM]
```

```python
class Glm4MoeForCausalLMNextN(Glm4MoeForCausalLM):
    def load_weights(self, weights, is_nextn=True):
        super().load_weights(weights, is_nextn=True)
```

- 验证影响：GLM-4.5 后续任何改动都要覆盖 base MoE、NextN/MTP、`--reasoning-parser glm45`、`--tool-call-parser glm45`、TP/EP 和量化加载。

### PR #8456 - compressed_tensors 启动修复

- 链接：https://github.com/sgl-project/sglang/pull/8456
- 状态：已合入，`2025-07-28`，merge commit `25f73c6cf3c2b20441266693ad12030157c1cbef`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：`zai-org/GLM-4.5-Air-FP8` / compressed-tensors checkpoint 启动失败，因为 shared experts fusion 只允许 `fp8` 和 `blockwise_int8`，没有接受 `compressed_tensors`。
- 关键实现：把 `compressed_tensors` 加入 shared-expert fusion 量化 allowlist；PR body 给出 GSM8K accuracy `0.935`、output throughput `1582.320 tok/s`。
- 关键代码片段：

```diff
 elif (
     self.quant_config.get_name() == "fp8"
     or self.quant_config.get_name() == "blockwise_int8"
+    or self.quant_config.get_name() == "compressed_tensors"
 ):
```

- 验证影响：GLM-4.5-Air 量化 smoke 不能只测 FP8/NVFP4，也要包含 compressed-tensors。

### PR #8647 - EP 下禁用 shared experts TP 切分

- 链接：https://github.com/sgl-project/sglang/pull/8647
- 状态：已合入，`2025-08-01`，merge commit `2ae95d17e80710d5ed1189398f36905ad43f5baa`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：`--tp 8 --enable-ep-moe` 跑 GLM-4.5 FP8 per-block 时，shared experts 仍被 TP 切分，导致 `192` 维 gate/up 输出不能被 FP8 block `128` 整除。
- 关键实现：记录 `ep_size`；EP>1 时给 shared-expert linears 传 `tp_rank=0,tp_size=1`；forward 中按 EP 路径处理 all-reduce 和 shared output addition。PR body 给出 GSM8K accuracy `0.955`、throughput `479.302 tok/s`。
- 关键代码片段：

```python
self.ep_size = get_moe_expert_parallel_world_size()
...
**(dict(tp_rank=0, tp_size=1) if self.ep_size > 1 else {}),
```

```python
if self.ep_size > 1:
    if self.tp_size > 1 and not can_fuse_mlp_allreduce:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    if shared_output is not None:
        final_hidden_states += shared_output
```

- 验证影响：EP+TP 的 GLM-4.5-Air FP8 per-block 是必测回归。

### PR #8729 - GLM-4.5 router correction bias 保持 FP32

- 链接：https://github.com/sgl-project/sglang/pull/8729
- 状态：已合入，`2025-08-03`，merge commit `760286e3d378780546b88c6d9e932bc178d39669`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：`e_score_correction_bias` 是 router/top-k 的数值敏感参数，应保持 FP32，避免低精度扰动专家选择。
- 关键实现：parameter 初始化显式指定 `dtype=torch.float32`。
- 关键代码片段：

```python
self.e_score_correction_bias = nn.Parameter(
    torch.empty((config.n_routed_experts), dtype=torch.float32)
)
```

- 验证影响：GLM-4.5 BF16/FP8 router 稳定性和 top-k 分布需要关注。

### PR #8804 - GLM-4.5 与 GLM-4.5-Air 都支持 shared experts fusion

- 链接：https://github.com/sgl-project/sglang/pull/8804
- 状态：已合入，`2025-08-05`，merge commit `a4b0d5c9e5cb2b36eacdc30bc9259a213cd17a16`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：旧逻辑把 shared experts fusion 和固定 `n_routed_experts == 128` 绑定，GLM-4.5-Air 会被误禁用。
- 关键实现：默认 architecture 改成 `Glm4MoeForCausalLM`，删除专家数硬编码 guard。
- 关键代码片段：

```diff
-        self, architecture: str = "DeepseekV3ForCausalLM"
+        self, architecture: str = "Glm4MoeForCausalLM"
...
-            or self.config.n_routed_experts != 128
```

- 验证影响：GLM-4.5 与 GLM-4.5-Air 都要测 shared experts fusion。

### PR #8883 - reduce-scatter 接口补齐

- 链接：https://github.com/sgl-project/sglang/pull/8883
- 状态：已合入，`2025-08-07`，merge commit `5b6acc1495f4c4d44bfdb0ce8090426de280b002`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：GLM4 继承 DeepSeek MoE 后，没有同步 reduce-scatter 相关接口变化，DP/TP communicator 场景 logits 会出错。
- 关键实现：`forward_normal` / `forward_normal_dual_stream` 增加 `use_reduce_scatter`；当 reduce-scatter 接管 reduction 时跳过 TP all-reduce；历史上曾把 `allow_reduce_scatter=True` 传入 `LayerCommunicator`。
- 关键代码片段：

```python
if (
    self.tp_size > 1
    and not can_fuse_mlp_allreduce
    and not use_reduce_scatter
):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

```python
allow_reduce_scatter=True
```

- 验证影响：这是历史背景，当前行为要以 #11665 为准。

### PR #9136 - DP Attention buffer/flag utility

- 链接：https://github.com/sgl-project/sglang/pull/9136
- 状态：已合入，`2025-08-14`，merge commit `b87aacb5c55d673ead0a2bc501a58f7d02a5e2cd`
- Diff 覆盖：`21` 个文件，读了 DP attention buffer、flag、logits 和 GLM/DeepSeek call-site。
- Motivation：GLM MoE + DP attention 需要统一地拿到 gathered buffer 和 DP 开关，不能继续直接依赖 `global_server_args_dict["enable_dp_attention"]`。
- 关键实现：新增 global/local DP buffer getter 和 `is_dp_attention_enabled()`。
- 关键代码片段：

```python
def get_global_dp_buffer() -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer()

def get_local_dp_buffer() -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_local_dp_buffer()
```

```python
def is_dp_attention_enabled():
    assert _ENABLE_DP_ATTENTION_FLAG is not None, "dp attention not initialized!"
    return _ENABLE_DP_ATTENTION_FLAG
```

- 验证影响：DP attention、cuda graph、logits gather 是 GLM MoE 回归点。

### PR #9223 - MoE `TopKOutput` 接口清理

- 链接：https://github.com/sgl-project/sglang/pull/9223
- 状态：已合入，`2025-08-15`，merge commit `84b006b27833d93045ae5552e2cebb13f5140ab5`
- Diff 覆盖：`3` 个文件，完整阅读。
- Motivation：MoE refactor 后统一用 `TopKOutput`，GLM4 MoE 不能继续传散落的 router logits/top-k tensor。
- 关键实现：`topk_output = self.topk(...)` 后直接传给 `self.experts`；MXFP4 TRTLLM 路径读取 `topk_output.topk_config.top_k`。
- 关键代码片段：

```python
topk_output = self.topk(hidden_states, router_logits)
final_hidden_states = self.experts(hidden_states, topk_output)
```

```python
assert TopKOutputChecker.format_is_bypassed(topk_output)
top_k = topk_output.topk_config.top_k
```

- 验证影响：GLM4 MoE 出现 top-k format 错误时，优先检查 `TopKOutput` 合约。

### PR #9264 - GLM PP metadata quick fix

- 链接：https://github.com/sgl-project/sglang/pull/9264
- 状态：已合入，`2025-08-17`，merge commit `e47800e176b86d7d95309ab23d6cb3bd76d6c2be`
- Diff 覆盖：`2` 个文件，完整阅读。
- Motivation：#8846 后 GLM inference 缺少 PP metadata；nightly GSM8K threshold 也用错 benchmark 标尺。
- 关键实现：模型持有 `pp_group/start_layer/end_layer`；GLM-4.5-Air-FP8 threshold 从 `0.94` 改到 `0.78`。
- 关键代码片段：

```python
self.pp_group = get_pp_group()
self.start_layer = 0
self.end_layer = config.num_hidden_layers
```

```diff
-"zai-org/GLM-4.5-Air-FP8": 0.94,
+"zai-org/GLM-4.5-Air-FP8": 0.78,
```

- 验证影响：GLM wrapper 改动后要跑 PP metadata smoke。

### PR #10008 - MXFP4/AITER 性能路径兼容 GLM

- 链接：https://github.com/sgl-project/sglang/pull/10008
- 状态：已合入，`2025-09-04`，merge commit `e96973742c326a129da772a115bdeb925643d95a`
- Diff 覆盖：`8` 个文件，读了 DeepSeek MXFP4/AITER 与 GLM4 MoE signature 兼容 hunk。
- Motivation：PR 主目标是 DeepSeek MXFP4 性能，但共享 DeepSeek/GLM 量化接口变化曾破坏 GLM-4.5-Air。
- 关键实现：activation 支持 fused MXFP4 prequant；GLM4 MoE forward 接受 `gemm_output_zero_allocator`，避免共享调用签名漂移。
- 关键代码片段：

```python
def forward_cuda(self, x: torch.Tensor, fused_mxfp4_prequant: Optional[bool] = False):
    if fused_mxfp4_prequant:
        out = act_mul_and_mxfp4_quant(x, "silu")
```

```python
def forward_normal(
    self,
    hidden_states: torch.Tensor,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
    gemm_output_zero_allocator: BumpAllocator = None,
) -> torch.Tensor:
```

- 验证影响：即使 PR 标题不含 GLM，只要改 shared DeepSeek quant/kernel 签名，就要回归 GLM-4.5-Air。

### PR #11017 - GLM-4.5/4.6 文档和 router mapping

- 链接：https://github.com/sgl-project/sglang/pull/11017
- 状态：已合入，`2025-09-28`，merge commit `abb6781573a86c7e7b22e41fd2924094a7d4a135`
- Diff 覆盖：`5` 个文件，完整阅读。
- Motivation：GLM-4.6 复用了 GLM-4.5 runtime/parser 形态，文档和 router mapping 要从单 GLM-4.5 扩展为 GLM-4.5/4.6。
- 关键实现：router 把 `glm-4.5*` 和 `glm-4.6*` 都映射到 `glm4_moe`。
- 关键代码片段：

```rust
self.map_model("glm-4.5*", "glm4_moe");
self.map_model("glm-4.6*", "glm4_moe");
```

- 验证影响：GLM-4.5/4.6 使用 `glm45` parser，不能和 GLM-4.7 的 `glm47` 混掉。

### PR #11665 - GLM45 禁用 reduce-scatter

- 链接：https://github.com/sgl-project/sglang/pull/11665
- 状态：已合入，`2025-10-18`，merge commit `f7ab9554554fbd3d07ffa4ad34c5fcbef69591b6`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：GLM45 MoE 当时仍不安全支持 reduce-scatter，保留 `allow_reduce_scatter=True` 有正确性风险。
- 关键实现：把 GLM45 `LayerCommunicator` 的 reduce-scatter 关掉。
- 关键代码片段：

```diff
-            allow_reduce_scatter=True,
+            allow_reduce_scatter=False,
```

- 验证影响：不要在没有完整正确性矩阵的情况下重新打开 GLM45 reduce-scatter。

### PR #11692 - GLM4.5 MoE Block A2A backend init

- 链接：https://github.com/sgl-project/sglang/pull/11692
- 状态：已合入，`2025-10-16`，merge commit `476c67d7fcfea316f23d24afe90a8f679f0490a4`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：GLM-4.5 MoE block 缺少 A2A backend 初始化，DeepEP/Mooncake 抽象后 CI 失败。
- 关键实现：统一判断 `is_deepep()` 或 `is_mooncake()`，内部变量改为 `_enable_a2a_moe`。
- 关键代码片段：

```python
if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
    ...
self._enable_a2a_moe = (
    get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
)
```

- 验证影响：A2A 测试要同时覆盖 DeepEP 和 Mooncake。

### PR #11800 - GLM-4.5/4.5V 实现重构

- 链接：https://github.com/sgl-project/sglang/pull/11800
- 状态：已合入，`2025-10-24`，merge commit `4060ed37cb67262b0cc7af2bcbbdf37ba12d3501`
- Diff 覆盖：`4` 个文件，读了 text MoE、GLM4V MoE、PP、shared-expert 和 NextN hunk。
- Motivation：GLM-4.5/4.5V 直接继承 DeepSeek-V2 过多，后续 shared experts、PP、VLM 和 GLM 演进都容易被 DeepSeek 变化牵连。
- 关键实现：GLM 定义自己的 `Glm4MoeSparseMoeBlock`、`Glm4MoeDecoderLayer`、`Glm4MoeModel`；通过 `make_layers`、`PPMissingLayer`、`PPProxyTensors` 支持 PP；GLM4V MoE 简化为复用 GLM text model。
- 关键代码片段：

```python
self.layers, self.start_layer, self.end_layer = make_layers(
    config.num_hidden_layers,
    lambda idx, prefix: Glm4MoeDecoderLayer(...),
    pp_rank=self.pp_group.rank_in_group,
    pp_size=self.pp_group.world_size,
)
```

```python
if self.pp_group.is_last_rank:
    self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
else:
    self.norm = PPMissingLayer(return_tuple=True)
```

- 验证影响：重构后必须测 text、GLM-4.5V 共享文本路径、PP 首尾 rank、NextN、shared expert fusion。

### PR #11847 - MoE dispatcher interface 清理

- 链接：https://github.com/sgl-project/sglang/pull/11847
- 状态：已合入，`2025-10-20`，merge commit `bfc3b3f786829b3ba73504cda07b6ec74908564f`
- Diff 覆盖：`24` 个文件，读了 dispatcher、DP state、DeepEP/Mooncake 和 GLM call-site。
- Motivation：Standard/DeepEP/Mooncake dispatcher 签名不一致，GLM4 MoE 的 A2A/overlap 路径依赖这些接口。
- 关键实现：dispatcher 统一使用 `TopKOutput`，extend/decode 状态放到 DP attention wrapper。
- 关键代码片段：

```python
def set_is_extend_in_batch(is_extend_in_batch: bool):
    _DpGatheredBufferWrapper.set_is_extend_in_batch(is_extend_in_batch)
```

```python
def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, ...):
    return single_batch_overlap.execute_sbo(
        hidden_states=hidden_states,
        topk_output=topk_output,
        experts=self,
    )
```

- 验证影响：GLM4 MoE A2A 要测 DeepEP、Mooncake、TBO/SBO、decode/extend 切换。

### PR #12162 - return routed experts

- 链接：https://github.com/sgl-project/sglang/pull/12162
- 状态：已合入，`2025-12-21`，merge commit `bed301a5acaa9577c9aa706468bdf242f6a43051`
- Diff 覆盖：`27` 个文件，读了 capturer、scheduler/model-runner、FusedMoE capture、detokenizer 和 GLM4 patch hunk。
- Motivation：RL/training 对齐需要 inference 返回 routed expert IDs/top-k，以降低训练和推理 logprob 差异。
- 关键实现：`RoutedExpertsCapturer` 根据 `--enable-return-routed-experts` 创建；FusedMoE 捕获每层 `topk_output`；detokenizer 返回 base64 int32 expert id。
- 关键代码片段：

```python
self.routed_experts_capturer = RoutedExpertsCapturer.create(
    get_global_server_args().enable_return_routed_experts
)
```

```python
self.routed_experts_capturer.capture(
    layer_id=self.layer_id,
    topk_output=topk_output,
)
```

- 验证影响：GLM MoE 的 `layer_id`、`num_experts_per_tok`、fused shared-expert top-k 必须对齐。

### PR #12456 - GLM tool-call escaped character 修复

- 链接：https://github.com/sgl-project/sglang/pull/12456
- 状态：已合入，`2025-11-05`，merge commit `44da737770e4bcd9bfa27751f0a0751c9b5c06e1`
- Diff 覆盖：`2` 个文件，完整阅读。
- Motivation：GLM tool-call 输出中的 literal escaped chars，例如 `\n`、`\"`，会让旧 regex/JSON 解析失败，随后 fallback 又二次序列化，造成 double escape。
- 关键实现：regex 同时支持真实 newline 和 literal `\\n`；`parse_arguments()` 依次尝试 direct JSON、JSON string unescape、reparse、`ast.literal_eval`。
- 关键代码片段：

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
)
```

```python
wrapped = json.loads('{"tmp": "' + json_value + '"}')
parsed_value = json.loads(wrapped["tmp"])
```

- 验证影响：escaped array、Windows path、literal `\n`、quotes、stream/non-stream 都要测。

### PR #12497 - GLM-4.5 NVFP4 weight scale padding assertion

- 链接：https://github.com/sgl-project/sglang/pull/12497
- 状态：已合入，`2026-01-15`，merge commit `4346db5fafee11513799ebb57ec3e6ad5d95f6e9`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：`iAzure/GLM-4.5-NVFP4` 在 `modelopt_fp4` load 时命中 `weight_scale.shape[2] % 16 == 0`，但 swizzle padding 已经能处理非 16 对齐。
- 关键实现：硬 assert 改成 K-prime 非 4 倍数时 warning。PR body 记录 TP8 GSM8K accuracy `0.945`、throughput `2362.182 tok/s`。
- 关键代码片段：

```python
if weight_scale.shape[assert_dim] % 4 != 0:
    logger.warning(
        "NVFP4 %s_weight_scale K' not multiple of 4: shape=%s, group_size=%s",
        name,
        tuple(weight_scale.shape),
    )
```

- 验证影响：NVFP4 加载验证要检查 warning 和 GSM8K，而不是假设 scale 完美对齐。

### PR #12572 - symmetric memory 注册 allgather/reducescatter buffer

- 链接：https://github.com/sgl-project/sglang/pull/12572
- 状态：已合入，`2025-11-05`，merge commit `2340798353bc58398b6d45f582c7c79b670d0256`
- Diff 覆盖：`19` 个文件，读了 symmetric memory、PyNccl collectives、DP buffer 和 GLM MoE allocation。
- Motivation：MoE/TP collectives 的 allgather/reducescatter buffer 需要 symmetric memory 保证，以支持 overlap 和通信路径。
- 关键实现：新增 `use_symmetric_memory()` context；GLM4 MoE shared output 分配放进 symmetric memory context。
- 关键代码片段：

```python
def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    disabled = not is_symmetric_memory_enabled() or disabled or group_coordinator.world_size == 1
    return SymmetricMemoryContext(group_coordinator) if not disabled else nullcontext()
```

```python
with use_symmetric_memory(
    parallel_state.get_tp_group(), disabled=not is_allocation_symmetric()
):
    final_hidden_states_out = torch.empty_like(final_hidden_states)
```

- 验证影响：TP collectives、cuda graph、shared output allocation 是回归点。

### PR #12834 - KTransformers heterogeneous compute refactor

- 链接：https://github.com/sgl-project/sglang/pull/12834
- 状态：已合入，`2025-11-10`，merge commit `ddd1440d0f027e85af6be53bbb309683ed7ea2c4`
- Diff 覆盖：`10` 个文件，读了 KT wrapper、server args、quant fallback、GLM routed scaling。
- Motivation：KTransformers CPU/GPU heterogeneous expert 路径以前由分散硬编码和 env 控制，需要统一 wrapper 组合 CPU experts 与 GPU quant 方法。
- 关键实现：`KTEPWrapperMethod` 包装 GPU MoE method 和 CPU AMX/AVX experts；CPU expert ids 在 GPU 上 mask；GLM4 MoE 在 routed scaling 上识别 KT wrapper。
- 关键代码片段：

```python
topk_ids[topk_ids >= num_gpu_experts] = -1
```

```python
if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):
    final_hidden_states *= self.routed_scaling_factor
```

- 验证影响：GLM4 MoE + KT 要检查 CPU/GPU expert correctness 和 routed scaling。

### PR #12957 - 清理 #12834 冗余代码

- 链接：https://github.com/sgl-project/sglang/pull/12957
- 状态：已合入，`2025-11-10`，merge commit `9cfe78dd3076749c9ac1eec0a91d941d3d3a76c7`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：#12834 留下重复的 `forward_normal_dual_stream()`，容易让 GLM routed scaling、symmetric memory、all-reduce 后续只改一个分支。
- 关键实现：删除冗余 method 和未使用的 `KTEPWrapperMethod` import。
- 关键代码片段：

```diff
-from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
...
-    def forward_normal_dual_stream(...):
-        ...
```

- 验证影响：当前 GLM4 MoE 的 dual-stream 要看后续 #13786，而不是这段被删掉的历史实现。

### PR #13786 - GLM MoE GEMM 双 CUDA stream overlap

- 链接：https://github.com/sgl-project/sglang/pull/13786
- 状态：已合入，`2025-11-25`，merge commit `4b45d556a7e66d1d978e6df14098a8ba87606a4b`
- Diff 覆盖：`1` 个文件，完整阅读。
- Motivation：GLM MoE 的 shared experts 和 routed experts 在 cuda graph capture 下可用两个 stream overlap。PR body 记录 GLM-4.6-FP8 从 `60.40` 提升到 `66.31 token/s`，GSM8K accuracy `0.952`。
- 关键实现：当 `alt_stream` 存在、hidden states 非空且处于 capture mode 时走 `forward_normal_dual_stream()`。
- 关键代码片段：

```python
if (
    self.alt_stream is not None
    and hidden_states.shape[0] > 0
    and get_is_capture_mode()
):
    return self.forward_normal_dual_stream(...)
```

```python
torch.add(final_hidden_states, shared_output, out=final_hidden_states_out)
```

- 验证影响：cuda graph on/off、shared experts fused/non-fused 都要测。

### PR #13873 - GLM-4.6 shared experts fusion

- 链接：https://github.com/sgl-project/sglang/pull/13873
- 状态：已合入，`2025-12-01`，merge commit `982db4ebac260ef4b0597796541724c81a78fe94`
- Diff 覆盖：`7` 个文件，读了 fused-MoE config、lookup、`glm4_moe.py`、NextN。
- Motivation：标题是 GLM-4.6，但实现改的是 GLM4-MoE 共享基线：把 shared experts 融到 routed experts 中以降低延迟。
- 关键实现：把 shared expert 作为额外 expert slot；`num_experts/top_k` 都加 `num_fused_shared_experts`；load weights 时把 `mlp.shared_experts` remap 到 `mlp.experts.{n_routed_experts}`。
- 关键代码片段：

```python
self.experts = get_moe_impl_class(quant_config)(
    num_experts=config.n_routed_experts + self.num_fused_shared_experts,
    num_fused_shared_experts=self.num_fused_shared_experts,
    top_k=self.top_k + self.num_fused_shared_experts,
)
```

```python
name = name.replace(
    "mlp.shared_experts",
    f"mlp.experts.{self.config.n_routed_experts}",
)
```

- 验证影响：expert count、top-k、weight remap、fused-MoE config lookup 必须一起测。

### PR #13989 - GLM tool-call arguments 流式输出

- 链接：https://github.com/sgl-project/sglang/pull/13989
- 状态：已合入，`2025-12-13`，merge commit `80554598d33b68636be645856fce43403c7be1cb`
- Diff 覆盖：`2` 个文件，完整阅读 `glm4_moe_detector.py` 和测试。
- Motivation：旧 parser 要等完整 `</tool_call>` 才重 parse 整段 XML，streaming arguments 不能逐字符/逐片段输出。
- 关键实现：引入 `StreamState` 状态机，把 `<arg_key>/<arg_value>` XML fragment 增量转成 JSON fragment；先发 tool name，再用 `_streamed_raw_length` 控制新增 raw 片段。
- 关键代码片段：

```python
class StreamState(str, Enum):
    INIT = "INIT"
    BETWEEN = "BETWEEN"
    IN_KEY = "IN_KEY"
    WAITING_VALUE = "WAITING_VALUE"
    IN_VALUE = "IN_VALUE"
```

```python
raw_increment = func_args_raw[self._streamed_raw_length :]
json_increment = self._process_xml_to_json_streaming(raw_increment, func_name, tools)
```

- 验证影响：tag/value 分块、多 tool、quotes、unknown type、stream/non-stream 都要测。

### PR #14668 - FlashInfer A2A MoE dispatcher

- 链接：https://github.com/sgl-project/sglang/pull/14668
- 状态：已合入，`2026-01-24`，merge commit `2c2c4e446b99c529896b3377b24e1b48b6a52e61`
- Diff 覆盖：`14` 个文件，读了 dispatcher、server args、env、modelopt quant、DeepSeek/GLM call-site 和测试。
- Motivation：FlashInfer 提供了 one-sided NVLink all-to-all MoE kernels，SGLang 需要 `--moe-a2a-backend=flashinfer` 支持 FlashInfer CUTLASS MoE 和 FP4/NVFP4 dispatch。
- 关键实现：新增 `FlashinferDispatcher` / `FlashinferDispatchOutput` / `MoeA2ABackend.FLASHINFER`；server args 自动把 EP size 设成 TP size、禁用 shared experts fusion、设置 `SGLANG_MOE_NVFP4_DISPATCH=True`。
- 关键代码片段：

```python
elif a2a_backend.is_flashinfer():
    return FlashinferDispatcher(
        group=get_tp_group().device_group,
        router_topk=moe_runner_config.top_k,
        num_experts=moe_runner_config.num_experts,
    )
```

```python
if self.moe_a2a_backend == "flashinfer":
    self.ep_size = self.tp_size
    self.disable_shared_experts_fusion = True
    envs.SGLANG_MOE_NVFP4_DISPATCH.set(True)
```

- 验证影响：FlashInfer A2A 必须和 `flashinfer_cutlass`、NVFP4 dispatch、dummy token、empty rank、shared experts fusion disabled 一起测。

### PR #15333 - GLM-4.7 parser split

- 链接：https://github.com/sgl-project/sglang/pull/15333
- 状态：已合入，`2025-12-20`，merge commit `b82c7a0ae7444d4fa5a44185643f7c1cc6f372eb`
- Diff 覆盖：`7` 个文件，读了 docs、parser registry、新 `glm47_moe_detector.py` 和测试。
- Motivation：GLM-4.7 tool-call 格式和 GLM-4.5/4.6 不同，需要单独的 `glm47` parser，避免继续挤在 `glm45`。
- 关键实现：docs 明确 GLM-4.7 用 `--tool-call-parser glm47`，GLM-4.5/4.6 用 `glm45`；parser registry 加 `Glm47MoeDetector`。
- 关键代码片段：

```python
"glm45": Glm4MoeDetector,
"glm47": Glm47MoeDetector,
```

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
)
```

- 验证影响：GLM45 和 GLM47 parser 要分开测，不能相互偷规则。

### PR #15753 - GLM parser 复杂 JSON Schema 类型推断

- 链接：https://github.com/sgl-project/sglang/pull/15753
- 状态：已合入，`2026-01-09`，merge commit `8ef5b9052825c2624e3ac91852b16998f6f6ee3c`
- Diff 覆盖：`4` 个文件，读了 GLM45/GLM47 detector、shared utils 和测试。
- Motivation：`anyOf`、`oneOf`、`allOf`、enum-only、`type: ["string","null"]` 等复杂 schema 会让 array/object 被误当成 string。
- 关键实现：新增 `infer_type_from_json_schema()`，GLM45/GLM47 parser 都用它推断参数类型；streaming value detection 先尝试 JSON parse。
- 关键代码片段：

```python
return infer_type_from_json_schema(properties[arg_key])
```

```python
if "properties" in schema:
    return "object"
if "items" in schema:
    return "array"
```

- 验证影响：GLM45 也要测复杂 schema，不只是 GLM47。

### PR #15754 - GLM MoE detectors 空 func_name / None robustness

- 链接：https://github.com/sgl-project/sglang/pull/15754
- 状态：已合入，`2025-12-30`，merge commit `bc8b526edad7cb0b53658a6d230d4f4f5a1d1949`
- Diff 覆盖：`4` 个文件，读了 GLM45/GLM47 error handling 和边界测试。
- Motivation：streaming 中模型可能只吐 `<tool_call>` 或 partial tag，旧 parser 会因为 empty function name 或 `None.strip()` 崩溃。
- 关键实现：安全提取 regex group，function name 完整后才发 tool name；即使 chunk 没生成 JSON，也先更新 `_streamed_raw_length`，避免重复消费。
- 关键代码片段：

```python
is_func_name_complete = has_arg_key or is_tool_end == self.eot_token
if not is_func_name_complete:
    return None
if not func_name:
    logger.warning("Empty function name detected, skipping tool call")
    return None
```

```python
self._streamed_raw_length = current_raw_length
if not json_increment:
    return None
```

- 验证影响：要 fuzz partial chunks、split tag、no-arg tool、normal text、undefined tool、不完整 stream。

### PR #17714 - GLM45 reasoning tool interruption

- 链接：https://github.com/sgl-project/sglang/pull/17714
- 状态：已合入，`2026-03-02`，merge commit `da2a0240f7784fa8e4c7e978e4357a5908a4ee64`
- Diff 覆盖：`2` 个文件，完整阅读 reasoning parser 和测试。
- Motivation：GLM-4.5 会出现 `<think>...<tool_call>`，用 tool call 起始隐式结束 reasoning；旧 parser 等 `</think>`，会把 tool call 吃进 reasoning。
- 关键实现：`BaseReasoningFormatDetector` 增加 `tool_start_token`；`Glm45Detector` 使用 `<think>`、`</think>`、`<tool_call>`。
- 关键代码片段：

```python
if in_reasoning and self.tool_start_token is not None and self.tool_start_token in processed_text:
    tool_idx = processed_text.find(self.tool_start_token)
    reasoning_text = processed_text[:tool_idx].strip()
    normal_text = processed_text[tool_idx:]
```

```python
class Glm45Detector(BaseReasoningFormatDetector):
    super().__init__(
        "<think>",
        "</think>",
        tool_start_token="<tool_call>",
    )
```

- 验证影响：normal reasoning、truncated reasoning、tool interruption、split token、forced reasoning 都要测。

### PR #20543 - GLM tool-call values 保留 whitespace

- 链接：https://github.com/sgl-project/sglang/pull/20543
- 状态：已合入，`2026-04-09`，merge commit `8eb235ab512528de4c55200c09e2cbc3159a94ba`
- Diff 覆盖：`3` 个文件，完整阅读。
- Motivation：代码编辑工具参数常带缩进，`arg_value.strip()` 会破坏 exact old/new string。
- 关键实现：只 strip `arg_key`，不 strip `arg_value`；测试覆盖 GLM45 和 GLM47 detector。
- 关键代码片段：

```diff
 for arg_key, arg_value in pairs:
     arg_key = arg_key.strip()
-    arg_value = arg_value.strip()
     arg_type = get_argument_type(func_name, arg_key, tools)
```

- 验证影响：parser 测试要比较 exact whitespace。

## 已读 open PR 风险卡片

### PR #13711 - RTX Pro 6000 fused-MoE TP2 config

- 链接：https://github.com/sgl-project/sglang/pull/13711
- 状态：open（2026-04-23）
- Diff 覆盖：`5` 个文件，`+585/-0`，读了 fused-MoE benchmark util 和 config JSON。
- Motivation：支持 2x RTX Pro 6000 Blackwell 上 GLM-4.5-Air/4.5V 的 FP8 W8A8 TP2 fused-MoE 配置。
- 关键实现：加入 `Glm4vMoeForConditionalGeneration`，新增 `E=128,N=704` 与 `E=129,N=704` 的 Triton `3.4.0`/`3.5.1` 配置。
- 关键代码片段：

```python
"Glm4vMoeForConditionalGeneration",
```

```json
{ "1": { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64 } }
```

- 验证影响：合入前只能当硬件调优 radar，不能当主线事实。

### PR #19106 - GLM4 MoE Lite CompressedTensors / TF version

- 链接：https://github.com/sgl-project/sglang/pull/19106
- 状态：open（2026-04-23）
- Diff 覆盖：`12` 个文件，读了 model config、attention backend、weight loader、DeepSeek/GLM mapping、GLM4 MoE Lite 和测试。
- Motivation：CompressedTensors GLM4 MoE Lite checkpoint 因 `ReplicatedLinear` 没有 `.weight` 失败；`glm4_moe_lite` 又被误提示降级 transformers。
- 关键实现：MLA fast path guard `.weight`；CT WNA16 `kv_b_proj` dequant 后生成 `w_kc/w_vc`；fused q/kv 和 gate/up mapping；shared experts 被 quant ignore 时禁用 fusion；`glm4_moe_lite` 要求 TF>=5。
- 关键代码片段：

```python
fused_qkv_a_proj = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
if fused_qkv_a_proj is not None and getattr(fused_qkv_a_proj, "weight", None) is not None:
    ...
```

```python
qweight = unpack_from_int32(qweight, num_bits=4, packed_dim=1)
return (qweight * scales).reshape(out_features, in_features)
```

```python
any(".mlp.shared_experts." in item for item in self.quant_config.ignore)
```

- 验证影响：AWQ baseline、CT WNA16、shared expert ignore、TF>=5 warning 都要测。

### PR #19728 - ROCm GLM-4.5V-FP8 startup

- 链接：https://github.com/sgl-project/sglang/pull/19728
- 状态：open（2026-04-23）
- Diff 覆盖：`4` 个文件，读了 fused-MoE padding guard、HIP FP8 fallback copy helper 和测试。
- Motivation：MI300X 上 GLM-4.5V-FP8 因 unpadded MoE weights + global padding、以及 HIP FP8 fallback copy 到 padded buffer 失败而启动不了。
- 关键实现：`hidden_states.shape[1] == w1.shape[2]` 时不再扣 padding；HIP fallback 用 `_copy_with_optional_row_padding()` 填 tail rows。
- 关键代码片段：

```python
elif hidden_states.shape[1] == w1.shape[2]:
    padded_size = 0
```

```python
dst[: src.shape[0]].copy_(src)
if dst.shape[0] > src.shape[0]:
    dst[src.shape[0] :].fill_(pad_value)
```

- 验证影响：VLM 线 PR，但 touching shared fused-MoE/FP8 code，所以本页保留。

### PR #20917 - `/v1/responses` 尊重 `enable_thinking`

- 链接：https://github.com/sgl-project/sglang/pull/20917
- 状态：open（2026-04-23）
- Diff 覆盖：读了 `serving_responses.py` reasoning gating hunk 和 PR description；其他无关依赖/attention hunk 不计入 GLM motivation。
- Motivation：`/v1/responses` 对 GLM45/Qwen3 等 reasoning model 没有像 `/v1/chat/completions` 一样检查 `chat_template_kwargs.enable_thinking=false`。
- 关键实现：仅在 `enable_thinking` 没显式 false 时启用 reasoning parser。
- 关键代码片段：

```python
if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
    enable_reasoning = (
        not request.chat_template_kwargs
        or request.chat_template_kwargs.get("enable_thinking") is not False
    )
```

- 验证影响：`/v1/responses` 和 `/v1/chat/completions` 的 GLM45 thinking 行为要对齐。

### PR #23067 - Glm45Detector 转发 `continue_final_message`

- 链接：https://github.com/sgl-project/sglang/pull/23067
- 状态：open（2026-04-23）
- Diff 覆盖：`2` 个文件，完整阅读。
- Motivation：`ReasoningParser` 在 `continue_final_message=true` 时会传 `continue_final_message` 和 `previous_content`，但 `Glm45Detector.__init__` 不接这些参数，导致 GLM-4.5/GLM-5 chat completion HTTP 500。
- 关键实现：扩展 `Glm45Detector.__init__` signature，并把参数转给 base detector。
- 关键代码片段：

```python
def __init__(
    self,
    stream_reasoning: bool = True,
    force_reasoning: bool = False,
    continue_final_message: bool = False,
    previous_content: str = "",
):
    super().__init__(
        "<think>",
        "</think>",
        tool_start_token="<tool_call>",
        continue_final_message=continue_final_message,
        previous_content=previous_content,
    )
```

- 验证影响：GLM45/GLM5 radar 必须保留；所有 ReasoningParser detector subclass 都应接受 base path 传入的 kwargs。

## sgl-cookbook / 公开资料

- `sgl-cookbook#92`：GLM-4.5 AMD MI300X/MI325X/MI355X 部署背景。
- `sgl-cookbook#95`：GLM-4.5V AMD 部署背景，仅在触及 shared text MoE/fused-MoE/quant/parser 时进入本页。
- SGLang 官方 GLM 文档覆盖 GLM-4.5/4.6/4.7 launch、`glm45` vs `glm47` parser split、EAGLE/MTP、thinking budget/custom logit processor。
- LMSYS GLM-4.5 发布博客记录 day-one SGLang support、128k context、native function calling、MTP；这些是部署事实，不替代 PR diff review。

## 下一步优化建议

1. 固化 BF16、FP8、compressed-tensors、NVFP4、A2A/DeepEP/Mooncake/FlashInfer、parser streaming 的最小 smoke。
2. GLM45 parser 改动要分 reasoning parser 和 tool parser 两条验证，不要和 MoE 性能 PR 混在一起。
3. shared experts fusion 改动必须同时看 expert count、top-k、weight remap、quant ignore 和 fused-MoE config。
4. open PR 只能作为风险卡片，合入前不能写成 current-main 行为。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `GLM-4.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-21 | [#8224](https://github.com/sgl-project/sglang/pull/8224) | merged | GLM-4.5 Model Support | model wrapper, MoE/router, kernel, tests/benchmarks, docs/config | `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-07-28 | [#8456](https://github.com/sgl-project/sglang/pull/8456) | merged | fix GLM4_MOE launch with compressed_tensor quant model | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-01 | [#8647](https://github.com/sgl-project/sglang/pull/8647) | merged | Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647) | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-03 | [#8729](https://github.com/sgl-project/sglang/pull/8729) | merged | use fp32 for e_score_correction_bias in GLM-4.5 | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-05 | [#8804](https://github.com/sgl-project/sglang/pull/8804) | merged | GLM-4.5 and GLM-4.5-Air both support | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-07 | [#8883](https://github.com/sgl-project/sglang/pull/8883) | merged | fix glm4 moe | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-13 | [#9136](https://github.com/sgl-project/sglang/pull/9136) | merged | [DP Attention] Refactor: adding some utility functions | model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime | `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2025-08-15 | [#9223](https://github.com/sgl-project/sglang/pull/9223) | merged | Cleanup MoE Refactor | model wrapper, MoE/router, quantization | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-17 | [#9264](https://github.com/sgl-project/sglang/pull/9264) | merged | Quick Fix GLM | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py` |
| 2025-09-04 | [#10008](https://github.com/sgl-project/sglang/pull/10008) | merged | Optimized deepseek-v3/r1 model performance on mxfp4 run | model wrapper, MoE/router, quantization | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` |
| 2025-09-28 | [#11017](https://github.com/sgl-project/sglang/pull/11017) | merged | Update GLM-4.5 Model Doc | model wrapper, MoE/router | `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `sgl-router/src/tool_parser/registry.rs` |
| 2025-10-15 | [#11665](https://github.com/sgl-project/sglang/pull/11665) | merged | fix(glm45): disable reduce scatter | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-16 | [#11692](https://github.com/sgl-project/sglang/pull/11692) | merged | Fix missing a2a backend init of GLM4.5 MoE Block | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-18 | [#11800](https://github.com/sgl-project/sglang/pull/11800) | merged | Refactoring GLM-4.5 and GLM-4.5V related implementations | model wrapper, MoE/router, multimodal/processor | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-10-20 | [#11847](https://github.com/sgl-project/sglang/pull/11847) | merged | [9/N] MoE Refactor: cleanup dispatcher interfaces | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-10-26 | [#12162](https://github.com/sgl-project/sglang/pull/12162) | merged | [Feature] Enable return routed experts | model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `test/srt/rl/test_return_routed_experts.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-10-31 | [#12456](https://github.com/sgl-project/sglang/pull/12456) | merged | [fix] Handle escaped characters in GLM tool call parser to prevent double serialization | MoE/router, tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-11-01 | [#12497](https://github.com/sgl-project/sglang/pull/12497) | merged | [Fix] Remove assertion for padding for NVFP4 weight scales to fix GLM 4.5 NVFP4 | quantization | `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2025-11-04 | [#12572](https://github.com/sgl-project/sglang/pull/12572) | merged | Register allgather/reducescatter buffers with symm memory | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2025-11-07 | [#12834](https://github.com/sgl-project/sglang/pull/12834) | merged | Refactor KTransformers heterogeneous compute with unified GPU-quantization backend | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/server_args.py` |
| 2025-11-10 | [#12957](https://github.com/sgl-project/sglang/pull/12957) | merged | clean redundant code in previous PR | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-11-21 | [#13711](https://github.com/sgl-project/sglang/pull/13711) | open | [fused-moe] Add TP2 RTX Pro 6000 for GLM-4.5-Air and GLM-4.5V | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` |
| 2025-11-23 | [#13786](https://github.com/sgl-project/sglang/pull/13786) | merged | Overlap glm moe gemms in two cuda streams | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-11-25 | [#13873](https://github.com/sgl-project/sglang/pull/13873) | merged | Feat: GLM-4.6 supports shared experts fusion | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2025-11-26 | [#13989](https://github.com/sgl-project/sglang/pull/13989) | merged | Fix GLM-4.6 tool calls don't support streaming output for arguments i… | MoE/router, tests/benchmarks | `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-08 | [#14668](https://github.com/sgl-project/sglang/pull/14668) | merged | [NVIDIA] Add flashinfer all-to-all MOE dispatcher | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` |
| 2025-12-17 | [#15333](https://github.com/sgl-project/sglang/pull/15333) | merged | [GLM-4.7] GLM-4.7 Tool Parser and Doc Update | model wrapper, MoE/router, tests/benchmarks, docs/config | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-12-24 | [#15753](https://github.com/sgl-project/sglang/pull/15753) | merged | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py` |
| 2025-12-24 | [#15754](https://github.com/sgl-project/sglang/pull/15754) | merged | Fix: Handle empty func_name and None values in GLM MoE detectors | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-25 | [#17714](https://github.com/sgl-project/sglang/pull/17714) | merged | Add GLM45 tool interruption support | tests/benchmarks | `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-02-21 | [#19106](https://github.com/sgl-project/sglang/pull/19106) | open | Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks | model wrapper, attention/backend, MoE/router, tests/benchmarks, docs/config | `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-03 | [#19728](https://github.com/sgl-project/sglang/pull/19728) | open | Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback | MoE/router, quantization, kernel, tests/benchmarks | `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py` |
| 2026-03-13 | [#20543](https://github.com/sgl-project/sglang/pull/20543) | merged | fix: do not strip whitespace from GLM tool call values | MoE/router, tests/benchmarks | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-03-19 | [#20917](https://github.com/sgl-project/sglang/pull/20917) | open | fix(serving_responses): check enable_thinking for qwen3/glm45 models | attention/backend, multimodal/processor | `PR_DESCRIPTION.md`, `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2026-04-17 | [#23067](https://github.com/sgl-project/sglang/pull/23067) | open | Fix: forward continue_final_message kwargs in Glm45Detector | tests/benchmarks | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |

### 逐 PR 代码 diff 阅读记录

### PR #8224 - GLM-4.5 Model Support

- 链接：https://github.com/sgl-project/sglang/pull/8224
- 状态/时间：`merged`，created 2025-07-21, merged 2025-07-28；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`14` 个文件，`+1673/-7`；代码面：model wrapper, MoE/router, kernel, tests/benchmarks, docs/config；关键词：moe, kv, config, expert, test, attention, fp4, processor, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` added +1034/-0 (1034 lines); hunk: +# Copyright 2025-2026 SGLang Team; 符号: Glm4MoeMLP, __init__, forward, Glm4MoeAttention
  - `test/srt/test_function_call_parser.py` modified +184/-0 (184 lines); hunk: from sglang.srt.entrypoints.openai.protocol import Function, Tool; def setUp(self):; 符号: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf
  - `python/sglang/srt/models/glm4_moe_nextn.py` added +167/-0 (167 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: Glm4MoeModelNextN, __init__, forward, Glm4MoeForCausalLMNextN
  - `python/sglang/srt/function_call/glm4_moe_detector.py` added +165/-0 (165 lines); hunk: +import ast; 符号: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
  - `test/srt/openai_server/features/test_enable_thinking.py` modified +55/-0 (55 lines); hunk: def setUpClass(cls):; def test_chat_completion_with_reasoning(self):; 符号: setUpClass, tearDownClass, test_chat_completion_with_reasoning, test_chat_completion_without_reasoning
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；patch 关键词为 moe, kv, config, expert, test, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/models/glm4_moe_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8456 - fix GLM4_MOE launch with compressed_tensor quant model

- 链接：https://github.com/sgl-project/sglang/pull/8456
- 状态/时间：`merged`，created 2025-07-28, merged 2025-07-28；作者 `zminglei`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper, MoE/router；关键词：config, fp8, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 config, fp8, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8647 - Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647)

- 链接：https://github.com/sgl-project/sglang/pull/8647
- 状态/时间：`merged`，created 2025-08-01, merged 2025-08-01；作者 `zminglei`。
- 代码 diff 已读范围：`1` 个文件，`+73/-5`；代码面：model wrapper, MoE/router；关键词：config, cuda, deepep, expert, kv, moe, quant, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +73/-5 (78 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, forward_normal_dual_stream
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 config, cuda, deepep, expert, kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8729 - use fp32 for e_score_correction_bias in GLM-4.5

- 链接：https://github.com/sgl-project/sglang/pull/8729
- 状态/时间：`merged`，created 2025-08-03, merged 2025-08-03；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper, MoE/router；关键词：config, expert, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 config, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8804 - GLM-4.5 and GLM-4.5-Air both support

- 链接：https://github.com/sgl-project/sglang/pull/8804
- 状态/时间：`merged`，created 2025-08-05, merged 2025-08-05；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`1` 个文件，`+1/-2`；代码面：model wrapper, MoE/router；关键词：config, cuda, expert, kv, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-2 (3 lines); hunk: def __init__(; def determine_num_fused_shared_experts(; 符号: __init__, determine_num_fused_shared_experts, determine_num_fused_shared_experts
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 config, cuda, expert, kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8883 - fix glm4 moe

- 链接：https://github.com/sgl-project/sglang/pull/8883
- 状态/时间：`merged`，created 2025-08-07, merged 2025-08-07；作者 `ch-wan`。
- 代码 diff 已读范围：`1` 个文件，`+19/-4`；代码面：model wrapper, MoE/router；关键词：attention, cuda, deepep, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +19/-4 (23 lines); hunk: def __init__(; def forward_normal_dual_stream(; 符号: __init__, forward_normal_dual_stream, forward_normal_dual_stream, forward_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 attention, cuda, deepep, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9136 - [DP Attention] Refactor: adding some utility functions

- 链接：https://github.com/sgl-project/sglang/pull/9136
- 状态/时间：`merged`，created 2025-08-13, merged 2025-08-14；作者 `ch-wan`。
- 代码 diff 已读范围：`21` 个文件，`+216/-159`；代码面：model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime；关键词：attention, config, cuda, cache, processor, spec, eagle, moe, expert, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/dp_attention.py` modified +114/-27 (141 lines); hunk: import logging; tensor_model_parallel_all_reduce,; 符号: DPPaddingMode, DpPaddingMode, is_max_len, is_sum_len
  - `python/sglang/srt/layers/logits_processor.py` modified +12/-18 (30 lines); hunk: tensor_model_parallel_all_gather,; class LogitsMetadata:; 符号: LogitsMetadata:, from_forward_batch, compute_dp_attention_metadata, compute_dp_attention_metadata
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +8/-21 (29 lines); hunk: ); def __init__(self, model_runner: ModelRunner):; 符号: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
  - `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` modified +7/-21 (28 lines); hunk: import torch; def __init__(self, eagle_worker: EAGLEWorker):; 符号: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +7/-21 (28 lines); hunk: import torch; def __init__(self, eagle_worker: EAGLEWorker):; 符号: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`；patch 关键词为 attention, config, cuda, cache, processor, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9223 - Cleanup MoE Refactor

- 链接：https://github.com/sgl-project/sglang/pull/9223
- 状态/时间：`merged`，created 2025-08-15, merged 2025-08-15；作者 `ch-wan`。
- 代码 diff 已读范围：`3` 个文件，`+18/-16`；代码面：model wrapper, MoE/router, quantization；关键词：expert, router, topk, cuda, moe, config, flash, fp4, fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-7 (13 lines); hunk: def forward_normal_dual_stream(; def forward_normal(; 符号: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +8/-3 (11 lines); hunk: def apply(; def apply(; 符号: apply, apply
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-6 (10 lines); hunk: def forward_normal_dual_stream(; def forward_normal(; 符号: forward_normal_dual_stream, forward_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 expert, router, topk, cuda, moe, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9264 - Quick Fix GLM

- 链接：https://github.com/sgl-project/sglang/pull/9264
- 状态/时间：`merged`，created 2025-08-17, merged 2025-08-17；作者 `hebiao064`。
- 代码 diff 已读范围：`2` 个文件，`+6/-1`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：config, expert, fp8, moe, quant, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-0 (5 lines); hunk: from sglang.srt.distributed import (; def __init__(; 符号: __init__, __init__
  - `test/srt/test_nightly_gsm8k_eval.py` modified +1/-1 (2 lines); hunk: "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.83,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py`；patch 关键词为 config, expert, fp8, moe, quant, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_nightly_gsm8k_eval.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10008 - Optimized deepseek-v3/r1 model performance on mxfp4 run

- 链接：https://github.com/sgl-project/sglang/pull/10008
- 状态/时间：`merged`，created 2025-09-04, merged 2025-09-04；作者 `kkHuang-amd`。
- 代码 diff 已读范围：`8` 个文件，`+489/-67`；代码面：model wrapper, MoE/router, quantization；关键词：cuda, fp4, quant, triton, expert, moe, cache, flash, fp8, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +228/-32 (260 lines); hunk: is_cpu,; _is_cpu_amx_available = cpu_has_amx_support(); 符号: forward, __init__, forward, forward
  - `python/sglang/srt/layers/quantization/quark/utils.py` modified +97/-0 (97 lines); hunk: from types import MappingProxyType; def _is_equal_or_regex_match(; 符号: deep_compare, _is_equal_or_regex_match, b_dynamic_mxfp4_quant, mxfp4_to_f32
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` modified +49/-30 (79 lines); hunk: from aiter.ops.gemm_op_a4w4 import gemm_a4w4; def get_min_capability(cls) -> int:; 符号: get_min_capability, process_weights_after_loading, create_weights, apply_weights
  - `python/sglang/srt/layers/rocm_linear_utils.py` added +44/-0 (44 lines); hunk: +import torch; 符号: aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size
  - `python/sglang/srt/layers/communicator.py` modified +36/-4 (40 lines); hunk: from sglang.srt.managers.schedule_batch import global_server_args_dict; def prepare_attn(; 符号: prepare_attn, prepare_attn
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`；patch 关键词为 cuda, fp4, quant, triton, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/utils.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11017 - Update GLM-4.5 Model Doc

- 链接：https://github.com/sgl-project/sglang/pull/11017
- 状态/时间：`merged`，created 2025-09-28, merged 2025-09-28；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`5` 个文件，`+11/-12`；代码面：model wrapper, MoE/router；关键词：moe, config, router, expert, fp4, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +3/-3 (6 lines); hunk: def parse_arguments(json_value):; def __init__(self):; 符号: parse_arguments, Glm4MoeDetector, __init__, has_tool_call
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunk: # limitations under the License.; def determine_num_fused_shared_experts(; 符号: determine_num_fused_shared_experts
  - `sgl-router/src/tool_parser/registry.rs` modified +2/-3 (5 lines); hunk: impl ParserRegistry {
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-2 (4 lines); hunk: # limitations under the License.; def __init__(; 符号: __init__
  - `sgl-router/src/reasoning_parser/README.md` modified +1/-1 (2 lines); hunk: classDiagram
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `sgl-router/src/tool_parser/registry.rs`；patch 关键词为 moe, config, router, expert, fp4, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`, `sgl-router/src/tool_parser/registry.rs` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11665 - fix(glm45): disable reduce scatter

- 链接：https://github.com/sgl-project/sglang/pull/11665
- 状态/时间：`merged`，created 2025-10-15, merged 2025-10-18；作者 `jimmy-evo`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper, MoE/router；关键词：attention, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 attention, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11692 - Fix missing a2a backend init of GLM4.5 MoE Block

- 链接：https://github.com/sgl-project/sglang/pull/11692
- 状态/时间：`merged`，created 2025-10-16, merged 2025-10-16；作者 `ShangmingCai`。
- 代码 diff 已读范围：`1` 个文件，`+4/-2`；代码面：model wrapper, MoE/router；关键词：config, deepep, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-2 (6 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, forward_normal_dual_stream
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 config, deepep, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11800 - Refactoring GLM-4.5 and GLM-4.5V related implementations

- 链接：https://github.com/sgl-project/sglang/pull/11800
- 状态/时间：`merged`，created 2025-10-18, merged 2025-10-24；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`4` 个文件，`+356/-565`；代码面：model wrapper, MoE/router, multimodal/processor；关键词：config, spec, expert, moe, processor, quant, attention, awq, cache, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +322/-354 (676 lines); hunk: """Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights"""; get_pp_group,; 符号: __init__, forward, __init__, forward
  - `python/sglang/srt/models/glm4v_moe.py` modified +29/-196 (225 lines); hunk: import torch.nn as nn; def __init__(; 符号: __init__, __init__, determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-14 (18 lines); hunk: # limitations under the License.; from sglang.srt.model_executor.forward_batch_info import ForwardBatch; 符号: forward, forward, forward, Glm4MoeForCausalLMNextN
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +1/-1 (2 lines); hunk: class Glm4vImageProcessor(SGLangBaseProcessor):; 符号: Glm4vImageProcessor, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；patch 关键词为 config, spec, expert, moe, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11847 - [9/N] MoE Refactor: cleanup dispatcher interfaces

- 链接：https://github.com/sgl-project/sglang/pull/11847
- 状态/时间：`merged`，created 2025-10-20, merged 2025-10-20；作者 `ch-wan`。
- 代码 diff 已读范围：`24` 个文件，`+394/-428`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime；关键词：expert, moe, topk, deepep, config, quant, router, triton, flash, fp4。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunk: from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder; DispatchOutput,; 符号: DeepEPNormalOutput, format, DeepEPLLOutput, __init__
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunk: tma_align_input_scale,; def __init__(; 符号: __init__, __init__, forward, forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunk: get_moe_expert_parallel_world_size,; from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod; 符号: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunk: from typing import NamedTuple, Optional, Tuple; import torch; 符号: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_a
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-46 (60 lines); hunk: ); from sglang.srt.server_args import get_global_server_args; 符号: __init__, forward_deepep, forward_deepep, _forward_shared_experts_and_put_results
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；patch 关键词为 expert, moe, topk, deepep, config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12162 - [Feature] Enable return routed experts

- 链接：https://github.com/sgl-project/sglang/pull/12162
- 状态/时间：`merged`，created 2025-10-26, merged 2025-12-21；作者 `ocss884`。
- 代码 diff 已读范围：`27` 个文件，`+646/-10`；代码面：model wrapper, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks；关键词：expert, moe, attention, cache, config, cuda, topk, processor, scheduler, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/routed_experts_capturer.py` added +289/-0 (289 lines); hunk: +import logging; 符号: get_tensor_size_bytes, _RoutedExpertsDeviceCache:, __init__, get_buffer_size_bytes
  - `test/srt/rl/test_return_routed_experts.py` added +187/-0 (187 lines); hunk: +import asyncio; 符号: TestReturnRoutedExperts, setUpClass, test_return_routed_experts, fetch_result
  - `python/sglang/srt/model_executor/model_runner.py` modified +61/-0 (61 lines); hunk: set_is_extend_in_batch,; def initialize(self, min_per_gpu_memory: float):; 符号: initialize, initialize, init_routed_experts_capturer, remote_instance_init_transfer_engine
  - `python/sglang/srt/managers/detokenizer_manager.py` modified +18/-0 (18 lines); hunk: from typing import Dict, List, Union; def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):; 符号: _decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out, handle_batch_token_id_out
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +18/-0 (18 lines); hunk: from sglang.srt.disaggregation.utils import DisaggregationMode; def process_batch_result_prebuilt(self: Scheduler, batch: ScheduleBatch):; 符号: process_batch_result_prebuilt, maybe_collect_routed_experts, process_batch_result_prefill, process_batch_result_prefill
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `test/srt/rl/test_return_routed_experts.py`, `python/sglang/srt/model_executor/model_runner.py`；patch 关键词为 expert, moe, attention, cache, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `test/srt/rl/test_return_routed_experts.py`, `python/sglang/srt/model_executor/model_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12456 - [fix] Handle escaped characters in GLM tool call parser to prevent double serialization

- 链接：https://github.com/sgl-project/sglang/pull/12456
- 状态/时间：`merged`，created 2025-10-31, merged 2025-11-05；作者 `soaringk`。
- 代码 diff 已读范围：`2` 个文件，`+127/-13`；代码面：MoE/router, tests/benchmarks；关键词：moe, test。
- 代码 diff 细节：
  - `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunk: def test_partial_tool_call(self):; 符号: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunk: def get_argument_type(func_name: str, arg_key: str, defined_tools: list):; def __init__(self):; 符号: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；patch 关键词为 moe, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12497 - [Fix] Remove assertion for padding for NVFP4 weight scales to fix GLM 4.5 NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/12497
- 状态/时间：`merged`，created 2025-11-01, merged 2026-01-15；作者 `b8zhong`。
- 代码 diff 已读范围：`1` 个文件，`+7/-4`；代码面：quantization；关键词：config, fp4, fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +7/-4 (11 lines); hunk: def _slice_scale(w):; 符号: _slice_scale
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/modelopt_quant.py`；patch 关键词为 config, fp4, fp8, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/modelopt_quant.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12572 - Register allgather/reducescatter buffers with symm memory

- 链接：https://github.com/sgl-project/sglang/pull/12572
- 状态/时间：`merged`，created 2025-11-04, merged 2025-11-05；作者 `nvcastet`。
- 代码 diff 已读范围：`19` 个文件，`+250/-114`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime；关键词：attention, moe, expert, topk, config, router, cuda, flash, fp4, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/distributed/parallel_state.py` modified +76/-19 (95 lines); hunk: def reg_all_gather_into_tensor_fake(; def __init__(; 符号: reg_all_gather_into_tensor_fake, reg_reduce_scatter_tensor, reg_reduce_scatter_tensor_fake, GroupCoordinator:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +30/-22 (52 lines); hunk: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; def apply(; 符号: apply, apply, apply, apply
  - `python/sglang/srt/layers/dp_attention.py` modified +34/-11 (45 lines); hunk: get_tp_group,; class _DpGatheredBufferWrapper:; 符号: _DpGatheredBufferWrapper:, set_dp_buffer_len, get_global_dp_buffer, get_local_dp_buffer
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +16/-21 (37 lines); hunk: import os; def is_symmetric_memory_enabled():; 符号: is_symmetric_memory_enabled, get_nccl_mem_pool, use_symmetric_memory:, SymmetricMemoryContext:
  - `python/sglang/srt/layers/moe/topk.py` modified +21/-9 (30 lines); hunk: import torch.nn.functional as F; def forward_cuda(; 符号: forward_cuda, forward_cpu, forward_npu, empty_topk_output
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py`；patch 关键词为 attention, moe, expert, topk, config, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/dp_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12834 - Refactor KTransformers heterogeneous compute with unified GPU-quantization backend

- 链接：https://github.com/sgl-project/sglang/pull/12834
- 状态/时间：`merged`，created 2025-11-07, merged 2025-11-10；作者 `Atream`。
- 代码 diff 已读范围：`10` 个文件，`+494/-507`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime；关键词：expert, moe, config, quant, cuda, fp8, triton, topk, deepep, cache。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-411 (412 lines); hunk: import enum; except ImportError:; 符号: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, GPTQMarlinState
  - `python/sglang/srt/layers/moe/kt_ep_wrapper.py` added +393/-0 (393 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: KTConfig:, create_kt_config_from_server_args, mask_cpu_expert_ids, KTEPWrapperMethod
  - `python/sglang/srt/server_args.py` modified +6/-43 (49 lines); hunk: class ServerArgs:; def __post_init__(self):; 符号: ServerArgs:, __post_init__, _handle_grammar_backend, _handle_ktransformers_configs
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +24/-19 (43 lines); hunk: get_moe_a2a_backend,; FusedMoEMethodBase,; 符号: __init__, __init__, _weight_loader_physical, _weight_loader_impl
  - `python/sglang/srt/models/glm4_moe.py` modified +37/-0 (37 lines); hunk: ); def forward(; 符号: forward, forward_normal_dual_stream, forward_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/server_args.py`；patch 关键词为 expert, moe, config, quant, cuda, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/kt_ep_wrapper.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12957 - clean redundant code in previous PR

- 链接：https://github.com/sgl-project/sglang/pull/12957
- 状态/时间：`merged`，created 2025-11-10, merged 2025-11-10；作者 `Atream`。
- 代码 diff 已读范围：`1` 个文件，`+0/-37`；代码面：model wrapper, MoE/router；关键词：config, cuda, deepep, expert, flash, fp4, fp8, moe, quant, router。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +0/-37 (37 lines); hunk: ); def forward(; 符号: forward, forward_normal_dual_stream, forward_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 config, cuda, deepep, expert, flash, fp4。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13711 - [fused-moe] Add TP2 RTX Pro 6000 for GLM-4.5-Air and GLM-4.5V

- 链接：https://github.com/sgl-project/sglang/pull/13711
- 状态/时间：`open`，created 2025-11-21；作者 `mratsim`。
- 代码 diff 已读范围：`5` 个文件，`+585/-0`；代码面：MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：config, moe, triton, fp8, quant, benchmark, kv。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunk: +{
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunk: def get_model_config(; 符号: get_model_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`；patch 关键词为 config, moe, triton, fp8, quant, benchmark。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13786 - Overlap glm moe gemms in two cuda streams

- 链接：https://github.com/sgl-project/sglang/pull/13786
- 状态/时间：`merged`，created 2025-11-23, merged 2025-11-25；作者 `Qiaolin-Yu`。
- 代码 diff 已读范围：`1` 个文件，`+47/-3`；代码面：model wrapper, MoE/router；关键词：cuda, deepep, expert, flash, fp4, moe, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunk: def forward(; 符号: forward, forward_normal_dual_stream, forward_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 cuda, deepep, expert, flash, fp4, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13873 - Feat: GLM-4.6 supports shared experts fusion

- 链接：https://github.com/sgl-project/sglang/pull/13873
- 状态/时间：`merged`，created 2025-11-25, merged 2025-12-01；作者 `UranusSeven`。
- 代码 diff 已读范围：`7` 个文件，`+252/-24`；代码面：model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, moe, quant, triton, expert, topk, benchmark, cuda, deepep, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunk: is_cuda,; def __init__(; 符号: __init__, __init__, __init__, forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +19/-2 (21 lines); hunk: def try_get_optimal_moe_config(; def try_get_optimal_moe_config(; 符号: try_get_optimal_moe_config, try_get_optimal_moe_config, try_get_optimal_moe_config
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +7/-3 (10 lines); hunk: def get_model_config(; def get_model_config(; 符号: get_model_config, get_model_config
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunk: def __init__(; 符号: __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`；patch 关键词为 config, moe, quant, triton, expert, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13989 - Fix GLM-4.6 tool calls don't support streaming output for arguments i…

- 链接：https://github.com/sgl-project/sglang/pull/13989
- 状态/时间：`merged`，created 2025-11-26, merged 2025-12-13；作者 `cynial`。
- 代码 diff 已读范围：`2` 个文件，`+527/-81`；代码面：MoE/router, tests/benchmarks；关键词：cache, moe, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunk: import json; def get_argument_type(func_name: str, arg_key: str, defined_tools: list):; 符号: get_argument_type, StreamState, get_argument_type, get_argument_type
  - `test/registered/function_call/test_function_call_parser.py` modified +29/-15 (44 lines); hunk: def test_streaming_tool_call(self):; def test_streaming_multiple_tool_calls(self):; 符号: test_streaming_tool_call, test_streaming_multiple_tool_calls, test_invalid_tool_call, test_partial_tool_call
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py`；patch 关键词为 cache, moe, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14668 - [NVIDIA] Add flashinfer all-to-all MOE dispatcher

- 链接：https://github.com/sgl-project/sglang/pull/14668
- 状态/时间：`merged`，created 2025-12-08, merged 2026-01-24；作者 `trevor-m`。
- 代码 diff 已读范围：`14` 个文件，`+723/-16`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：flash, moe, expert, config, deepep, fp4, router, topk, quant, attention。
- 代码 diff 细节：
  - `python/sglang/test/test_flashinfer_dispatcher.py` added +322/-0 (322 lines); hunk: +import unittest; 符号: TestFlashinferDispatcher, setUpClass, tearDownClass, create_dispatcher
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunk: +from __future__ import annotations; 符号: FlashinferDispatchOutput, format, FlashinferCombineInput, format
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunk: +import torch.distributed as dist; 符号: CommBackend:, when, TorchDistributedCommBackend, __init__
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunk: MoeRunner,; def _slice_scale(w):; 符号: _slice_scale, apply, apply, apply
  - `python/sglang/srt/server_args.py` modified +23/-2 (25 lines); hunk: "cutlass",; class ServerArgs:; 符号: ServerArgs:, _handle_a2a_moe, _handle_eplb_and_dispatch
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`；patch 关键词为 flash, moe, expert, config, deepep, fp4。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15333 - [GLM-4.7] GLM-4.7 Tool Parser and Doc Update

- 链接：https://github.com/sgl-project/sglang/pull/15333
- 状态/时间：`merged`，created 2025-12-17, merged 2025-12-20；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`7` 个文件，`+809/-394`；代码面：model wrapper, MoE/router, tests/benchmarks, docs/config；关键词：moe, kv, cache, doc, spec, config, fp8, processor, test。
- 代码 diff 细节：
  - `test/registered/function_call/test_function_call_parser.py` modified +212/-388 (600 lines); hunk: from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector; def setUp(self):; 符号: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_json_format
  - `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunk: +import ast; 符号: StreamState, get_argument_type, _convert_to_number, parse_arguments
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunk: def get_argument_type(; 符号: get_argument_type, _convert_to_number
  - `docs/basic_usage/glm45.md` modified +4/-2 (6 lines); hunk: -## Launch GLM-4.5 / GLM-4.6 with SGLang; python3 -m sglang.launch_server \
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunk: Please consult the documentation below and [server_args.py](https://github.com/s
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；patch 关键词为 moe, kv, cache, doc, spec, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15753 - Fix GLM-4.7 MoE Detector complex JSON Schema type parsing

- 链接：https://github.com/sgl-project/sglang/pull/15753
- 状态/时间：`merged`，created 2025-12-24, merged 2026-01-09；作者 `Leoyzen`。
- 代码 diff 已读范围：`4` 个文件，`+869/-20`；代码面：MoE/router, tests/benchmarks；关键词：moe, spec, config, test。
- 代码 diff 细节：
  - `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunk: from sglang.srt.entrypoints.openai.protocol import Function, Tool; def test_streamed_raw_length_multiple_empty_returns(self):; 符号: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type
  - `python/sglang/srt/function_call/utils.py` modified +104/-1 (105 lines); hunk: from json import JSONDecodeError, JSONDecoder; def _get_tool_schema(tool: Tool) -> dict:; 符号: _get_tool_schema, infer_type_from_json_schema, get_json_schema_constraint
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunk: ToolCallItem,; def get_argument_type(; 符号: get_argument_type, get_argument_type, _get_value_type, _format_value_complete
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunk: ToolCallItem,; def get_argument_type(; 符号: get_argument_type, get_argument_type, _convert_to_number, _get_value_type
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`；patch 关键词为 moe, spec, config, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15754 - Fix: Handle empty func_name and None values in GLM MoE detectors

- 链接：https://github.com/sgl-project/sglang/pull/15754
- 状态/时间：`merged`，created 2025-12-24, merged 2025-12-30；作者 `Leoyzen`。
- 代码 diff 已读范围：`4` 个文件，`+1513/-140`；代码面：MoE/router, tests/benchmarks；关键词：moe, spec, test, cache, config。
- 代码 diff 细节：
  - `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunk: +import json; 符号: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunk: def get_argument_type(; def __init__(self):; 符号: get_argument_type, _convert_to_number, __init__, _reset_streaming_state
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunk: def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; def parse_streaming_increment(; 符号: detect_and_parse, parse_streaming_increment, parse_streaming_increment, parse_streaming_increment
  - `test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunk: def check_single_todos(tool_result, expected):; 符号: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；patch 关键词为 moe, spec, test, cache, config。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17714 - Add GLM45 tool interruption support

- 链接：https://github.com/sgl-project/sglang/pull/17714
- 状态/时间：`merged`，created 2026-01-25, merged 2026-03-02；作者 `Leoyzen`。
- 代码 diff 已读范围：`2` 个文件，`+238/-3`；代码面：tests/benchmarks；关键词：test。
- 代码 diff 细节：
  - `test/registered/parser/test_reasoning_parser.py` modified +182/-0 (182 lines); hunk: from sglang.srt.parser.reasoning_parser import (; def test_streaming_kimi_format(self):; 符号: test_streaming_kimi_format, TestGlm45Detector, setUp, test_init
  - `python/sglang/srt/parser/reasoning_parser.py` modified +56/-3 (59 lines); hunk: def __init__(; def detect_and_parse(self, text: str) -> StreamingParseResult:; 符号: __init__, detect_and_parse, parse_streaming_increment, parse_streaming_increment
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`；patch 关键词为 test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19106 - Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks

- 链接：https://github.com/sgl-project/sglang/pull/19106
- 状态/时间：`open`，created 2026-02-21；作者 `lesj0610`。
- 代码 diff 已读范围：`12` 个文件，`+505/-37`；代码面：model wrapper, attention/backend, MoE/router, tests/benchmarks, docs/config；关键词：kv, quant, config, moe, test, expert, mla, awq, cuda, attention。
- 代码 diff 细节：
  - `test/registered/core/test_deepseek_weight_loader.py` added +86/-0 (86 lines); hunk: +from types import SimpleNamespace; 符号: _pack_int4_row, test_dequantize_ct_wna16_weight, test_post_load_weights_dequantizes_ct_kv_b_proj, _DummyLoader
  - `test/registered/core/test_model_config_transformers_version.py` added +84/-0 (84 lines); hunk: +import logging; 符号: _build_model_config_stub, _mock_transformers_version, test_verify_transformers_version_glm4_moe_lite_no_downgrade_warning, test_verify_transformers_version_glm4_moe_lite_requires_tf5
  - `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, DeepseekV2ForCausalLM, __init__
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunk: def forward(; def __init__(; 符号: forward, __init__, Glm4MoeLiteForCausalLM, __init__
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunk: normalize_e4m3fn_to_e4m3fnuz,; class DeepseekV2WeightLoaderMixin:; 符号: DeepseekV2WeightLoaderMixin:, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 kv, quant, config, moe, test, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19728 - Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback

- 链接：https://github.com/sgl-project/sglang/pull/19728
- 状态/时间：`open`，created 2026-03-03；作者 `andyluo7`。
- 代码 diff 已读范围：`4` 个文件，`+104/-4`；代码面：MoE/router, quantization, kernel, tests/benchmarks；关键词：fp8, quant, expert, moe, test, triton, cache, config, cuda, mla。
- 代码 diff 细节：
  - `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunk: import unittest; def test_various_configurations(self):; 符号: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunk: def per_token_group_quant_mla_deep_gemm_masked_fp8(; def _native_dynamic_per_token_quant_fp8(output, input, scale):; 符号: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_token_quant_fp8
  - `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunk: import pytest; def test_scaled_fp8_quant_with_padding(dtype) -> None:; 符号: test_scaled_fp8_quant_with_padding
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunk: def fused_experts_impl(; 符号: fused_experts_impl
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py`；patch 关键词为 fp8, quant, expert, moe, test, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20543 - fix: do not strip whitespace from GLM tool call values

- 链接：https://github.com/sgl-project/sglang/pull/20543
- 状态/时间：`merged`，created 2026-03-13, merged 2026-04-09；作者 `lawrence-harmonic`。
- 代码 diff 已读范围：`3` 个文件，`+66/-2`；代码面：MoE/router, tests/benchmarks；关键词：moe, test。
- 代码 diff 细节：
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunk: def test_empty_function_name_handling(self):; def check_single_todos(tool_result, expected):; 符号: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunk: def _parse_argument_pairs(; 符号: _parse_argument_pairs
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunk: def _parse_argument_pairs(; 符号: _parse_argument_pairs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；patch 关键词为 moe, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20917 - fix(serving_responses): check enable_thinking for qwen3/glm45 models

- 链接：https://github.com/sgl-project/sglang/pull/20917
- 状态/时间：`open`，created 2026-03-19；作者 `xueliangyang-oeuler`。
- 代码 diff 已读范围：`8` 个文件，`+130/-19`；代码面：attention/backend, multimodal/processor；关键词：spec, test, attention, cache, cuda, doc, flash, fp8, kv, quant。
- 代码 diff 细节：
  - `PR_DESCRIPTION.md` added +69/-0 (69 lines); hunk: +## PR Motivation
  - `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py` modified +19/-8 (27 lines); hunk: from sglang.multimodal_gen.runtime.server_args import ServerArgs; def _apply_torch_compile(self, pipe: Any, server_args: ServerArgs) -> Any:; 符号: _apply_torch_compile
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +18/-9 (27 lines); hunk: def forward_extend(; def forward_decode(; 符号: forward_extend, forward_decode
  - `python/sglang/srt/entrypoints/openai/serving_responses.py` modified +10/-1 (11 lines); hunk: def _make_response_output_items(; 符号: _make_response_output_items
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/hunyuan3d_paint.py` modified +7/-1 (8 lines); hunk: ); def _do_load_paint(self, server_args: ServerArgs) -> None:; 符号: _do_load_paint, _convert_pil_list_to_tensor
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `PR_DESCRIPTION.md`, `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`；patch 关键词为 spec, test, attention, cache, cuda, doc。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `PR_DESCRIPTION.md`, `python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23067 - Fix: forward continue_final_message kwargs in Glm45Detector

- 链接：https://github.com/sgl-project/sglang/pull/23067
- 状态/时间：`open`，created 2026-04-17；作者 `huwwds`。
- 代码 diff 已读范围：`2` 个文件，`+66/-1`；代码面：tests/benchmarks；关键词：test。
- 代码 diff 细节：
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunk: def test_forced_reasoning_mode(self):; def test_continue_final_message_with_request(self):; 符号: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous
  - `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunk: class Glm45Detector(BaseReasoningFormatDetector):; 符号: Glm45Detector, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`；patch 关键词为 test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：35；open PR 数：5。
- 仍需跟进的 open PR：[#13711](https://github.com/sgl-project/sglang/pull/13711), [#19106](https://github.com/sgl-project/sglang/pull/19106), [#19728](https://github.com/sgl-project/sglang/pull/19728), [#20917](https://github.com/sgl-project/sglang/pull/20917), [#23067](https://github.com/sgl-project/sglang/pull/23067)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
