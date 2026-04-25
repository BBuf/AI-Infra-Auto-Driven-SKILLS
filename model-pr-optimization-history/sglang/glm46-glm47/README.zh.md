# GLM-4.6 / GLM-4.7 模型优化 PR 历史

本文档记录 SGLang 中 GLM-4.6、GLM-4.7、GLM-4.7-Flash 相关的模型优化、解析器、量化、MTP、AMD/NPU 后端 PR。所有列出的 PR 都按要求打开过 GitHub diff，并回填了 motivation、关键实现思路和关键代码片段。

证据快照：

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- 手工 diff 阅读日期：`2026-04-23`
- 对应 skill：`skills/model-optimization/sglang/sglang-glm46-glm47-optimization`
- 详细英文 PR 卡片：`skills/model-optimization/sglang/sglang-glm46-glm47-optimization/references/pr-history.md`
- 生产优化博客证据：LMSYS / Novita 的 GLM4-MoE 优化博客将这一系列和 shared-expert fusion、QK-Norm-RoPE fusion、async transfer、suffix decoding、EAGLE/NEXTN 部署参数关联起来。

## 关键运行面

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`
- `sgl-router/src/tool_parser/parsers/glm47_moe_parser.rs`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx`

## 总体脉络

GLM-4.6/4.7 这一组不是单一优化点，而是五条线交织：

- GLM-4.6：shared-expert fusion、双 CUDA stream 重叠 routed/shared expert GEMM、GLM4 XML tool-call streaming。
- GLM-4.7：新增 `glm47` tool parser，但 reasoning 仍使用 `glm45`；同时引入 NVFP4/FP8/MTP/NextN 风险。
- GLM-4.7-Flash：新增 `glm4_moe_lite` / `Glm4MoeLiteForCausalLM`，需要独立处理 config、rope、量化 packed module、EAGLE 不支持、AMD/NPU 路径。
- Parser：GLM-4.6、GLM-4.7、GLM-5 共享大量 GLM XML tool/reasoning 行为，parser PR 经常跨模型生效。
- 后端：AMD AITER FP8、NPU fused attention / QKNorm / RoPE / dual stream、FlashInfer A2A、Blackwell FP4 都需要单独验证。

## 已合入 PR

### #12456 - 处理 GLM tool-call 中的转义字符

- 链接：https://github.com/sgl-project/sglang/pull/12456
- 状态：已合入，merge commit `44da737770e4bcd9bfa27751f0a0751c9b5c06e1`
- Diff：`2` files，`+127/-13`
- Motivation：GLM-4.x tool-call 可能输出字面量 `\n`、转义引号，以及嵌套在 `<arg_value>` 里的 JSON。旧 parser 只匹配真实换行，解析失败后把数组/对象当字符串，导致二次 JSON 序列化。
- 实现思路：`parse_arguments()` 先直接 `json.loads`，失败后把值包进临时 JSON 字段做一次 JSON 级别 unescape，再二次解析；同时编译 regex，使函数名和参数标签之间既支持真实换行也支持字面量 `\\n`。
- 关键代码：

```python
wrapped = json.loads('{"tmp": "' + json_value + '"}')
parsed_value = json.loads(wrapped["tmp"])
```

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
)
```

- 读过的文件：`glm4_moe_detector.py`、`test/srt/test_function_call_parser.py`
- 验证含义：GLM parser 测试必须覆盖 escaped JSON、literal newline、Windows path、数组参数。

### #13786 - GLM MoE GEMM 双 CUDA stream 重叠

- 链接：https://github.com/sgl-project/sglang/pull/13786
- 状态：已合入，merge commit `4b45d556a7e66d1d978e6df14098a8ba87606a4b`
- Diff：`1` file，`+47/-3`
- Motivation：GLM-4.6 decode 中 shared expert 和 routed expert GEMM 串行执行，单并发输出速度受限。PR 描述里单并发 output speed 从 `60.40` 提升到 `66.31 tok/s`，GSM8K accuracy `0.952`。
- 实现思路：在 CUDA graph capture、存在 `alt_stream` 且 batch 非空时进入 `forward_normal_dual_stream()`，让 shared experts 和 routed experts 分 stream 执行，最后同步并把 shared output 加回。
- 关键代码：

```python
if (
    self.alt_stream is not None
    and hidden_states.shape[0] > 0
    and get_is_capture_mode()
):
    return self.forward_normal_dual_stream(...)
```

```python
torch.add(final_hidden_states, shared_output, out=final_hidden_states)
```

- 读过的文件：`python/sglang/srt/models/glm4_moe.py`
- 验证含义：必须单独验证 CUDA graph decode、empty-token guard、输出一致性，然后再和 shared-expert fusion 合并测试。

### #13873 - GLM-4.6 shared-expert fusion

- 链接：https://github.com/sgl-project/sglang/pull/13873
- 状态：已合入，merge commit `982db4ebac260ef4b0597796541724c81a78fe94`
- Diff：`7` files，`+252/-24`
- Motivation：GLM-4.6 的 shared experts 和 routed experts 分开跑会增加 GEMM 和同步成本。生产优化博客也把 shared-expert fusion 作为 GLM4-MoE 的核心优化之一。
- 实现思路：把 shared experts 表示成 routed expert 之后的额外 expert slot；`num_experts` 和 `top_k` 都加上 fused shared experts 数；加载权重时把 `mlp.shared_experts` 重映射到 `mlp.experts.{n_routed_experts}`。
- 关键代码：

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

- 读过的文件：`glm4_moe.py`、fused-MoE config、相关测试/文档
- 验证含义：shared-expert fusion 必须做 logits/accuracy 对比，再做性能 profile，不能和双 stream 优化混为一个开关。

### #13989 - GLM-4.6 tool-call 参数 streaming

- 链接：https://github.com/sgl-project/sglang/pull/13989
- 状态：已合入，merge commit `80554598d33b68636be645856fce43403c7be1cb`
- Diff：`2` files，`+527/-81`
- Motivation：GLM-4.6 tool-call 之前要等到完整 `</tool_call>` 才输出参数，用户看到的是最后一次性吐出，而不是逐步 streaming。
- 实现思路：给 GLM detector 增加状态机，先流式输出 function name，再根据 `_streamed_raw_length` 只输出新增参数片段，并把 XML arg 片段转换成 JSON fragment。
- 关键代码：

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
json_increment = self._process_xml_to_json_streaming(
    raw_increment, func_name, tools
)
```

- 读过的文件：`glm4_moe_detector.py`、parser tests
- 验证含义：streaming parser 要覆盖 name-only chunk、argument delta、完整 block、malformed partial XML、多 tool-call。

### #14585 - GLM-4.6V launch/accuracy 修复中的共享 GLM4-MoE 改动

- 链接：https://github.com/sgl-project/sglang/pull/14585
- 状态：已合入，merge commit `cf0478d602ce3259e24bc17a463575484920e166`
- Diff：`12` files，`+308/-29`
- Motivation：GLM-4.6V 有 accuracy drop 和 server launch 问题；虽然目标是 VLM，但 PR 触碰了共享 GLM4-MoE text path、shared-expert fusion 和 PP/DP encoder 逻辑。
- 实现思路：补 `attn_qkv_bias`、修 video grid flattening、注册 GLM4V FA3、增加 GLM thinking-budget token；MoE 侧增加 PP group、DP encoder、最后 PP rank 才创建 `lm_head`、shared expert 权重重映射。
- 关键代码：

```python
class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198
```

```python
if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
    name = name.replace(
        "mlp.shared_experts",
        f"mlp.experts.{self.config.n_routed_experts}",
    )
```

- 读过的文件：`glm4v.py`、`glm4v_moe.py`、`glm4_moe.py`、GLM docs/tests
- 验证含义：VLM 验证单独跑，但任何共享 `glm4_moe.py` 改动都要回归 GLM-4.6 text MoE。

### #14668 - FlashInfer A2A MoE dispatcher

- 链接：https://github.com/sgl-project/sglang/pull/14668
- 状态：已合入，merge commit `2c2c4e446b99c529896b3377b24e1b48b6a52e61`
- Diff：`14` files，`+723/-16`
- Motivation：GLM4-MoE FP4/NVFP4 类路径需要 FlashInfer A2A MoE dispatcher，通用 token dispatch 不一定是最快或最兼容的路径。
- 实现思路：dispatcher factory 增加 `flashinfer` 后端；GLM4-MoE 在这个后端下把 EP size 设成 TP size，关闭 shared-expert fusion，并设置 NVFP4 dispatch env。
- 关键代码：

```python
elif a2a_backend.is_flashinfer():
    return FlashinferDispatcher(...)
```

```python
if self.moe_a2a_backend == "flashinfer":
    self.ep_size = self.tp_size
    self.disable_shared_experts_fusion = True
    envs.SGLANG_MOE_NVFP4_DISPATCH.set(True)
```

- 读过的文件：MoE token dispatcher、`glm4_moe.py`、server args/env
- 验证含义：FlashInfer A2A 不应默认和 shared-expert fusion 同时开启，除非兼容 guard 明确允许。

### #15333 - GLM-4.7 tool parser 和文档

- 链接：https://github.com/sgl-project/sglang/pull/15333
- 状态：已合入，merge commit `b82c7a0ae7444d4fa5a44185643f7c1cc6f372eb`
- Diff：`7` files，`+809/-394`
- Motivation：GLM-4.7 的 tool-call 格式去掉了 tool name 后面的换行，旧 GLM-4.5/4.6 parser 会错解析 `<tool_call>name<arg_key>...`。
- 实现思路：新增 `glm47` parser，GLM-4.7/Flash 使用 `--tool-call-parser glm47`，reasoning 继续使用 `--reasoning-parser glm45`。
- 关键代码：

```python
"glm45": Glm4MoeDetector,
"glm47": Glm47MoeDetector,
```

```python
self.func_detail_regex = re.compile(
    r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
)
```

- 读过的文件：`glm47_moe_detector.py`、parser registry、GLM-4.7 docs/snippets、tests
- 验证含义：GLM-4.7 启动文档必须显式写 `glm47` tool parser 和 `glm45` reasoning parser。

### #15520 - model-gateway GLM-4.7 tool parser

- 链接：https://github.com/sgl-project/sglang/pull/15520
- 状态：已合入，merge commit `26704c23c056e426c6bc86ea1289e82b5fd37e59`
- Diff：`8` files，`+179/-26`
- Motivation：Rust model-gateway 也需要和 Python server 一致地区分 GLM-4.5/4.6 与 GLM-4.7 tool-call 格式。
- 实现思路：注册 `glm45_moe` 和 `glm47_moe` 两个 parser，模型名 `glm-4.5*`/`glm-4.6*` 映射到前者，`glm-4.7*` 映射到后者。
- 关键代码：

```rust
registry.register_parser("glm45_moe", || Box::new(Glm4MoeParser::glm45()));
registry.register_parser("glm47_moe", || Box::new(Glm4MoeParser::glm47()));
registry.map_model("glm-4.6*", "glm45_moe");
registry.map_model("glm-4.7*", "glm47_moe");
```

```rust
pub fn glm47() -> Self {
    Self::new(r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>")
}
```

- 读过的文件：`sgl-router/src/tool_parser/parsers/*glm*_parser.rs`、registry、Rust tests
- 验证含义：Python parser 改了以后，Rust model-gateway 测试也要同步补。

### #15753 - GLM detector 支持复杂 JSON Schema

- 链接：https://github.com/sgl-project/sglang/pull/15753
- 状态：已合入，merge commit `8ef5b9052825c2624e3ac91852b16998f6f6ee3c`
- Diff：`4` files，`+869/-20`
- Motivation：真实 tool schema 会有 array、object、nullable、enum、anyOf。旧 parser 只做简单值解析时容易把非字符串参数解析错。
- 实现思路：按工具定义里的 schema 获取每个 arg 的类型，再按类型解析 `<arg_value>`，而不是所有值都按字符串处理。
- 关键代码：

```python
arg_type = get_argument_type(func_name, arg_key, tools)
parsed_value, is_good_json = parse_arguments(arg_value, arg_type)
```

- 读过的文件：`glm4_moe_detector.py`、`glm47_moe_detector.py`、function-call parser tests
- 验证含义：GLM-4.7 parser 测试必须有复杂 schema，不只测字符串。

### #15754 - GLM detector 空函数名和 None 值处理

- 链接：https://github.com/sgl-project/sglang/pull/15754
- 状态：已合入，merge commit `bc8b526edad7cb0b53658a6d230d4f4f5a1d1949`
- Diff：`4` files，`+1513/-140`
- Motivation：模型输出可能出现空函数名、无效 tool name、`None` 风格值或部分 XML。旧 parser 可能抛异常或输出非法 tool-call。
- 实现思路：解析后先校验 function name，空名或无效名走安全返回；参数值进入统一 Python/JSON null-like 解析路径。
- 关键代码：

```python
if not func_name:
    return StreamingParseResult(normal_text=text)
```

```python
if func_name not in tool_indices:
    logger.warning("Invalid tool name ...")
    return StreamingParseResult()
```

- 读过的文件：`glm4_moe_detector.py`、`glm47_moe_detector.py`、parser tests
- 验证含义：malformed GLM tool-call 是生产 parser 合同的一部分。

### #17166 - GLM-4.7 NVFP4 和 MTP 修复

- 链接：https://github.com/sgl-project/sglang/pull/17166
- 状态：已合入，merge commit `2ff0880a0ed1b81f0dc34e45fbccaa244cf80cf8`
- Diff：`6` files，`+114/-9`
- Motivation：GLM-4.7 FP4/NVFP4 + MTP 存在三类问题：draft model quantization 被错误覆盖，`mtp.safetensors` 存在但没有进入 index，Blackwell modelopt FP4 需要更合适的 MoE backend。
- 实现思路：保留兼容的 CLI/HF quant method；GLM4-MoE NextN 自动把 `mtp.safetensors` 加入权重列表；Blackwell + `modelopt_fp4` 自动选择 `flashinfer_trtllm`。
- 关键代码：

```python
if is_compatible:
    logger.info("Using CLI-specified quantization ...")
elif self.is_draft_model:
    self.quantization = quant_method
```

```python
if (
    arch in ["Glm4MoeForCausalLM", "Glm4MoeForCausalLMNextN"]
    and num_nextn_layers > 0
):
    return hf_weights_files + [mtp_path]
```

```python
if self.quantization == "modelopt_fp4" and self.moe_runner_backend == "auto":
    if check_pkg_version_at_least("flashinfer-python", "0.6.2"):
        self.moe_runner_backend = "flashinfer_trtllm"
```

- 读过的文件：`model_config.py`、loader、weight utils、`glm4_moe.py`、server args
- 验证含义：GLM-4.7-FP4/NVFP4 要检查 MTP 权重、draft accept length、Blackwell backend 自动选择。

### #17247 - GLM-4.7-Flash 模型支持

- 链接：https://github.com/sgl-project/sglang/pull/17247
- 状态：已合入，merge commit `76b06bee03e8d5e5fbd57dfbdbc80688705988ac`
- Diff：`6` files，`+842/-12`
- Motivation：GLM-4.7-Flash 使用 `Glm4MoeLiteForCausalLM`，SGLang 需要独立的 lite 模型类、MTP/NextN 配置、chat template 兼容和 shape 推导。
- 实现思路：新增 `glm4_moe_lite.py`，复用 DeepSeek MLA/MoE 结构但实现 GLM Lite gate、SparseMoeBlock、shared-expert fusion、`EntryClass`；config 里将 Lite draft 改写为 NextN，Lite scaling 设为 `1`。
- 关键代码：

```python
if is_draft_model and self.hf_config.architectures[0] in [
    "Glm4MoeForCausalLM",
    "Glm4MoeLiteForCausalLM",
]:
    self.hf_config.architectures[0] = "Glm4MoeForCausalLMNextN"
```

```python
if "Glm4MoeLiteForCausalLM" in self.hf_config.architectures:
    self.scaling = 1
    self.hf_config.rope_scaling = None
```

- 读过的文件：`glm4_moe_lite.py`、`model_config.py`、server args、serving chat、attention backend
- 验证含义：GLM-4.7-Flash 不能只靠 GLM-4.7 full model 测试，需要 BF16、量化、MTP、parser flags、chat template 单独覆盖。

### #19246 - NPU optimize GLM-4.7

- 链接：https://github.com/sgl-project/sglang/pull/19246
- 状态：已合入，merge commit `ad0516d9c1f8235edf594f14b76106dcc8b7e469`
- Diff：`4` files，`+146/-15`
- Motivation：GLM-4.7 在 NPU 上需要更好的 decode 性能和 draft 行为。PR 描述给出 GSM8K accuracy `0.915`、latency `86.270s`、output throughput `318.951 tok/s`。
- 实现思路：NPU utils 增加 shared/routed expert stream；GLM4-MoE decode 使用 `split_qkv_rmsnorm_rope` 合并 split、QK norm、RoPE；NextN 支持 unquant draft，通过临时 env 切换 BF16 dispatch。
- 关键代码：

```python
def process_shared_expert(hidden_states, forward_func):
    stream = get_share_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_share_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states)
    return shared_output
```

```python
q, k, v = split_qkv_rmsnorm_rope(
    qkv,
    self.rotary_emb.position_sin,
    self.rotary_emb.position_cos,
    self.q_size,
    self.kv_size,
    self.head_dim,
    eps=self.q_norm.variance_epsilon,
)
```

- 读过的文件：NPU utils、ModelSlim RMSNorm、`glm4_moe.py`、`glm4_moe_nextn.py`
- 验证含义：NPU 路径要测 fused QKNorm/RoPE、stream 同步、MTP draft；GPU draft quant 回归由后续 #22823 修。

### #20543 - GLM tool-call value 不再 strip 空白

- 链接：https://github.com/sgl-project/sglang/pull/20543
- 状态：已合入，merge commit `8eb235ab512528de4c55200c09e2cbc3159a94ba`
- Diff：`3` files，`+66/-2`
- Motivation：tool-call 经常用于代码编辑或 diff，`arg_value.strip()` 会破坏缩进。
- 实现思路：保留 `arg_key.strip()`，移除 GLM4 和 GLM47 detector 里的 `arg_value.strip()`，并增加缩进字符串测试。
- 关键代码：

```diff
 for arg_key, arg_value in pairs:
     arg_key = arg_key.strip()
-    arg_value = arg_value.strip()
     arg_type = get_argument_type(func_name, arg_key, tools)
```

```python
self.assertEqual(params["old_string"], "    indented code")
self.assertEqual(params["new_string"], "        also indented")
```

- 读过的文件：`glm4_moe_detector.py`、`glm47_moe_detector.py`、parser tests
- 验证含义：agentic coding 场景下 parser 必须精确保留空白。

### #21135 - 用 `get_rope_config()` 兼容没有 `rope_parameters` 的 config

- 链接：https://github.com/sgl-project/sglang/pull/21135
- 状态：已合入，merge commit `646573e4e8d10c2684e0563bc40915b4bef874f4`
- Diff：`18` files，`+44/-42`
- Motivation：Transformers 升级后很多代码直接读 `config.rope_parameters["rope_theta"]`，但 GLM4-MoE 等 trust-remote-code config 不一定有这个字段。
- 实现思路：GLM4 和 GLM4-MoE 改为通过 `get_rope_config(config)` 获取 rope theta/scaling，partial rotary factor 从 rope scaling 或 config fallback。
- 关键代码：

```python
rope_theta, rope_scaling = get_rope_config(config)
partial_rotary_factor = (rope_scaling or {}).get("partial_rotary_factor")
if partial_rotary_factor is None:
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
```

- 读过的文件：`glm4.py`、`glm4_moe.py`、`hf_transformers_utils.py` 以及批量模型改动
- 验证含义：GLM-4.6/4.7 config loading 失败时先查 rope config 路径。

### #21403 - AMD GLM-4.7-FP8 融合 RMSNorm + per-token FP8 quant

- 链接：https://github.com/sgl-project/sglang/pull/21403
- 状态：已合入，merge commit `7e4e1dcd7ac85f20e48e442515c352aa201049fb`
- Diff：`3` files，`+149/-13`
- Motivation：AMD GLM-4.7-FP8 中 RMSNorm 后的 per-token FP8 quant 有额外 global memory round trip。PR 报告 MI355X TP8 decode ITL 约 `+1%`，GSM8K `0.948 -> 0.943`，在波动范围。
- 实现思路：`LayerCommunicator.prepare_attn()` 支持 `quant_format="fp8_per_token"`，调用 AITER fused RMSNorm quant；FP8 linear 接受 `(q_input, x_scale)` tuple，避免重复 quant；GLM4-MoE 自动识别 CompressedTensors W8A8 FP8 channel strategy。
- 关键代码：

```python
def _fused_rmsnorm_fp8_per_token_quant(...):
    out_fp8 = torch.empty((M, N), dtype=_aiter_fp8_dtype, device=hidden_states.device)
    scale = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
    _aiter_rmsnorm_quant(out_fp8, hidden_states, scale, weight, epsilon, 0)
    return (out_fp8, scale.unsqueeze(1))
```

```python
if isinstance(input, tuple):
    q_input, x_scale = input
    output = aiter.gemm_a8w8_bpreshuffle(
        q_input, weight, x_scale, weight_scale, None, torch.bfloat16
    )
```

- 读过的文件：`communicator.py`、`fp8_utils.py`、`glm4_moe.py`
- 验证含义：AMD FP8 要比较开启/关闭 fused RMSNorm quant 的 accuracy 和 ITL，tuple hidden states 是主要风险点。

### #21534 - AMD GLM-4.7-FP8 MI35x accuracy CI

- 链接：https://github.com/sgl-project/sglang/pull/21534
- 状态：已合入，merge commit `7078e385ea137e380b091caf41f460444867ba85`
- Diff：`2` files，`+96/-0`
- Motivation：GLM-4.7-FP8 需要 AMD MI35x nightly accuracy gate，防止 ROCm/AITER/GLM MoE 改动静默回归。
- 实现思路：ROCm nightly 增加 8-GPU MI35x job，测试 `zai-org/GLM-4.7-FP8`、TP8、baseline accuracy `0.92`，并固定 `glm47` tool parser 和 `glm45` reasoning parser。
- 关键代码：

```yaml
- nightly-8-gpu-mi35x-glm47-fp8-rocm720
```

```python
base_args = [
    "--trust-remote-code",
    "--tool-call-parser=glm47",
    "--reasoning-parser=glm45",
]
```

- 读过的文件：AMD ROCm workflow、registered AMD test
- 验证含义：AMD GLM-4.7 性能改动必须关注这个 accuracy gate。

### #21660 - GLM gate projection 转 FP32

- 链接：https://github.com/sgl-project/sglang/pull/21660
- 状态：已合入，merge commit `ad064c2f4e33e1ad2f5ad50b40bb1ab2fb3e4657`
- Diff：`1` file，`+6/-1`
- Motivation：GLM expert routing 对 gate logits 精度敏感，低精度 gate projection 会扰动 expert selection。
- 实现思路：`Glm4MoeGate` 缓存一份 FP32 gate weight，并把 hidden states cast 到 FP32 后做线性投影。
- 关键代码：

```python
self.register_buffer("_weight_fp32", None, persistent=False)
```

```python
if self._weight_fp32 is None:
    self._weight_fp32 = self.weight.data.to(torch.float32)
logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)
```

- 读过的文件：`glm4_moe.py`
- 验证含义：如果运行时会更新 gate weight，必须失效 `_weight_fp32`；常规路径要看 routing-sensitive accuracy。

### #21851 - GLM-4.7 / GLM-4.7-Flash loading 和 import format

- 链接：https://github.com/sgl-project/sglang/pull/21851
- 状态：已合入，merge commit `b7ae3b5a9a57236c64e513276ab15bbabad4c4e7`
- Diff：`2` files，`+139/-86`
- Motivation：GLM-4.7-Flash 没有 EAGLE 实现，import/comment 路径陈旧，`glm4_moe.py` 与 `deepseek_v2.py` 行为漂移，导致 Flash/lite、A2A backend、shared-expert fusion、rope config 都不稳定。
- 实现思路：GLM4-MoE 支持更多 A2A backend guard，A2A/FP4 allgather 下 shared experts 使用 `tp_size=1`；shared-expert fusion 支持 AMD gfx942 并禁用 W4AFP8；GLM4-MoE-Lite 使用 `get_rope_config` 并移除 EAGLE 相关逻辑。
- 关键代码：

```python
dict(tp_rank=0, tp_size=1)
if get_moe_a2a_backend().is_deepep()
or get_moe_a2a_backend().is_flashinfer()
or should_use_flashinfer_cutlass_moe_fp4_allgather()
else {}
```

```python
rope_theta, rope_scaling = get_rope_config(config)
```

- 读过的文件：`glm4_moe.py`、`glm4_moe_lite.py`
- 验证含义：GLM-4.7-Flash 不应走 EAGLE 路径；每个 A2A backend 要独立确认 shared-expert fusion guard。

### #22509 - NPU GLM-4.7-Flash 修复

- 链接：https://github.com/sgl-project/sglang/pull/22509
- 状态：已合入，merge commit `92f28e9ba80b81bba9f82a4c0a69dccf81ff581c`
- Diff：`2` files，`+4/-2`
- Motivation：GPU op 优化让 GLM-4.7-Flash 在 NPU 上失败，原因是 CUDA kernel import 和 AMD quant-format attribute 假设泄漏到 NPU。
- 实现思路：去掉模块级 `sgl_kernel.dsv3_router_gemm` import；DeepSeek V2 attention 调用 `_gfx95_quant_format` 时用 `getattr` 给默认值。
- 关键代码：

```diff
-from sgl_kernel import dsv3_router_gemm
```

```diff
-            self._gfx95_quant_format,
+            getattr(self, "_gfx95_quant_format", ""),
```

- 读过的文件：`glm4_moe_lite.py`、`deepseek_v2.py`
- 验证含义：NPU smoke test 要覆盖 import、prefill/decode 和 GLM-4.7 parser flags。

### #22720 - GLM-4.7-Flash 检测 `gfx95_quant_format`

- 链接：https://github.com/sgl-project/sglang/pull/22720
- 状态：已合入，merge commit `6b2bf66cd9cd0448b0e9f3af8a54e9e10686fdf2`
- Diff：`1` file，`+2/-0`
- Motivation：`Glm4MoeLiteDecoderLayer` 缺少 `_gfx95_quant_format`，但 DeepSeek V2 路径会读取它，导致 GLM-4.7-Flash 启动失败。
- 实现思路：在创建 communicator 之前调用 `_detect_gfx95_quant_format()` 初始化属性。
- 关键代码：

```python
self._gfx95_quant_format = self._detect_gfx95_quant_format()
```

- 读过的文件：`glm4_moe_lite.py`
- 验证含义：这是小 diff 但影响启动；AMD quantized GLM-4.7-Flash 要重点回归。

### #22823 - 保留 GLM NextN draft auto-detected `quant_config`

- 链接：https://github.com/sgl-project/sglang/pull/22823
- 状态：已合入，merge commit `28e915b474eba6d132a65b28c8325b1bbc3f572a`
- Diff：`1` file，`+2/-1`
- Motivation：#19246 让 NextN draft quant 依赖 `server_args.speculative_draft_model_quantization`。自动检测 compressed-tensors FP8 时用户通常不传 `--quantization`，draft 被加载成 BF16，accept length 从约 `2.0` 掉到 `1.0`，吞吐从 `1018.8 tok/s` 级别掉到 `489.22 tok/s`。
- 实现思路：只要 loader 传入了 `quant_config`，就认为 draft 需要量化，不再依赖 CLI 显式参数。
- 关键代码：

```python
self.needs_quant_draft = (
    get_global_server_args().speculative_draft_model_quantization is not None
    or quant_config is not None
)
quant_config = quant_config if self.needs_quant_draft else None
```

- 读过的文件：`glm4_moe_nextn.py`
- 验证含义：GLM-4.7-FP8 / GLM-4.6-FP8 EAGLE/NEXTN 必须检查 draft quant config 和 average accept length。

## Open PR Radar

### #11951 - WIP GLM-4.6 tool-call streaming parser

- 链接：https://github.com/sgl-project/sglang/pull/11951
- 状态：Open，`3` files，`+450/-105`
- Motivation：早期尝试解决 GLM-4.6 tool-call 参数无法 streaming 的问题。
- 实现思路：Python/Rust parser 都添加 `current_tool_name_sent`、partial parser、argument diff。
- 关键代码：

```python
if not self.current_tool_name_sent:
    self.current_tool_name_sent = True
    calls.append(ToolCallItem(tool_index=tool_id, name=func_name, parameters=""))
```

- 备注：已合入的 #13989 是当前主线实现，#11951 只作为设计背景。

### #17869 - NPU 支持 GLM-4.7-Flash

- 链接：https://github.com/sgl-project/sglang/pull/17869
- 状态：Open，`4` files，`+86/-5`
- Motivation：GLM-4.7-Flash 之前不支持 NPU，PR 描述给出 `81%` accuracy 和启动命令。
- 实现思路：NPU `forward_extend` 增加 `qk_head_dim == v_head_dim` 分支，逐请求调用 `torch.ops.npu.npu_fused_infer_attention_score`，并加 Ascend GLM-4.7-Flash GSM8K 测试。
- 关键代码：

```python
if layer.qk_head_dim == layer.v_head_dim:
    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    torch.ops.npu.npu_fused_infer_attention_score(...)
```

- 验证含义：需要和已合入 #19246、#22509、open #22801 对齐后再采用启动参数。

### #18930 - AMD GLM-4.7 MTP 测试

- 链接：https://github.com/sgl-project/sglang/pull/18930
- 状态：Open，`2` files，`+120/-1`
- Motivation：MI300 上 GLM-4.7-FP8 + speculative decoding 出现 garbage output，`spec_accept_rate` 接近 0。
- 实现思路：新增失败型 canary 测试，启动 TP8、EAGLE、3 steps、draft tokens 4，然后检查 GSM8K accuracy、accept rate、average accept length。
- 关键代码：

```python
self.assertGreater(spec_accept_rate, 0.5)
self.assertGreater(avg_spec_accept_length, 2.0)
```

- 验证含义：#22823 修的就是同类 draft quant / accept length 问题，这个测试适合作为 AMD MTP canary。

### #19040 - `Glm4MoeLiteConfig` 和 `enable_a2a_moe`

- 链接：https://github.com/sgl-project/sglang/pull/19040
- 状态：Open，`4` files，`+52/-0`
- Motivation：`glm4_moe_lite` 没有被 Transformers 原生注册，GLM-4.7-Flash config 解析失败；同时 Lite model 绕过 DeepSeek init 后没有 `enable_a2a_moe`。
- 实现思路：新增 `Glm4MoeLiteConfig`，注册到 SGLang config registry，并在 `Glm4MoeLiteModel.__init__` 设置 `self.enable_a2a_moe = False`。
- 关键代码：

```python
class Glm4MoeLiteConfig(Glm4MoeConfig):
    model_type = "glm4_moe_lite"
```

```python
self.enable_a2a_moe = False
```

- 验证含义：如果合入，需要重测 Flash 不带 `trust_remote_code` 的 config loading 和 A2A guard。

### #19106 - GLM4 MoE Lite CompressedTensors / AWQ serving

- 链接：https://github.com/sgl-project/sglang/pull/19106
- 状态：Open，`12` files，`+505/-37`
- Motivation：`GLM-4.7-Flash-REAP-23B-A3B-AWQ-4bit` 因 packed module 没有 `.weight` 启动失败，同时 `glm4_moe_lite` 收到错误的 Transformers 降级提示。
- 实现思路：MLA fused path 读取 `.weight` 前先 guard；CT WNA16 `kv_b_proj` 反量化生成 `w_kc/w_vc`；补 packed module mapping；如果 quant config ignore 了 shared experts，则禁用 shared-expert fusion；`glm4_moe_lite` 走 TF>=5 检查。
- 关键代码：

```python
fused_qkv_a_proj = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
if (
    fused_qkv_a_proj is not None
    and getattr(fused_qkv_a_proj, "weight", None) is not None
    and use_intel_amx_backend(attn)
):
    return AttnForwardMethod.MLA_FUSED_ROPE_CPU
```

```python
packed_modules_mapping = {
    "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

- 验证含义：这是 GLM-4.7-Flash compressed-tensors/AWQ 的主要 open 风险。

### #22315 - GLM-4.7-FP8 EAGLE accept_len=1.00 修复尝试

- 链接：https://github.com/sgl-project/sglang/pull/22315
- 状态：Open，`1` file，`+7/-5`
- Motivation：#19246 的 NPU unquant draft 逻辑让 GPU GLM-4.7-FP8 draft 丢失 quant_config，accept length 变成 `1.00`。
- 实现思路：只在 `is_npu()` 时允许根据 `speculative_draft_model_quantization` 清空 quant_config；GPU 始终保留 draft quant。
- 关键代码：

```python
self.needs_quant_draft = True
if is_npu():
    self.needs_quant_draft = (
        get_global_server_args().speculative_draft_model_quantization
    )
    quant_config = quant_config if self.needs_quant_draft else None
```

- 备注：后续已合入 #22823 用更通用的方式解决了自动检测 quant_config 保留问题。

### #22801 - NPU dual-stream / DeepEP 支持 GLM-4.7-Flash

- 链接：https://github.com/sgl-project/sglang/pull/22801
- 状态：Open，`2` files，`+14/-3`
- Motivation：GLM-4.7-Flash 在 NPU 上需要 dual-stream 和 DeepEP 支持。
- 实现思路：DeepEP 在 `SGLANG_DEEPEP_BF16_DISPATCH` 下不强制 FP8 dispatch；Lite gate 增加 `forward_batch` 参数；NPU 可通过 `SGLANG_NPU_USE_MULTI_STREAM` 创建 `alt_stream`。
- 关键代码：

```python
elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
    use_fp8 = True
```

```python
self.alt_stream = (
    torch.cuda.Stream()
    if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
    else None
)
```

- 验证含义：如果合入，要分别测开启/关闭 `SGLANG_NPU_USE_MULTI_STREAM` 的 NPU GLM-4.7-Flash。

### #23067 - `Glm45Detector` 支持 `continue_final_message`

- 链接：https://github.com/sgl-project/sglang/pull/23067
- 状态：Open，`2` files，`+66/-1`
- Motivation：`ReasoningParser` 在 `continue_final_message=true` 时会传 `continue_final_message` 和 `previous_content`，但 `Glm45Detector` 不接受这两个参数，导致使用 `--reasoning-parser glm45` 的 GLM-4.7/GLM-5 请求 HTTP 500。
- 实现思路：给 `Glm45Detector.__init__` 增加两个参数并转发给 base detector。
- 关键代码：

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
        continue_final_message=continue_final_message,
        previous_content=previous_content,
    )
```

- 验证含义：GLM-4.7 虽然 tool parser 是 `glm47`，reasoning parser 仍是 `glm45`，所以这个 open PR 和 GLM-4.7 相关。

## 推荐验证矩阵

- GLM-4.6 BF16 + `glm45` tool/reasoning parser。
- GLM-4.6 shared-expert fusion 单独开关。
- GLM-4.6 CUDA graph dual-stream decode。
- GLM-4.7 BF16 + `glm47` tool parser + `glm45` reasoning parser。
- GLM-4.7-FP8 TP8，无 MTP。
- GLM-4.7-FP8 TP8 + EAGLE/NEXTN，检查 average accept length。
- GLM-4.7 NVFP4/modelopt FP4 on Blackwell，检查 `flashinfer_trtllm` auto-selection。
- GLM-4.7-Flash BF16 / quantized / compressed-tensors-AWQ。
- AMD MI35x/MI355X FP8 path，检查 AITER fused RMSNorm quant。
- NPU GLM-4.7/Flash，检查 fused QKNorm/RoPE、dual stream、parser flags。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `GLM-4.6 / GLM-4.7`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-10-22 | [#11951](https://github.com/sgl-project/sglang/pull/11951) | open | WIP: Fix glm-4.6 tool call streaming parse | MoE/router, tests/benchmarks | `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs` |
| 2025-10-31 | [#12456](https://github.com/sgl-project/sglang/pull/12456) | merged | [fix] Handle escaped characters in GLM tool call parser to prevent double serialization | MoE/router, tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-11-23 | [#13786](https://github.com/sgl-project/sglang/pull/13786) | merged | Overlap glm moe gemms in two cuda streams | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2025-11-25 | [#13873](https://github.com/sgl-project/sglang/pull/13873) | merged | Feat: GLM-4.6 supports shared experts fusion | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2025-11-26 | [#13989](https://github.com/sgl-project/sglang/pull/13989) | merged | Fix GLM-4.6 tool calls don't support streaming output for arguments i… | MoE/router, tests/benchmarks | `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-07 | [#14585](https://github.com/sgl-project/sglang/pull/14585) | merged | [Glm46v] Bug fix for accuracy drop and unable to launch server | model wrapper, MoE/router, multimodal/processor, docs/config | `docs/basic_usage/glmv.md`, `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glm45.md` |
| 2025-12-08 | [#14668](https://github.com/sgl-project/sglang/pull/14668) | merged | [NVIDIA] Add flashinfer all-to-all MOE dispatcher | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/test/test_flashinfer_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` |
| 2025-12-17 | [#15333](https://github.com/sgl-project/sglang/pull/15333) | merged | [GLM-4.7] GLM-4.7 Tool Parser and Doc Update | model wrapper, MoE/router, tests/benchmarks, docs/config | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-12-20 | [#15520](https://github.com/sgl-project/sglang/pull/15520) | merged | [model-gateway]: Tool parser for glm47 | MoE/router, tests/benchmarks | `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` |
| 2025-12-24 | [#15753](https://github.com/sgl-project/sglang/pull/15753) | merged | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/utils.py`, `python/sglang/srt/function_call/glm47_moe_detector.py` |
| 2025-12-24 | [#15754](https://github.com/sgl-project/sglang/pull/15754) | merged | Fix: Handle empty func_name and None values in GLM MoE detectors | MoE/router, tests/benchmarks | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-15 | [#17166](https://github.com/sgl-project/sglang/pull/17166) | merged | [Fix] GLM 4.7 + NVFP4 + MTP | model wrapper, MoE/router, docs/config | `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py` |
| 2026-01-17 | [#17247](https://github.com/sgl-project/sglang/pull/17247) | merged | [New Model] GLM4.7-Flash | model wrapper, attention/backend, MoE/router, docs/config | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| 2026-01-28 | [#17869](https://github.com/sgl-project/sglang/pull/17869) | open | [NPU]Support model GLM-4.7-Flash for npu, accuracy 81% | model wrapper, attention/backend, tests/benchmarks | `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-02-17 | [#18930](https://github.com/sgl-project/sglang/pull/18930) | open | [AMD] Unit tests for mtp in GLM-4.7 | attention/backend, quantization, tests/benchmarks | `test/registered/amd/test_glm4v_fp8_mtp.py`, `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-02-20 | [#19040](https://github.com/sgl-project/sglang/pull/19040) | open | feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash | model wrapper, MoE/router, docs/config | `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/utils/hf_transformers_utils.py` |
| 2026-02-21 | [#19106](https://github.com/sgl-project/sglang/pull/19106) | open | Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks | model wrapper, attention/backend, MoE/router, tests/benchmarks, docs/config | `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_model_config_transformers_version.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-24 | [#19246](https://github.com/sgl-project/sglang/pull/19246) | merged | [NPU] optimize glm4.7 | model wrapper, MoE/router, quantization | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-03-13 | [#20543](https://github.com/sgl-project/sglang/pull/20543) | merged | fix: do not strip whitespace from GLM tool call values | MoE/router, tests/benchmarks | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-03-22 | [#21135](https://github.com/sgl-project/sglang/pull/21135) | merged | fix: use get_rope_config() to support models without rope_parameters | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py` |
| 2026-03-25 | [#21403](https://github.com/sgl-project/sglang/pull/21403) | merged | [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8 | model wrapper, MoE/router, quantization | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-03-27 | [#21534](https://github.com/sgl-project/sglang/pull/21534) | merged | [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x | quantization, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-30 | [#21660](https://github.com/sgl-project/sglang/pull/21660) | merged | [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model. | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2026-04-01 | [#21851](https://github.com/sgl-project/sglang/pull/21851) | merged | GLM-4.7 and GLM-4.7-Flash Loading and import format | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-08 | [#22315](https://github.com/sgl-project/sglang/pull/22315) | open | [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-04-10 | [#22509](https://github.com/sgl-project/sglang/pull/22509) | merged | [NPU]Fix GLM-4.7-Flash failed on NPU | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-13 | [#22720](https://github.com/sgl-project/sglang/pull/22720) | merged | fix[glm4.7 flash]: properly detect `gfx95_quant_format` | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-14 | [#22801](https://github.com/sgl-project/sglang/pull/22801) | open | [NPU]add dual-stream and deepep support for GLM-4.7-Flash | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |
| 2026-04-14 | [#22823](https://github.com/sgl-project/sglang/pull/22823) | merged | [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-04-17 | [#23067](https://github.com/sgl-project/sglang/pull/23067) | open | Fix: forward continue_final_message kwargs in Glm45Detector | tests/benchmarks | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |

### 逐 PR 代码 diff 阅读记录

### PR #11951 - WIP: Fix glm-4.6 tool call streaming parse

- 链接：https://github.com/sgl-project/sglang/pull/11951
- 状态/时间：`open`，created 2025-10-22；作者 `tonylt`。
- 代码 diff 已读范围：`3` 个文件，`+450/-105`；代码面：MoE/router, tests/benchmarks；关键词：moe, router, config, test。
- 代码 diff 细节：
  - `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86 (284 lines); hunk: pub struct Glm4MoeParser {; impl Glm4MoeParser {; 符号: Glm4MoeParser
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19 (199 lines); hunk: from sglang.srt.entrypoints.openai.protocol import Tool; def parse_streaming_increment(; 符号: parse_streaming_increment, parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix
  - `sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0 (72 lines); hunk: async fn test_glm4_nested_json_in_arg_values() {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs`；patch 关键词为 moe, router, config, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12456 - [fix] Handle escaped characters in GLM tool call parser to prevent double serialization

- 链接：https://github.com/sgl-project/sglang/pull/12456
- 状态/时间：`merged`，created 2025-10-31, merged 2025-11-05；作者 `soaringk`。
- 代码 diff 已读范围：`2` 个文件，`+127/-13`；代码面：MoE/router, tests/benchmarks；关键词：moe, test。
- 代码 diff 细节：
  - `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunk: def test_partial_tool_call(self):; 符号: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunk: def get_argument_type(func_name: str, arg_key: str, defined_tools: list):; def __init__(self):; 符号: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；patch 关键词为 moe, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #14585 - [Glm46v] Bug fix for accuracy drop and unable to launch server

- 链接：https://github.com/sgl-project/sglang/pull/14585
- 状态/时间：`merged`，created 2025-12-07, merged 2025-12-08；作者 `byjiang1996`。
- 代码 diff 已读范围：`12` 个文件，`+308/-29`；代码面：model wrapper, MoE/router, multimodal/processor, docs/config；关键词：moe, config, attention, processor, doc, quant, vision, cache, cuda, eagle。
- 代码 diff 细节：
  - `docs/basic_usage/glmv.md` added +136/-0 (136 lines); hunk: +# GLM-4.6V / GLM-4.5V Usage
  - `python/sglang/srt/models/glm4v_moe.py` modified +68/-15 (83 lines); hunk: import torch.nn as nn; def __init__(; 符号: __init__, __init__, determine_num_fused_shared_experts, load_weights
  - `docs/basic_usage/glm45.md` added +70/-0 (70 lines); hunk: +## Launch GLM-4.5 / GLM-4.6 with SGLang
  - `python/sglang/srt/models/glm4v.py` modified +19/-3 (22 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, get_video_feature
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0 (8 lines); hunk: def __call__(self, logits, custom_param_list: list[dict[str, Any]]):; 符号: __call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/glmv.md`, `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glm45.md`；patch 关键词为 moe, config, attention, processor, doc, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/glmv.md`, `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glm45.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #15520 - [model-gateway]: Tool parser for glm47

- 链接：https://github.com/sgl-project/sglang/pull/15520
- 状态/时间：`merged`，created 2025-12-20, merged 2025-12-21；作者 `UbeCc`。
- 代码 diff 已读范围：`8` 个文件，`+179/-26`；代码面：MoE/router, tests/benchmarks；关键词：moe, spec, test, config, benchmark, cache, doc, router。
- 代码 diff 细节：
  - `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0 (132 lines); hunk: +//! GLM-4.7 MoE Parser Integration Tests
  - `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8 (30 lines); hunk: use crate::{; pub struct Glm4MoeParser {; 符号: Glm4MoeParser
  - `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7 (14 lines); hunk: use common::create_test_tools;; The weather will be..."#;
  - `sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3 (8 lines); hunk: impl ParserFactory {; impl ParserFactory {
  - `sgl-model-gateway/benches/tool_parser_benchmark.rs` modified +5/-2 (7 lines); hunk: Let me examine the scan results and provide recommendations."#;; analyze_customer_behavior
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`；patch 关键词为 moe, spec, test, config, benchmark, cache。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #17166 - [Fix] GLM 4.7 + NVFP4 + MTP

- 链接：https://github.com/sgl-project/sglang/pull/17166
- 状态/时间：`merged`，created 2026-01-15, merged 2026-01-21；作者 `b8zhong`。
- 代码 diff 已读范围：`6` 个文件，`+114/-9`；代码面：model wrapper, MoE/router, docs/config；关键词：config, quant, fp4, moe, cache, flash, spec, attention, expert, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/model_loader/weight_utils.py` modified +38/-0 (38 lines); hunk: def filter_duplicate_safetensors_files(; 符号: filter_duplicate_safetensors_files, maybe_add_mtp_safetensors, filter_files_not_needed_for_inference
  - `python/sglang/srt/configs/model_config.py` modified +17/-8 (25 lines); hunk: def _verify_quantization(self) -> None:; 符号: _verify_quantization
  - `python/sglang/srt/server_args.py` modified +22/-0 (22 lines); hunk: from sglang.srt.utils.common import (; def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
  - `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunk: def assert_pkg_version(pkg: str, min_version: str, message: str):; 符号: assert_pkg_version, check_pkg_version_at_least, kill_process_tree
  - `python/sglang/srt/model_loader/loader.py` modified +14/-0 (14 lines); hunk: get_quant_config,; class Source:; 符号: Source:, init_new, __init__, _get_weights_iterator
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`；patch 关键词为 config, quant, fp4, moe, cache, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17247 - [New Model] GLM4.7-Flash

- 链接：https://github.com/sgl-project/sglang/pull/17247
- 状态/时间：`merged`，created 2026-01-17, merged 2026-01-20；作者 `Qiaolin-Yu`。
- 代码 diff 已读范围：`6` 个文件，`+842/-12`；代码面：model wrapper, attention/backend, MoE/router, docs/config；关键词：kv, moe, attention, config, cuda, flash, fp8, mla, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0 (808 lines); hunk: +# Copyright 2025-2026 SGLang Team; 符号: Glm4MoeLiteMLP, __init__, forward, Glm4MoeLiteGate
  - `python/sglang/srt/configs/model_config.py` modified +19/-9 (28 lines); hunk: def _config_draft_model(self):; def _derive_model_shapes(self):; 符号: _config_draft_model, _derive_model_shapes, _derive_model_shapes
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2 (9 lines); hunk: _use_aiter_gfx95,; def _concat_and_cast_mha_k(; 符号: _concat_and_cast_mha_k
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/server_args.py` modified +3/-0 (3 lines); hunk: def _handle_model_specific_adjustments(self):; def _handle_speculative_decoding(self):; 符号: _handle_model_specific_adjustments, _handle_speculative_decoding, auto_choose_speculative_params
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；patch 关键词为 kv, moe, attention, config, cuda, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17869 - [NPU]Support model GLM-4.7-Flash for npu, accuracy 81%

- 链接：https://github.com/sgl-project/sglang/pull/17869
- 状态/时间：`open`，created 2026-01-28；作者 `McZyWu`。
- 代码 diff 已读范围：`4` 个文件，`+86/-5`；代码面：model wrapper, attention/backend, tests/benchmarks；关键词：attention, flash, cache, kv, lora, test, cuda, mla, quant, topk。
- 代码 diff 细节：
  - `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0 (54 lines); hunk: +import os; 符号: TestGLM47Flash
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4 (34 lines); hunk: def forward_extend(; 符号: forward_extend
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunk: def forward_mha_prepare_npu(; 符号: forward_mha_prepare_npu
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0 (1 lines); hunk: DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH = os.path.join(
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`；patch 关键词为 attention, flash, cache, kv, lora, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18930 - [AMD] Unit tests for mtp in GLM-4.7

- 链接：https://github.com/sgl-project/sglang/pull/18930
- 状态/时间：`open`，created 2026-02-17；作者 `almaslof`。
- 代码 diff 已读范围：`2` 个文件，`+120/-1`；代码面：attention/backend, quantization, tests/benchmarks；关键词：attention, cache, config, cuda, eagle, fp8, kv, mla, spec, test。
- 代码 diff 细节：
  - `test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0 (118 lines); hunk: +import unittest; 符号: TestGLM47FP8TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1 (3 lines); hunk: def init_forward_metadata_capture_cuda_graph(; 符号: init_forward_metadata_capture_cuda_graph
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/test_glm4v_fp8_mtp.py`, `python/sglang/srt/layers/attention/aiter_backend.py`；patch 关键词为 attention, cache, config, cuda, eagle, fp8。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/test_glm4v_fp8_mtp.py`, `python/sglang/srt/layers/attention/aiter_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19040 - feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash

- 链接：https://github.com/sgl-project/sglang/pull/19040
- 状态/时间：`open`，created 2026-02-20；作者 `lujangus`。
- 代码 diff 已读范围：`4` 个文件，`+52/-0`；代码面：model wrapper, MoE/router, docs/config；关键词：config, moe, attention, flash, kv, lora, mla, spec。
- 代码 diff 细节：
  - `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0 (47 lines); hunk: +# Copyright 2025-2026 SGLang Team; 符号: Glm4MoeLiteConfig, with, __init__
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunk: from sglang.srt.configs.dots_vlm import DotsVLMConfig; "DotsVLMConfig",
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunk: DotsVLMConfig,; KimiLinearConfig,
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/utils/hf_transformers_utils.py`；patch 关键词为 config, moe, attention, flash, kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/utils/hf_transformers_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #19246 - [NPU] optimize glm4.7

- 链接：https://github.com/sgl-project/sglang/pull/19246
- 状态/时间：`merged`，created 2026-02-24, merged 2026-04-03；作者 `randgun`。
- 代码 diff 已读范围：`4` 个文件，`+146/-15`；代码面：model wrapper, MoE/router, quantization；关键词：expert, config, deepep, moe, quant, topk, attention, cuda, kv, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +61/-11 (72 lines); hunk: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; is_cuda,; 符号: Glm4MoeMLP, __init__, forward_prepare, forward_deepep
  - `python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0 (64 lines); hunk: def get_indexer_weight_stream():; 符号: get_indexer_weight_stream, get_share_stream, set_share_stream, get_routed_stream
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2 (21 lines); hunk: """Inference-only GLM-4.5, GLM-4.6 Speculative Decoding."""; from transformers import PretrainedConfig; 符号: __init__, forward
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2 (4 lines); hunk: def _rmsnorm_forward_oot(; 符号: _rmsnorm_forward_oot
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；patch 关键词为 expert, config, deepep, moe, quant, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/models/glm4_moe_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #21135 - fix: use get_rope_config() to support models without rope_parameters

- 链接：https://github.com/sgl-project/sglang/pull/21135
- 状态/时间：`merged`，created 2026-03-22, merged 2026-03-26；作者 `alphabetc1`。
- 代码 diff 已读范围：`18` 个文件，`+44/-42`；代码面：model wrapper, MoE/router；关键词：config, attention, kv, cuda, moe, cache, expert, lora。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunk: log_info_on_rank0,; def __init__(; 符号: __init__
  - `python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunk: kv_cache_scales_loader,; def __init__(; 符号: __init__
  - `python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunk: from sglang.srt.model_loader.loader import DefaultModelLoader; def __init__(; 符号: __init__
  - `python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunk: is_npu,; def __init__(; 符号: __init__
  - `python/sglang/srt/models/deepseek.py` modified +2/-2 (4 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`；patch 关键词为 config, attention, kv, cuda, moe, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21403 - [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8

- 链接：https://github.com/sgl-project/sglang/pull/21403
- 状态/时间：`merged`，created 2026-03-25, merged 2026-04-11；作者 `Jacob0226`。
- 代码 diff 已读范围：`3` 个文件，`+149/-13`；代码面：model wrapper, MoE/router, quantization；关键词：fp8, quant, cache, moe, attention, config, expert, flash, fp4, kv。
- 代码 diff 细节：
  - `python/sglang/srt/layers/communicator.py` modified +76/-8 (84 lines); hunk: _is_npu = is_npu(); def model_input_output():; 符号: _fused_rmsnorm_fp8_per_token_quant, model_input_output, AttentionInputs:, __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +58/-3 (61 lines); hunk: def forward_prepare(; def __init__(; 符号: forward_prepare, __init__, _detect_fp8_per_token_quant, _detect_attn_quant_format
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +15/-2 (17 lines); hunk: import logging; def can_auto_enable_marlin_fp8() -> bool:; 符号: can_auto_enable_marlin_fp8, apply_fp8_ptpc_linear, apply_fp8_ptpc_linear
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`；patch 关键词为 fp8, quant, cache, moe, attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21534 - [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x

- 链接：https://github.com/sgl-project/sglang/pull/21534
- 状态/时间：`merged`，created 2026-03-27, merged 2026-03-28；作者 `Jacob0226`。
- 代码 diff 已读范围：`2` 个文件，`+96/-0`；代码面：quantization, tests/benchmarks；关键词：fp8, test, benchmark, cache, doc, fp4。
- 代码 diff 细节：
  - `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0 (61 lines); hunk: +"""MI35x GLM-4.7-FP8 GSM8K Accuracy Evaluation Test (8-GPU); 符号: TestGLM47FP8EvalMI35x, test_glm_47_fp8
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +35/-0 (35 lines); hunk: on:; jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml`；patch 关键词为 fp8, test, benchmark, cache, doc, fp4。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21660 - [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.

- 链接：https://github.com/sgl-project/sglang/pull/21660
- 状态/时间：`merged`，created 2026-03-30, merged 2026-03-30；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`1` 个文件，`+6/-1`；代码面：model wrapper, MoE/router；关键词：cache, config, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunk: def __init__(; 符号: __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 cache, config, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21851 - GLM-4.7 and GLM-4.7-Flash Loading and import format

- 链接：https://github.com/sgl-project/sglang/pull/21851
- 状态/时间：`merged`，created 2026-04-01, merged 2026-04-04；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`2` 个文件，`+139/-86`；代码面：model wrapper, MoE/router；关键词：config, cuda, eagle, expert, flash, kv, moe, router, spec, attention。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +130/-57 (187 lines); hunk: """Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 model compatible with HuggingFace weights"""; ); 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29 (38 lines); hunk: # limitations under the License.; get_tensor_model_parallel_world_size,; 符号: forward, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；patch 关键词为 config, cuda, eagle, expert, flash, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22315 - [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config

- 链接：https://github.com/sgl-project/sglang/pull/22315
- 状态/时间：`open`，created 2026-04-08；作者 `maodoudou168`。
- 代码 diff 已读范围：`1` 个文件，`+7/-5`；代码面：model wrapper, MoE/router；关键词：config, moe, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5 (12 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe_nextn.py`；patch 关键词为 config, moe, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22509 - [NPU]Fix GLM-4.7-Flash failed on NPU

- 链接：https://github.com/sgl-project/sglang/pull/22509
- 状态/时间：`merged`，created 2026-04-10, merged 2026-04-22；作者 `Todobe`。
- 代码 diff 已读范围：`2` 个文件，`+4/-2`；代码面：model wrapper, MoE/router；关键词：config, cuda, moe, quant, router。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1 (4 lines); hunk: import torch; _is_cuda = is_cuda()
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 config, cuda, moe, quant, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22720 - fix[glm4.7 flash]: properly detect `gfx95_quant_format`

- 链接：https://github.com/sgl-project/sglang/pull/22720
- 状态/时间：`merged`，created 2026-04-13, merged 2026-04-13；作者 `ishandhanani`。
- 代码 diff 已读范围：`1` 个文件，`+2/-0`；代码面：model wrapper, MoE/router；关键词：config, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe_lite.py`；patch 关键词为 config, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe_lite.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22801 - [NPU]add dual-stream and deepep support for GLM-4.7-Flash

- 链接：https://github.com/sgl-project/sglang/pull/22801
- 状态/时间：`open`，created 2026-04-14；作者 `Estrella-xx`。
- 代码 diff 已读范围：`2` 个文件，`+14/-3`；代码面：model wrapper, MoE/router；关键词：config, moe, attention, cuda, deepep, expert, fp4, fp8, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2 (15 lines); hunk: get_pp_group,; ParallelLMHead,; 符号: __init__, forward, forward, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunk: def _dispatch_core(; 符号: _dispatch_core
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`；patch 关键词为 config, moe, attention, cuda, deepep, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22823 - [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model

- 链接：https://github.com/sgl-project/sglang/pull/22823
- 状态/时间：`merged`，created 2026-04-14, merged 2026-04-15；作者 `Jiminator`。
- 代码 diff 已读范围：`1` 个文件，`+2/-1`；代码面：model wrapper, MoE/router；关键词：config, moe, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1 (3 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe_nextn.py`；patch 关键词为 config, moe, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

- 已覆盖 PR 数：30；open PR 数：8。
- 仍需跟进的 open PR：[#11951](https://github.com/sgl-project/sglang/pull/11951), [#17869](https://github.com/sgl-project/sglang/pull/17869), [#18930](https://github.com/sgl-project/sglang/pull/18930), [#19040](https://github.com/sgl-project/sglang/pull/19040), [#19106](https://github.com/sgl-project/sglang/pull/19106), [#22315](https://github.com/sgl-project/sglang/pull/22315), [#22801](https://github.com/sgl-project/sglang/pull/22801), [#23067](https://github.com/sgl-project/sglang/pull/23067)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
