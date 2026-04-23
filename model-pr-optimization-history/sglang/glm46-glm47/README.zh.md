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
