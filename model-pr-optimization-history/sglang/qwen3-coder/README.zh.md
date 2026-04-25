# SGLang Qwen3-Coder 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理。覆盖 Qwen3-Coder-480B-A35B、Qwen3-Coder-Next、`qwen3_coder` tool parser、streaming tool arguments、NVFP4/FP8、AMD/NPU/Blackwell cookbook。

结论：Qwen3-Coder 必须拆成两条线看。`qwen3_coder_detector.py` 是独立的 parser 风险面，而且被 Qwen3.6 文档复用；Qwen3-Coder-Next 的模型 runtime 则大多落在 Qwen3-Next hybrid lane，涉及 GDN/Mamba/MTP/cache、MoE、ModelOpt、AMD/NPU 后端。

## 代码面

- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `python/sglang/srt/function_call/base_format_detector.py`
- `python/sglang/srt/function_call/function_call_parser.py`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/layers/attention/aiter_backend.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`
- `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`
- `test/registered/amd/test_qwen3_coder_next_8gpu.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx`

## 手工 diff 审阅 PR 卡片

### PR #8357 - XML-ish grammar 与 Qwen3-Coder detector 修复

- 链接：https://github.com/sgl-project/sglang/pull/8357
- 状态：已合入，`2025-07-25T05:08:06Z`
- Diff 覆盖：完整阅读 patch，`6` 个文件，`+305/-58`。
- Motivation：PR #8260 后，Qwen3-Coder 的 `tool_choice=required` 和指定函数选择仍会失败，因为当时 `EBNFComposer` 没有 XML-like grammar。detector 还存在 streaming index、structural-tag 兼容性和 registry key 命名问题。
- 关键实现：给 `EBNFComposer` 补 XML format，统一基础类型映射与 format-specific override；把 detector 从泛化 Qwen3 名称收敛到 `qwen3_coder`；新增 `supports_structural_tag()`，Qwen3-Coder 与 Pythonic 都返回 `False`，避免 structural tag 包裹 XML-like tool-call 流。
- 关键代码片段：

```python
FORMAT_TYPE_OVERRIDES = {
    "pythonic": {"boolean": '"True" | "False"', "null": '"None"'},
    "xml": {"string": "xml_text"},
}
```

```python
def supports_structural_tag(self) -> bool:
    return False
```

- 已读文件：`base_format_detector.py`、`ebnf_composer.py`、`function_call_parser.py`、`pythonic_detector.py`、`qwen3_coder_detector.py`、`test/srt/test_function_call_parser.py`。
- 验证影响：需要覆盖 XML 参数标签、`tool_choice=required`、指定函数、streaming index 和 structural-tag disabled 行为。

### PR #8371 - Qwen3-Coder streaming parser 改造

- 链接：https://github.com/sgl-project/sglang/pull/8371
- 状态：已合入，`2025-08-08T06:42:29Z`
- Diff 覆盖：完整阅读 patch，`2` 个文件，`+304/-54`。
- Motivation：旧 parser 在 streaming 下缓冲过多，可能抛 `AttributeError` 导致连接断开；客户端也无法在 `<function=...>` 完整后立即收到 tool name。
- 关键实现：新增 `_current_function_name`、`_current_parameters`、`_streamed_parameters`、`_in_tool_call`、`_function_name_sent` 等状态。`parse_streaming_increment()` 先发带 name、空参数的 delta，再在参数块可解析后按 JSON diff 发参数。
- 关键代码片段：

```python
self._current_function_name: str = ""
self._current_parameters: Dict[str, Any] = {}
self._streamed_parameters: Dict[str, str] = {}
self._in_tool_call: bool = False
self._function_name_sent: bool = False
```

```python
calls.append(ToolCallItem(tool_index=self.current_tool_id, name=function_name, parameters=""))
```

```python
argument_diff = current_args_json[sent_length:]
calls.append(ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=argument_diff))
```

- 已读文件：`qwen3_coder_detector.py`、`test/srt/test_function_call_parser.py`。
- 验证影响：必须用很小 chunk 的 parser-only 单测验证状态转移，不能只跑模型 smoke。

### PR #8445 - GLM-4.5 follow-up 中的 Qwen3-Coder EBNF separator 修复

- 链接：https://github.com/sgl-project/sglang/pull/8445
- 状态：已合入，`2025-07-28T06:35:20Z`
- Diff 覆盖：完整阅读 patch，`6` 个文件，`+44/-15`。
- Motivation：这个 PR 主要处理 GLM-4.5 review comment，但同时修了 Qwen3-Coder XML 参数 grammar。多个 `<parameter=...>` 之间缺少换行 separator 时，constrained generation 容易生成相邻拼接的畸形标签。
- 关键实现：`Qwen3CoderDetector.build_ebnf()` 调 `EBNFComposer.build_ebnf()` 时显式传入 `key_value_separator="\n"`；同 PR 还扩大 tool-choice 测试的 `max_tokens`，降低被截断误判的概率。
- 关键代码片段：

```python
return EBNFComposer.build_ebnf(
    tools,
    function_format="xml",
    call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
    key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
    key_value_separator="\\n",
)
```

- 已读文件：`glm4_moe_detector.py`、`qwen3_coder_detector.py`、`test_tool_choice.py`、`test_function_call_parser.py`。
- 验证影响：多参数 constrained generation 不能只测单参数，必须检查参数间换行。

### PR #12226 - unknown tool call 不再强制丢弃

- 链接：https://github.com/sgl-project/sglang/pull/12226
- 状态：已合入，`2025-11-01T02:10:35Z`
- Diff 覆盖：完整阅读 patch，`7` 个文件，`+145/-60`。
- Motivation：模型可能输出请求 `tools` 中不存在的函数名。旧行为直接丢弃，客户端 orchestrator 无法知道模型想调用什么。PR 保持默认兼容，同时给出 opt-in forward 行为。
- 关键实现：新增 `SGLANG_FORWARD_UNKNOWN_TOOLS`；base、GPT-OSS、Pythonic、Qwen3-Coder detector 都改成在 env 打开时继续发出 unknown tool call。Qwen3-Coder streaming 路径中 invalid function name 不再无条件 reset 和 flush。
- 关键代码片段：

```python
if not (name and name in tool_indices):
    logger.warning(f"Model attempted to call undefined function: {name}")
    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
        continue
```

```python
if not is_valid:
    logger.warning(f"Invalid function name: {function_name}")
    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
        self._reset_streaming_state()
        normal += self._buf
        self._buf = ""
        break
```

- 已读文件：`environ.py`、`base_format_detector.py`、`gpt_oss_detector.py`、`pythonic_detector.py`、`qwen3_coder_detector.py`、`test_unknown_tool_name.py`、`environment_variables.md`。
- 验证影响：默认 drop 和 opt-in forward 都要测；Qwen3-Coder 还要专门测 invalid function streaming state。

### PR #13163 - 移除 EBNF Composer

- 链接：https://github.com/sgl-project/sglang/pull/13163
- 状态：已合入，`2025-11-13T01:55:31Z`
- Diff 覆盖：完整阅读 patch，`18` 个文件，`+6/-1081`。
- Motivation：SGLang 已用 JSON Schema 约束 required/named `tool_choice`，继续保留 detector 自己的 EBNF 会造成双约束系统，Qwen3-Coder XML grammar 也会持续增加维护成本。
- 关键实现：删除 `ebnf_composer.py`，从所有 detector 删除 `build_ebnf()`，包括 `Qwen3CoderDetector`；`FunctionCallParser.get_structure_constraint()` 成为结构化约束入口，required/named tool choice 走 JSON Schema。
- 关键代码片段：

```python
elif tool_choice == "required" or isinstance(tool_choice, ToolChoice):
    json_schema = get_json_schema_constraint(self.tools, tool_choice)
    return ("json_schema", json_schema)
```

```diff
-    def build_ebnf(self, tools: List[Tool]):
-        return EBNFComposer.build_ebnf(...)
```

- 已读文件：`base_format_detector.py`、`function_call_parser.py`、`qwen3_coder_detector.py`、`glm4_moe_detector.py`、`json_array_parser.py`、`test_json_schema_constraint.py`、`test_function_call_parser.py`。
- 验证影响：后续不要为 Qwen3-Coder 恢复 `build_ebnf()`；新约束逻辑应测 JSON Schema 和 parser 两条链路。

### PR #13411 - schema-aware Qwen3-Coder 参数类型转换

- 链接：https://github.com/sgl-project/sglang/pull/13411
- 状态：open，`2026-04-23` 时未合入
- Diff 覆盖：完整阅读 open patch，`2` 个文件，`+155/-10`。
- Motivation：Qwen3-Coder 参数是 XML-like 文本。旧 `_safe_val()` 用 `json.loads()` 和 `ast.literal_eval()` 猜类型，可能把 zip code `03106`、字符串 `"42"` 或 JSON-looking string 转错。工具 schema 才是参数类型的准确信息。
- 关键实现：用 `_convert_param_value(param_value, param_name, param_config, func_name)` 替换 `_safe_val(raw)`；streaming 和 non-streaming 路径都构造 tool name 到 parameter schema 的映射；string 类型即使内容像 int/float/bool/object，也保留为 string。
- 关键代码片段：

```python
def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    param_value = html.unescape(param_value.strip())
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
```

```python
self._tool_parameter_configs = {
    tool.function.name: tool.function.parameters.get("properties", {})
    for tool in tools
    if tool.function.name
}
```

```python
self.assertEqual(params["str_param_int_content"], "42")
self.assertEqual(params["str_param_float_content"], "3.14")
self.assertEqual(params["str_param_bool_content"], "true")
self.assertEqual(params["str_param_obj_content"], '{"key": "value"}')
```

- 已读文件：`qwen3_coder_detector.py`、`test/per_commit/function_call/test_function_call_parser.py`。
- 验证影响：这是未合入设计证据，不代表当前主线能力。后续落地必须同时测 streaming 和 non-streaming 的 schema-aware 转换。

### PR #16744 - 新版 Qwen3-Coder detector

- 链接：https://github.com/sgl-project/sglang/pull/16744
- 状态：已合入，`2026-01-19T02:22:41Z`
- Diff 覆盖：完整阅读 patch，`2` 个文件，`+637/-667`。
- Motivation：SGLang 需要生产可用的 Qwen3-Coder XML-like tool-call parser，覆盖 `<tool_call>`、`<function=...>`、`<parameter=...>`，并且 Qwen 团队确认了行为。
- 关键实现：重写 `qwen3_coder_detector.py`，显式定义 sentinel token、function/parameter 正则、参数转换 helper 和 cursor-based streaming parser。解析完成后输出带稳定 tool index 的 `ToolCallItem`，参数用 `json.dumps(..., ensure_ascii=False)`。
- 关键代码片段：

```python
self.tool_call_start_token = "<tool_call>"
self.tool_call_end_token = "</tool_call>"
self.tool_call_prefix = "<function="
self.function_end_token = "</function>"
self.parameter_prefix = "<parameter="
self.parameter_end_token = "</parameter>"
```

```python
calls.append(
    ToolCallItem(
        tool_index=tool_idx,
        name=func_name,
        parameters=json.dumps(parsed_params, ensure_ascii=False),
    )
)
```

- 已读文件：`qwen3_coder_detector.py`、`test/registered/function_call/test_function_call_parser.py`。
- 验证影响：这是后续 Qwen3-Coder parser 的主线基线，回归要覆盖 one-shot、streaming、多参数、类型转换和 tool index。

### PR #21829 - Qwen3-Coder tool-call arguments 增量 streaming

- 链接：https://github.com/sgl-project/sglang/pull/21829
- 状态：open，`2026-04-23` 时未合入
- Diff 覆盖：完整阅读 open patch，`1` 个文件，`+140/-0`。
- Motivation：开启 `--tool-call-parser qwen3_coder` 和 streaming 时，长代码/长文本参数会一直缓存到 `</parameter>`，再一次性发出巨大 delta。客户端期望参数也能增量到达。
- 关键实现：增加 active parameter streaming state、已发送游标和 leading newline 处理；只有 string-like schema 类型增量发，因为 int/bool/array/object 必须拿完整文本做转换；`_find_safe_emit_end()` 避免把 `</parameter>`、`<parameter=`、`</function>` 切断。
- 关键代码片段：

```python
self._streaming_param_active: bool = False
self._streaming_param_emitted: int = 0
self._streaming_param_leading_checked: bool = False
```

```python
return p_type in ("string", "str", "text", "varchar", "char", "enum")
```

```python
for tag in [self.parameter_end_token, self.parameter_prefix, self.function_end_token]:
    if tag.startswith(suffix):
        return last_angle
```

- 已读文件：`qwen3_coder_detector.py`。
- 验证影响：未合入。若后续采用，必须新增 exact delta sequence 单测，包括 key prefix、escaped partial content、最后剩余内容和 closing quote。

### PR #17965 - Triton TP MoE SwapAB 调优覆盖 Qwen3-Coder

- 链接：https://github.com/sgl-project/sglang/pull/17965
- 状态：已合入，`2026-01-31T21:57:39Z`
- Diff 覆盖：完整阅读 patch，`12` 个文件，`+765/-13`。
- Motivation：在 H200 上启用 SwapAB，并为 DeepSeek V3 和 Qwen3-Coder 重调 TP MoE。Qwen3-Coder 目标模型是 `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`，TP=8，EP=2，并搭配 EAGLE3 draft。
- 关键实现：tuning 脚本新增 `ep_size`，EP 打开时把全局 top-k expert id 映射成本地 expert id；新增 H200 FP8 Triton MoE config，包括 Qwen3-Coder 的大 MoE shape。
- 关键代码片段：

```python
if ep_size > 1:
    topk_ids = (topk_ids // ep_size).to(
        device=moe_inputs[k].topk_ids.device,
        dtype=moe_inputs[k].topk_ids.dtype,
    )
```

- 已读文件：`tuning_fused_moe_triton_sep.py`、H200 Triton MoE config JSONs、`fused_moe_triton_kernels.py`。
- 验证影响：Qwen3-Coder MoE 性能回归要同时看 TP 和 EP，EP top-k remap 错会表现成 kernel 慢或输出异常。

### PR #18195 - Qwen3-Coder-Next FP8 H100 TP=2 fused MoE config

- 链接：https://github.com/sgl-project/sglang/pull/18195
- 状态：已合入，`2026-02-04T19:38:25Z`
- Diff 覆盖：完整阅读 patch，`1` 个文件，`+70/-0`。
- Motivation：Qwen3-Coder-Next-FP8 在 H100 TP=2 上需要专门的 Triton MoE config。PR body 记录 output throughput `+2.2%`、peak `+7.3%`、median TTFT `-40.8%`、p99 E2E `-9.6%`、median ITL `-8.2%`。
- 关键实现：新增 `E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`，按 token count 配置 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`。
- 关键代码片段：

```json
"2048": {
  "BLOCK_SIZE_M": 64,
  "BLOCK_SIZE_N": 128,
  "BLOCK_SIZE_K": 128,
  "GROUP_SIZE_M": 16,
  "num_warps": 4,
  "num_stages": 4
}
```

- 已读文件：`python/sglang/srt/layers/moe/fused_moe_triton/configs/...E=512,N=256...json`。
- 验证影响：复测必须固定 H100、TP=2、FP8、block shape，否则无法判断配置本身收益。

### PR #18224 - Qwen3-Coder-Next ModelOpt NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/18224
- 状态：已合入，`2026-02-08T06:38:39Z`
- Diff 覆盖：完整阅读 patch，`1` 个文件，`+23/-12`。
- Motivation：`vincentzed-hf/Qwen3-Coder-Next-NVFP4` 需要通过 `--quantization modelopt_fp4` 加载。PR body 给出 B300 上 GSM8K Platinum accuracy `0.969`、throughput `4610.959 tok/s`。
- 关键实现：把 `quant_config` 传入 Qwen3-Next attention；注册 `qkv_proj` 和 `gate_up_proj` 的 packed-module mapping；把 ModelOpt FP8 KV scale key 从 split `k_proj/v_proj` 重映射到 SGLang 的 `attn.k_scale/v_scale`。
- 关键代码片段：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

```python
if name.endswith(".k_proj.k_scale"):
    name = name.replace(".k_proj.k_scale", ".attn.k_scale")
elif name.endswith(".v_proj.v_scale"):
    name = name.replace(".v_proj.v_scale", ".attn.v_scale")
```

- 已读文件：`python/sglang/srt/models/qwen3_next.py`。
- 验证影响：NVFP4 加载要同时测 fused projection packing 和 KV scale key remap，单纯不报错不够。

### PR #18355 - AMD Qwen3-Coder-Next 支持

- 链接：https://github.com/sgl-project/sglang/pull/18355
- 状态：已合入，`2026-02-25T00:29:30Z`
- Diff 覆盖：完整阅读 patch，`2` 个文件，`+72/-12`。
- Motivation：让 Qwen3-Coder-Next 跑在 AMD GPU 上，覆盖 non-MTP + FP8 KV cache 和 MTP。AITER backend 需要正确处理 hybrid linear-attention 模型的 value head dim。
- 关键实现：AITER 按 MLA、hybrid GDN/Kimi-linear、普通 KV pool 三类来源计算 `v_head_dim`；Qwen3-Next dual stream 保持 CUDA-only；CuTe DSL GDN import 改成仅在显式打开时强制要求。
- 关键代码片段：

```python
if self.use_mla:
    self.v_head_dim = model_runner.model_config.v_head_dim
elif model_runner.hybrid_gdn_config is not None or model_runner.kimi_linear_config is not None:
    self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
else:
    self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
```

```python
alt_stream = torch.cuda.Stream() if _is_cuda else None
```

- 已读文件：`aiter_backend.py`、`qwen3_next.py`。
- 验证影响：AMD 验证要覆盖 AITER、hybrid GDN state、FP8 KV、MTP/non-MTP，不要假设 dual stream 可用。

### PR #18608 - AMD MI35x Qwen3-Coder-Next 测试

- 链接：https://github.com/sgl-project/sglang/pull/18608
- 状态：已合入，`2026-03-02T21:52:04Z`
- Diff 覆盖：完整阅读 patch，`2` 个文件，`+246/-0`。
- Motivation：AMD runtime 支持合入后，Qwen3-Coder-Next 需要 MI35x accuracy 和 functionality 注册测试。这个模型同时有 full attention、GDN、512-expert MoE、FP8 KV、chunked prefill、MTP，普通 AMD smoke 覆盖不足。
- 关键实现：新增 nightly MI35x accuracy suite 和 stage-c functionality suite。basic 路径使用 `--attention-backend aiter`、`--chunked-prefill-size 131072`、`--disable-radix-cache`、`--kv-cache-dtype fp8_e4m3`、TP=8。MTP 路径使用 EAGLE 参数，但当时不带 FP8 KV，因为 gfx950 的 Triton extend_attention 还不支持。
- 关键代码片段：

```python
register_amd_ci(est_time=3600, suite="nightly-amd-8-gpu-mi35x", nightly=True)
```

```python
other_args=[
    "--attention-backend", "aiter",
    "--chunked-prefill-size", "131072",
    "--disable-radix-cache",
    "--kv-cache-dtype", "fp8_e4m3",
]
```

- 已读文件：`test_qwen3_coder_next_eval_mi35x.py`、`test_qwen3_coder_next_8gpu.py`。
- 验证影响：AMD 改动要跑 basic 和 MTP 两条线，FP8 KV skip 的原因要随 backend 支持更新。

### PR #18700 - NPU Qwen3-Coder-Next weight transpose 修复

- 链接：https://github.com/sgl-project/sglang/pull/18700
- 状态：已合入，`2026-02-25T06:02:41Z`
- Diff 覆盖：完整阅读 patch，`2` 个文件，`+7/-9`。
- Motivation：NPU 上 Qwen3-Coder-Next 的 load postprocess 和 fused MoE runtime 都对权重做转置，导致 shape 错；hybrid attention 还会导入 NPU 不需要的 CuTe DSL。
- 关键实现：NPU fused MoE 传权重时去掉重复 `.permute(0, 2, 1)`；CuTe DSL import 用 `is_npu()` 和 env 开关保护。
- 关键代码片段：

```python
weight=[layer.w13_weight],
...
weight=[layer.w2_weight],
```

```python
if not is_npu() or use_cutedsl:
    from sglang.jit_kernel.cutedsl_gdn import ...
```

- 已读文件：`fused_moe_method_npu.py`、`hybrid_linear_attn_backend.py`。
- 验证影响：NPU 不能只看 server launch，要检查 `w13` 和 `w2` 实际 shape。

### PR #19736 - AMD AITER extend_attention k/v scale 参数修复

- 链接：https://github.com/sgl-project/sglang/pull/19736
- 状态：已合入，`2026-03-04T17:20:38Z`
- Diff 覆盖：完整阅读 patch，`1` 个文件，`+4/-0`。
- Motivation：PR #18882 给 `extend_attention_fwd()` 增加必需的 `k_scale`、`v_scale` 后，Triton backend 改了，但 AITER non-MLA `target_verify` / `draft_extend` 路径漏改。Qwen3-Coder-Next MTP 在 AMD 上因此报缺少 `v_scale`。
- 关键实现：在 AITER call site 补默认 `1.0` 的 k/v scale，匹配新函数签名。
- 关键代码片段：

```python
1.0,  # k_scale
1.0,  # v_scale
layer.scaling,
```

- 已读文件：`python/sglang/srt/layers/attention/aiter_backend.py`。
- 验证影响：`extend_attention_fwd()` 签名变化后，AITER MTP target-verify 和 draft-extend 要有直接 smoke。

### PR #13979 - Qwen3-Coder-480B nightly performance tests

- 链接：https://github.com/sgl-project/sglang/pull/13979
- 状态：open，`2026-04-23` 时未合入
- Diff 覆盖：完整阅读 open patch，`3` 个文件，`+288/-171`。
- Motivation：为 `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` 补 8-GPU H200/B200 nightly performance coverage。480B Coder 的 MoE/EP footprint 与小 Qwen 不同，需要独立 perf 信号。
- 关键实现：新增 `test/nightly/test_qwen3_coder_480b_perf.py`，使用 `NightlyBenchmarkRunner`、TP=8、EP=8、多线程 load、batch sizes `[1, 1, 8, 16, 64]`、input len `4096`、output len `512`、`server_start_timeout=3600`。workflow patch 中临时注释了其它 nightly perf job，不能直接照搬为最终 CI 形态。
- 关键代码片段：

```python
QWEN3_CODER_480B_MODEL_PATH = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
cls.other_args = [
    "--tp",
    "8",
    "--ep",
    "8",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]
```

```python
self.runner.run_benchmark_for_model(
    model_path=self.model,
    batch_sizes=self.batch_sizes,
    input_lens=self.input_lens,
    output_lens=self.output_lens,
    other_args=self.other_args,
    server_start_timeout=3600,
)
```

- 已读文件：`.github/workflows/nightly-test-nvidia.yml`、`test_qwen3_coder_480b_perf.py`、`nightly_utils.py`。
- 验证影响：可复用 test 文件和 timeout hook，但不要复制注释其它 nightly job 的 workflow hunk。

## sgl-cookbook / 文档

- `sgl-cookbook#86`：Qwen3-Coder-480B-A35B AMD MI300X。
- `sgl-cookbook#112`：MI325X/MI355X。
- `sgl-cookbook#143`：Qwen3-Coder-Next。
- `sgl-cookbook#174`：NVIDIA B200/GB200。

## 下一步优化建议

1. 先补 parser-only 单测：复杂 schema、空函数名、多工具、增量 streaming、unknown tool、string-looking number。
2. Qwen3-Coder-Next runtime 改动应同步跑 Qwen3-Next MTP/cache 测试。
3. AMD 改动覆盖 AITER basic 和 MTP，NPU 改动检查 MoE 权重 shape。
4. Cookbook 命令必须显式带 `--tool-call-parser qwen3_coder`，并说明 parser 与模型性能是两条验证线。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3 Coder`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-22 | [#8260](https://github.com/sgl-project/sglang/pull/8260) | merged | Preliminary Support for Qwen3XMLDetector | misc | `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` |
| 2025-07-25 | [#8357](https://github.com/sgl-project/sglang/pull/8357) | merged | [Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector | tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py` |
| 2025-07-26 | [#8371](https://github.com/sgl-project/sglang/pull/8371) | merged | Update qwen3_coder_detector.py for streaming | tests/benchmarks | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py` |
| 2025-07-28 | [#8445](https://github.com/sgl-project/sglang/pull/8445) | merged | GLM-4.5 Model Support Follow-up | MoE/router, tests/benchmarks | `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/features/test_enable_thinking.py` |
| 2025-10-27 | [#12226](https://github.com/sgl-project/sglang/pull/12226) | merged | Forward unknown tool calls instead of dropping | tests/benchmarks, docs/config | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py` |
| 2025-11-12 | [#13163](https://github.com/sgl-project/sglang/pull/13163) | merged | Remove EBNF Composer | MoE/router, tests/benchmarks | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py` |
| 2025-11-17 | [#13411](https://github.com/sgl-project/sglang/pull/13411) | open | Improve Qwen3CoderDetector with schema-aware parameter type conversion | tests/benchmarks | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py` |
| 2025-11-26 | [#13979](https://github.com/sgl-project/sglang/pull/13979) | open | Add Qwen3-Coder-480B to nightly tests | tests/benchmarks | `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py` |
| 2026-01-08 | [#16744](https://github.com/sgl-project/sglang/pull/16744) | merged | support new qwen3_coder_detector | tests/benchmarks | `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2026-01-30 | [#17965](https://github.com/sgl-project/sglang/pull/17965) | merged | [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-02-03 | [#18195](https://github.com/sgl-project/sglang/pull/18195) | merged | Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2 | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2026-02-04 | [#18224](https://github.com/sgl-project/sglang/pull/18224) | merged | [ModelOPT] Support Qwen 3 Next Coder NVFP4 | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-06 | [#18355](https://github.com/sgl-project/sglang/pull/18355) | merged | [AMD] Support Qwen3-Coder-Next on AMD platform | model wrapper, attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-11 | [#18608](https://github.com/sgl-project/sglang/pull/18608) | merged | [AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU | tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py` |
| 2026-02-12 | [#18700](https://github.com/sgl-project/sglang/pull/18700) | merged | [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu. | attention/backend, MoE/router, quantization | `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-02-16 | [#18882](https://github.com/sgl-project/sglang/pull/18882) | merged | feat: Add FP8 KV cache support for Triton attention backend | attention/backend, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/triton_backend.py`, `test/registered/quant/test_fp8kv_triton.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| 2026-03-03 | [#19736](https://github.com/sgl-project/sglang/pull/19736) | merged | [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-04-01 | [#21829](https://github.com/sgl-project/sglang/pull/21829) | open | [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector | misc | `python/sglang/srt/function_call/qwen3_coder_detector.py` |

### 逐 PR 代码 diff 阅读记录

### PR #8260 - Preliminary Support for Qwen3XMLDetector

- 链接：https://github.com/sgl-project/sglang/pull/8260
- 状态/时间：`merged`，created 2025-07-22, merged 2025-07-22；作者 `yhyang201`。
- 代码 diff 已读范围：`3` 个文件，`+153/-0`；代码面：misc；关键词：kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/qwen3_detector.py` added +150/-0 (150 lines); hunk: +import ast; 符号: _safe_val, Qwen3XMLDetector, __init__, has_tool_call
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunk: from sglang.srt.function_call.llama32_detector import Llama32Detector; class FunctionCallParser:; 符号: FunctionCallParser:, __init__
  - `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunk: def add_cli_args(parser: argparse.ArgumentParser):; 符号: add_cli_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`；patch 关键词为 kv, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/qwen3_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8357 - [Bugfix][Feat] Add XML-ish grammar in EBNFComposer and fix misc bugs in Qwen3 detector

- 链接：https://github.com/sgl-project/sglang/pull/8357
- 状态/时间：`merged`，created 2025-07-25, merged 2025-07-25；作者 `CatherineSue`。
- 代码 diff 已读范围：`7` 个文件，`+574/-83`；代码面：tests/benchmarks；关键词：spec, kv, doc, test。
- 代码 diff 细节：
  - `test/srt/test_function_call_parser.py` modified +455/-0 (455 lines); hunk: from sglang.srt.function_call.llama32_detector import Llama32Detector; def setUp(self):; 符号: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_qwen3_coder_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` modified +95/-63 (158 lines); hunk: -from typing import Literal, Optional; class EBNFComposer:; 符号: EBNFComposer:, EBNFComposer:, get_value_rule, _handle_enum
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` renamed +10/-9 (19 lines); hunk: from sglang.srt.function_call.base_format_detector import BaseFormatDetector; def _safe_val(raw: str) -> Any:; 符号: _safe_val, Qwen3XMLDetector, Qwen3CoderDetector, _parse_block
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-5 (9 lines); hunk: from sglang.srt.function_call.base_format_detector import BaseFormatDetector; def _get_parameter_value(self, val):; 符号: _get_parameter_value, structure_info, info, supports_structural_tag
  - `python/sglang/srt/function_call/function_call_parser.py` modified +4/-4 (8 lines); hunk: from sglang.srt.function_call.llama32_detector import Llama32Detector; class FunctionCallParser:; 符号: FunctionCallParser:, __init__, get_structure_constraint
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py`；patch 关键词为 spec, kv, doc, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `python/sglang/srt/function_call/qwen3_coder_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8371 - Update qwen3_coder_detector.py for streaming

- 链接：https://github.com/sgl-project/sglang/pull/8371
- 状态/时间：`merged`，created 2025-07-26, merged 2025-08-08；作者 `maocheng23`。
- 代码 diff 已读范围：`2` 个文件，`+348/-67`；代码面：tests/benchmarks；关键词：moe, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +219/-9 (228 lines); hunk: def __init__(self):; def parse_streaming_increment(; 符号: __init__, has_tool_call, parse_streaming_increment, _parse_and_stream_parameters
  - `test/srt/test_function_call_parser.py` modified +129/-58 (187 lines); hunk: def test_parse_streaming_simple(self):; def test_parse_streaming_incomplete(self):; 符号: test_parse_streaming_simple, test_parse_streaming_incomplete, test_edge_case_no_parameters, test_extract_tool_calls_type_conversion
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py`；patch 关键词为 moe, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/test_function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8445 - GLM-4.5 Model Support Follow-up

- 链接：https://github.com/sgl-project/sglang/pull/8445
- 状态/时间：`merged`，created 2025-07-28, merged 2025-07-28；作者 `byjiang1996`。
- 代码 diff 已读范围：`6` 个文件，`+44/-15`；代码面：MoE/router, tests/benchmarks；关键词：test, moe, spec。
- 代码 diff 细节：
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +39/-10 (49 lines); hunk: def get_test_messages(self):; def test_tool_choice_auto_non_streaming(self):; 符号: get_test_messages, test_tool_choice_auto_non_streaming, test_tool_choice_auto_streaming, test_tool_choice_required_non_streaming
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +1/-2 (3 lines); hunk: def build_ebnf(self, tools: List[Tool]):; 符号: build_ebnf
  - `test/srt/openai_server/features/test_enable_thinking.py` modified +1/-1 (2 lines); hunk: def test_stream_chat_completion_without_reasoning(self):; 符号: test_stream_chat_completion_without_reasoning, TestGLM45EnableThinking, setUpClass
  - `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +1/-1 (2 lines); hunk: def test_pythonic_tool_call_streaming(self):; 符号: test_pythonic_tool_call_streaming, TestGLM45ServerFunctionCalling, setUpClass
  - `test/srt/test_function_call_parser.py` modified +1/-1 (2 lines); hunk: def test_streaming_multiple_tool_calls(self):; 符号: test_streaming_multiple_tool_calls, test_tool_call_completion, test_tool_call_id
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/features/test_enable_thinking.py`；patch 关键词为 test, moe, spec。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/openai_server/function_call/test_tool_choice.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `test/srt/openai_server/features/test_enable_thinking.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12226 - Forward unknown tool calls instead of dropping

- 链接：https://github.com/sgl-project/sglang/pull/12226
- 状态/时间：`merged`，created 2025-10-27, merged 2025-11-01；作者 `Surya-Gunukula`。
- 代码 diff 已读范围：`7` 个文件，`+145/-60`；代码面：tests/benchmarks, docs/config；关键词：config, cache, doc, expert, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +41/-37 (78 lines); hunk: from typing import Any, Dict, List, Tuple; def parse_streaming_increment(; 符号: parse_streaming_increment
  - `test/srt/function_call/test_unknown_tool_name.py` added +69/-0 (69 lines); hunk: +import json; 符号: DummyDetector, has_tool_call, detect_and_parse, test_unknown_tool_name_dropped_default
  - `python/sglang/srt/function_call/base_format_detector.py` modified +15/-12 (27 lines); hunk: from partial_json_parser.core.options import Allow; def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:; 符号: parse_base_json
  - `docs/references/environment_variables.md` modified +10/-9 (19 lines); hunk: SGLang supports various environment variables that can be used to configure its
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +4/-1 (5 lines); hunk: from typing import List, Optional; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; 符号: detect_and_parse
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py`；patch 关键词为 config, cache, doc, expert, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/srt/function_call/test_unknown_tool_name.py`, `python/sglang/srt/function_call/base_format_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13163 - Remove EBNF Composer

- 链接：https://github.com/sgl-project/sglang/pull/13163
- 状态/时间：`merged`，created 2025-11-12, merged 2025-11-13；作者 `TJ5`。
- 代码 diff 已读范围：`18` 个文件，`+6/-1081`；代码面：MoE/router, tests/benchmarks；关键词：spec, kv, moe, test, doc。
- 代码 diff 细节：
  - `test/srt/test_function_call_parser.py` modified +5/-459 (464 lines); hunk: import json; def test_detect_and_parse_with_text_before_tool_call(self):; 符号: test_detect_and_parse_with_text_before_tool_call, TestEBNFGeneration, setUp, test_pythonic_detector_ebnf
  - `python/sglang/srt/function_call/ebnf_composer.py` removed +0/-344 (344 lines); hunk: -from typing import Any, Dict, Literal, Optional; 符号: EBNFComposer:, get_value_rule, _handle_enum, format_enum_val
  - `test/srt/function_call/test_json_schema_constraint.py` modified +0/-52 (52 lines); hunk: def test_tools_without_parameters(self):; 符号: test_tools_without_parameters, test_json_schema_vs_ebnf_constraint_generation, test_conflicting_defs_raises_valueerror
  - `python/sglang/srt/function_call/function_call_parser.py` modified +0/-38 (38 lines); hunk: def get_structure_constraint(; 符号: get_structure_constraint, get_ebnf
  - `python/sglang/srt/function_call/step3_detector.py` modified +0/-29 (29 lines); hunk: ToolCallItem,; def supports_structural_tag(self) -> bool:; 符号: supports_structural_tag, structure_info, build_ebnf
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py`；patch 关键词为 spec, kv, moe, test, doc。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/ebnf_composer.py`, `test/srt/function_call/test_json_schema_constraint.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13411 - Improve Qwen3CoderDetector with schema-aware parameter type conversion

- 链接：https://github.com/sgl-project/sglang/pull/13411
- 状态/时间：`open`，created 2025-11-17；作者 `00INDEX`。
- 代码 diff 已读范围：`2` 个文件，`+155/-10`；代码面：tests/benchmarks；关键词：config, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +135/-10 (145 lines); hunk: logger = logging.getLogger(__name__); def parse_streaming_increment(; 符号: _safe_val, _convert_param_value, Qwen3CoderDetector, parse_streaming_increment
  - `test/per_commit/function_call/test_function_call_parser.py` modified +20/-0 (20 lines); hunk: def test_extract_tool_calls_type_conversion(self):; def test_extract_tool_calls_type_conversion(self):; 符号: test_extract_tool_calls_type_conversion, test_extract_tool_calls_type_conversion, test_extract_tool_calls_type_conversion, test_parse_streaming_incremental
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py`；patch 关键词为 config, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/per_commit/function_call/test_function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13979 - Add Qwen3-Coder-480B to nightly tests

- 链接：https://github.com/sgl-project/sglang/pull/13979
- 状态/时间：`open`，created 2025-11-26；作者 `Kangyan-Zhou`。
- 代码 diff 已读范围：`3` 个文件，`+288/-171`；代码面：tests/benchmarks；关键词：test, benchmark, config, fp8, moe。
- 代码 diff 细节：
  - `.github/workflows/nightly-test-nvidia.yml` modified +232/-170 (402 lines); hunk: jobs:; jobs:
  - `test/nightly/test_qwen3_coder_480b_perf.py` added +53/-0 (53 lines); hunk: +import unittest; 符号: TestNightlyQwen3Coder480BPerformance, setUpClass, test_bench_one_batch
  - `test/nightly/nightly_utils.py` modified +3/-1 (4 lines); hunk: def run_benchmark_for_model(; def run_benchmark_for_model(; 符号: run_benchmark_for_model, run_benchmark_for_model, run_benchmark_for_model
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py`；patch 关键词为 test, benchmark, config, fp8, moe。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `.github/workflows/nightly-test-nvidia.yml`, `test/nightly/test_qwen3_coder_480b_perf.py`, `test/nightly/nightly_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16744 - support new qwen3_coder_detector

- 链接：https://github.com/sgl-project/sglang/pull/16744
- 状态/时间：`merged`，created 2026-01-08, merged 2026-01-19；作者 `attack204`。
- 代码 diff 已读范围：`2` 个文件，`+637/-667`；代码面：tests/benchmarks；关键词：config, moe, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +392/-271 (663 lines); hunk: import ast; logger = logging.getLogger(__name__); 符号: _safe_val, Qwen3CoderDetector, __init__, already
  - `test/registered/function_call/test_function_call_parser.py` modified +245/-396 (641 lines); hunk: def test_streaming_no_parameters_with_whitespace(self):; 符号: test_streaming_no_parameters_with_whitespace, TestQwen3CoderDetector, setUp, test_has_tool_call
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py`；patch 关键词为 config, moe, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/qwen3_coder_detector.py`, `test/registered/function_call/test_function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17965 - [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB

- 链接：https://github.com/sgl-project/sglang/pull/17965
- 状态/时间：`merged`，created 2026-01-30, merged 2026-01-31；作者 `b8zhong`。
- 代码 diff 已读范围：`6` 个文件，`+573/-16`；代码面：MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：moe, triton, config, fp8, cuda, benchmark, cache, expert, quant, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +128/-0 (128 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +114/-0 (114 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +4/-16 (20 lines); hunk: import triton; from sglang.srt.utils import (; 符号: support_tensor_descriptor, should_enable_swap_ab, is_h20_device_and_sm90_supported
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`；patch 关键词为 moe, triton, config, fp8, cuda, benchmark。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18195 - Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2

- 链接：https://github.com/sgl-project/sglang/pull/18195
- 状态/时间：`merged`，created 2026-02-03, merged 2026-02-04；作者 `mmangkad`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/18224
- 状态/时间：`merged`，created 2026-02-04, merged 2026-02-08；作者 `vincentzed`。
- 代码 diff 已读范围：`1` 个文件，`+35/-6`；代码面：model wrapper；关键词：cache, config, expert, fp8, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunk: def __init__(; class HybridLayerType(enum.Enum):; 符号: __init__, HybridLayerType, Qwen3NextForCausalLM, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 cache, config, expert, fp8, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- 链接：https://github.com/sgl-project/sglang/pull/18355
- 状态/时间：`merged`，created 2026-02-06, merged 2026-02-25；作者 `yichiche`。
- 代码 diff 已读范围：`2` 个文件，`+213/-74`；代码面：model wrapper, attention/backend；关键词：cuda, attention, cache, config, flash, kv, mla, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunk: class ForwardMetadata:; def __init__(; 符号: ForwardMetadata:, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunk: def _forward_input_proj(self, hidden_states: torch.Tensor):; 符号: _forward_input_proj
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 cuda, attention, cache, config, flash, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18608 - [AMD] Add Qwen3-Coder-Next accuracy and functionality test scripts for MI35x 8-GPU

- 链接：https://github.com/sgl-project/sglang/pull/18608
- 状态/时间：`merged`，created 2026-02-11, merged 2026-03-02；作者 `yichiche`。
- 代码 diff 已读范围：`2` 个文件，`+486/-0`；代码面：tests/benchmarks；关键词：attention, cache, config, eagle, fp8, kv, spec, test, topk, triton。
- 代码 diff 细节：
  - `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py` added +302/-0 (302 lines); hunk: +"""MI35x Qwen3-Coder-Next GSM8K Completion Evaluation Test (8-GPU); 符号: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `test/registered/amd/test_qwen3_coder_next_8gpu.py` added +184/-0 (184 lines); hunk: +"""MI35x Qwen3-Coder-Next Functionality Test (8-GPU); 符号: TestQwen3CoderNext, setUpClass, tearDownClass, test_a_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py`；patch 关键词为 attention, cache, config, eagle, fp8, kv。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/accuracy/mi35x/test_qwen3_coder_next_eval_mi35x.py`, `test/registered/amd/test_qwen3_coder_next_8gpu.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18700 - [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu.

- 链接：https://github.com/sgl-project/sglang/pull/18700
- 状态/时间：`merged`，created 2026-02-12, merged 2026-02-25；作者 `Hexq0210`。
- 代码 diff 已读范围：`2` 个文件，`+3/-3`；代码面：attention/backend, MoE/router, quantization；关键词：attention, cuda, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` modified +2/-2 (4 lines); hunk: def npu_fused_moe_without_routing_weights_bf16(; def npu_fused_moe_without_routing_weights_bf16(; 符号: npu_fused_moe_without_routing_weights_bf16, npu_fused_moe_without_routing_weights_bf16
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +1/-1 (2 lines); hunk: from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；patch 关键词为 attention, cuda, moe, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18882 - feat: Add FP8 KV cache support for Triton attention backend

- 链接：https://github.com/sgl-project/sglang/pull/18882
- 状态/时间：`merged`，created 2026-02-16, merged 2026-03-03；作者 `zack041`。
- 代码 diff 已读范围：`6` 个文件，`+180/-27`；代码面：attention/backend, quantization, kernel, tests/benchmarks；关键词：attention, kv, triton, test, cache, quant, config, cuda, flash, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +63/-6 (69 lines); hunk: import triton; def __init__(; 符号: __init__, forward_extend, forward_extend, forward_extend
  - `test/registered/quant/test_fp8kv_triton.py` added +58/-0 (58 lines); hunk: +import unittest; 符号: TestFP8KVCacheTritonBackend, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +26/-15 (41 lines); hunk: def _fwd_kernel_stage1(; def _fwd_kernel_stage1(; 符号: _fwd_kernel_stage1, _fwd_kernel_stage1, _decode_att_m_fwd, _decode_att_m_fwd
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +16/-6 (22 lines); hunk: def _fwd_kernel(; def _fwd_kernel(; 符号: _fwd_kernel, _fwd_kernel, _fwd_kernel, extend_attention_fwd
  - `test/registered/attention/test_triton_attention_kernels.py` modified +14/-0 (14 lines); hunk: def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; 符号: _test_extend_attention_once, _test_extend_attention_once, _test_extend_attention_sliding_window_once, _test_decode_attention_once
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/triton_backend.py`, `test/registered/quant/test_fp8kv_triton.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`；patch 关键词为 attention, kv, triton, test, cache, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/triton_backend.py`, `test/registered/quant/test_fp8kv_triton.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19736 - [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend

- 链接：https://github.com/sgl-project/sglang/pull/19736
- 状态/时间：`merged`，created 2026-03-03, merged 2026-03-04；作者 `michaelzhang-ai`。
- 代码 diff 已读范围：`1` 个文件，`+2/-0`；代码面：attention/backend；关键词：attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-0 (2 lines); hunk: def forward_extend(; 符号: forward_extend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/aiter_backend.py`；patch 关键词为 attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/aiter_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21829 - [Feature] Support incremental streaming for tool_call arguments in Qwen3CoderDetector

- 链接：https://github.com/sgl-project/sglang/pull/21829
- 状态/时间：`open`，created 2026-04-01；作者 `yunkchen`。
- 代码 diff 已读范围：`1` 个文件，`+140/-0`；代码面：misc；关键词：config。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/qwen3_coder_detector.py` modified +140/-0 (140 lines); hunk: def __init__(self):; def _convert_param_value(; 符号: __init__, has_tool_call, _convert_param_value, _should_stream_param
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/qwen3_coder_detector.py`；patch 关键词为 config。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/qwen3_coder_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：18；open PR 数：3。
- 仍需跟进的 open PR：[#13411](https://github.com/sgl-project/sglang/pull/13411), [#13979](https://github.com/sgl-project/sglang/pull/13979), [#21829](https://github.com/sgl-project/sglang/pull/21829)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
