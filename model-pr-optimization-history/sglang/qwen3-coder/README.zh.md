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
