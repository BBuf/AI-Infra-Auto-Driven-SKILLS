# vLLM DeepSeek V4 支持与 PR 历史

本文记录 vLLM 在提交 `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
附近对 DeepSeek V4 的真实状态。

- 状态: 当前 mainline 尚未支持，只有 open PR 证据

## 核心结论

- 当前 mainline 的 `vllm/model_executor/models/registry.py` 里还没有
  `DeepseekV4ForCausalLM`。
- 真正的 bring-up 工作集中在 open PR `#40760`，它不是只加 alias，而是
  一次性改了模型实现、MTP、tokenizer、renderer、tool parser、测试和
  spec-decode 配套。
- 另外两个 open PR 也已经构成 DeepSeek V4 证据链的一部分:
  `#40811` 负责 BF16 persistent top-k，
  `#40806` 负责 DSML 流式解析不泄漏 sentinel。

## 主要代码面

- 当前 mainline 检查点: `vllm/vllm/model_executor/models/registry.py`
- open-radar 代码面:
  `vllm/vllm/model_executor/models/deepseek_v4.py`,
  `vllm/vllm/model_executor/models/deepseek_v4_mtp.py`,
  `vllm/vllm/tokenizers/deepseek_v4.py`,
  `vllm/vllm/renderers/deepseek_v4.py`,
  `vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`,
  `vllm/vllm/v1/spec_decode/eagle.py`,
  `vllm/csrc/persistent_topk.cuh`

## Open PR 雷达

- [#40760](https://github.com/vllm-project/vllm/pull/40760)
  `[New Model] Support DeepseekV4`
  已审 diff: `156` 个文件，`16193` 行新增，`760` 行删除。
  这组改动提出了 DeepSeek V4 主模型、MTP 草稿模型、tokenizer、renderer、
  parser、config 映射和 speculative decode 连接层。
- [#40811](https://github.com/vllm-project/vllm/pull/40811)
  `[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4`
  已审 diff: `3` 个文件，`886` 行新增，`330` 行删除。
  它把稀疏 top-k kernel 从默认 FP32 排序逻辑扩展到 BF16，并补了 BF16
  kernel 测试。
- [#40806](https://github.com/vllm-project/vllm/pull/40806)
  `[Bugfix] Fix the DSML token leakage in DSV4/3.2`
  已审 diff: `2` 个文件，`30` 行新增，`1` 行删除。
  它修掉了流式输出里半截 DSML sentinel 被当作普通文本吐出的 parser 问题。

## 当前结论

在 model alias 真正进入 mainline 之前，不要把 DeepSeek V4 写成 vLLM
已支持。等这些 PR 合入后，要把模型加载、tool calling、spec decode 和
BF16 sparse top-k 作为一整条链路一起回归。
