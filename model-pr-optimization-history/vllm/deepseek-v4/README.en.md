# vLLM DeepSeek V4 Support and PR History

This note tracks the vLLM status for DeepSeek V4 at commit
`0f7be0f2f76814f80f9091220a5fbbb53912ad00`.

- Status: not supported on current mainline; only open PR evidence exists

## Key Conclusions

- Current mainline still does not register `DeepseekV4ForCausalLM` in
  `vllm/model_executor/models/registry.py`.
- The real bring-up work is concentrated in open PR `#40760`, which spans the
  model, MTP draft path, tokenizer, renderer, parser, tests, and spec-decode
  glue.
- Two additional open PRs are already relevant even before merge:
  `#40811` for BF16 persistent top-k and `#40806` for DSML streaming safety.

## Main Runtime Surfaces

- Current-main check point: `vllm/vllm/model_executor/models/registry.py`
- Open-radar files:
  `vllm/vllm/model_executor/models/deepseek_v4.py`,
  `vllm/vllm/model_executor/models/deepseek_v4_mtp.py`,
  `vllm/vllm/tokenizers/deepseek_v4.py`,
  `vllm/vllm/renderers/deepseek_v4.py`,
  `vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`,
  `vllm/vllm/v1/spec_decode/eagle.py`,
  `vllm/csrc/persistent_topk.cuh`

## Open PR Radar

- [#40760](https://github.com/vllm-project/vllm/pull/40760)
  `[New Model] Support DeepseekV4`
  Diff reviewed: `156` files, `16193` additions, `760` deletions.
  The PR adds the proposed model alias, V4 MTP class, tokenizer, renderer,
  parser, config mapping, and speculative-decode wiring.
- [#40811](https://github.com/vllm-project/vllm/pull/40811)
  `[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4`
  Diff reviewed: `3` files, `886` additions, `330` deletions.
  The patch teaches the sparse top-k kernel to handle BF16 ordered keys and adds
  BF16 kernel tests.
- [#40806](https://github.com/vllm-project/vllm/pull/40806)
  `[Bugfix] Fix the DSML token leakage in DSV4/3.2`
  Diff reviewed: `2` files, `30` additions, `1` deletion.
  The parser now buffers partial DSML sentinels instead of leaking them as plain
  text during chunked streaming.

## Current Contract

Do not claim DeepSeek V4 support in vLLM until the model alias appears on
mainline and the tokenizer/parser path merges with it. If these PRs merge,
validate model load, tool calling, speculative decoding, and BF16 sparse top-k
as one connected stack.
