# vLLM DeepSeek V4 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not landed on current mainline
- Scope: open-radar DeepSeek V4 bring-up, BF16 sparse top-k, and DSML parser
  correctness before merge

## Landed PRs

No merged DeepSeek V4 family PR is present in the checked mainline commit.
Treat every item below as open radar.

## Open PR Radar

### Open PR #40760 - Support DeepseekV4

- Link: https://github.com/vllm-project/vllm/pull/40760
- State: open
- Diff coverage: full file list reviewed via GitHub API, `156` files, `16193`
  additions, `760` deletions
- Motivation:
  - vLLM had DeepSeek V3/V3.2 paths but no DeepSeek V4 architecture alias,
    config class, tokenizer/renderer, DSML tool parser, or MTP draft model.
  - The PR is trying to land the whole bring-up stack instead of only adding a
    registry alias.
- Key implementation:
  - Adds `DeepseekV4ForCausalLM` and `DeepSeekV4MTPModel` to the registry.
  - Introduces new runtime files `deepseek_v4.py` and `deepseek_v4_mtp.py`.
  - Adds `DeepseekV4Config`, `DeepseekV4Tokenizer`, `DeepseekV4Renderer`, and a
    dedicated `DeepSeekV4ToolParser`.
  - Extends spec-decode buffers for the V4 hyper-compressed residual path and
    shares the target `lm_head` into MTP `shared_head.head`.
- Key code excerpts:

```diff
+    "DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM"),
+    "DeepSeekV4MTPModel": ("deepseek_v4_mtp", "DeepSeekV4MTP"),
```

```diff
+class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
+    tool_call_start_token: str = "<｜DSML｜tool_calls>"
+    tool_call_end_token: str = "</｜DSML｜tool_calls>"
```

```diff
+if hasattr(draft_hf_config, "compress_ratios") and hasattr(
+    draft_hf_config, "hc_mult"
+):
+    self.hidden_size = self.hidden_size * draft_hf_config.hc_mult
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py`,
    `vllm/model_executor/models/deepseek_v4_mtp.py`,
    `vllm/model_executor/models/registry.py`,
    `vllm/model_executor/models/config.py`
  - tokenizer/renderer: `vllm/tokenizers/deepseek_v4.py`,
    `vllm/renderers/deepseek_v4.py`,
    `vllm/renderers/registry.py`,
    `vllm/tokenizers/registry.py`
  - tool calling: `vllm/tool_parsers/deepseekv4_tool_parser.py`,
    `vllm/tool_parsers/__init__.py`,
    `tests/tool_parsers/test_deepseekv4_tool_parser.py`
  - spec decode: `vllm/v1/spec_decode/eagle.py`,
    `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`,
    `vllm/v1/worker/gpu/spec_decode/eagle/utils.py`
  - tests: `tests/models/registry.py`, `tests/tokenizers_/test_deepseek_v4.py`
- Validation implications:
  - Do not present DeepSeek V4 as supported until the registry alias and parser
    are merged on mainline.
  - After merge, validate model load, chat templating, tool calls, and MTP
    speculative decode together because the PR spans all of those layers.

### Open PR #40811 - BF16 input support for persistent topK - DeepSeekV4

- Link: https://github.com/vllm-project/vllm/pull/40811
- State: open
- Diff coverage: full diff reviewed via GitHub API, `3` files, `886`
  additions, `330` deletions
- Motivation:
  - The persistent top-k path used by the DeepSeek sparse indexer assumed FP32
    ordering logic. BF16 inputs need ordered-key conversion and dedicated test
    coverage.
- Key implementation:
  - Adds BF16 ordered-key helpers and dtype traits to `persistent_topk.cuh`.
  - Makes `topk.cu` explicitly support both `float32` and `bfloat16`.
  - Extends `tests/kernels/test_top_k_per_row.py` with BF16 decode and long
    context cases.
- Key code excerpts:

```diff
+__device__ __forceinline__ auto convert_to_uint16_bf16(__nv_bfloat16 x)
+    -> uint16_t {
+  uint16_t bits;
+  memcpy(&bits, &x, sizeof(uint16_t));
+  return (bits & 0x8000) ? static_cast<uint16_t>(~bits)
+                         : static_cast<uint16_t>(bits | 0x8000);
+}
```

```diff
-// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.
+// Persistent TopK kernel for DeepSeek V3/V4 sparse attention indexer.
+// Supports float32 and bfloat16 input dtypes.
```

- Reviewed files:
  - kernel: `csrc/persistent_topk.cuh`, `csrc/topk.cu`
  - tests: `tests/kernels/test_top_k_per_row.py`
- Validation implications:
  - Re-run the kernel suite on CUDA after merge.
  - Keep BF16 and FP32 lanes separate when debugging DeepSeek V4 indexer output,
    because this patch changes the dtype-ordering logic rather than only adding
    a new call site.

### Open PR #40806 - Fix the DSML token leakage in DSV4/3.2

- Link: https://github.com/vllm-project/vllm/pull/40806
- State: open
- Diff coverage: full diff reviewed via GitHub API, `2` files, `30` additions,
  `1` deletion
- Motivation:
  - When the `<｜DSML｜function_calls>` start token arrived split across streaming
    chunks, the parser could emit partial sentinel bytes as plain content.
  - DeepSeek V4 uses a sibling DSML grammar, so the parser had to buffer partial
    prefixes instead of forwarding them.
- Key implementation:
  - Precomputes all strict prefixes of the start token.
  - Adds `_is_potential_tool_call_prefix(...)` and returns `None` while the
    sentinel is still incomplete.
  - Adds a chunk-size-1 regression test to guarantee no token leakage.
- Key code excerpts:

```diff
+self._start_prefixes = {
+    self.tool_call_start_token[:i]
+    for i in range(1, len(self.tool_call_start_token))
+}
...
+elif self._is_potential_tool_call_prefix(current_text):
+    return None
```

```diff
+def test_no_leakage_during_chunked_streaming(self, parser):
+    full_text = build_tool_call("get_weather", {"location": "SF"})
+    deltas = self._stream_chunked(parser, full_text, chunk_size=1)
+    assert "<｜DSML｜function_calls>" not in content
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py`
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py`
- Validation implications:
  - Even before a DeepSeek V4 parser lands, this is relevant because V4 tooling
    builds on the same DSML streaming behavior.
  - Streaming tool-call tests need to cover partial-sentinel chunk boundaries,
    not only full-token boundaries.
