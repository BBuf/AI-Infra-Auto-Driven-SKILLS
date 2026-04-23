# SGLang Qwen3-Coder Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen3-Coder-480B-A35B, Qwen3-Coder-Next, the `qwen3_coder` parser, streaming tool arguments, NVFP4/FP8, and AMD/NPU/Blackwell recipes.

## Summary

Keep parser correctness separate from model performance. Qwen3-Coder-Next runtime mostly belongs to the Qwen3-Next lane, but `qwen3_coder` parser behavior is a standalone risk and is reused by Qwen3.6 docs.

## Code Surfaces

- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_moe.py`

## Merged PRs

Key merged PRs: `#16744`, `#17965`, `#18195`, `#18224`, `#18355`, `#18608`, `#18700`, and `#19736`.

## Open Radar

Track `#13411`, `#13979`, and `#21829`.

## Cookbook Evidence

Track `sgl-cookbook#86`, `#112`, `#143`, and `#174`.

## Next Work

Add parser-only tests for complex schemas, empty names, multiple tools, and incremental streaming. Run Qwen3-Next MTP/cache tests for Coder-Next runtime changes.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. Parser history now also includes `#8357`, `#8371`, `#8445`, `#12226`, and `#13163` for XML-like grammar, streaming, unknown tool calls, EBNF Composer removal, and shared detector semantics.
