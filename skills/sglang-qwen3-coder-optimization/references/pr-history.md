# Qwen3-Coder PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: `qwen3_coder_detector.py`, Qwen3-Coder docs/snippets, Qwen3-Next runtime files, AMD/NPU Coder tests.
- Searched PR terms: `Qwen3-Coder`, `Qwen3 Coder`, `qwen3_coder`, `Qwen3-Coder-Next`, `qwen3-coder-next`.

## Runtime Surfaces

- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `test/registered/amd/test_qwen3_coder_next_eval_mi35x.py`

## Merged/Current-Main PRs

- [#16744](https://github.com/sgl-project/sglang/pull/16744): add `qwen3_coder_detector`.
- [#17965](https://github.com/sgl-project/sglang/pull/17965): Triton TP MoE SwapAB tuning for DeepSeek V3 and Qwen3-Coder.
- [#18195](https://github.com/sgl-project/sglang/pull/18195): MoE fused config for Qwen3-Coder-Next FP8 on H100 TP=2.
- [#18224](https://github.com/sgl-project/sglang/pull/18224): ModelOpt Qwen3-Next-Coder NVFP4 support.
- [#18355](https://github.com/sgl-project/sglang/pull/18355): AMD Qwen3-Coder-Next support.
- [#18608](https://github.com/sgl-project/sglang/pull/18608): AMD Qwen3-Coder-Next accuracy/functionality on MI35x.
- [#18700](https://github.com/sgl-project/sglang/pull/18700): NPU Qwen3-Coder-Next weight transpose bugfix.
- [#19736](https://github.com/sgl-project/sglang/pull/19736): AMD Qwen3-Coder-Next AITER backend k/v scale argument fix.

## Open PR Radar

- [#13411](https://github.com/sgl-project/sglang/pull/13411): schema-aware Qwen3CoderDetector conversion.
- [#13979](https://github.com/sgl-project/sglang/pull/13979): Qwen3-Coder-480B nightly tests.
- [#21829](https://github.com/sgl-project/sglang/pull/21829): incremental streaming tool-call arguments.

## Cookbook Evidence

- sgl-cookbook [#86](https://github.com/sgl-project/sgl-cookbook/pull/86): Qwen3-Coder-480B-A35B AMD MI300X deployment guide.
- sgl-cookbook [#112](https://github.com/sgl-project/sgl-cookbook/pull/112): MI325X and MI355X support.
- sgl-cookbook [#143](https://github.com/sgl-project/sgl-cookbook/pull/143): Qwen3-Coder-Next cookbook.
- sgl-cookbook [#174](https://github.com/sgl-project/sgl-cookbook/pull/174): NVIDIA B200/GB200 support.

## Validation Notes

- Parser-only tests are mandatory; model smoke tests alone miss incremental streaming bugs.
- Coder-Next model changes should be validated with Qwen3-Next MTP/cache tests when hybrid state is touched.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional parser PRs found by detector history:

- [#8357](https://github.com/sgl-project/sglang/pull/8357): XML-like grammar and Qwen3 detector bugfixes.
- [#8371](https://github.com/sgl-project/sglang/pull/8371): streaming update.
- [#8445](https://github.com/sgl-project/sglang/pull/8445): GLM-4.5 follow-up that changed shared detector expectations.
- [#12226](https://github.com/sgl-project/sglang/pull/12226): forward unknown tool calls instead of dropping them.
- [#13163](https://github.com/sgl-project/sglang/pull/13163): remove EBNF Composer.

Keep runtime work for Qwen3-Coder-Next cross-linked to the Qwen3-Next skill because GDN/Mamba/MTP/cache behavior is shared.
