# vLLM GLM-5 / 5.1 Support and PR History

This note tracks the landed GLM-5 / 5.1 path in vLLM at commit
`0f7be0f2f76814f80f9091220a5fbbb53912ad00`.

- Status: partially supported on current mainline

## Key Conclusions

- GLM-5 support is currently an adaptation layer on top of the DeepSeek-V2/V3
  runtime, not an independent `glm5.py` implementation.
- The two key landed steps are:
  `#34124` for architecture/config adaptation and `#34385` for MTP accuracy.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/config/speculative.py`
- `vllm/vllm/transformers_utils/model_arch_config_convertor.py`
- `vllm/vllm/v1/spec_decode/eagle.py`

## Landed PRs

- [#34124](https://github.com/vllm-project/vllm/pull/34124)
  `GLM adaptation`
  Diff reviewed: `7` files, `13` additions, `3` deletions.
  Adds `GlmMoeDsaForCausalLM` as a DeepSeek-V2-derived alias, extends
  speculative config conversion, and respects `indexer_rope_interleave`.
- [#34385](https://github.com/vllm-project/vllm/pull/34385)
  `Fix MTP accuracy for GLM-5`
  Diff reviewed: `1` file, `18` additions.
  Shares the target `lm_head` into MTP `shared_head.head` so GLM-5 draft logits
  stop producing NaNs or uninitialized outputs.

## Current Contract

If GLM-5 breaks, inspect the DeepSeek-based MLA/MoE runtime and speculative
decode stack first. Do not assume the older `glm4*` files are the source of
truth for GLM-5 behavior.
