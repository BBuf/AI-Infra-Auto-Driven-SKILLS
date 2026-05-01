# SGLang Model PR Optimization History

Current model families:

- `deepseek-v3-r1`
- `deepseek-v31`
- `deepseek-v32`
- `deepseek-v4`
- `ernie45`
- `gemma4`
- `glm-vlm-ocr`
- `glm45`
- `glm46-glm47`
- `glm5-glm51`
- `gpt-oss`
- `hunyuan3-preview`
- `intern-s1`
- `internvl35`
- `kimi`
- `llama4`
- `mimo-v2-flash`
- `minimax`
- `mistral-small-4`
- `mixtral-quark-int4fp8-moe`
- `moss-vl`
- `nemotron-super`
- `qwen-vlm-omni-asr`
- `qwen3-coder`
- `qwen3-core`
- `qwen3-next`
- `qwen35`
- `qwen36`
- `step35`

## Open Optimization Items

Refresh: `2026-05-01`. Keep these rows close to the relevant model histories and
skills; move them into the per-model PR history once they land or become
directly traceable through model implementation files.

| PR | Model / area | Current signal | Why it matters |
| --- | --- | --- | --- |
| [#23882](https://github.com/sgl-project/sglang/pull/23882) | DeepSeek-V4 | rebase tracking | Keeps DSV4 branch integration moving; check before assuming local DeepSeek-V4 support is final. |
| [#24047](https://github.com/sgl-project/sglang/pull/24047) | DeepSeek-V4 / SM120 | Blackwell-lite support | Affects FP4, MoE, and attention kernel eligibility on SM120 hardware. |
| [#18612](https://github.com/sgl-project/sglang/pull/18612) | NVFP4 CUTLASS MoE | fused SiLU+Mul into expert quant | Covers a concrete activation-plus-quant MoE fuse opportunity. |
| [#22918](https://github.com/sgl-project/sglang/pull/22918) | NVFP4 MoE | FlashInfer per-token NVFP4 MoE | May replace standalone per-token quant/dequant support kernels. |
| [#22851](https://github.com/sgl-project/sglang/pull/22851) | NSA | top-k backend selection | Affects whether NSA top-k is fused, FlashInfer-backed, or PyTorch fallback. |
| [#24125](https://github.com/sgl-project/sglang/pull/24125) | GLM-5 / NSA | skip redundant TileLang decode cat/copy | Directly relevant when decode traces show `CatArrayBatchedCopy` or copy bursts. |
| [#24007](https://github.com/sgl-project/sglang/pull/24007) | MoE LoRA | csgmv backend with virtual experts | Batches adapter work that otherwise appears as many small MoE-LoRA kernels. |
| [#24150](https://github.com/sgl-project/sglang/pull/24150) | torch.compile / decode | local compilation support | Can hide or replace handwritten decode fusions with Inductor-generated kernels. |
| [#21878](https://github.com/sgl-project/sglang/pull/21878) | TTFT / TPOT | torch.compile-oriented latency work | Check before proposing one-off fixes for many small compiler-visible decode ops. |
| [#23965](https://github.com/sgl-project/sglang/pull/23965) | DSV32 / GLM5 | PDL for selected kernels | Changes launch-overlap interpretation around dependent decode kernels. |
| [#24168](https://github.com/sgl-project/sglang/pull/24168) | logprobs / embeddings | batched GPU-to-CPU sync | Relevant when profiler gaps point to per-request sync rather than GPU kernels. |
