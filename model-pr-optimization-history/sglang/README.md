# SGLang Model PR Optimization History

Current model families:

- `deepseek-ocr`
- `deepseek-ocr-2`
- `deepseek-v3-r1`
- `deepseek-v31`
- `deepseek-v32`
- `deepseek-v4`
- `gemma4`
- `glm-vlm-ocr`
- `glm45`
- `glm46-glm47`
- `glm5-glm51`
- `gpt-oss`
- `intern-s1`
- `internvl35`
- `kimi`
- `ling25`
- `llada21`
- `llama31`
- `llama4`
- `mimo-v2-flash`
- `minimax`
- `mistral-small-4`
- `mixtral-quark-int4fp8-moe`
- `nemotron-super`
- `qwen-vlm-omni-asr`
- `qwen3-coder`
- `qwen3-core`
- `qwen3-next`
- `qwen35`
- `ring25`
- `step35`

## Current Optimization Items

Refresh: `2026-06-26`. Source head:
`sgl-project/sglang@b91348071e6ca21f9357fd8b1adc83a18781df30`.

Keep active rows close to the relevant model histories;
landed or closed rows remain here only as cross-model navigation hints until
they are mirrored in a per-model PR history.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#23882](https://github.com/sgl-project/sglang/pull/23882) | DeepSeek-V4 | merged | rebase tracking | Keeps DSV4 branch integration moving; check before assuming local DeepSeek-V4 support is final. |
| [#24047](https://github.com/sgl-project/sglang/pull/24047) | DeepSeek-V4 / SM120 | closed | Blackwell-lite support | Affects FP4, MoE, and attention kernel eligibility on SM120 hardware. |
| [#18612](https://github.com/sgl-project/sglang/pull/18612) | NVFP4 CUTLASS MoE | open | fused SiLU+Mul into expert quant | Covers a concrete activation-plus-quant MoE fuse opportunity. |
| [#22918](https://github.com/sgl-project/sglang/pull/22918) | NVFP4 MoE | merged | FlashInfer per-token NVFP4 MoE | May replace standalone per-token quant/dequant support kernels. |
| [#22851](https://github.com/sgl-project/sglang/pull/22851) | NSA | merged | top-k backend selection | Affects whether NSA top-k is fused, FlashInfer-backed, or PyTorch fallback. |
| [#24125](https://github.com/sgl-project/sglang/pull/24125) | GLM-5 / NSA | merged | skip redundant TileLang decode cat/copy | Directly relevant when decode traces show `CatArrayBatchedCopy` or copy bursts. |
| [#24007](https://github.com/sgl-project/sglang/pull/24007) | MoE LoRA | merged | csgmv backend with virtual experts | Batches adapter work that otherwise appears as many small MoE-LoRA kernels. |
| [#24150](https://github.com/sgl-project/sglang/pull/24150) | torch.compile / decode | open | local compilation support | Can hide or replace handwritten decode fusions with Inductor-generated kernels. |
| [#21878](https://github.com/sgl-project/sglang/pull/21878) | TTFT / TPOT | open | torch.compile-oriented latency work | Check before proposing one-off fixes for many small compiler-visible decode ops. |
| [#23965](https://github.com/sgl-project/sglang/pull/23965) | DSV32 / GLM5 | merged | PDL for selected kernels | Changes launch-overlap interpretation around dependent decode kernels. |
| [#24168](https://github.com/sgl-project/sglang/pull/24168) | logprobs / embeddings | open | batched GPU-to-CPU sync | Relevant when profiler gaps point to per-request sync rather than GPU kernels. |
