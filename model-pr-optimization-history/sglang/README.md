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

Refresh: `2026-06-27`. Source head:
`sgl-project/sglang@9214b9338fcb1df5b1a9ab5835aa631539f55556`.

Keep active rows close to the relevant model histories;
landed or closed rows remain here only as cross-model navigation hints until
they are mirrored in a per-model PR history.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#29472](https://github.com/sgl-project/sglang/pull/29472) | KDA linear attention | open | FlashKDA prefill backend | Open safe-gated linear-attention work; SOTA loops for Qwen3-Next/GDN-style models should check it before calling missing KDA prefill support a local gap. |
| [#29470](https://github.com/sgl-project/sglang/pull/29470) | GLM-5 | open | router GEMM threshold tuning | Current GLM-5 serving traces may change around router GEMM dispatch once this lands. |
| [#29361](https://github.com/sgl-project/sglang/pull/29361) | LTX-2.3 diffusion | merged | residual-gate CUDA fast path | Adds `diffusion_residual_gate_add`; profiler loops should classify split `residual + update * gate` ladders as a missing shipped fusion. |
| [#29390](https://github.com/sgl-project/sglang/pull/29390) | LTX-2.3 diffusion | merged | fused Ada values | Adds a Triton `ltx2_ada_values9` path that removes repeated Ada table materialization; profiler loops should classify split Ada add/reshape/slice ladders as a missing shipped fusion. |
| [#29335](https://github.com/sgl-project/sglang/pull/29335) | Qwen3.5 / GB300 | open | FP4 disaggregated serving modes | Benchmark plans for Qwen3.5-397B on GB300 should record whether this candidate path is included. |
| [#29081](https://github.com/sgl-project/sglang/pull/29081) | DeepSeek / MLA | open | cache dequantized absorbed MLA weights | Can remove per-step MLA weight dequant overhead; check before proposing duplicate DeepSeek MLA decode work. |
| [#27976](https://github.com/sgl-project/sglang/pull/27976) | reasoning parser | open | DeepSeek V4, Qwen3.5, Kimi K2.5/K2.6 parsing | Agentic or reasoning benchmarks should account for parser parity, not just token throughput. |
| [#29250](https://github.com/sgl-project/sglang/pull/29250) | MiniMax / MSA | merged | optional `fmha_sm100` fallback | Converts missing/incompatible MSA plan API into Triton sparse-attention fallback; benchmark comparisons should not count an image with broken optional MSA as a model-level SGLang regression without this fix. |
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
