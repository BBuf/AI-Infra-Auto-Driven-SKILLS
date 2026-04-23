# Qwen3 Core PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched code paths: `qwen3.py`, `qwen3_moe.py`, Qwen parser/config files, registered Qwen tests, Qwen cookbook/docs.
- Searched PR terms: `Qwen3`, `qwen3`, `Qwen3 MoE`, `Qwen3 235B`, `Qwen3 30B`, `qwen3_moe`, `qwen25_detector`.

## Runtime Surfaces

- `python/sglang/srt/models/qwen3.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/function_call/qwen25_detector.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/configs/`
- `docs/basic_usage/qwen3.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`
- `test/registered/models/test_qwen_models.py`
- `test/registered/4-gpu-models/test_qwen3_30b.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- `test/srt/models/test_lora_qwen3.py`

## Merged/Current-Main PRs

- [#7312](https://github.com/sgl-project/sglang/pull/7312): LoRA hidden-dim support for Qwen3.
- [#7912](https://github.com/sgl-project/sglang/pull/7912): Qwen FP8/NVFP4 ModelOpt quantization support.
- [#9973](https://github.com/sgl-project/sglang/pull/9973): Qwen3-MoE FlashInfer fused all-reduce optimization.
- [#10574](https://github.com/sgl-project/sglang/pull/10574): Ascend Qwen3 optimization.
- [#10749](https://github.com/sgl-project/sglang/pull/10749): fuse write-KV-buffer into RoPE for Qwen3-MoE and adjacent models.
- [#10911](https://github.com/sgl-project/sglang/pull/10911): Qwen3-Omni thinker-only support, also touched Qwen3 MoE plumbing.
- [#12002](https://github.com/sgl-project/sglang/pull/12002): EAGLE3 DP attention for Qwen3 MoE.
- [#12078](https://github.com/sgl-project/sglang/pull/12078): Ascend Qwen optimization.
- [#13489](https://github.com/sgl-project/sglang/pull/13489): FlashInfer TRTLLM-GEN-MoE path for Qwen3.
- [#13715](https://github.com/sgl-project/sglang/pull/13715): EPLB plus FP4 path for Qwen3 MoE.
- [#13998](https://github.com/sgl-project/sglang/pull/13998): fused QK-norm/RoPE for Qwen3-MoE.
- [#14093](https://github.com/sgl-project/sglang/pull/14093): fused FP8 KV-cache write for TRTLLM MHA.
- [#15203](https://github.com/sgl-project/sglang/pull/15203): NPU GPTQ quantization for Qwen3.
- [#15223](https://github.com/sgl-project/sglang/pull/15223): Qwen3 model loading under PP.
- [#15390](https://github.com/sgl-project/sglang/pull/15390): NPU Qwen3 PP.
- [#15835](https://github.com/sgl-project/sglang/pull/15835): JIT fused QK norm.
- [#15890](https://github.com/sgl-project/sglang/pull/15890): tied embedding logic under PP.
- [#16115](https://github.com/sgl-project/sglang/pull/16115): NPU DP LM-head path.
- [#17535](https://github.com/sgl-project/sglang/pull/17535): Qwen3 embedding weight rename.
- [#17784](https://github.com/sgl-project/sglang/pull/17784): transformers 5.3.0 compatibility changes touching Qwen3.
- [#18189](https://github.com/sgl-project/sglang/pull/18189): ModelOpt Qwen3-235B NVFP4 launch.
- [#18233](https://github.com/sgl-project/sglang/pull/18233): Qwen3 MoE context parallel.
- [#19059](https://github.com/sgl-project/sglang/pull/19059): fused QK-norm/RoPE JIT follow-up.
- [#19532](https://github.com/sgl-project/sglang/pull/19532): NPU speculative condition for Qwen3 and Qwen3 MoE.
- [#20931](https://github.com/sgl-project/sglang/pull/20931): Qwen3 RoPE parameter compatibility.
- [#21195](https://github.com/sgl-project/sglang/pull/21195): Qwen3 registered test enablement.
- [#21654](https://github.com/sgl-project/sglang/pull/21654): deduplicate `sincosf` in fused QK-norm/RoPE.
- [#22003](https://github.com/sgl-project/sglang/pull/22003): MoE/attention-CP topology relaxation that affects Qwen3 MoE deployments.
- [#22358](https://github.com/sgl-project/sglang/pull/22358): DFLASH support across Qwen model paths.
- [#22739](https://github.com/sgl-project/sglang/pull/22739): restore Qwen3 RoPE config fallback.

## Open PR Radar

- [#9147](https://github.com/sgl-project/sglang/pull/9147): Qwen3-MoE W4AFP8.
- [#20127](https://github.com/sgl-project/sglang/pull/20127): tied embeddings for Qwen MoE and Qwen3-Next.
- [#20474](https://github.com/sgl-project/sglang/pull/20474): Intel XPU Qwen3 layernorm/MRoPE.
- [#22529](https://github.com/sgl-project/sglang/pull/22529): sliding window attention for Qwen3.
- [#22674](https://github.com/sgl-project/sglang/pull/22674): NPU Qwen3.5-MoE and Qwen3-Next quantization.
- [#22837](https://github.com/sgl-project/sglang/pull/22837): Qwen3 reasoning detector swallowing tool calls.

## Cookbook/Docs Evidence

- sgl-cookbook [#74](https://github.com/sgl-project/sgl-cookbook/pull/74): AMD support for Qwen3 and tool-calling typo fixes.
- sgl-cookbook [#245](https://github.com/sgl-project/sgl-cookbook/pull/245): Qwen cookbook refresh.

## Validation Notes

- Keep Qwen3 dense and MoE paths in separate tests when possible.
- Quantized checkpoint support should cite the exact exporter and backend because ModelOpt, compressed-tensors, GPTQ, and platform-native quantization use different loader assumptions.
- Parser fixes must be checked with streaming tool-call chunks and reasoning content in the same response.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional merged/current-main PRs found by path-level `git log`:

- [#4693](https://github.com/sgl-project/sglang/pull/4693): initial Qwen3/Qwen3MoE support.
- [#5917](https://github.com/sgl-project/sglang/pull/5917), [#6120](https://github.com/sgl-project/sglang/pull/6120), [#6121](https://github.com/sgl-project/sglang/pull/6121): Qwen3 MoE EP/DeepEP/DP-attention bring-up.
- [#6250](https://github.com/sgl-project/sglang/pull/6250), [#6546](https://github.com/sgl-project/sglang/pull/6546), [#6709](https://github.com/sgl-project/sglang/pull/6709): PP and tied-weight support for Qwen2/Qwen3/Qwen3 MoE.
- [#6533](https://github.com/sgl-project/sglang/pull/6533), [#6818](https://github.com/sgl-project/sglang/pull/6818), [#6964](https://github.com/sgl-project/sglang/pull/6964), [#7580](https://github.com/sgl-project/sglang/pull/7580), [#8448](https://github.com/sgl-project/sglang/pull/8448): EPLB and expert-distribution evolution.
- [#6598](https://github.com/sgl-project/sglang/pull/6598), [#6652](https://github.com/sgl-project/sglang/pull/6652), [#8280](https://github.com/sgl-project/sglang/pull/8280), [#9101](https://github.com/sgl-project/sglang/pull/9101): two-batch-overlap, DP LM-head, DP-attention padding, and throughput follow-ups.
- [#6820](https://github.com/sgl-project/sglang/pull/6820), [#7222](https://github.com/sgl-project/sglang/pull/7222), [#7723](https://github.com/sgl-project/sglang/pull/7723), [#7966](https://github.com/sgl-project/sglang/pull/7966), [#8036](https://github.com/sgl-project/sglang/pull/8036), [#8421](https://github.com/sgl-project/sglang/pull/8421), [#8450](https://github.com/sgl-project/sglang/pull/8450), [#8658](https://github.com/sgl-project/sglang/pull/8658), [#8751](https://github.com/sgl-project/sglang/pull/8751), [#9338](https://github.com/sgl-project/sglang/pull/9338): fused MoE, FlashInfer, `select_experts`, and MoE argument evolution.
- [#6990](https://github.com/sgl-project/sglang/pull/6990), [#7634](https://github.com/sgl-project/sglang/pull/7634), [#7681](https://github.com/sgl-project/sglang/pull/7681), [#7740](https://github.com/sgl-project/sglang/pull/7740), [#7745](https://github.com/sgl-project/sglang/pull/7745), [#8987](https://github.com/sgl-project/sglang/pull/8987), [#10975](https://github.com/sgl-project/sglang/pull/10975): embedding, layer-wise prefill, DP attention, two-stream norm, EAGLE3, hidden-dim helper, and memory-fraction heuristics.

Additional open radar: [#20520](https://github.com/sgl-project/sglang/pull/20520), [#21412](https://github.com/sgl-project/sglang/pull/21412), [#21770](https://github.com/sgl-project/sglang/pull/21770), [#22429](https://github.com/sgl-project/sglang/pull/22429), [#22446](https://github.com/sgl-project/sglang/pull/22446), [#22450](https://github.com/sgl-project/sglang/pull/22450), [#22687](https://github.com/sgl-project/sglang/pull/22687), [#23372](https://github.com/sgl-project/sglang/pull/23372), [#23397](https://github.com/sgl-project/sglang/pull/23397), [#23434](https://github.com/sgl-project/sglang/pull/23434).

Extra validation surfaces: `test/manual/test_qwen3_235b.py`, `test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`, `test/registered/stress/test_stress_qwen3_235b.py`, and NPU Qwen3 tests under `test/registered/npu/`.
