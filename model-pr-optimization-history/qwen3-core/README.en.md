# SGLang Qwen3 Core Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen3 dense, Qwen3 MoE, Qwen3-30B, Qwen3-235B-A22B, and shared parser/quant/LoRA/platform paths.

## Summary

Qwen3 core is the baseline for Qwen3.5, Qwen3-Next, and Qwen3.6. Current main includes dense/MoE support, RoPE config fallback, fused QK-norm/RoPE, FP8/NVFP4 ModelOpt support, EPLB/FP4, FlashInfer TRTLLM-GEN-MoE, EAGLE3 DP attention, LoRA, PP, NPU, AMD, and Blackwell-oriented validation.

## Code Surfaces

- `python/sglang/srt/models/qwen3.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/function_call/qwen25_detector.py`
- `test/registered/models/test_qwen_models.py`
- `test/registered/4-gpu-models/test_qwen3_30b.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- `test/srt/models/test_lora_qwen3.py`
- `docs/basic_usage/qwen3.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`

## Merged PRs

Key merged PRs: `#7312`, `#7912`, `#9973`, `#10574`, `#10749`, `#12002`, `#12078`, `#13489`, `#13715`, `#13998`, `#14093`, `#15203`, `#15223`, `#15390`, `#15835`, `#15890`, `#16115`, `#17535`, `#17784`, `#18189`, `#18233`, `#19059`, `#19532`, `#20931`, `#21195`, `#21654`, `#22003`, `#22358`, `#22739`.

## Open Radar

Track `#9147`, `#20127`, `#20474`, `#22529`, `#22674`, and `#22837` before starting new Qwen3 core work.

## Cookbook Evidence

- `sgl-cookbook#74`: Qwen3 AMD support and tool-calling doc fixes.
- `sgl-cookbook#245`: Qwen cookbook refresh.

## Next Work

Prioritize parser tests for streaming reasoning plus tool calls, separate W4AFP8/NPU/XPU/sliding-window validation, and always regression-test Qwen3.5/Qwen3-Next when shared Qwen3 code changes.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. This pass adds early Qwen3/Qwen3-MoE support and MoE runtime PRs: `#4693`, `#5917`, `#6120`, `#6121`, `#6250`, `#6546`, `#6533`, `#6598`, `#6652`, `#6709`, `#6818`, `#6820`, `#6964`, `#6990`, `#7222`, `#7634`, `#7681`, `#7723`, `#7740`, `#7745`, `#7966`, `#8036`, `#8280`, `#8421`, `#8448`, `#8450`, `#8658`, `#8751`, `#8987`, `#9101`, `#9338`, and `#10975`.

Additional open radar: `#20520`, `#21412`, `#21770`, `#22429`, `#22446`, `#22450`, `#22687`, `#23372`, `#23397`, and `#23434`. Additional validation surfaces include `test/manual/test_qwen3_235b.py`, `test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`, and NPU Qwen3 tests.
