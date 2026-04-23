# SGLang Qwen3-Next Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen3-Next, Qwen3-Next MTP, and the shared hybrid paths used by Qwen3-Coder-Next.

## Summary

Qwen3-Next is a dedicated optimization lane for hybrid GDN/Mamba state, RadixLinearAttention, MTP, quantized loading, CPU offload, GDN projection fusion, FlashInfer all-reduce, and AMD/NPU/Blackwell validation.

## Code Surfaces

- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_next_mtp.py`
- `python/sglang/srt/configs/qwen3_next.py`
- `test/registered/4-gpu-models/test_qwen3_next_models.py`
- `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py`
- `test/registered/models/test_qwen3_next_models_fp4.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx`

## Merged PRs

Key merged PRs: `#10912`, `#11487`, `#11969`, `#12508`, `#12525`, `#13081`, `#13708`, `#14607`, `#14855`, `#16164`, `#16863`, `#17016`, `#17373`, `#17570`, `#17613`, `#17627`, `#17660`, `#18224`, `#18355`, `#18489`, `#19220`, `#19321`, `#19767`, `#21019`, `#21313`, `#21496`, `#21662`, `#22073`, `#22458`, `#22664`.

## Open Radar

Track `#10657`, `#12892`, `#13964`, `#14502`, `#16488`, `#17981`, `#17983`, `#19812`, `#20397`, and `#23474`.

## Cookbook Evidence

- `sgl-cookbook#100`: AMD MI300X/MI355X support.
- `sgl-cookbook#123`: AMD MI325X support.
- `sgl-cookbook#143`: Qwen3-Coder-Next cookbook.

## Next Work

Add combined Mamba/radix-cache/MTP tests, keep GDN fusion tied to logits and profile evidence, and split CPU offload, PCG, EAGLE3, and Blackwell kernels into separate validation lanes.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. This pass adds official-doc initial support `#10233`, plus `#10322`, `#10379`, `#10392`, `#10466`, `#10622`, `#15631`, `#17373`, `#17627`, `#18917`, `#19434`, and `#22358`.

Additional open radar: `#21684`, `#21698`, `#22876`, `#23075`, and `#23273`. Preserve official-doc Mamba cache/SSM/scheduler flags: `--max-mamba-cache-size`, `--mamba-ssm-dtype`, `--mamba-full-memory-ratio`, `--mamba-scheduler-strategy extra_buffer`, `--page-size 64`, NEXTN/EAGLE flags, `tool-call-parser qwen`, and `reasoning-parser qwen3`.
