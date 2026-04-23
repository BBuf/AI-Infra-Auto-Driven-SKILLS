# Qwen3-Next PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: `qwen3_next.py`, `qwen3_next_mtp.py`, `qwen3_next.py` config, registered Qwen3-Next tests, docs/cookbook snippets.
- Searched PR terms: `Qwen3-Next`, `qwen3-next`, `qwen3_next`, `GDN`, `RadixLinearAttention`, `Qwen3 Next MTP`.

## Runtime Surfaces

- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_next_mtp.py`
- `python/sglang/srt/configs/qwen3_next.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx`
- `test/registered/4-gpu-models/test_qwen3_next_models.py`
- `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py`
- `test/registered/models/test_qwen3_next_models_fp4.py`

## Merged/Current-Main PRs

- [#10912](https://github.com/sgl-project/sglang/pull/10912): PD support for Qwen3-Next and hybrid models.
- [#11487](https://github.com/sgl-project/sglang/pull/11487): early Qwen3-Next support path.
- [#11969](https://github.com/sgl-project/sglang/pull/11969): hybrid-model follow-up touching Qwen3-Next.
- [#12508](https://github.com/sgl-project/sglang/pull/12508): Qwen3-Next runtime cleanup.
- [#12525](https://github.com/sgl-project/sglang/pull/12525): Qwen3-Next follow-up.
- [#13081](https://github.com/sgl-project/sglang/pull/13081): Qwen3-Next model path update.
- [#13708](https://github.com/sgl-project/sglang/pull/13708): Qwen3-Next accuracy/perf follow-up.
- [#14607](https://github.com/sgl-project/sglang/pull/14607): hybrid runtime update for Qwen3-Next.
- [#14855](https://github.com/sgl-project/sglang/pull/14855): Qwen3-Next scheduler/runtime update.
- [#16164](https://github.com/sgl-project/sglang/pull/16164): Qwen3-Next model update.
- [#16863](https://github.com/sgl-project/sglang/pull/16863): Qwen3-Next state/runtime update.
- [#17016](https://github.com/sgl-project/sglang/pull/17016): Qwen3-Next follow-up.
- [#17373](https://github.com/sgl-project/sglang/pull/17373): Qwen3-Next follow-up.
- [#17570](https://github.com/sgl-project/sglang/pull/17570): Qwen3-Next and hybrid loader/runtime update.
- [#17613](https://github.com/sgl-project/sglang/pull/17613): Qwen3-Next follow-up.
- [#17627](https://github.com/sgl-project/sglang/pull/17627): Qwen3-Next follow-up.
- [#17660](https://github.com/sgl-project/sglang/pull/17660): Qwen3-Next model/runtime update.
- [#18224](https://github.com/sgl-project/sglang/pull/18224): ModelOpt Qwen3-Next-Coder NVFP4; relevant to shared Qwen3-Next loading.
- [#18355](https://github.com/sgl-project/sglang/pull/18355): AMD Qwen3-Coder-Next support; touches Qwen3-Next architecture.
- [#18489](https://github.com/sgl-project/sglang/pull/18489): Qwen3.5 support also added/refactored Qwen3-Next-adjacent classes.
- [#18917](https://github.com/sgl-project/sglang/pull/18917): Qwen3-Next runtime follow-up.
- [#19220](https://github.com/sgl-project/sglang/pull/19220): PCG fix that affected Qwen3-Next/Qwen3.5 hybrid paths.
- [#19321](https://github.com/sgl-project/sglang/pull/19321): fuse Qwen3-Next GDN `qkvz_proj` and `ba_proj`.
- [#19434](https://github.com/sgl-project/sglang/pull/19434): Qwen3-Next follow-up.
- [#19767](https://github.com/sgl-project/sglang/pull/19767): Qwen3.5 MTP/EPLB changes also touched Next-family MTP.
- [#21019](https://github.com/sgl-project/sglang/pull/21019): fuse split/reshape/cat ops in GDN projection.
- [#21313](https://github.com/sgl-project/sglang/pull/21313): Qwen3-Next weight loading bugfix.
- [#21496](https://github.com/sgl-project/sglang/pull/21496): revert of an incorrect Qwen3-Next loading fix.
- [#21662](https://github.com/sgl-project/sglang/pull/21662): FP8 `weight_loader` property assignment fix.
- [#22073](https://github.com/sgl-project/sglang/pull/22073): Qwen3-ASR support; shares some Next-family hybrid plumbing.
- [#22458](https://github.com/sgl-project/sglang/pull/22458): fix NCCL AllGather hang for Qwen3-Next MTP.
- [#22664](https://github.com/sgl-project/sglang/pull/22664): auto-enable FlashInfer all-reduce for Qwen3-Next.

## Open PR Radar

- [#10657](https://github.com/sgl-project/sglang/pull/10657): EAGLE3 for Qwen3-Next.
- [#12892](https://github.com/sgl-project/sglang/pull/12892): avoid SSM/conv state copy during speculative decoding.
- [#13964](https://github.com/sgl-project/sglang/pull/13964): kernel performance optimization.
- [#14502](https://github.com/sgl-project/sglang/pull/14502): PCG optimization.
- [#16488](https://github.com/sgl-project/sglang/pull/16488): TBO support.
- [#17981](https://github.com/sgl-project/sglang/pull/17981): CuteDSL decode/MTP Blackwell kernels.
- [#17983](https://github.com/sgl-project/sglang/pull/17983): prefill kernel, GDN Gluon, and cumsum work.
- [#19812](https://github.com/sgl-project/sglang/pull/19812): Qwen3.5/Qwen3-Next MTP and EPLB compatibility.
- [#20397](https://github.com/sgl-project/sglang/pull/20397): NPU MTP support.
- [#23474](https://github.com/sgl-project/sglang/pull/23474): CPU offload bugfix for hybrid linear-attention models.

## Cookbook Evidence

- sgl-cookbook [#100](https://github.com/sgl-project/sgl-cookbook/pull/100): AMD MI300X/MI355X support.
- sgl-cookbook [#123](https://github.com/sgl-project/sgl-cookbook/pull/123): AMD MI325X support.
- sgl-cookbook [#143](https://github.com/sgl-project/sgl-cookbook/pull/143): Qwen3-Coder-Next cookbook, adjacent to Next deployment flags.

## Validation Notes

- Keep MTP and non-MTP tests separate.
- For GDN fusion, compare logits before/after fusion on short, long, and prefix-cache prompts.
- For CPU offload, test both BF16 and quantized checkpoints because loader state and hybrid cache state fail differently.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional merged/current-main PRs found by doc/code/git-log sweep:

- [#10233](https://github.com/sgl-project/sglang/pull/10233): initial Qwen3-Next support referenced by the official SGLang docs.
- [#10322](https://github.com/sgl-project/sglang/pull/10322), [#10379](https://github.com/sgl-project/sglang/pull/10379), [#10392](https://github.com/sgl-project/sglang/pull/10392), [#10466](https://github.com/sgl-project/sglang/pull/10466), [#10622](https://github.com/sgl-project/sglang/pull/10622): early norm/NPU/MTP+DP/FP8 DeepEP fixes.
- [#15631](https://github.com/sgl-project/sglang/pull/15631): CuTe DSL GDN decode kernel.
- [#17373](https://github.com/sgl-project/sglang/pull/17373): RadixLinearAttention path.
- [#17627](https://github.com/sgl-project/sglang/pull/17627): NVFP4 quantized Qwen3-Next model path.
- [#18917](https://github.com/sgl-project/sglang/pull/18917), [#19434](https://github.com/sgl-project/sglang/pull/19434), [#22358](https://github.com/sgl-project/sglang/pull/22358): later runtime/DFLASH follow-ups not called out in the first draft.

Additional open radar: [#21684](https://github.com/sgl-project/sglang/pull/21684), [#21698](https://github.com/sgl-project/sglang/pull/21698), [#22876](https://github.com/sgl-project/sglang/pull/22876), [#23075](https://github.com/sgl-project/sglang/pull/23075), [#23273](https://github.com/sgl-project/sglang/pull/23273).

Official-doc flags that must be preserved in optimization playbooks: `--max-mamba-cache-size`, `--mamba-ssm-dtype`, `--mamba-full-memory-ratio`, `--mamba-scheduler-strategy extra_buffer`, `--page-size 64`, NEXTN/EAGLE speculative flags, `--tool-call-parser qwen`, and `--reasoning-parser qwen3`.
