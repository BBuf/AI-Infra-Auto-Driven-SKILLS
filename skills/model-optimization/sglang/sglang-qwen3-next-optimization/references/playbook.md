# Qwen3-Next Optimization Playbook

This playbook is a working guide. Use `pr-history.md` as the authoritative PR-by-PR dossier.

## Symptom Map

| Symptom | First files to inspect | PR trail |
| --- | --- | --- |
| Base model load or architecture mismatch | `configs/qwen3_next.py`, `qwen3_next.py`, `qwen3_next_mtp.py` | `#10233`, `#10322`, `#10392` |
| MTP draft/verify accuracy mismatch | `qwen3_next_mtp.py`, `hybrid_linear_attn_backend.py`, `speculative/` | `#10392`, `#14607`, `#19767`, `#22458`, `#12892` |
| GDN projection overhead | `qwen3_next.py`, `jit_kernel/triton/gdn_fused_proj.py` | `#18917`, `#19321`, `#21019` |
| Norm/gate/GDN decode bottleneck | `layers/attention/linear/`, `layers/attention/fla/` | `#12508`, `#15631`, `#17981`, `#17983`, `#19434`, `#23273` |
| PCG compile/capture issue | `radix_linear_attention.py`, `model_runner.py`, split-op registry | `#13081`, `#16863`, `#17613`, `#19220`, `#14502` |
| FP8/NVFP4/W8A8 loading failure | `qwen3_next.py`, quant configs, `layers/linear.py` | `#10466`, `#10622`, `#17627`, `#18224`, `#21313`, `#21496`, `#21662`, `#21698` |
| CPU offload crash or garbage output | `utils/offloader.py`, `RadixLinearAttention.conv_weights` | `#23474` |
| Mixed chunk accuracy drop | `server_args.py`, `mamba2_metadata.py`, `schedule_batch.py` | `#22876`, `#23075` |
| Memory leak/cache alias issue | `mem_cache/allocator.py`, `memory_pool.py` | `#21684` |
| NPU path breakage | Ascend backend, NPU memory pool, GDN backend | `#10379`, `#11969`, `#16164`, `#20397`, `#21698` |
| AMD path breakage | AITER backend, dual-stream guard | `#17016`, `#18355` |

## Investigation Order

1. Reproduce without MTP, with `--mamba-scheduler-strategy no_buffer`, and without CPU offload.
2. Add quantization while keeping speculative decoding disabled.
3. Add radix/prefix cache and `extra_buffer`.
4. Add mixed chunk only after the prefill/decode state metadata is known-good.
5. Enable MTP/NEXTN/EAGLE and test TP>1.
6. Only then profile GDN fusion, FlashInfer, CuTe DSL, Gluon, all-reduce, or TBO.

## Official Flag Checklist

Preserve and record the official deployment flags when reproducing cookbook behavior:

- Mamba cache: `--max-mamba-cache-size`, `--mamba-full-memory-ratio`
- SSM dtype: `--mamba-ssm-dtype`
- Scheduler: `--mamba-scheduler-strategy extra_buffer`
- Page size: `--page-size 64` or backend-specific page-size override
- Speculative decoding: `SGLANG_ENABLE_SPEC_V2=1`, `--speculative-algorithm NEXTN`, EAGLE flags
- Parser flags: `--tool-call-parser qwen`, `--reasoning-parser qwen3`
- Backend choice: `--attention-backend`, `--linear-attn-decode-backend`, `--linear-attn-prefill-backend`
- Quantization: `--quantization modelopt_fp4`, FP8/W8A8-specific loader configs
- Offload: `--cpu-offload-gb`

## Change Rules

- If changing GDN projection layout, prove both split-checkpoint and packed-checkpoint loading.
- If changing Mamba state, prove radix-cache reuse and target-verify rollback.
- If changing CPU offload, test tied parameters and plain tensor views; dense-model offload tests are not enough.
- If changing NPU/AMD code, avoid CUDA-only assumptions around stream, conv state layout, and custom kernel availability.
- If changing PCG, document which operations remain eager and why.
- If changing FlashInfer/CuTe/Gluon paths, record SM architecture, kernel package version, and fallback path.

## Minimal Validation Commands

```bash
python -m pytest test/registered/4-gpu-models/test_qwen3_next_models.py
python -m pytest test/registered/4-gpu-models/test_qwen3_next_models_mtp.py
python -m pytest test/registered/models/test_qwen3_next_models_fp4.py
```

## Lane-Specific Validation

- MTP: run no-MTP, MTP greedy, MTP sampling, TP>1, radix cache on/off.
- Mixed chunk: run concurrency 1 and concurrency 16+ with `extra_buffer`; compare GSM8K or equivalent.
- CPU offload: compare greedy outputs at `--cpu-offload-gb 0` and `>0`; include a hybrid model with cached conv weight views.
- GDN kernels: collect TTFT, TPOT, output throughput, kernel time, and accuracy.
- NPU: run W8A8 and BF16 separately; check fused projection loader scale/offset tensors.
- AMD: verify alt-stream remains disabled when unsupported and that AITER gets correct `v_head_dim`.

## Documentation Standard

When adding a PR to Qwen3-Next docs, include:

- Why the PR was needed
- Which files and execution path changed
- The essential code fragment
- Benchmark/accuracy evidence, or a clear note that validation was not provided
- Current status: merged/current-main, open radar, adjacent, reverted, or superseded
