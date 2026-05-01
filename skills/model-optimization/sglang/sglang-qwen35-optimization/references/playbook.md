# Qwen3.5 Optimization Playbook

Use this playbook together with `references/pr-history.md`. Do not diagnose a Qwen3.5 issue from PR titles alone; first map the symptom to the diff-reviewed PR cards and then inspect the current SGLang source.

## Symptom Map

| Symptom | First files to inspect | Diff-reviewed PR trail |
| --- | --- | --- |
| Initial architecture, class registration, config mismatch | `qwen3_5.py`, `qwen3_5_mtp.py`, `configs/qwen3_5.py`, model registry | `#18489`, `#18538`, `#18544` |
| FP8/NVFP4/MXFP4 load failure | `linear.py`, `fp8.py`, `quantization/utils.py`, `qwen3_5.py`, `qwen3_5_mtp.py` | `#18926`, `#18937`, `#21234`, `#21692`, `#22948`, `#23467` |
| Dense model accuracy issue under TP | `qwen3_5.py`, communicator/layernorm paths | `#19070`, `#19411`, `#19961` |
| MTP/spec-v2 failure or bad acceptance | `qwen3_5_mtp.py`, `eagle_worker_v2.py`, `eagle_info_v2.py`, `server_args.py` | `#19391`, `#19767`, `#20864`, `#22908`, `#23034` |
| GDN projection speed or correctness regression | `gdn_fused_proj.py`, `qwen3_5.py`, FLA GDN kernels | `#20386`, `#21019`, `#22312` |
| Shared expert fusion issue | `qwen2_moe.py`, Qwen3.5 weight loading | `#20736`, `#22948` |
| Pipeline parallel load/OOM/tied embedding bug | `qwen3_5.py`, `memory_pool.py`, PP tests | `#19670`, `#21070`, `#21347`, `#21448` |
| AMD performance or ROCm behavior | AMD registered tests, `qwen2_moe.py`, `server_args.py` | `#20736`, `#21234`, `#21669`, `#22908`, `#22948` |
| NPU quant behavior | `modelslim.py`, `model_loader/loader.py`, `qwen3_5.py` | `#18544`, `#21692` |
| VLM / processor / encoder-disaggregation issue | `qwen_vl.py`, `encode_server.py`, `server_args.py`, EPD tests | `#21849`, `#22431` |
| PD / NIXL / heterogeneous TP issue | `disaggregation/nixl/conn.py` | `#22145`, `#22240` |
| Retraction under memory pressure | `schedule_batch.py`, `scheduler.py`, `memory_pool.py`, Mamba unit tests | `#22493` |
| GB300/B200 CI instability | GB300/B200 registered tests, workflow partitioning | `#21487`, `#22913` |
| DFLASH aux hidden capture | `qwen3_5.py`, layer communicator | `#22358` |

## Investigation Order

1. Reproduce on a reference BF16 or unquantized checkpoint with MTP disabled.
2. Confirm the exact model class: dense text, MoE text, dense VLM, or MoE VLM.
3. Inspect weight-loader mappings before kernel code: `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, `in_proj_ba`, MTP prefix, PP skip rules, tied embedding rules.
4. Add quantization only after base loading is proven. Check exporter-specific exclusions before enabling shared-expert fusion.
5. Add MTP/spec-v2 after base quantized decoding works. Confirm `SGLANG_ENABLE_SPEC_V2`, `extra_buffer`, radix-cache behavior, and acceptance length.
6. Add PP/EP/EPLB/PD only one dimension at a time. For hybrid Qwen3.5, validate Mamba state as well as attention KV.
7. Profile GDN projection and shared expert kernels only after logits and acceptance match the baseline.

## Hardware Lanes

- CUDA/H200/B200/GB300: check `trtllm_mha`, FlashInfer all-reduce fusion, `modelopt_fp4`, FP8 KV caveats, and the B200/GB300 registered tests.
- AMD MI300/MI325/MI350/MI355: check `SGLANG_USE_AITER=1`, `--attention-backend triton`, ROCm radix-cache behavior, and whether shared-expert fusion is allowed for the target quantization.
- NPU: check ModelSlim names and packed-module mapping after GDN fusion.
- CPU/offload: keep view/tied-parameter aliasing in mind; open PR `#23474` is relevant but not merged history.

## Validation Commands

```bash
python -m pytest test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py
python -m pytest test/registered/4-gpu-models/test_qwen35_fp4_triton.py
python -m pytest test/registered/8-gpu-models/test_qwen35.py
python -m pytest test/registered/gb300/test_qwen35_fp8.py
python -m pytest test/registered/gb300/test_qwen35_nvfp4.py
```

## Documentation Rules

- Do not add a PR to the playbook or history unless the diff/source has been opened.
- Each PR entry must include motivation, implementation idea, a real code snippet, and validation meaning.
- Keep merged PR history separate from open PR candidates.
- Cookbook updates must mention the exact model, quantization, parser, backend, Mamba scheduler, KV dtype, and hardware lane they are meant for.
