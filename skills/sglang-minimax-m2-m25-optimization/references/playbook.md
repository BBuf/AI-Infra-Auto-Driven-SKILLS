# MiniMax M2/M2.5 Optimization Playbook



## Fast Mapping

| Symptom | Check first | Historical precedent | Likely fix direction |
| --- | --- | --- | --- |
| Accuracy or stability regresses around QK norm | `MiniMaxM2RMSNormTP` in `minimax_m2.py` | [#12186](https://github.com/sgl-project/sglang/pull/12186), [#14416](https://github.com/sgl-project/sglang/pull/14416), [#16483](https://github.com/sgl-project/sglang/pull/16483) | Start with precision and aligned all-reduce, then consider newer fused TP QK norm work |
| TP decode is QK-norm bound | `MiniMaxM2RMSNormTP`, `forward_qk`, JIT qknorm path | [#14416](https://github.com/sgl-project/sglang/pull/14416), [#16483](https://github.com/sgl-project/sglang/pull/16483), [#20673](https://github.com/sgl-project/sglang/pull/20673) | Prefer the specialized MiniMax QK norm path before inventing a new kernel |
| Spec decode or Eagle3 hooks fail | `set_eagle3_layers_to_capture`, `get_embed_and_head` | [#12798](https://github.com/sgl-project/sglang/pull/12798), [#13297](https://github.com/sgl-project/sglang/pull/13297) | Restore the MiniMax capture surface instead of patching speculative code around it |
| MiniMax MoE with DeepEP fails or uses the wrong forward path | `forward_deepep`, `ExpertLocationDispatchInfo`, launch dtype | [#13892](https://github.com/sgl-project/sglang/pull/13892), [#19468](https://github.com/sgl-project/sglang/pull/19468) | Fix the MiniMax MoE contract first; on M2.5 also validate DeepEP hidden-size and bf16 requirements |
| Router-side decode spends too much time in generic top-k work | `topk.py` and the MiniMax gate path | [#14047](https://github.com/sgl-project/sglang/pull/14047) | Use the MiniMax sigmoid top-k fast path before changing generic MoE code |
| Piecewise CUDA graph or PP launch breaks | `pp_proxy_tensors`, layer-range-aware loading, graph contexts | [#18217](https://github.com/sgl-project/sglang/pull/18217), [#19577](https://github.com/sgl-project/sglang/pull/19577), [#18310](https://github.com/sgl-project/sglang/pull/18310) | Rebuild the graph or PP contract first; do not start with kernels |
| M2.5 checkpoint load fails on packed modules or KV scales | `packed_modules_mapping`, `maybe_remap_kv_scale_name`, ModelSlim hooks | [#19995](https://github.com/sgl-project/sglang/pull/19995), [#20870](https://github.com/sgl-project/sglang/pull/20870), [#20905](https://github.com/sgl-project/sglang/pull/20905) | Preserve the mainline loader contract before adding new quant features |
| AWQ or fused-expert M2.5 weights fail to load | `load_weights(...)` expert mapping order | [#20031](https://github.com/sgl-project/sglang/pull/20031) | Try fused `w13` mapping before older `w1/w2/w3` assumptions |
| M2.5 DP-attention crashes, mis-shards, or scales poorly | attention TP group plumbing and post-MoE communication | [#17826](https://github.com/sgl-project/sglang/pull/17826), [#20067](https://github.com/sgl-project/sglang/pull/20067), [#20489](https://github.com/sgl-project/sglang/pull/20489), [#20975](https://github.com/sgl-project/sglang/pull/20975) | Start from the #20067 mainline contract, then inspect still-open DP-attention cleanup PRs for remaining gaps |
| M2.5 at `tp16` produces repeated or garbled outputs | KV-head replica handling in `MiniMaxM2RMSNormTP` | [#20967](https://github.com/sgl-project/sglang/pull/20967) | Make norm weight sharding and reduction replica-aware rather than whole-TP-aware |
| NVFP4 MiniMax-family checkpoint crashes on non-Blackwell GPUs | NVFP4 Marlin fallback path | [#19652](https://github.com/sgl-project/sglang/pull/19652) | Use the generic FP4 Marlin fallback track; the blocker may be outside MiniMax model code |

## Investigation Commands

Commands to run before editing:

```bash
git -C /path/to/sglang log --first-parent --oneline main -- python/sglang/srt/models/minimax_m2.py
git -C /path/to/sglang log --all --oneline --grep='MiniMax'
rg -n "MiniMaxM2RMSNormTP|forward_qk|packed_modules_mapping|get_embed_and_head|layers_to_capture|pp_proxy_tensors|maybe_remap_kv_scale_name" python/sglang/srt/models/minimax_m2.py
rg -n "enable-dp-attention|attn_tp|reduce_scatter|allreduce_fusion|deepep|modelslim|nvfp4" python/sglang/srt python/sglang/jit_kernel
rg -n "MiniMax|minimax" test/registered python/sglang/jit_kernel/tests
```

If the issue looks like a still-open upstream gap rather than a mainline regression, inspect the PR patches or branch code for:

- [#17826](https://github.com/sgl-project/sglang/pull/17826)
- [#19468](https://github.com/sgl-project/sglang/pull/19468)
- [#20031](https://github.com/sgl-project/sglang/pull/20031)
- [#20489](https://github.com/sgl-project/sglang/pull/20489)
- [#20975](https://github.com/sgl-project/sglang/pull/20975)

Mainline PRs to verify in the local `main` tree before porting old branch code:

- [#19652](https://github.com/sgl-project/sglang/pull/19652)
- [#20067](https://github.com/sgl-project/sglang/pull/20067)
- [#20673](https://github.com/sgl-project/sglang/pull/20673)
- [#20967](https://github.com/sgl-project/sglang/pull/20967)

## Workflow

### 1. Classify the runtime shape

Record all of these before editing:

- exact model id
- M2, M2.1, or M2.5
- quant format
- TP / DP / EP / PP sizes
- DP-attention enabled or not
- DeepEP or other MoE communication backend
- piecewise CUDA graph enabled or not
- speculative decoding enabled or not
- exact GPU family and backend


- M2 with TP-only QK norm hotspot
- M2 with DeepEP MoE path
- M2.5 with packed quantized checkpoints
- M2.5 with DP attention plus DEP
- M2.5 with replicated KV heads at high TP

### 2. Start from the narrowest MiniMax-specific hotspot

Prefer this order:

1. loader contract or topology contract
2. model-local MiniMax path
3. active communication strategy
4. kernel code

This matches the PR history. Many MiniMax issues were solved without inventing a brand-new kernel.

### 3. Distinguish `main` from still-open upstream

As of `2026-04-17`, several MiniMax-M2.5 optimization PRs that were previously active are now part of `main`, but not every important gap is closed.

Treat the paths differently:

- If the issue is covered by mainline PRs, patch against current code and validate locally.
- If the issue only appears in still-open upstream PRs, decide explicitly whether you are:
  - porting the PR
  - reimplementing the same idea locally
  - or just documenting the gap

Do not silently describe an open PR as if it were already shipped.

### 4. Reuse MiniMax's actual communication contract

MiniMax-M2.5 scale-out work depends on three separate communication questions:

- which group owns attention partitioning
- whether the MoE output should all-reduce or reduce-scatter
- whether a fused or FP4-aware transport path exists for the active backend

Inference from the still-open and mainline PR history:

- the old "just all-reduce the MoE result" contract is too generic for the best M2.5 DEP path
- the old "model TP group equals attention TP group" assumption is invalid once DP attention is enabled


## Validation Order

### Current mainline MiniMax path

Use the lightest targeted launch first:

```bash
pytest -q test/registered/8-gpu-models/test_minimax_m25.py
pytest -q test/registered/ascend/llm_models/test_ascend_minimax_m2.py
```

If you are working on AMD-specific MiniMax-M2.5 evaluation:

```bash
pytest -q test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py
pytest -q test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py
```

### QK norm changes

For mainline QK norm work:

- validate the normal MiniMax registered launch first
- then compare decode throughput on the exact TP shape that was slow

For the [#20673](https://github.com/sgl-project/sglang/pull/20673) JIT path, also run its focused tests when changing TP QK norm behavior:

```bash
pytest -q python/sglang/jit_kernel/tests/test_tp_qknorm.py
python -m sglang.jit_kernel.benchmark.bench_tp_qknorm
```

### Loader changes

For mainline loader work:

- validate the normal registered MiniMax launch
- confirm the checkpoint no longer needs manual renaming

If you port still-open PR [#20031](https://github.com/sgl-project/sglang/pull/20031), also add or run a focused weight-loading test similar to:

```bash
pytest -q tests/registered/models/test_minimax_m2_weights.py
```

That file is part of the PR, not current `main`, so create or port it when adopting that work.

### DP-attention or DeepEP changes

Do not validate only a TP-only text path.

When debugging DP-attention or DEP:

- use the exact `TP + DP + EP` shape that triggered the issue
- include empty-batch or padded-batch cases
- confirm whether the path should all-reduce, reduce-scatter, or stay fused

### NVFP4 fallback changes

If the real work is the generic NVFP4 fallback from [#19652](https://github.com/sgl-project/sglang/pull/19652), validate with its dedicated fallback tests, not only with a MiniMax server launch.

## Anti-Patterns

- Do not start from generic MoE kernels if the bug is really MiniMax loader or topology plumbing.
- Do not assume the full TP group is the right reduction group for replicated KV heads.
- Do not "fix" M2.5 by bypassing `packed_modules_mapping` or KV-scale remapping.
- Do not validate a DP-attention change only on TP-only traffic.
- Do not claim a still-open upstream MiniMax optimization is already on `main` without checking the local tree.
