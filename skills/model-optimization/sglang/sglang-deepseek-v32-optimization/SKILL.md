---
name: sglang-deepseek-v32-optimization
description: PR-backed and current-main optimization manual for DeepSeek V3.2, V3.2-Exp, V3.2-Speciale, NVFP4, and MXFP4 in SGLang. Use when Codex needs to recover, extend, or audit DSA/NSA sparse attention, NSA indexer, FP8/BF16/FP4 KV cache, context parallel, MTP, IndexCache, DSML tool calling, V3.2 docs/tests, AMD/NPU/Blackwell backends, or open NSA/DSA PRs.
---

# SGLang DeepSeek V3.2 Optimization

## Overview

This skill covers the DeepSeek V3.2 support and optimization ladder active in SGLang main. V3.2 shares the DeepSeek V3/R1 model backbone, but it is a separate optimization problem because it activates DeepSeek Sparse Attention, called DSA in docs and NSA in SGLang code.

Current-main snapshot:

- SGLang `origin/main`: `929e00eea` on `2026-04-21`
- sgl-cookbook `origin/main`: `8ec4d03` on `2026-04-21`
- V3.2 runtime entry: `DeepseekV32ForCausalLM` in `python/sglang/srt/models/deepseek_v2.py`
- NSA backend: `python/sglang/srt/layers/attention/nsa_backend.py`
- NSA indexer: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
- V3.2 tool parser: `python/sglang/srt/function_call/deepseekv32_detector.py`

The historical evidence lives in:

- [references/pr-history.md](references/pr-history.md): chronological PR evidence and code-level notes
- [references/playbook.md](references/playbook.md): investigation order, symptom mapping, validation commands

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Record the exact serving shape first:

- model: V3.2-Exp, V3.2, V3.2-Speciale, V3.2-NVFP4, or V3.2-MXFP4
- whether `is_deepseek_nsa(config)` is true
- `--attention-backend`, `--nsa-prefill-backend`, `--nsa-decode-backend`
- KV cache dtype: `auto`, `bfloat16`, `fp8_e4m3`, or experimental FP4 tracks
- TP / DP / EP / PP / PD topology
- `--enable-dp-attention`
- `--enable-nsa-prefill-context-parallel`
- `--nsa-prefill-cp-mode`: `round-robin-split` or `in-seq-split`
- MTP enabled or not
- IndexCache knobs: `index_topk_freq`, `index_topk_pattern`
- tool parser: V3.2-Exp may use `deepseekv31` in the cookbook path, standard V3.2 uses `deepseekv32`
- reasoning parser: `--reasoning-parser deepseek-v3`
- hardware: H200, B200/GB200/GB300, AMD MI300/MI355, NPU, or another backend

## Core Principle

Do not treat V3.2 as ordinary DeepSeek V3.

- V3.2 turns on DSA/NSA through `is_deepseek_nsa(config)`.
- The attention hot path is split between the indexer, top-k transform, sparse MLA backend, and KV-cache quant/dequant.
- Server defaults are model-specific: attention backend becomes `nsa`, KV cache dtype defaults differ by architecture, and NSA prefill/decode backends are auto-selected.
- Context parallel is experimental and has strict mode-specific constraints.
- MTP spans the NextN layer, NSA metadata, target_verify, draft_extend, CP positions, and speculative overlap.
- V3.2 parser behavior is DSML for standard V3.2, while V3.2-Exp docs still point at the V3.1-style parser path.

The optimization order matters:

1. confirm DSA detection and server defaults
2. confirm KV cache dtype and NSA backend pair
3. validate indexer top-k generation and transform
4. validate MTP, CP, PP, or DP attention only after base DSA is correct
5. then tune backend-specific kernels for Blackwell, Hopper, AMD, or NPU
6. add model-backed tests for any IndexCache, MTP, CP, or backend change

## Main Runtime Surfaces

Start from these files before changing behavior:

- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/speculative/eagle_worker_v2.py`
- `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`
- `python/sglang/srt/layers/attention/nsa_backend.py`
- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
- `python/sglang/srt/layers/attention/nsa/utils.py`
- `python/sglang/srt/layers/attention/nsa/transform_index.py`
- `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`
- `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`
- `python/sglang/srt/layers/communicator_nsa_cp.py`
- `python/sglang/srt/function_call/deepseekv32_detector.py`
- `examples/chat_template/tool_chat_template_deepseekv32.jinja`

## Open PRs to Track

Check these before declaring a V3.2 gap:

- [#11191](https://github.com/sgl-project/sglang/pull/11191): sparse attention and CPU/GPU KV scheduling for GQA/DSA, open.
- [#12820](https://github.com/sgl-project/sglang/pull/12820): TP-SP for Qwen and DeepSeek V2/V3/V3.2, open.
- [#16148](https://github.com/sgl-project/sglang/pull/16148): V3.2 W4AFP8 MTP with FP8 draft model, open.
- [#17185](https://github.com/sgl-project/sglang/pull/17185): TP `o_proj` linear in context-parallel NSA, open.
- [#17761](https://github.com/sgl-project/sglang/pull/17761): missing Assistant token after V3.1/V3.2 tool output, open.
- [#18167](https://github.com/sgl-project/sglang/pull/18167): DCP support for V3.2, open.
- [#18275](https://github.com/sgl-project/sglang/pull/18275): NPU all-gather after qlora for V3.2, open.
- [#18733](https://github.com/sgl-project/sglang/pull/18733): V3.2 PD disaggregation test, open.
- [#19211](https://github.com/sgl-project/sglang/pull/19211): extract V3.2/NSA logic into `DeepseekV32Mixin`, open.
- [#19299](https://github.com/sgl-project/sglang/pull/19299): O(1) expert weight matching in DeepSeek weight loader, open.
- [#19609](https://github.com/sgl-project/sglang/pull/19609): TP indexer weight in NSA attention, open.
- [#19975](https://github.com/sgl-project/sglang/pull/19975): AMD context parallel for V3.2, open.
- [#20360](https://github.com/sgl-project/sglang/pull/20360): AMD CP round-robin split garbage output, open.
- [#20531](https://github.com/sgl-project/sglang/pull/20531): NSA indexer ragged gather mismatch in CP round-robin split, open.
- [#20809](https://github.com/sgl-project/sglang/pull/20809): add `DeepseekV32ForCausalLM` to MTP draft mapping, open.
- [#20880](https://github.com/sgl-project/sglang/pull/20880): reject HiCache L3 for NSA models, open.
- [#21179](https://github.com/sgl-project/sglang/pull/21179): preserve V3.2 tool-call markers in reasoning parsing, open.
- [#21194](https://github.com/sgl-project/sglang/pull/21194): AMD `PPMissingLayer` fix in DeepSeek AITER gfx95 path, open.
- [#21506](https://github.com/sgl-project/sglang/pull/21506): V3.2 NPU torch compile, open.
- [#21529](https://github.com/sgl-project/sglang/pull/21529): ROCm MXFP4 / Quark W4A4 support for DeepSeek architecture, open.
- [#21530](https://github.com/sgl-project/sglang/pull/21530): ROCm fused MLA decode RoPE fix for DeepSeek-variant models, open.
- [#21546](https://github.com/sgl-project/sglang/pull/21546): catch malformed JSON in V3.2 partial function-call parsing, open.
- [#21889](https://github.com/sgl-project/sglang/pull/21889): AMD FP4 KV cache quantization for NSA TileLang, open.
- [#22268](https://github.com/sgl-project/sglang/pull/22268): DeepSeek MLA LoRA adapter bypass in `prepare_qkv_latent`, open.
- [#22473](https://github.com/sgl-project/sglang/pull/22473): dense MLA decode fallback for short sequences, open.
- [#22774](https://github.com/sgl-project/sglang/pull/22774): MUSA backend support for DeepSeek V2/V3/R1-class layers, merged at `2026-04-24T01:59:51Z`.
- [#22851](https://github.com/sgl-project/sglang/pull/22851): `--nsa-topk-backend` and FlashInfer/PyTorch top-k, open.
- [#22865](https://github.com/sgl-project/sglang/pull/22865): sparsity framework extension for non-NSA sparse algorithms, open.
- [#14332](https://github.com/sgl-project/sglang/pull/14332): V3.2 tool-call parsing without DSML tag, open.
- [#14524](https://github.com/sgl-project/sglang/pull/14524): NSA backend test suite, open.
- [#15322](https://github.com/sgl-project/sglang/pull/15322): V3.2 `o_proj` TP support, open.
- [#18094](https://github.com/sgl-project/sglang/pull/18094): V3.2 piecewise CUDA graph, open and related to [#23351](https://github.com/sgl-project/sglang/pull/23351).
- [#18542](https://github.com/sgl-project/sglang/pull/18542): EAGLE3 plus NSA CP aux-hidden-state index bug, open.
- [#19987](https://github.com/sgl-project/sglang/pull/19987): AMD FP8 KV cache for TileLang NSA backend, open.
- [#20534](https://github.com/sgl-project/sglang/pull/20534): transfer FP8 K/K-scale for CP indexer prefill gather, open.
- [#21623](https://github.com/sgl-project/sglang/pull/21623): unit tests for `encoding_dsv32.py`, open.
- [#22792](https://github.com/sgl-project/sglang/pull/22792): AITER `indexer_k_quant_and_cache`, open.
- [#23268](https://github.com/sgl-project/sglang/pull/23268): NPU accuracy fix for NSA CP plus prefix cache, open.
- [#22938](https://github.com/sgl-project/sglang/pull/22938): restore DeepSeek MLA MI300X paths after the MLA refactor, open.
- [#23195](https://github.com/sgl-project/sglang/pull/23195): guard `.weight` access in DeepSeek MLA for AWQ/compressed-tensors, open.
- [#23241](https://github.com/sgl-project/sglang/pull/23241): 3FS backend for DSA/mamba, open.
- [#23257](https://github.com/sgl-project/sglang/pull/23257): CuteDSL EP plus DP-attention double-reduce fix in `DeepseekV2MoE`, open.
- [#23336](https://github.com/sgl-project/sglang/pull/23336): adaptive speculative-num-steps support for spec v2 EAGLE workers, open.
- [#23351](https://github.com/sgl-project/sglang/pull/23351): piecewise CUDA graph with NSA, open.

## Additional PR Coverage

Additional all-state PR coverage includes V3.2 bugfixes, closed experiments, tool-parser updates, and platform-specific backend work:

- Early bring-up polish: [#11063](https://github.com/sgl-project/sglang/pull/11063), [#11194](https://github.com/sgl-project/sglang/pull/11194), [#11308](https://github.com/sgl-project/sglang/pull/11308), [#11309](https://github.com/sgl-project/sglang/pull/11309), [#11450](https://github.com/sgl-project/sglang/pull/11450), [#11557](https://github.com/sgl-project/sglang/pull/11557), [#11565](https://github.com/sgl-project/sglang/pull/11565), [#11682](https://github.com/sgl-project/sglang/pull/11682), [#11815](https://github.com/sgl-project/sglang/pull/11815), and [#11835](https://github.com/sgl-project/sglang/pull/11835).
- Short-sequence MHA / Indexer fixes: [#11892](https://github.com/sgl-project/sglang/pull/11892), [#12094](https://github.com/sgl-project/sglang/pull/12094), [#12582](https://github.com/sgl-project/sglang/pull/12582), [#12583](https://github.com/sgl-project/sglang/pull/12583), [#12645](https://github.com/sgl-project/sglang/pull/12645), [#12788](https://github.com/sgl-project/sglang/pull/12788), [#12816](https://github.com/sgl-project/sglang/pull/12816), [#12964](https://github.com/sgl-project/sglang/pull/12964), [#13022](https://github.com/sgl-project/sglang/pull/13022), [#13459](https://github.com/sgl-project/sglang/pull/13459), and [#13544](https://github.com/sgl-project/sglang/pull/13544).
- DSML/tool/parser path: [#14304](https://github.com/sgl-project/sglang/pull/14304), [#14307](https://github.com/sgl-project/sglang/pull/14307), [#14353](https://github.com/sgl-project/sglang/pull/14353), [#14573](https://github.com/sgl-project/sglang/pull/14573), [#14750](https://github.com/sgl-project/sglang/pull/14750), [#15064](https://github.com/sgl-project/sglang/pull/15064), [#15278](https://github.com/sgl-project/sglang/pull/15278), [#16091](https://github.com/sgl-project/sglang/pull/16091), [#18126](https://github.com/sgl-project/sglang/pull/18126), [#18174](https://github.com/sgl-project/sglang/pull/18174), and [#17951](https://github.com/sgl-project/sglang/pull/17951).
- NSA backend / metadata / sparse-cache work: [#14781](https://github.com/sgl-project/sglang/pull/14781), [#14901](https://github.com/sgl-project/sglang/pull/14901), [#15040](https://github.com/sgl-project/sglang/pull/15040), [#15086](https://github.com/sgl-project/sglang/pull/15086), [#15242](https://github.com/sgl-project/sglang/pull/15242), [#15429](https://github.com/sgl-project/sglang/pull/15429), [#16520](https://github.com/sgl-project/sglang/pull/16520), [#16758](https://github.com/sgl-project/sglang/pull/16758), [#16841](https://github.com/sgl-project/sglang/pull/16841), [#17205](https://github.com/sgl-project/sglang/pull/17205), [#17554](https://github.com/sgl-project/sglang/pull/17554), and [#18319](https://github.com/sgl-project/sglang/pull/18319).
- HiSparse/HiCache and platform fixes: [#14741](https://github.com/sgl-project/sglang/pull/14741), [#17409](https://github.com/sgl-project/sglang/pull/17409), [#17518](https://github.com/sgl-project/sglang/pull/17518), [#17523](https://github.com/sgl-project/sglang/pull/17523), [#17633](https://github.com/sgl-project/sglang/pull/17633), [#18297](https://github.com/sgl-project/sglang/pull/18297), [#18526](https://github.com/sgl-project/sglang/pull/18526), [#20343](https://github.com/sgl-project/sglang/pull/20343), [#21932](https://github.com/sgl-project/sglang/pull/21932), and [#22238](https://github.com/sgl-project/sglang/pull/22238).
- Closed or superseded experiments to cite as history, not current support: [#11109](https://github.com/sgl-project/sglang/pull/11109), [#11596](https://github.com/sgl-project/sglang/pull/11596), [#11761](https://github.com/sgl-project/sglang/pull/11761), [#12017](https://github.com/sgl-project/sglang/pull/12017), [#12052](https://github.com/sgl-project/sglang/pull/12052), [#13531](https://github.com/sgl-project/sglang/pull/13531), [#13546](https://github.com/sgl-project/sglang/pull/13546), [#14619](https://github.com/sgl-project/sglang/pull/14619), [#14904](https://github.com/sgl-project/sglang/pull/14904), [#15051](https://github.com/sgl-project/sglang/pull/15051), [#15217](https://github.com/sgl-project/sglang/pull/15217), [#15310](https://github.com/sgl-project/sglang/pull/15310), [#15807](https://github.com/sgl-project/sglang/pull/15807), [#16079](https://github.com/sgl-project/sglang/pull/16079), [#16881](https://github.com/sgl-project/sglang/pull/16881), [#17024](https://github.com/sgl-project/sglang/pull/17024), [#17199](https://github.com/sgl-project/sglang/pull/17199), [#17310](https://github.com/sgl-project/sglang/pull/17310), and [#17647](https://github.com/sgl-project/sglang/pull/17647).
- Round-2 runtime additions: [#21249](https://github.com/sgl-project/sglang/pull/21249) adds all-reduce fusion with context parallel, [#22003](https://github.com/sgl-project/sglang/pull/22003) relaxes `moe_dp_size == 1` with different `attention_cp_size` values, [#21599](https://github.com/sgl-project/sglang/pull/21599) adds adaptive EAGLE top-k=1 draft steps, [#22128](https://github.com/sgl-project/sglang/pull/22128) allows PCG with speculative decoding, [#23219](https://github.com/sgl-project/sglang/pull/23219) touches shared DSA/NextN infrastructure through `deepseek_nextn.py`, [#22950](https://github.com/sgl-project/sglang/pull/22950) is the closed predecessor for reasoning radix-cache stripping, [#23315](https://github.com/sgl-project/sglang/pull/23315) is the merged opt-in thinking-token strip from radix cache, and [#23336](https://github.com/sgl-project/sglang/pull/23336) is the open spec-v2 adaptive-spec follow-up.

## Evolution Path

### Stage V32-0: Bring up DSA/NSA as a separate DeepSeek class

Key PR:

- [#11061](https://github.com/sgl-project/sglang/pull/11061)

Success check:

- `DeepseekV32ForCausalLM` exists
- `is_deepseek_nsa(config)` is true
- `server_args.py` selects `attention_backend = "nsa"`
- `NativeSparseAttnBackend` and `Indexer` are active

### Stage V32-1: Server defaults, KV cache dtype, and backend pair

V3.2 has model-specific defaults:

- DSA KV cache defaults to `fp8_e4m3` on SM100 and `bfloat16` otherwise
- only `bfloat16` and `fp8_e4m3` are mainline DSA KV cache dtypes
- ROCm defaults to TileLang NSA backends
- Blackwell defaults now prefer TRTLLM NSA kernels
- Hopper often uses `flashmla_sparse`, `flashmla_kv`, or `fa3`

Key PRs:

- [#11936](https://github.com/sgl-project/sglang/pull/11936)
- [#18389](https://github.com/sgl-project/sglang/pull/18389)
- [#18931](https://github.com/sgl-project/sglang/pull/18931)
- [#21783](https://github.com/sgl-project/sglang/pull/21783)
- [#21914](https://github.com/sgl-project/sglang/pull/21914)

### Stage V32-2: Indexer correctness and performance

The NSA indexer computes sparse indices through q/k projection, weights projection, top-k, transforms, and optional KV-cache store.

Key PRs:

- [#12044](https://github.com/sgl-project/sglang/pull/12044)
- [#13812](https://github.com/sgl-project/sglang/pull/13812)
- [#16637](https://github.com/sgl-project/sglang/pull/16637)
- [#17688](https://github.com/sgl-project/sglang/pull/17688)
- [#19041](https://github.com/sgl-project/sglang/pull/19041)
- [#19148](https://github.com/sgl-project/sglang/pull/19148)
- [#19319](https://github.com/sgl-project/sglang/pull/19319)
- [#22232](https://github.com/sgl-project/sglang/pull/22232)
- [#22424](https://github.com/sgl-project/sglang/pull/22424)
- [#22850](https://github.com/sgl-project/sglang/pull/22850)

Success check:

- `weights_proj` avoids FP32 precision loss
- K/S buffers use fused kernels where available
- FP8 KV cache store is fused or padded correctly for the selected backend
- AMD and NPU have separate indexer paths where needed

### Stage V32-3: Context parallel, PP, and DP attention

Context parallel for NSA is powerful but constrained.

Key PRs:

- [#12065](https://github.com/sgl-project/sglang/pull/12065)
- [#13959](https://github.com/sgl-project/sglang/pull/13959)
- [#16119](https://github.com/sgl-project/sglang/pull/16119)
- [#16156](https://github.com/sgl-project/sglang/pull/16156)
- [#16305](https://github.com/sgl-project/sglang/pull/16305)
- [#16380](https://github.com/sgl-project/sglang/pull/16380)
- [#18613](https://github.com/sgl-project/sglang/pull/18613)
- [#20438](https://github.com/sgl-project/sglang/pull/20438)
- [#21192](https://github.com/sgl-project/sglang/pull/21192)
- [#22914](https://github.com/sgl-project/sglang/pull/22914)

Success check:

- `round-robin-split` is the current default CP token split method
- `in-seq-split` requires DeepEP and `ep_size == tp_size`
- CP in PD decode mode is asserted away
- CP positions match EAGLE NextN
- key all-gather can overlap query computation

### Stage V32-4: MTP and speculative decoding

V3.2 MTP must cooperate with NSA metadata, target verify, draft extend, and context parallel.

Key PRs:

- [#11652](https://github.com/sgl-project/sglang/pull/11652)
- [#15088](https://github.com/sgl-project/sglang/pull/15088)
- [#15307](https://github.com/sgl-project/sglang/pull/15307)
- [#16961](https://github.com/sgl-project/sglang/pull/16961)
- [#17662](https://github.com/sgl-project/sglang/pull/17662)
- [#19016](https://github.com/sgl-project/sglang/pull/19016)
- [#19062](https://github.com/sgl-project/sglang/pull/19062)
- [#19367](https://github.com/sgl-project/sglang/pull/19367)
- [#19536](https://github.com/sgl-project/sglang/pull/19536)
- [#20492](https://github.com/sgl-project/sglang/pull/20492)

### Stage V32-5: Quantized checkpoints and platform lanes

Separate the backend tracks:

- NVFP4 Blackwell: [#17657](https://github.com/sgl-project/sglang/pull/17657), [#18389](https://github.com/sgl-project/sglang/pull/18389), [#20086](https://github.com/sgl-project/sglang/pull/20086)
- AMD MXFP4/TileLang/FP8 KV: [#17783](https://github.com/sgl-project/sglang/pull/17783), [#19945](https://github.com/sgl-project/sglang/pull/19945), [#20840](https://github.com/sgl-project/sglang/pull/20840), [#21511](https://github.com/sgl-project/sglang/pull/21511), [#22258](https://github.com/sgl-project/sglang/pull/22258), [#22850](https://github.com/sgl-project/sglang/pull/22850)
- NPU: [#14541](https://github.com/sgl-project/sglang/pull/14541), [#14572](https://github.com/sgl-project/sglang/pull/14572), [#15381](https://github.com/sgl-project/sglang/pull/15381), [#16990](https://github.com/sgl-project/sglang/pull/16990), [#17007](https://github.com/sgl-project/sglang/pull/17007), [#21468](https://github.com/sgl-project/sglang/pull/21468)
- HiSparse/HiCache: [#21259](https://github.com/sgl-project/sglang/pull/21259), [#22065](https://github.com/sgl-project/sglang/pull/22065), [#22425](https://github.com/sgl-project/sglang/pull/22425)

### Stage V32-6: IndexCache

IndexCache reuses NSA top-k indices across layers.

Key PR:

- [#21405](https://github.com/sgl-project/sglang/pull/21405)

Success check:

- `skip_topk` and `next_skip_topk` are set per layer
- `index_topk_freq` and `index_topk_pattern` override behavior correctly
- `prev_topk_indices` is carried through layers
- `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` remains accurate

### Stage V32-7: DSML tool calling and reasoning interaction

Standard V3.2 uses DSML:

```text
<｜DSML｜function_calls><｜DSML｜invoke name="tool">...</｜DSML｜invoke></｜DSML｜function_calls>
```

The detector supports XML parameter tags and direct JSON. Track open parser bugs:

- [#21179](https://github.com/sgl-project/sglang/pull/21179): reasoning parser should preserve V3.2 tool-call markers.
- [#21546](https://github.com/sgl-project/sglang/pull/21546): catch malformed JSON while parsing partial function calls.

## Validation Surface

Use the narrowest lane that matches the change:

- V3.2 base/MTP/DP/TP/tool-calling: `test/registered/8-gpu-models/test_deepseek_v32.py`
- NSA backend pair: `test_deepseek_v32_nsa_backends` inside that file
- IndexCache: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`
- chat template argument types: `test/manual/test_deepseek_chat_templates.py`
- CP and DeepEP-specific changes: use the dedicated CP/DeepEP suites referenced by the PR
- AMD changes: MI300/MI355 registered lanes
- NPU changes: Ascend/NPU model deployment and backend tests
