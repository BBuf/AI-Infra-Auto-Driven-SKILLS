# DeepSeek V3/R1 Playbook

Use this playbook when a V3/R1 optimization or regression report is ambiguous. The goal is to map the symptom to the right layer before changing code.

## 1. Identify The Model Shape

Collect these first:

- `hf_config.architectures`
- `quantization_config`
- `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, `n_group`, `topk_group`
- `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`
- TP / DP / EP / PP / PD
- `--enable-dp-attention`
- `--moe-a2a-backend`, `--deepep-mode`, `--moe-runner-backend`
- `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend`
- `--speculative-algorithm`, `--speculative-*`, `SGLANG_ENABLE_SPEC_V2`
- `--reasoning-parser`, `--tool-call-parser`, chat template

If `is_deepseek_nsa(config)` is true, switch to `sglang-deepseek-v32-optimization`.
If the report is only about DeepSeek V3.1 tool/thinking behavior, switch to `sglang-deepseek-v31-optimization`.

## 2. Loader Failures

Check in this order:

1. `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
2. `python/sglang/srt/layers/quantization/w4afp8.py`
3. `python/sglang/srt/layers/quantization/modelopt_quant.py`
4. `python/sglang/srt/layers/quantization/quark/`
5. `python/sglang/srt/models/deepseek_nextn.py` when MTP is enabled

Common clues:

- missing `kv_b_proj` scale means the MLA post-load path did not remap `k_scale` / `v_scale`
- `.weight` access on quantized layers may be an AWQ/compressed-tensors bug; see open [#23195](https://github.com/sgl-project/sglang/pull/23195)
- R1 MXFP4 draft/NextN can use `model.layers.61*` naming; check `deepseek_nextn.py`
- W4AFP8 shared and routed experts may use different quant methods, which disables shared expert fusion

## 3. MLA Or Attention Regressions

Start with:

- `server_args.py` model-specific adjustments
- `attention_backend_handler.py`
- `attention_forward_methods/forward_mla.py`
- `attention_forward_methods/forward_mha.py`
- `deepseek_weight_loader.py` post-load `kv_b_proj` handling

Useful distinctions:

- SM100 defaults to `trtllm_mla` for V3/R1 when no MLA backend is set.
- PCG and deterministic inference can force a different MLA/MHA method.
- ROCm fused MLA decode RoPE is still tracked by open [#21530](https://github.com/sgl-project/sglang/pull/21530).
- MI300X path restoration after the MLA refactor is still tracked by open [#22938](https://github.com/sgl-project/sglang/pull/22938).

## 4. H200 Performance Ladder

If the report is about single-node H200 DeepSeek V3/R1 throughput, compare against the local H200 note before changing code:

- FP8 Block GEMM / DeepGEMM: check `fp8_kernel.py`, `deep_gemm.py`, `compile_deep_gemm.py`, [#5432](https://github.com/sgl-project/sglang/pull/5432), [#5473](https://github.com/sgl-project/sglang/pull/5473), [#5549](https://github.com/sgl-project/sglang/pull/5549), and [#5628](https://github.com/sgl-project/sglang/pull/5628). `#5432` is an explored BMM path, not automatically the default.
- Fused MoE preprocessing: check `moe_fused_gate.cu`, `moe_align_kernel.cu`, `per_token_group_quant_8bit`, [#4530](https://github.com/sgl-project/sglang/pull/4530), [#5086](https://github.com/sgl-project/sglang/pull/5086), [#5716](https://github.com/sgl-project/sglang/pull/5716), [#5740](https://github.com/sgl-project/sglang/pull/5740), and routed-scaling fusion in [#6220](https://github.com/sgl-project/sglang/pull/6220) / [#6970](https://github.com/sgl-project/sglang/pull/6970).
- MLA backend hot path: check FlashMLA [#4514](https://github.com/sgl-project/sglang/pull/4514), FA3 MLA [#4831](https://github.com/sgl-project/sglang/pull/4831) / [#5210](https://github.com/sgl-project/sglang/pull/5210), Cutlass MLA [#5390](https://github.com/sgl-project/sglang/pull/5390), and docs [#6034](https://github.com/sgl-project/sglang/pull/6034).
- MLA model-file kernels: check chunked MHA prefill [#5113](https://github.com/sgl-project/sglang/pull/5113), merge-state kernel [#5381](https://github.com/sgl-project/sglang/pull/5381), DeepSeek CUDA RoPE [#5385](https://github.com/sgl-project/sglang/pull/5385), removed `forward_absorb` copy [#5578](https://github.com/sgl-project/sglang/pull/5578), fused `q_a_proj` / `kv_a_proj_with_mqa` [#5619](https://github.com/sgl-project/sglang/pull/5619), fused MLA KV-cache writes [#5748](https://github.com/sgl-project/sglang/pull/5748), and q/k norm overlap [#5977](https://github.com/sgl-project/sglang/pull/5977).
- Closed or exploratory paths: hybrid attention [#6151](https://github.com/sgl-project/sglang/pull/6151) closed, and DeepGEMM BMM [#5432](https://github.com/sgl-project/sglang/pull/5432) must be checked against current defaults before being described as shipped throughput behavior.

## 5. MoE Routing And Shared Experts

Read `DeepseekV2MoE.__init__` and `DeepseekV2ForCausalLM.determine_num_fused_shared_experts()`.

Check:

- whether `num_fused_shared_experts` is `0` or `1`
- whether DeepEP is active
- whether `--enforce-shared-experts-fusion` was set
- whether TBO/SBO disables fusion
- whether W4AFP8 disables fusion because routed and shared experts have different quantization
- whether `TopKOutputFormat.STANDARD` is being forced for unquantized MTP layers

If the router is hot:

- `fused_topk_deepseek` is the current preferred optimized path when its constraints are met
- `dsv3_router_gemm` still has an open JIT migration track in [#21531](https://github.com/sgl-project/sglang/pull/21531)
- AMD can select `aiter_dsv3_router_gemm` when expert count is at most `256`

## 6. R1 W4AFP8 / W4A8

Do not debug this as ordinary FP8.

Checklist:

- `W4AFp8Config` is selected from `hf_quant_config.json` or quantization config
- `cutlass_w4a8_moe.py` has the active runner for the selected mode
- `ep_moe/layer.py` maps experts correctly under EP
- `token_dispatcher/deepep.py` emits the expected normal or low-latency dispatch format
- if low-latency DeepEP is enabled, account for [#14162](https://github.com/sgl-project/sglang/pull/14162), [#21719](https://github.com/sgl-project/sglang/pull/21719), and [#22316](https://github.com/sgl-project/sglang/pull/22316)

## 7. MTP / NextN

If `--speculative-algorithm EAGLE` is set:

- inspect `deepseek_nextn.py`
- verify target and draft quant configs separately
- check the MTP layer id and naming
- check whether the draft MoE backend is overridden by server args
- query `/server_info` for `avg_spec_accept_length`

Expected H200 V3 MTP registered behavior:

- GSM8K score above `0.935`
- average spec accept length above `2.8`
- batch-size-1 speed above the non-MTP lane

## 8. Parser Issues

For V3/R1:

- V3 tool calling uses `--tool-call-parser deepseekv3`
- V3 thinking-style output uses `--reasoning-parser deepseek-v3`
- R1 reasoning uses `--reasoning-parser deepseek-r1`
- R1 can produce reasoning without a starting `<think>` tag; `DeepSeekR1Detector` forces reasoning until `</think>`

If the parser issue is specifically V3.1 or V3.2, switch skills.

## 9. Validation Order

Pick the smallest lane that covers the risk:

1. loader-only: import or unit-level weight-loading test if available
2. parser-only: function-call or reasoning parser unit tests
3. MLA/backend: `test/registered/mla/test_mla_deepseek_v3.py`
4. V3 base: `test/registered/8-gpu-models/test_deepseek_v3_basic.py`
5. V3 MTP: `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`
6. W4A8/W4AFP8: `test/registered/quant/test_w4a8_deepseek_v3.py`
7. R1 backend: `test/registered/backends/test_deepseek_r1_fp8_trtllm_backend.py`
8. AMD/R1 MXFP4: `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`
9. LoRA: `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`

## 10. Common False Conclusions

- "DeepSeek V3 official weights need `--quantization fp8`." They do not; the official V3 checkpoint is already FP8.
- "DeepEP automatically means shared experts are fused." Current main requires `--enforce-shared-experts-fusion` under DeepEP.
- "R1 W4AFP8 equals FP8." It has its own quant config, Cutlass W4A8 MoE kernels, and DeepEP paths.
- "MTP uses the same quant config as the target model." The NextN layer may be BF16 or have special quant handling.
- "A PR title mentioning support means current main still contains that path." Check reverts and relands, especially [#14162](https://github.com/sgl-project/sglang/pull/14162), [#21719](https://github.com/sgl-project/sglang/pull/21719), and [#22316](https://github.com/sgl-project/sglang/pull/22316).
