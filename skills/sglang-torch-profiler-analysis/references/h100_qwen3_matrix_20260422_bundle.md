# Unified LLM Torch Profiler Triage Bundle

_Generated on 2026-04-22 16:07:47 UTC_

## Contents

- [Qwen/Qwen3-14B](#qwen-qwen3-14b)
- [Qwen/Qwen3-30B-A3B](#qwen-qwen3-30b-a3b)
- [Qwen/Qwen3-8B](#qwen-qwen3-8b)

## Qwen/Qwen3-14B

Model id: `qwen3_14b`

### SGLang

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/analysis_sglang.txt`

#### Triage View
Mode: single-trace
Framework: SGLang
Input traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/sglang_profile_live/1776865830.2463052/triage-trace-1776865830.269116-TP-0-EXTEND.trace.json.gz, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/sglang_profile_live/1776865830.2463052/triage-trace-1776865830.269116-TP-0-DECODE.trace.json.gz
Model: Qwen/Qwen3-14B

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_384x24_64x4_4x1_v_bz_TNT | gemm | 4.99 ms | 44.2% | 40 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| nvjet_tst_64x24_64x16_4x1_v_bz_TNT | gemm | 4.46 ms | 39.5% | 120 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.51 ms | 4.5% | 1 | python/sglang/srt/layers/logits_processor.py:878 _compute_lm_head | aten::mm |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, false, true, true, true, true, true> > > > | gemm | 0.37 ms | 3.3% | 40 | python/sglang/jit_kernel/flash_attention_v3.py:89 flash_attn_with_kvcache | sgl_kernel::fwd |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.21 ms | 1.9% | 80 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| void flash::prepare_varlen_num_blocks_kernel<1, true> | gemm | 0.14 ms | 1.3% | 40 | python/sglang/jit_kernel/flash_attention_v3.py:89 flash_attn_with_kvcache | sgl_kernel::fwd |
| void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))> | activation | 0.12 ms | 1.1% | 40 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 0.12 ms | 1.1% | 40 | python/sglang/srt/layers/attention/flashattention_backend.py:597 forward_extend | aten::copy_ |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_384x8_64x4_4x1_v_bz_TNT | gemm | 24.65 ms | 45.7% | 200 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 16.47 ms | 30.6% | 400 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 5.44 ms | 10.1% | 200 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 2.54 ms | 4.7% | 5 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true> > > > | gemm | 1.35 ms | 2.5% | 200 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o512051201_tensorptrbf16gmemalign128o512051201_tensorptrbf16gmemalign_0 | gemm | 0.99 ms | 1.8% | 400 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))> | activation | 0.60 ms | 1.1% | 200 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |

#### Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No actionable overlap rows. Use mapping/formal two-trace triage for stronger overlap conclusions. | - | - | - | - |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 9.96 ms | 88.3% | nvjet_tst_384x24_64x4_4x1_v_bz_TNT (44.2%)<br>nvjet_tst_64x24_64x16_4x1_v_bz_TNT (39.5%)<br>nvjet_tst_384x8_64x4_2x1_v_bz_TNT (4.5%) | trampoline @ python/sglang/srt/compilation/compile.py:188<br>_compute_lm_head @ python/sglang/srt/layers/logits_processor.py:878<br>forward @ python/sglang/srt/layers/sampler.py:77 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 88.3% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RoPE reshape + KV cache write | Likely | 0.66 ms | 5.9% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, false, true, true, true, true, true> > > > (3.3%)<br>void flash::prepare_varlen_num_blocks_kernel<1, true> (1.3%) | trampoline @ python/sglang/srt/compilation/compile.py:188<br>flash_attn_with_kvcache @ python/sglang/jit_kernel/flash_attention_v3.py:89<br>_set_kv_buffer_impl @ python/sglang/srt/mem_cache/memory_pool.py:89 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 5.9% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| Fused residual add + RMSNorm | Likely | 0.21 ms | 1.9% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (1.9%) | trampoline @ python/sglang/srt/compilation/compile.py:188 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 1.9% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 49.11 ms | 91.1% | nvjet_tst_384x8_64x4_4x1_v_bz_TNT (45.7%)<br>nvjet_tst_64x8_64x16_4x1_v_bz_TNT (30.6%)<br>nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT (10.1%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895<br>forward @ python/sglang/srt/layers/sampler.py:77 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 91.1% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RoPE reshape + KV cache write | Conditional | 0.58 ms | 1.1% | - | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 1.1% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| Fused residual add + RMSNorm | Likely | 0.99 ms | 1.8% | kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o512051201_tensorptrbf16gmemalign128o512051201_tensorptrbf16gmemalign_0 (1.8%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 1.8% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |

### vLLM

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/analysis_vllm.txt`

#### Triage View
Mode: mapping-formal
Framework: vLLM
Mapping traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/vllm_mapping/vllm_profile/rank0.1776866445850274472.pt.trace.json.gz
Formal traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/vllm_formal/vllm_profile/rank0.1776865979877683213.pt.trace.json.gz

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_384x24_64x4_4x1_v_bz_TNT | gemm | 4.92 ms | 44.1% | 40 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x24_64x16_4x1_v_bz_TNT | gemm | 4.50 ms | 40.3% | 120 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.51 ms | 4.6% | 1 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, true, true, true, true, false, true> > > > | gemm | 0.49 ms | 4.4% | 40 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2 | memory | 0.14 ms | 1.2% | 40 | vllm/model_executor/models/qwen2.py:422 forward | cudaGraphLaunch |
| void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > | gemm | 0.12 ms | 1.1% | 40 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_384x8_64x4_4x1_v_bz_TNT | gemm | 53.76 ms | 45.2% | 440 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 35.89 ms | 30.2% | 880 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 11.95 ms | 10.1% | 440 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 5.60 ms | 4.7% | 11 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, false, true> > > > | gemm | 4.26 ms | 3.6% | 440 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2 | memory | 1.44 ms | 1.2% | 440 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |

#### Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P2 | headroom | nvjet_tst_384x8_64x4_4x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 53762.3 us, share 41.4%, excl 100.0% / hid 0.0% | unclear | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 35889.3 us, share 27.6%, excl 100.0% / hid 0.0% | unclear | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 11948.4 us, share 9.2%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 6116.4 us, share 4.7%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_384x24_64x4_4x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 4924.6 us, share 3.8%, excl 100.0% / hid 0.0% | unclear | check deps |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| vLLM-origin Attention + Quantization | Likely | 10.66 ms | 95.4% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, true, true, true, true, false, true> > > > (4.4%)<br>void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > (1.1%)<br>nvjet_tst_384x24_64x4_4x1_v_bz_TNT (44.1%) | flash_attn_varlen_func @ vllm/vllm_flash_attn/flash_attn_interface.py:176<br>get_scheduler_metadata @ vllm/vllm_flash_attn/flash_attn_interface.py:122<br>_prepare_inputs @ vllm/v1/worker/gpu_model_runner.py:1784<br>default_unquantized_gemm @ vllm/model_executor/layers/utils.py:92 | vllm/compilation/passes/fusion/attn_quant_fusion.py<br>vllm/v1/attention/ops/merge_attn_states.py<br>vllm/csrc/attention/merge_attn_states.cu<br>vllm/docs/design/fusions.md | This trace matches a reusable upstream precedent at 95.4% related GPU time. vLLM already combines attention merge and attention-epilogue quantization through reusable compile-time and kernel paths. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| vLLM-origin Attention + Quantization | Likely | 113.95 ms | 95.9% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, false, true> > > > (3.6%)<br>nvjet_tst_384x8_64x4_4x1_v_bz_TNT (45.2%)<br>nvjet_tst_64x8_64x16_4x1_v_bz_TNT (30.2%) | flash_attn_varlen_func @ vllm/vllm_flash_attn/flash_attn_interface.py:176<br>get_scheduler_metadata @ vllm/vllm_flash_attn/flash_attn_interface.py:122<br>_prepare_inputs @ vllm/v1/worker/gpu_model_runner.py:1784<br>default_unquantized_gemm @ vllm/model_executor/layers/utils.py:92 | vllm/compilation/passes/fusion/attn_quant_fusion.py<br>vllm/v1/attention/ops/merge_attn_states.py<br>vllm/csrc/attention/merge_attn_states.cu<br>vllm/docs/design/fusions.md | This trace matches a reusable upstream precedent at 95.9% related GPU time. vLLM already combines attention merge and attention-epilogue quantization through reusable compile-time and kernel paths. |

### TensorRT-LLM

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/analysis_trtllm.txt`

#### Triage View
Mode: mapping-formal
Framework: TensorRT-LLM
Mapping traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/trtllm_mapping/trace-prefill.json, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/trtllm_mapping/trace-decode.json
Formal traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/trtllm_formal/trace-prefill.json, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_14b/trtllm_formal/trace-decode.json

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_384x8_64x4_4x1_v_bz_TNT | gemm | 4.83 ms | 44.2% | 40 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 3.32 ms | 30.4% | 80 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 1.06 ms | 9.7% | 40 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| kernel_mha | other | 0.62 ms | 5.7% | 40 | tensorrt_llm/_torch/attention_backend/trtllm.py:298 run | trtllm::attention_inplace |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.51 ms | 4.7% | 1 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.22 ms | 2.0% | 80 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 flashinfer_fused_add_rmsnorm | norm::fused_add_rmsnorm |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_384x8_64x4_4x1_v_bz_TNT | gemm | 14.48 ms | 44.2% | 120 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 9.95 ms | 30.4% | 240 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 3.18 ms | 9.7% | 120 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| kernel_mha | other | 1.87 ms | 5.7% | 120 | tensorrt_llm/_torch/attention_backend/trtllm.py:298 run | trtllm::attention_inplace |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 1.52 ms | 4.7% | 3 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.65 ms | 2.0% | 240 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 flashinfer_fused_add_rmsnorm | norm::fused_add_rmsnorm |

#### Overlap Opportunity Table
##### extend/prefill
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P2 | headroom | nvjet_tst_384x8_64x4_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 4831.5 us, share 44.2%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 3317.7 us, share 30.4%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 1062.4 us, share 9.7%, excl 100.0% / hid 0.0% | high | check deps |
| P1 | headroom | kernel_mha | tensorrt_llm/_torch/attention_backend/trtllm.py(298): run | 622.2 us, share 5.7%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 509.6 us, share 4.7%, excl 100.0% / hid 0.0% | high | check deps |

##### decode
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P2 | headroom | nvjet_tst_384x8_64x4_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 14482.1 us, share 44.2%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 9946.8 us, share 30.4%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 3179.0 us, share 9.7%, excl 100.0% / hid 0.0% | high | check deps |
| P1 | headroom | kernel_mha | tensorrt_llm/_torch/attention_backend/trtllm.py(298): run | 1869.2 us, share 5.7%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 1524.2 us, share 4.7%, excl 100.0% / hid 0.0% | high | check deps |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| TensorRT-LLM FlashInfer residual add + RMSNorm | Likely | 0.22 ms | 2.0% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (2.0%) | flashinfer_fused_add_rmsnorm @ tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py<br>tensorrt_llm/_torch/modules/rms_norm.py<br>tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | This trace matches a reusable upstream precedent at 2.0% related GPU time. TensorRT-LLM already exposes a FlashInfer fused residual-add plus RMSNorm family, including AutoDeploy rewrites. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| TensorRT-LLM FlashInfer residual add + RMSNorm | Likely | 0.65 ms | 2.0% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (2.0%) | flashinfer_fused_add_rmsnorm @ tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py<br>tensorrt_llm/_torch/modules/rms_norm.py<br>tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | This trace matches a reusable upstream precedent at 2.0% related GPU time. TensorRT-LLM already exposes a FlashInfer fused residual-add plus RMSNorm family, including AutoDeploy rewrites. |

## Qwen/Qwen3-30B-A3B

Model id: `qwen3_30b_a3b`

### SGLang

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/analysis_sglang.txt`

#### Triage View
Mode: single-trace
Framework: SGLang
Input traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/sglang_profile_live/1776867215.1439197/triage-trace-1776867215.1651354-TP-0-EXTEND.trace.json.gz, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/sglang_profile_live/1776867215.1439197/triage-trace-1776867215.1651354-TP-0-DECODE.trace.json.gz
Model: Qwen/Qwen3-30B-A3B

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fused_moe_kernel | moe | 10.16 ms | 75.5% | 96 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 4/2 |
| nvjet_tst_64x24_64x16_4x1_v_bz_TNT | gemm | 0.46 ms | 3.4% | 48 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 4/2 |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, false, true, true, true, true, true> > > > | gemm | 0.46 ms | 3.4% | 48 | python/sglang/jit_kernel/flash_attention_v3.py:89 flash_attn_with_kvcache | sgl_kernel::fwd |
| nvjet_tst_64x24_64x16_4x1_v_bz_splitK_TNT | gemm | 0.41 ms | 3.0% | 48 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 4/2 |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | gemm | 0.26 ms | 1.9% | 48 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 4/2 |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.22 ms | 1.7% | 96 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 4/2 |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.20 ms | 1.5% | 1 | python/sglang/srt/layers/logits_processor.py:878 _compute_lm_head | aten::mm |
| void flash::prepare_varlen_num_blocks_kernel<1, true> | gemm | 0.18 ms | 1.3% | 48 | python/sglang/jit_kernel/flash_attention_v3.py:89 flash_attn_with_kvcache | sgl_kernel::fwd |
| void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16> | softmax | 0.16 ms | 1.2% | 48 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 4/2 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 0.14 ms | 1.0% | 48 | python/sglang/srt/layers/attention/flashattention_backend.py:597 forward_extend | aten::copy_ |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fused_moe_kernel | moe | 7.53 ms | 36.3% | 480 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 2.26 ms | 10.9% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | gemm | 1.98 ms | 9.5% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true> > > > | gemm | 1.65 ms | 8.0% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_64x8_64x16_2x1_v_bz_TNT | gemm | 1.19 ms | 5.8% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 1.02 ms | 4.9% | 5 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0 | gemm | 0.99 ms | 4.8% | 480 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16> | softmax | 0.79 ms | 3.8% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void moe_align_block_size_kernel<int> | moe | 0.63 ms | 3.0% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void (anonymous namespace)::fused_rope_store_kernel<true, 128l, true, __nv_bfloat16, long, 16u> | rope | 0.43 ms | 2.1% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.41 ms | 2.0% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16> | other | 0.36 ms | 1.7% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > | gemm | 0.31 ms | 1.5% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))> | activation | 0.31 ms | 1.5% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| triton_per_fused_copy__mul_sum_0 | memory | 0.31 ms | 1.5% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void count_and_sort_expert_tokens_kernel<int> | moe | 0.31 ms | 1.5% | 240 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |

#### Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No actionable overlap rows. Use mapping/formal two-trace triage for stronger overlap conclusions. | - | - | - | - |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 1.33 ms | 9.9% | nvjet_tst_64x24_64x16_4x1_v_bz_TNT (3.4%)<br>nvjet_tst_64x24_64x16_4x1_v_bz_splitK_TNT (3.0%)<br>nvjet_tst_64x8_64x16_2x4_h_bz_TNT (1.9%) | trampoline @ python/sglang/srt/compilation/compile.py:188<br>_compute_lm_head @ python/sglang/srt/layers/logits_processor.py:878<br>forward @ python/sglang/srt/layers/sampler.py:77 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 9.9% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RoPE reshape + KV cache write | Likely | 0.75 ms | 5.6% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, false, true, true, true, true, true> > > > (3.4%)<br>void flash::prepare_varlen_num_blocks_kernel<1, true> (1.3%) | trampoline @ python/sglang/srt/compilation/compile.py:188<br>flash_attn_with_kvcache @ python/sglang/jit_kernel/flash_attention_v3.py:89 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 5.6% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| Fused residual add + RMSNorm | Likely | 0.22 ms | 1.7% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (1.7%) | trampoline @ python/sglang/srt/compilation/compile.py:188 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 1.7% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 6.45 ms | 31.2% | nvjet_tst_64x8_64x16_4x1_v_bz_TNT (10.9%)<br>nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT (9.5%)<br>nvjet_tst_64x8_64x16_2x1_v_bz_TNT (5.8%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895<br>forward @ python/sglang/srt/layers/sampler.py:77 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 31.2% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RMSNorm + RoPE | Likely | 0.78 ms | 3.8% | void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16> (1.7%)<br>void (anonymous namespace)::fused_rope_store_kernel<true, 128l, true, __nv_bfloat16, long, 16u> (2.1%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895 | python/sglang/jit_kernel/fused_qknorm_rope.py<br>python/sglang/srt/models/qwen3_moe.py | Related split kernels occupy 3.8% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. SGLang already ships a fused QK-norm plus RoPE kernel family. |
| Fused residual add + RMSNorm | Likely | 0.99 ms | 4.8% | kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign128o204820481_tensorptrbf16gmemalign_0 (4.8%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 4.8% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |

### vLLM

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/analysis_vllm.txt`

#### Triage View
Mode: mapping-formal
Framework: vLLM
Mapping traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/vllm_mapping/vllm_profile/dp0_pp0_tp0_dcp0_ep0_rank0.1776867152271917712.pt.trace.json.gz
Formal traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/vllm_formal/vllm_profile/dp0_pp0_tp0_dcp0_ep0_rank0.1776866988759542644.pt.trace.json.gz

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fused_moe_kernel | moe | 10.21 ms | 73.8% | 96 | vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py:635 forward | vllm::moe_forward |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, true, true, true, true, false, true> > > > | gemm | 0.61 ms | 4.4% | 48 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| nvjet_tst_64x24_64x16_4x1_v_bz_TNT | gemm | 0.48 ms | 3.5% | 48 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x24_64x16_4x1_v_bz_splitK_TNT | gemm | 0.44 ms | 3.1% | 48 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> > | reduce_topk | 0.29 ms | 2.1% | 48 | vllm/_custom_ops.py:2129 moe_sum | aten::sum |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | gemm | 0.26 ms | 1.9% | 48 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.20 ms | 1.5% | 1 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void vllm::moe::topkGating<8, 128, 4, 16, 32, int, __nv_bfloat16, (vllm::moe::ScoringFunc)0> | moe | 0.20 ms | 1.4% | 48 | vllm/_custom_ops.py:2286 topk_softmax | _moe_C::topk_softmax |
| void vllm::moe::moe_align_block_size_kernel<int> | moe | 0.14 ms | 1.0% | 48 | vllm/_custom_ops.py:2133 moe_align_block_size | _moe_C::moe_align_block_size |
| void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > | gemm | 0.14 ms | 1.0% | 48 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fused_moe_kernel | moe | 17.02 ms | 32.8% | 1056 | vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py:635 forward | vllm::moe_forward |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, false, true> > > > | gemm | 5.25 ms | 10.1% | 528 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 4.97 ms | 9.6% | 528 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | gemm | 4.32 ms | 8.3% | 528 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_TNT | gemm | 2.60 ms | 5.0% | 528 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> > | reduce_topk | 2.60 ms | 5.0% | 528 | vllm/_custom_ops.py:2129 moe_sum | aten::sum |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 2.23 ms | 4.3% | 11 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void vllm::moe::topkGating<8, 128, 4, 16, 32, int, __nv_bfloat16, (vllm::moe::ScoringFunc)0> | moe | 2.06 ms | 4.0% | 528 | vllm/_custom_ops.py:2286 topk_softmax | _moe_C::topk_softmax |
| void vllm::moe::moe_align_block_size_kernel<int> | moe | 1.54 ms | 3.0% | 528 | vllm/_custom_ops.py:2133 moe_align_block_size | _moe_C::moe_align_block_size |
| void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0> | quantize | 1.17 ms | 2.2% | 528 | vllm/_custom_ops.py:2508 reshape_and_cache_flash | _C_cache_ops::reshape_and_cache_flash |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1 | memory | 1.11 ms | 2.1% | 528 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 | memory | 1.01 ms | 2.0% | 528 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |
| void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > | gemm | 0.95 ms | 1.8% | 528 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.89 ms | 1.7% | 528 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| triton_poi_fused_3 | other | 0.87 ms | 1.7% | 517 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |
| void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloat16 const&)), &(__nv_bfloat162 vllm::packed_silu_kernel<__nv_bfloat162>(__nv_bfloat162 const&)), true, true, false> | activation | 0.84 ms | 1.6% | 528 | vllm/model_executor/layers/fused_moe/activation.py:94 apply_moe_activation | _C::silu_and_mul |
| void vllm::moe::count_and_sort_expert_tokens_kernel<int> | moe | 0.69 ms | 1.3% | 528 | vllm/_custom_ops.py:2133 moe_align_block_size | _moe_C::moe_align_block_size |
| triton_red_fused_2 | other | 0.67 ms | 1.3% | 517 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |
| memcpy32_post | memory | 0.52 ms | 1.0% | 528 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |

#### Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 4968.1 us, share 7.6%, excl 100.0% / hid 0.0% | unclear | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 4322.7 us, share 6.6%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> > | vllm/_custom_ops.py(2129): moe_sum | 2886.3 us, share 4.4%, excl 100.0% / hid 0.0% | unclear | check deps |
| P2 | headroom | void vllm::moe::topkGating<8, 128, 4, 16, 32, int, __nv_bfloat16, (vllm::moe::ScoringFunc)0> | vllm/_custom_ops.py(2286): topk_softmax | 2265.4 us, share 3.5%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 2601.5 us, share 4.0%, excl 100.0% / hid 0.0% | unclear | check deps |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| vLLM-origin Attention + Quantization | Likely | 2.34 ms | 16.9% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, true, true, true, true, false, true> > > > (4.4%)<br>void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > (1.0%)<br>nvjet_tst_64x24_64x16_4x1_v_bz_TNT (3.5%) | flash_attn_varlen_func @ vllm/vllm_flash_attn/flash_attn_interface.py:176<br>get_scheduler_metadata @ vllm/vllm_flash_attn/flash_attn_interface.py:122<br>_prepare_inputs @ vllm/v1/worker/gpu_model_runner.py:1784<br>default_unquantized_gemm @ vllm/model_executor/layers/utils.py:92 | vllm/compilation/passes/fusion/attn_quant_fusion.py<br>vllm/v1/attention/ops/merge_attn_states.py<br>vllm/csrc/attention/merge_attn_states.cu<br>vllm/docs/design/fusions.md | This trace matches a reusable upstream precedent at 16.9% related GPU time. vLLM already combines attention merge and attention-epilogue quantization through reusable compile-time and kernel paths. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| vLLM-origin Attention + Quantization | Likely | 22.42 ms | 43.2% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, false, true> > > > (10.1%)<br>void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > (1.8%)<br>nvjet_tst_64x8_64x16_4x1_v_bz_TNT (9.6%) | flash_attn_varlen_func @ vllm/vllm_flash_attn/flash_attn_interface.py:176<br>get_scheduler_metadata @ vllm/vllm_flash_attn/flash_attn_interface.py:122<br>_prepare_inputs @ vllm/v1/worker/gpu_model_runner.py:1784<br>default_unquantized_gemm @ vllm/model_executor/layers/utils.py:92 | vllm/compilation/passes/fusion/attn_quant_fusion.py<br>vllm/v1/attention/ops/merge_attn_states.py<br>vllm/csrc/attention/merge_attn_states.cu<br>vllm/docs/design/fusions.md | This trace matches a reusable upstream precedent at 43.2% related GPU time. vLLM already combines attention merge and attention-epilogue quantization through reusable compile-time and kernel paths. |
| vLLM fused activation-and-mul | Likely | 0.84 ms | 1.6% | void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloat16 const&)), &(__nv_bfloat162 vllm::packed_silu_kernel<__nv_bfloat162>(__nv_bfloat162 const&)), true, true, false> (1.6%) | apply_moe_activation @ vllm/model_executor/layers/fused_moe/activation.py:94 | vllm/_custom_ops.py<br>vllm/compilation/passes/fusion/act_quant_fusion.py | This trace matches a reusable upstream precedent at 1.6% related GPU time. vLLM already ships fused activation-and-multiply kernels and quantization-aware variants for the MLP epilogue. |

### TensorRT-LLM

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/analysis_trtllm.txt`

#### Triage View
Mode: mapping-formal
Framework: TensorRT-LLM
Mapping traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/trtllm_mapping/trace-prefill.json, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/trtllm_mapping/trace-decode.json
Formal traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/trtllm_formal/trace-prefill.json, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_30b_a3b/trtllm_formal/trace-decode.json

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| void fused_moe::run_global<fused_moe::Fused_Moe_Kernel_sm80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, 32, 128, 64, 4, (fused_moe::Activation_Type)3> > | moe | 1.08 ms | 21.6% | 48 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| kernel_mha | other | 0.76 ms | 15.3% | 48 | tensorrt_llm/_torch/attention_backend/trtllm.py:298 run | trtllm::attention_inplace |
| _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalINS1_17GroupProblemShapeIN4cute5tupleIJlllEEEEENS1_10collective13CollectiveMmaINS1_39MainloopSm90ArrayTmaGmmaWarpSpecializedILi12ENS6_IJNS5_1CILi1EEESD_SD_EEENS1_43KernelPtrArrayTmaWarpSpecializedCooperativeEEENS6_IJNSC_ILi128EEENSC_ILi16EEENSC_ILi64EEEEEENS_10bfloat16_tEPNS6_IJlSD_NSC_ILi0EEEEEESL_SO_NS5_8TiledMMAINS5_8MMA_AtomIJNS5_4SM904GMMA27MMA_64x16x16_F32BF16BF16_SSILNSS_5MajorE0ELSU_0ELNSS_7ScaleInE1ELSV_1EEEEEENS5_6LayoutINS6_IJNSC_ILi2EEESD_SD_EEENS6_IJSD_SM_SM_EEEEENS6_IJNS5_10UnderscoreES13_S13_EEEEENS5_13SM90_TMA_LOADENS5_14ComposedLayoutINS5_7SwizzleILi3ELi4ELi3EEENS5_18smem_ptr_flag_bitsILi16EEENSY_INS6_IJNSC_ILi8EEESJ_EEENS6_IJSJ_SD_EEEEEEEvNS5_8identityES16_S1G_vS1H_EENS_8epilogue10collective31EpilogueMoeFusedFinalizeBuilderINS_4arch4Sm90ESK_SL_PNS6_IJSD_lSM_EEESL_S1O_ffSL_NS6_IJSD_SM_iEEEfNS6_IJSM_SD_EEEE44TmaWarpSpecializedAdapterWithSmemStorageImplINS1K_6detail29Sm90TmaWarpSpecializedAdapterINS1K_24EpilogueMoeFusedFinalizeIS1P_SL_S1O_NS1J_6thread17LinearCombinationIfLi1EffLNS1X_9ScaleType4KindE0ELNS_15FloatRoundStyleE2ESL_EESL_S1Q_fS1R_NS6_IJSH_SI_EEENS17_IS19_NS1A_ILi32EEENSY_INS6_IJNSC_ILi32EEES1C_EEENS6_IJSD_S25_EEEEEEENS5_39AutoVectorizingCopyWithAssumedAlignmentILi128EEENS2A_ILi8EEENS5_28SM90_RED_ADD_NOFTZ_BF16x2_V4EEEEES2E_EEvvEEEEvNT_6ParamsE | moe | 0.74 ms | 14.9% | 48 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 0.46 ms | 9.2% | 48 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | gemm | 0.39 ms | 7.9% | 48 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_TNT | gemm | 0.24 ms | 4.8% | 48 | tensorrt_llm/_torch/models/modeling_qwen3_moe.py:55 forward | trtllm::cublas_mm |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.23 ms | 4.5% | 96 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 flashinfer_fused_add_rmsnorm | norm::fused_add_rmsnorm |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.20 ms | 4.1% | 1 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void tensorrt_llm::kernels::cutlass_kernels::fusedBuildExpertMapsSortFirstTokenKernel<32, 8, 8> | moe | 0.15 ms | 2.9% | 48 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| void tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> | gemm | 0.12 ms | 2.5% | 48 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| void tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_bfloat16, __nv_bfloat16, (tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType)2, false> | gemm | 0.12 ms | 2.3% | 48 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| void tensorrt_llm::kernels::applyBiasRopeUpdateKVCacheV2<__nv_bfloat16, __nv_bfloat16, 256, 128, false, false, false, true, tensorrt_llm::kernels::KVBlockArray, (tensorrt_llm::kernels::RotaryPositionEmbeddingType)0> | rope | 0.11 ms | 2.1% | 48 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |
| void tensorrt_llm::kernels::fusedQKNormRopeKernel<128, false> | rope | 0.10 ms | 2.1% | 48 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |
| void tensorrt_llm::kernels::customMoeRoutingKernel<__nv_bfloat16, float, int, 128, 8, false> | moe | 0.09 ms | 1.7% | 48 | tensorrt_llm/_torch/modules/fused_moe/routing.py:113 apply | trtllm::renorm_moe_routing_op |
| Memset (Unknown) | memory | 0.08 ms | 1.7% | 49 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk (site share 98%)<br>tensorrt_llm/_torch/pyexecutor/sampler.py:446 _process_requests (site share 1%) | trtllm::fused_moe<br>aten::argmax |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.08 ms | 1.6% | 48 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| void fused_moe::run_global<fused_moe::Fused_Moe_Kernel_sm80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, 32, 128, 64, 4, (fused_moe::Activation_Type)3> > | moe | 3.24 ms | 21.5% | 144 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| kernel_mha | other | 2.32 ms | 15.4% | 144 | tensorrt_llm/_torch/attention_backend/trtllm.py:298 run | trtllm::attention_inplace |
| _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalINS1_17GroupProblemShapeIN4cute5tupleIJlllEEEEENS1_10collective13CollectiveMmaINS1_39MainloopSm90ArrayTmaGmmaWarpSpecializedILi12ENS6_IJNS5_1CILi1EEESD_SD_EEENS1_43KernelPtrArrayTmaWarpSpecializedCooperativeEEENS6_IJNSC_ILi128EEENSC_ILi16EEENSC_ILi64EEEEEENS_10bfloat16_tEPNS6_IJlSD_NSC_ILi0EEEEEESL_SO_NS5_8TiledMMAINS5_8MMA_AtomIJNS5_4SM904GMMA27MMA_64x16x16_F32BF16BF16_SSILNSS_5MajorE0ELSU_0ELNSS_7ScaleInE1ELSV_1EEEEEENS5_6LayoutINS6_IJNSC_ILi2EEESD_SD_EEENS6_IJSD_SM_SM_EEEEENS6_IJNS5_10UnderscoreES13_S13_EEEEENS5_13SM90_TMA_LOADENS5_14ComposedLayoutINS5_7SwizzleILi3ELi4ELi3EEENS5_18smem_ptr_flag_bitsILi16EEENSY_INS6_IJNSC_ILi8EEESJ_EEENS6_IJSJ_SD_EEEEEEEvNS5_8identityES16_S1G_vS1H_EENS_8epilogue10collective31EpilogueMoeFusedFinalizeBuilderINS_4arch4Sm90ESK_SL_PNS6_IJSD_lSM_EEESL_S1O_ffSL_NS6_IJSD_SM_iEEEfNS6_IJSM_SD_EEEE44TmaWarpSpecializedAdapterWithSmemStorageImplINS1K_6detail29Sm90TmaWarpSpecializedAdapterINS1K_24EpilogueMoeFusedFinalizeIS1P_SL_S1O_NS1J_6thread17LinearCombinationIfLi1EffLNS1X_9ScaleType4KindE0ELNS_15FloatRoundStyleE2ESL_EESL_S1Q_fS1R_NS6_IJSH_SI_EEENS17_IS19_NS1A_ILi32EEENSY_INS6_IJNSC_ILi32EEES1C_EEENS6_IJSD_S25_EEEEEEENS5_39AutoVectorizingCopyWithAssumedAlignmentILi128EEENS2A_ILi8EEENS5_28SM90_RED_ADD_NOFTZ_BF16x2_V4EEEEES2E_EEvvEEEEvNT_6ParamsE | moe | 2.25 ms | 15.0% | 144 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 1.38 ms | 9.2% | 144 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | gemm | 1.17 ms | 7.8% | 144 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_TNT | gemm | 0.71 ms | 4.7% | 144 | tensorrt_llm/_torch/models/modeling_qwen3_moe.py:55 forward | trtllm::cublas_mm |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.68 ms | 4.5% | 288 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 flashinfer_fused_add_rmsnorm | norm::fused_add_rmsnorm |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.61 ms | 4.0% | 3 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void tensorrt_llm::kernels::cutlass_kernels::fusedBuildExpertMapsSortFirstTokenKernel<32, 8, 8> | moe | 0.44 ms | 2.9% | 144 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| void tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> | gemm | 0.37 ms | 2.5% | 144 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| void tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_bfloat16, __nv_bfloat16, (tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType)2, false> | gemm | 0.35 ms | 2.3% | 144 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk | trtllm::fused_moe |
| void tensorrt_llm::kernels::applyBiasRopeUpdateKVCacheV2<__nv_bfloat16, __nv_bfloat16, 256, 128, false, false, false, true, tensorrt_llm::kernels::KVBlockArray, (tensorrt_llm::kernels::RotaryPositionEmbeddingType)0> | rope | 0.31 ms | 2.1% | 144 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |
| void tensorrt_llm::kernels::fusedQKNormRopeKernel<128, false> | rope | 0.31 ms | 2.1% | 144 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |
| void tensorrt_llm::kernels::customMoeRoutingKernel<__nv_bfloat16, float, int, 128, 8, false> | moe | 0.26 ms | 1.7% | 144 | tensorrt_llm/_torch/modules/fused_moe/routing.py:113 apply | trtllm::renorm_moe_routing_op |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.25 ms | 1.6% | 144 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| Memset (Device) | memory | 0.17 ms | 1.1% | 101 | tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:192 forward_chunk (site share 98%)<br>tensorrt_llm/_torch/modules/linear.py:276 apply (site share 1%) | trtllm::fused_moe<br>aten::mm |

#### Overlap Opportunity Table
##### extend/prefill
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | headroom | kernel_mha | tensorrt_llm/_torch/attention_backend/trtllm.py(298): run | 762.5 us, share 15.3%, excl 100.0% / hid 0.0% | low | try fusion |
| P1 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 460.8 us, share 9.2%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 392.7 us, share 7.9%, excl 100.0% / hid 0.0% | high | check deps |
| P1 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_TNT | tensorrt_llm/_torch/models/modeling_qwen3_moe.py(55): forward | 237.5 us, share 4.8%, excl 100.0% / hid 0.0% | low | try fusion |
| P1 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 202.5 us, share 4.1%, excl 100.0% / hid 0.0% | low | try fusion |

##### decode
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | headroom | kernel_mha | tensorrt_llm/_torch/attention_backend/trtllm.py(298): run | 2318.6 us, share 15.4%, excl 100.0% / hid 0.0% | low | try fusion |
| P1 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 1383.3 us, share 9.2%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 1170.7 us, share 7.8%, excl 100.0% / hid 0.0% | high | check deps |
| P1 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_TNT | tensorrt_llm/_torch/models/modeling_qwen3_moe.py(55): forward | 713.5 us, share 4.7%, excl 100.0% / hid 0.0% | low | try fusion |
| P1 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 607.2 us, share 4.0%, excl 100.0% / hid 0.0% | low | try fusion |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| TensorRT-LLM FlashInfer residual add + RMSNorm | Likely | 0.23 ms | 4.5% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (4.5%) | flashinfer_fused_add_rmsnorm @ tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py<br>tensorrt_llm/_torch/modules/rms_norm.py<br>tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | This trace matches a reusable upstream precedent at 4.5% related GPU time. TensorRT-LLM already exposes a FlashInfer fused residual-add plus RMSNorm family, including AutoDeploy rewrites. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| TensorRT-LLM FlashInfer residual add + RMSNorm | Likely | 0.68 ms | 4.5% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (4.5%) | flashinfer_fused_add_rmsnorm @ tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py<br>tensorrt_llm/_torch/modules/rms_norm.py<br>tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | This trace matches a reusable upstream precedent at 4.5% related GPU time. TensorRT-LLM already exposes a FlashInfer fused residual-add plus RMSNorm family, including AutoDeploy rewrites. |

## Qwen/Qwen3-8B

Model id: `qwen3_8b`

### SGLang

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/analysis_sglang.txt`

#### Triage View
Mode: single-trace
Framework: SGLang
Input traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/sglang_profile_live/1776864860.8510807/triage-trace-1776864860.872042-TP-0-EXTEND.trace.json.gz, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/sglang_profile_live/1776864860.8510807/triage-trace-1776864860.872042-TP-0-DECODE.trace.json.gz
Model: Qwen/Qwen3-8B

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_192x24_64x7_2x1_v_bz_TNT | gemm | 2.52 ms | 37.6% | 36 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| nvjet_tst_64x24_64x16_2x1_v_bz_splitK_TNT | gemm | 1.34 ms | 20.1% | 36 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| nvjet_tst_64x24_64x16_4x1_v_bz_TNT | gemm | 1.24 ms | 18.6% | 72 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.41 ms | 6.2% | 1 | python/sglang/srt/layers/logits_processor.py:878 _compute_lm_head | aten::mm |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, false, true, true, true, true, true> > > > | gemm | 0.30 ms | 4.4% | 36 | python/sglang/jit_kernel/flash_attention_v3.py:89 flash_attn_with_kvcache | sgl_kernel::fwd |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.19 ms | 2.8% | 72 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| void flash::prepare_varlen_num_blocks_kernel<1, true> | gemm | 0.12 ms | 1.8% | 36 | python/sglang/jit_kernel/flash_attention_v3.py:89 flash_attn_with_kvcache | sgl_kernel::fwd |
| void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))> | activation | 0.11 ms | 1.6% | 36 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 0.10 ms | 1.6% | 36 | python/sglang/srt/layers/attention/flashattention_backend.py:597 forward_extend | aten::copy_ |
| void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u> | rope | 0.08 ms | 1.2% | 36 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.07 ms | 1.1% | 36 | python/sglang/srt/compilation/compile.py:188 trampoline | Torch-Compiled Region: 0/0 |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_192x8_64x8_2x1_v_bz_TNT | gemm | 12.31 ms | 39.2% | 180 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 8.82 ms | 28.1% | 360 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 3.56 ms | 11.3% | 180 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 2.06 ms | 6.5% | 5 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true> > > > | gemm | 1.23 ms | 3.9% | 180 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign_0 | gemm | 0.83 ms | 2.7% | 360 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.64 ms | 2.0% | 360 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))> | activation | 0.53 ms | 1.7% | 180 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |
| void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u> | rope | 0.32 ms | 1.0% | 180 | python/sglang/srt/model_executor/model_runner.py:2895 _forward_raw | cudaGraphLaunch |

#### Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No actionable overlap rows. Use mapping/formal two-trace triage for stronger overlap conclusions. | - | - | - | - |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 5.52 ms | 82.4% | nvjet_tst_192x24_64x7_2x1_v_bz_TNT (37.6%)<br>nvjet_tst_64x24_64x16_2x1_v_bz_splitK_TNT (20.1%)<br>nvjet_tst_64x24_64x16_4x1_v_bz_TNT (18.6%) | trampoline @ python/sglang/srt/compilation/compile.py:188<br>_compute_lm_head @ python/sglang/srt/layers/logits_processor.py:878<br>forward @ python/sglang/srt/layers/sampler.py:77 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 82.4% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RoPE reshape + KV cache write | Likely | 0.54 ms | 8.1% | void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u> (1.2%)<br>void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, false, true, true, true, true, true> > > > (4.4%)<br>void flash::prepare_varlen_num_blocks_kernel<1, true> (1.8%) | trampoline @ python/sglang/srt/compilation/compile.py:188<br>flash_attn_with_kvcache @ python/sglang/jit_kernel/flash_attention_v3.py:89<br>_set_kv_buffer_impl @ python/sglang/srt/mem_cache/memory_pool.py:89 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 8.1% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| Fused residual add + RMSNorm | Likely | 0.19 ms | 2.8% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (2.8%) | trampoline @ python/sglang/srt/compilation/compile.py:188 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 2.8% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 26.76 ms | 85.2% | nvjet_tst_192x8_64x8_2x1_v_bz_TNT (39.2%)<br>nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT (28.1%)<br>nvjet_tst_64x8_64x16_4x1_v_bz_TNT (11.3%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895<br>forward @ python/sglang/srt/layers/sampler.py:77 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 85.2% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RoPE reshape + KV cache write | Conditional | 0.53 ms | 1.7% | void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u> (1.0%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 1.7% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| Fused residual add + RMSNorm | Likely | 0.83 ms | 2.7% | kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign_0 (2.7%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:2895 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 2.7% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |

### vLLM

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/analysis_vllm.txt`

#### Triage View
Mode: mapping-formal
Framework: vLLM
Mapping traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/vllm_mapping/vllm_profile/rank0.1776864935834059331.pt.trace.json.gz
Formal traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/vllm_formal/vllm_profile/rank0.1776864951336950340.pt.trace.json.gz

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_192x24_64x7_2x1_v_bz_TNT | gemm | 2.50 ms | 37.5% | 36 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x24_64x16_2x1_v_bz_splitK_TNT | gemm | 1.34 ms | 20.2% | 36 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x24_64x16_4x1_v_bz_TNT | gemm | 1.28 ms | 19.2% | 72 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, true, true, true, true, false, true> > > > | gemm | 0.44 ms | 6.6% | 36 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.42 ms | 6.4% | 1 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > | gemm | 0.11 ms | 1.7% | 36 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0> | quantize | 0.10 ms | 1.4% | 36 | vllm/_custom_ops.py:2508 reshape_and_cache_flash | _C_cache_ops::reshape_and_cache_flash |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2 | memory | 0.08 ms | 1.2% | 36 | vllm/model_executor/models/qwen2.py:422 forward | cudaGraphLaunch |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.07 ms | 1.1% | 36 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 | memory | 0.07 ms | 1.1% | 36 | vllm/model_executor/models/qwen2.py:422 forward | cudaGraphLaunch |
| triton_poi_fused_4 | other | 0.07 ms | 1.1% | 35 | vllm/model_executor/models/qwen2.py:422 forward | cudaGraphLaunch |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_192x8_64x8_2x1_v_bz_TNT | gemm | 27.20 ms | 39.0% | 396 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 19.49 ms | 28.0% | 792 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 7.97 ms | 11.4% | 396 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 4.62 ms | 6.6% | 11 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, false, true> > > > | gemm | 3.62 ms | 5.2% | 396 | vllm/vllm_flash_attn/flash_attn_interface.py:176 flash_attn_varlen_func | _vllm_fa3_C::fwd |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 1.37 ms | 2.0% | 792 | vllm/model_executor/layers/utils.py:92 default_unquantized_gemm | aten::mm |
| void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0> | quantize | 0.89 ms | 1.3% | 396 | vllm/_custom_ops.py:2508 reshape_and_cache_flash | _C_cache_ops::reshape_and_cache_flash |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2 | memory | 0.87 ms | 1.2% | 396 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |
| triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 | memory | 0.80 ms | 1.2% | 396 | vllm/v1/worker/gpu_model_runner.py:3491 _model_forward | cudaGraphLaunch |

#### Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P2 | headroom | nvjet_tst_192x8_64x8_2x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 27196.5 us, share 35.7%, excl 100.0% / hid 0.0% | unclear | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 19486.7 us, share 25.6%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 7966.3 us, share 10.4%, excl 100.0% / hid 0.0% | unclear | check deps |
| P2 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 5049.7 us, share 6.6%, excl 100.0% / hid 0.0% | high | check deps |
| P2 | headroom | nvjet_tst_192x24_64x7_2x1_v_bz_TNT | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm | 2499.3 us, share 3.3%, excl 100.0% / hid 0.0% | unclear | check deps |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| vLLM-origin Attention + Quantization | Likely | 6.27 ms | 94.1% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 128, 256, 128, true, true, true, true, false, true> > > > (6.6%)<br>void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > (1.7%)<br>nvjet_tst_192x24_64x7_2x1_v_bz_TNT (37.5%) | flash_attn_varlen_func @ vllm/vllm_flash_attn/flash_attn_interface.py:176<br>get_scheduler_metadata @ vllm/vllm_flash_attn/flash_attn_interface.py:122<br>_prepare_inputs @ vllm/v1/worker/gpu_model_runner.py:1784<br>default_unquantized_gemm @ vllm/model_executor/layers/utils.py:92 | vllm/compilation/passes/fusion/attn_quant_fusion.py<br>vllm/v1/attention/ops/merge_attn_states.py<br>vllm/csrc/attention/merge_attn_states.cu<br>vllm/docs/design/fusions.md | This trace matches a reusable upstream precedent at 94.1% related GPU time. vLLM already combines attention merge and attention-epilogue quantization through reusable compile-time and kernel paths. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| vLLM-origin Attention + Quantization | Likely | 65.79 ms | 94.4% | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, false, true> > > > (5.2%)<br>nvjet_tst_192x8_64x8_2x1_v_bz_TNT (39.0%)<br>nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT (28.0%) | flash_attn_varlen_func @ vllm/vllm_flash_attn/flash_attn_interface.py:176<br>get_scheduler_metadata @ vllm/vllm_flash_attn/flash_attn_interface.py:122<br>_prepare_inputs @ vllm/v1/worker/gpu_model_runner.py:1784<br>default_unquantized_gemm @ vllm/model_executor/layers/utils.py:92 | vllm/compilation/passes/fusion/attn_quant_fusion.py<br>vllm/v1/attention/ops/merge_attn_states.py<br>vllm/csrc/attention/merge_attn_states.cu<br>vllm/docs/design/fusions.md | This trace matches a reusable upstream precedent at 94.4% related GPU time. vLLM already combines attention merge and attention-epilogue quantization through reusable compile-time and kernel paths. |

### TensorRT-LLM

Source: `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/analysis_trtllm.txt`

#### Triage View
Mode: mapping-formal
Framework: TensorRT-LLM
Mapping traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/trtllm_mapping/trace-prefill.json, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/trtllm_mapping/trace-decode.json
Formal traces: /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/trtllm_formal/trace-prefill.json, /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2/qwen3_8b/trtllm_formal/trace-decode.json

#### Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_192x8_64x8_2x1_v_bz_TNT | gemm | 2.48 ms | 37.9% | 36 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 1.78 ms | 27.2% | 72 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 0.73 ms | 11.1% | 36 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| kernel_mha | other | 0.57 ms | 8.8% | 36 | tensorrt_llm/_torch/attention_backend/trtllm.py:298 run | trtllm::attention_inplace |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.41 ms | 6.2% | 1 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.19 ms | 2.9% | 72 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 flashinfer_fused_add_rmsnorm | norm::fused_add_rmsnorm |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.13 ms | 2.0% | 72 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void tensorrt_llm::kernels::applyBiasRopeUpdateKVCacheV2<__nv_bfloat16, __nv_bfloat16, 256, 128, false, false, false, true, tensorrt_llm::kernels::KVBlockArray, (tensorrt_llm::kernels::RotaryPositionEmbeddingType)0> | rope | 0.08 ms | 1.2% | 36 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |
| void tensorrt_llm::kernels::fusedQKNormRopeKernel<128, false> | rope | 0.08 ms | 1.2% | 36 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_tst_192x8_64x8_2x1_v_bz_TNT | gemm | 7.43 ms | 37.8% | 108 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 5.33 ms | 27.1% | 216 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 2.19 ms | 11.1% | 108 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| kernel_mha | other | 1.72 ms | 8.8% | 108 | tensorrt_llm/_torch/attention_backend/trtllm.py:298 run | trtllm::attention_inplace |
| nvjet_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 1.22 ms | 6.2% | 3 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.56 ms | 2.9% | 216 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 flashinfer_fused_add_rmsnorm | norm::fused_add_rmsnorm |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.40 ms | 2.0% | 216 | tensorrt_llm/_torch/modules/linear.py:276 apply | aten::mm |
| void tensorrt_llm::kernels::applyBiasRopeUpdateKVCacheV2<__nv_bfloat16, __nv_bfloat16, 256, 128, false, false, false, true, tensorrt_llm::kernels::KVBlockArray, (tensorrt_llm::kernels::RotaryPositionEmbeddingType)0> | rope | 0.24 ms | 1.2% | 108 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |
| void tensorrt_llm::kernels::fusedQKNormRopeKernel<128, false> | rope | 0.24 ms | 1.2% | 108 | tensorrt_llm/_torch/models/modeling_qwen3.py:98 apply_qk_norm_rope | trtllm::fused_qk_norm_rope |

#### Overlap Opportunity Table
##### extend/prefill
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | headroom | nvjet_tst_192x8_64x8_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 2476.3 us, share 37.9%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 1776.1 us, share 27.2%, excl 100.0% / hid 0.0% | high | check deps |
| P1 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 727.7 us, share 11.1%, excl 100.0% / hid 0.0% | low | try fusion |
| P1 | headroom | kernel_mha | tensorrt_llm/_torch/attention_backend/trtllm.py(298): run | 573.4 us, share 8.8%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 408.1 us, share 6.2%, excl 100.0% / hid 0.0% | high | check deps |

##### decode
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | headroom | nvjet_tst_192x8_64x8_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 7430.4 us, share 37.8%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 5328.0 us, share 27.1%, excl 100.0% / hid 0.0% | high | check deps |
| P1 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 2185.3 us, share 11.1%, excl 100.0% / hid 0.0% | low | try fusion |
| P1 | headroom | kernel_mha | tensorrt_llm/_torch/attention_backend/trtllm.py(298): run | 1724.6 us, share 8.8%, excl 100.0% / hid 0.0% | low | try fusion |
| P2 | headroom | nvjet_tst_384x8_64x4_2x1_v_bz_TNT | tensorrt_llm/_torch/modules/linear.py(276): apply | 1224.4 us, share 6.2%, excl 100.0% / hid 0.0% | high | check deps |

#### Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| TensorRT-LLM FlashInfer residual add + RMSNorm | Likely | 0.19 ms | 2.9% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (2.9%) | flashinfer_fused_add_rmsnorm @ tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py<br>tensorrt_llm/_torch/modules/rms_norm.py<br>tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | This trace matches a reusable upstream precedent at 2.9% related GPU time. TensorRT-LLM already exposes a FlashInfer fused residual-add plus RMSNorm family, including AutoDeploy rewrites. |

##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| TensorRT-LLM FlashInfer residual add + RMSNorm | Likely | 0.56 ms | 2.9% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (2.9%) | flashinfer_fused_add_rmsnorm @ tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py:54 | tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py<br>tensorrt_llm/_torch/modules/rms_norm.py<br>tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py | This trace matches a reusable upstream precedent at 2.9% related GPU time. TensorRT-LLM already exposes a FlashInfer fused residual-add plus RMSNorm family, including AutoDeploy rewrites. |
