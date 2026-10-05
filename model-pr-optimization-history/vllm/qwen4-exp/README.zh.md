# vLLM Qwen4-Exp (Qwen3.8-Flash-Next) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/evals/qwen4_exp/README.md` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/evals/qwen4_exp/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8-rocm.yaml` | 无直接 PR 号提交 |
| `tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8.yaml` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/evals/qwen4_exp/configs/models-b200.txt` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/evals/qwen4_exp/configs/models-h200.txt` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/evals/qwen4_exp/configs/models-rocm.txt` | 无直接 PR 号提交 |
| `tests/evals/qwen4_exp/conftest.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/evals/qwen4_exp/test_accuracy.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/models/qwen4_exp/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/models/qwen4_exp/test_config.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `tests/models/qwen4_exp/test_hc_ops.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54687](https://github.com/vllm-project/vllm/pull/54687), [#58957](https://github.com/vllm-project/vllm/pull/58957) |
| `tests/models/qwen4_exp/test_ple.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54371](https://github.com/vllm-project/vllm/pull/54371), [#54517](https://github.com/vllm-project/vllm/pull/54517), [#54722](https://github.com/vllm-project/vllm/pull/54722), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#55309](https://github.com/vllm-project/vllm/pull/55309), [#55375](https://github.com/vllm-project/vllm/pull/55375), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#57497](https://github.com/vllm-project/vllm/pull/57497), [#58489](https://github.com/vllm-project/vllm/pull/58489), [#59990](https://github.com/vllm-project/vllm/pull/59990) |
| `tests/models/qwen4_exp/test_qsa_amd.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `tests/models/qwen4_exp/test_qsa_prepare.py` | [#57097](https://github.com/vllm-project/vllm/pull/57097) |
| `tests/models/qwen4_exp/test_qsa_reference.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54513](https://github.com/vllm-project/vllm/pull/54513), [#54873](https://github.com/vllm-project/vllm/pull/54873), [#54890](https://github.com/vllm-project/vllm/pull/54890), [#54915](https://github.com/vllm-project/vllm/pull/54915), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#55309](https://github.com/vllm-project/vllm/pull/55309), [#55557](https://github.com/vllm-project/vllm/pull/55557), [#57097](https://github.com/vllm-project/vllm/pull/57097), [#58961](https://github.com/vllm-project/vllm/pull/58961) |
| `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py` | [#54513](https://github.com/vllm-project/vllm/pull/54513), [#54873](https://github.com/vllm-project/vllm/pull/54873) |
| `vllm/models/qwen4_exp/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/hyperconnection.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/indexer_qsa.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/qwen4_exp/amd/low_latency_gemm.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/model.py` | [#51289](https://github.com/vllm-project/vllm/pull/51289), [#53896](https://github.com/vllm-project/vllm/pull/53896), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/qwen4_exp/amd/model_state.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/mtp.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/qwen4_exp/amd/ops/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/ops/hc.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/ops/qsa.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/amd/ple_layer.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#57497](https://github.com/vllm-project/vllm/pull/57497) |
| `vllm/models/qwen4_exp/amd/qsa.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/qwen4_exp/common/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/common/hyperconnection.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/common/ngram_embedding.py` | [#57497](https://github.com/vllm-project/vllm/pull/57497), [#59990](https://github.com/vllm-project/vllm/pull/59990) |
| `vllm/models/qwen4_exp/common/ple.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/common/qsa_cache.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54513](https://github.com/vllm-project/vllm/pull/54513), [#54890](https://github.com/vllm-project/vllm/pull/54890), [#54915](https://github.com/vllm-project/vllm/pull/54915), [#58961](https://github.com/vllm-project/vllm/pull/58961) |
| `vllm/models/qwen4_exp/nvidia/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/nvidia/hyperconnection.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54687](https://github.com/vllm-project/vllm/pull/54687), [#58957](https://github.com/vllm-project/vllm/pull/58957) |
| `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54513](https://github.com/vllm-project/vllm/pull/54513), [#54873](https://github.com/vllm-project/vllm/pull/54873), [#54890](https://github.com/vllm-project/vllm/pull/54890), [#54915](https://github.com/vllm-project/vllm/pull/54915), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#57097](https://github.com/vllm-project/vllm/pull/57097), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54560](https://github.com/vllm-project/vllm/pull/54560), [#59214](https://github.com/vllm-project/vllm/pull/59214), [#59753](https://github.com/vllm-project/vllm/pull/59753) |
| `vllm/models/qwen4_exp/nvidia/model.py` | [#51289](https://github.com/vllm-project/vllm/pull/51289), [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54371](https://github.com/vllm-project/vllm/pull/54371), [#54517](https://github.com/vllm-project/vllm/pull/54517), [#54687](https://github.com/vllm-project/vllm/pull/54687), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#55309](https://github.com/vllm-project/vllm/pull/55309), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#58957](https://github.com/vllm-project/vllm/pull/58957) |
| `vllm/models/qwen4_exp/nvidia/model_state.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#55272](https://github.com/vllm-project/vllm/pull/55272) |
| `vllm/models/qwen4_exp/nvidia/mtp.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54517](https://github.com/vllm-project/vllm/pull/54517), [#54687](https://github.com/vllm-project/vllm/pull/54687), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | [#54371](https://github.com/vllm-project/vllm/pull/54371), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#57497](https://github.com/vllm-project/vllm/pull/57497), [#58489](https://github.com/vllm-project/vllm/pull/58489) |
| `vllm/models/qwen4_exp/nvidia/ops/__init__.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896) |
| `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/__init__.py` | [#58957](https://github.com/vllm-project/vllm/pull/58957) |
| `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py` | [#58957](https://github.com/vllm-project/vllm/pull/58957) |
| `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py` | [#58957](https://github.com/vllm-project/vllm/pull/58957) |
| `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py` | [#58957](https://github.com/vllm-project/vllm/pull/58957) |
| `vllm/models/qwen4_exp/nvidia/ops/hc.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54687](https://github.com/vllm-project/vllm/pull/54687) |
| `vllm/models/qwen4_exp/nvidia/ops/ple.py` | [#54517](https://github.com/vllm-project/vllm/pull/54517), [#55309](https://github.com/vllm-project/vllm/pull/55309), [#55375](https://github.com/vllm-project/vllm/pull/55375) |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54513](https://github.com/vllm-project/vllm/pull/54513), [#54873](https://github.com/vllm-project/vllm/pull/54873), [#55309](https://github.com/vllm-project/vllm/pull/55309), [#55557](https://github.com/vllm-project/vllm/pull/55557), [#57273](https://github.com/vllm-project/vllm/pull/57273) |
| `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` | [#54513](https://github.com/vllm-project/vllm/pull/54513), [#54873](https://github.com/vllm-project/vllm/pull/54873), [#54890](https://github.com/vllm-project/vllm/pull/54890), [#54915](https://github.com/vllm-project/vllm/pull/54915), [#57105](https://github.com/vllm-project/vllm/pull/57105) |
| `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py` | [#57097](https://github.com/vllm-project/vllm/pull/57097) |
| `vllm/models/qwen4_exp/nvidia/ple_layer.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54371](https://github.com/vllm-project/vllm/pull/54371), [#54517](https://github.com/vllm-project/vllm/pull/54517), [#54722](https://github.com/vllm-project/vllm/pull/54722), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#55309](https://github.com/vllm-project/vllm/pull/55309), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/qwen4_exp/nvidia/qsa.py` | [#53896](https://github.com/vllm-project/vllm/pull/53896), [#54873](https://github.com/vllm-project/vllm/pull/54873), [#55272](https://github.com/vllm-project/vllm/pull/55272), [#55309](https://github.com/vllm-project/vllm/pull/55309), [#55557](https://github.com/vllm-project/vllm/pull/55557), [#57097](https://github.com/vllm-project/vllm/pull/57097), [#57387](https://github.com/vllm-project/vllm/pull/57387) |

## PR 覆盖总览

- git 追溯 PR 数: 26
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 26
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-31 | [#53896](https://github.com/vllm-project/vllm/pull/53896) | merged | [Model] Support Qwen3.8-Flash-Next | `vllm/models/qwen4_exp/nvidia/ple_layer.py`, `vllm/models/qwen4_exp/amd/ple_layer.py`, `vllm/models/qwen4_exp/amd/ops/qsa.py` |
| 2026-09-01 | [#54560](https://github.com/vllm-project/vllm/pull/54560) | merged | [Kernel][Qwen] Add Hopper LL-GEMM tuning table for Qwen4Exp | `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` |
| 2026-09-02 | [#54513](https://github.com/vllm-project/vllm/pull/54513) | merged | [Qwen3.8-Flash-Next] Separate prefill and decode paths for QSA indexer | `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `tests/models/qwen4_exp/test_qsa_reference.py` |
| 2026-09-02 | [#54722](https://github.com/vllm-project/vllm/pull/54722) | merged | [Qwen4] validate FP8 PLE weight scale after loading | `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py` |
| 2026-09-02 | [#54517](https://github.com/vllm-project/vllm/pull/54517) | merged | [Qwen3.8-Flash-Next] Fuse Qwen4Exp PLE kernels | `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py`, `vllm/models/qwen4_exp/nvidia/ops/ple.py` |
| 2026-09-04 | [#54915](https://github.com/vllm-project/vllm/pull/54915) | merged | [Qwen3.8-Flash-Next] Compact indexer logits workspace to improve prefill efficiency | `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`, `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/common/qsa_cache.py` |
| 2026-09-04 | [#54687](https://github.com/vllm-project/vllm/pull/54687) | merged | [Kernel] Reuse Qwen4Exp HC combine-norm for MTP input | `vllm/models/qwen4_exp/nvidia/ops/hc.py`, `tests/models/qwen4_exp/test_hc_ops.py`, `vllm/models/qwen4_exp/nvidia/hyperconnection.py` |
| 2026-09-04 | [#54873](https://github.com/vllm-project/vllm/pull/54873) | merged | [Qwen3.8-Flash-Next] Improve QSA sparse GQA for prefill and short-ctx decode | `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py` |
| 2026-09-05 | [#55375](https://github.com/vllm-project/vllm/pull/55375) | merged | [Bugfix][Qwen4Exp] fix state index strides in fused PLE conv | `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/ops/ple.py` |
| 2026-09-07 | [#55272](https://github.com/vllm-project/vllm/pull/55272) | merged | [Qwen3.8-Flash-Next] Remove torch.compile for NVIDIA implementation | `vllm/models/qwen4_exp/nvidia/qsa.py`, `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py` |
| 2026-09-07 | [#54890](https://github.com/vllm-project/vllm/pull/54890) | merged | [Qwen3.8-Flash-Next] Support FP8 indexer cache for QSA | `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`, `vllm/models/qwen4_exp/common/qsa_cache.py` |
| 2026-09-09 | [#54371](https://github.com/vllm-project/vllm/pull/54371) | merged | [Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py`, `tests/models/qwen4_exp/test_ple.py` |
| 2026-09-14 | [#55309](https://github.com/vllm-project/vllm/pull/55309) | merged | [Qwen3.8-Flash-Next] Fuse PLE residual and QSA output gate | `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py`, `vllm/models/qwen4_exp/nvidia/ops/ple.py` |
| 2026-09-16 | [#55557](https://github.com/vllm-project/vllm/pull/55557) | merged | [Model] Qwen4Exp: fp8_e4m3 main KV cache on the QSA path | `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `vllm/models/qwen4_exp/nvidia/qsa.py`, `tests/models/qwen4_exp/test_qsa_reference.py` |
| 2026-09-17 | [#57273](https://github.com/vllm-project/vllm/pull/57273) | merged | [Perf][Model] Qwen4Exp QSA: sm_90 tuning table for _select_config | `vllm/models/qwen4_exp/nvidia/ops/qsa.py` |
| 2026-09-25 | [#58489](https://github.com/vllm-project/vllm/pull/58489) | merged | [Bugfix][Qwen4Exp] Keep pinned PLE prefetch ids out of the CUDA graph pool | `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` |
| 2026-09-25 | [#57497](https://github.com/vllm-project/vllm/pull/57497) | merged | [Qwen4Exp][ROCm] PLE n-gram table CPU offload | `vllm/models/qwen4_exp/common/ngram_embedding.py`, `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`, `tests/models/qwen4_exp/test_ple.py` |
| 2026-09-27 | [#57105](https://github.com/vllm-project/vllm/pull/57105) | merged | [Qwen3.8-Flash-Next] Avoid memory fragmentation in QSA indexer logits workspace | `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` |
| 2026-09-28 | [#58961](https://github.com/vllm-project/vllm/pull/58961) | merged | [Bugfix][Qwen4Exp] Release the profiling KV cache held by QSA key views | `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/common/qsa_cache.py` |
| 2026-09-28 | [#58957](https://github.com/vllm-project/vllm/pull/58957) | merged | [Perf][Qwen4Exp] Fuse HC down projection and SiLU on NVIDIA | `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py`, `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py`, `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py` |
| 2026-09-29 | [#51289](https://github.com/vllm-project/vllm/pull/51289) | merged | [Model] Extend device-side mm normalization to Qwen3VL/Qwen3.5/Qwen4Next | `vllm/models/qwen4_exp/amd/model.py`, `vllm/models/qwen4_exp/nvidia/model.py` |
| 2026-10-01 | [#59214](https://github.com/vllm-project/vllm/pull/59214) | merged | [Perf][Qwen4Exp] Add SM100 low-latency decode GEMM plans | `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` |
| 2026-10-01 | [#57097](https://github.com/vllm-project/vllm/pull/57097) | merged | [Qwen3.8-Flash-Next] Fuse main QK-norm/RoPE/gate and KV-cache write into the QSA pre-indexer launch | `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py`, `tests/models/qwen4_exp/test_qsa_prepare.py`, `vllm/models/qwen4_exp/nvidia/qsa.py` |
| 2026-10-02 | [#57387](https://github.com/vllm-project/vllm/pull/57387) | merged | [Model] Use upstream GLM-5.3 and Qwen4-Exp configs and processor | `vllm/models/glm5next/common/model.py`, `vllm/models/qwen4_exp/amd/model.py`, `vllm/models/qwen4_exp/nvidia/model.py` |
| 2026-10-02 | [#59753](https://github.com/vllm-project/vllm/pull/59753) | merged | [Perf][Qwen4Exp] Add SM121 TP=1 skinny-GEMM plans | `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` |
| 2026-10-05 | [#59990](https://github.com/vllm-project/vllm/pull/59990) | merged | [Bugfix] Fix Qwen4Exp PLE embedding rejecting INC (AutoRound) checkpoints | `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/common/ngram_embedding.py` |

## 逐 PR diff 审计卡

### PR #53896 - [Model] Support Qwen3.8-Flash-Next

- 链接: https://github.com/vllm-project/vllm/pull/53896
- 状态/时间: merged / 2026-08-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/qwen4_exp/README.md`, `tests/evals/qwen4_exp/__init__.py`, `tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8.yaml`, `tests/evals/qwen4_exp/configs/models-b200.txt`, `tests/evals/qwen4_exp/configs/models-h200.txt` 等 42 个文件；关联提交 `e126687a9a82`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 124 个文件，+19425/-451，可读 patch 22004 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ple_layer.py` added +1256/-0 (1256 lines); hunks: -0,0 +1,1256; symbols: Qwen4ExpPLEGroupedNorm, __init__, forward, Qwen4ExpPLEFp8EmbeddingMethod，涉及 `Qwen4ExpPLEGroupedNorm, __init__, forward`；`vllm/models/qwen4_exp/amd/ple_layer.py` added +1131/-0 (1131 lines); hunks: -0,0 +1,1131; symbols: Qwen4ExpPLEGroupedNorm, __init__, forward, Qwen4ExpNGramEmbedding，涉及 `Qwen4ExpPLEGroupedNorm, __init__, forward`；`vllm/models/qwen4_exp/amd/ops/qsa.py` added +1128/-0 (1128 lines); hunks: -0,0 +1,1128; symbols: _qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel，涉及 `_qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel`；`vllm/models/qwen4_exp/nvidia/ops/qsa.py` added +1115/-0 (1115 lines); hunks: -0,0 +1,1115; symbols: _qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel，涉及 `_qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ple_layer.py` added +1256/-0 (1256 lines); hunks: -0,0 +1,1256; symbols: Qwen4ExpPLEGroupedNorm, __init__, forward, Qwen4ExpPLEFp8EmbeddingMethod
  - `vllm/models/qwen4_exp/amd/ple_layer.py` added +1131/-0 (1131 lines); hunks: -0,0 +1,1131; symbols: Qwen4ExpPLEGroupedNorm, __init__, forward, Qwen4ExpNGramEmbedding
  - `vllm/models/qwen4_exp/amd/ops/qsa.py` added +1128/-0 (1128 lines); hunks: -0,0 +1,1128; symbols: _qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel
  - `vllm/models/qwen4_exp/nvidia/ops/qsa.py` added +1115/-0 (1115 lines); hunks: -0,0 +1,1115; symbols: _qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel
  - `vllm/models/qwen4_exp/amd/model.py` added +1065/-0 (1065 lines); hunks: -0,0 +1,1065; symbols: without_modelopt_fp4, _remap_qsa_cache_scale_name, Qwen4ExpSparseMoeBlock, __init__
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -0,0 +1,1256 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""GPU-resident Qwen4Exp position-learning enhancement layers."""
+import math
+from collections.abc import Iterable, Sequence
+import torch
diff -- vllm/models/qwen4_exp/amd/ple_layer.py
@@ -0,0 +1,1131 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""GPU-resident Qwen4Exp position-learning enhancement layers."""
+import math
+from collections.abc import Iterable, Sequence
+import torch
diff -- vllm/models/qwen4_exp/amd/ops/qsa.py
@@ -0,0 +1,1128 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ple_layer.py` added +1256/-0; `vllm/models/qwen4_exp/amd/ple_layer.py` added +1131/-0; `vllm/models/qwen4_exp/amd/ops/qsa.py` added +1128/-0; `vllm/models/qwen4_exp/nvidia/ops/qsa.py` added +1115/-0; `vllm/models/qwen4_exp/amd/model.py` added +1065/-0; `vllm/models/qwen4_exp/nvidia/model.py` added +1065/-0
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` added +973/-0
- 验证与风险: diff 自带测试面 `tests/config/test_config_utils.py`, `tests/config/test_speculative_draft_hf_overrides.py`, `tests/distributed/test_custom_all_reduce.py`, `tests/evals/qwen4_exp/README.md`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54560 - [Kernel][Qwen] Add Hopper LL-GEMM tuning table for Qwen4Exp

- 链接: https://github.com/vllm-project/vllm/pull/54560
- 状态/时间: merged / 2026-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`；关联提交 `191cecd51e25`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+151/-6，可读 patch 237 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +82/-4 (86 lines); hunks: -1,6 +1,6; -78,11 +78,88; symbols: _is_sm103, _is_sm90, _gemm_plans, _is_packed_row_major，涉及 `_is_sm103, _is_sm90, _gemm_plans`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +82/-4 (86 lines); hunks: -1,6 +1,6; -78,11 +78,88; symbols: _is_sm103, _is_sm90, _gemm_plans, _is_packed_row_major
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/low_latency_gemm.py
@@ -1,6 +1,6 @@
-"""Qwen4Exp decode GEMM selection on Blackwell.
+"""Qwen4Exp decode GEMM selection on Hopper and Blackwell.
@@ -78,11 +78,88 @@
+# H200 plans selected by exhaustive CUDA graph replay measurements over
+# M={1, 2, 4, 8, 16}. Only points that beat the standard linear implementation
+# in both hot-cache and L2-flush measurements are retained; other token counts
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +82/-4
- 验证与风险: diff 自带测试面 `tests/kernels/test_bf16_skinny_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54513 - [Qwen3.8-Flash-Next] Separate prefill and decode paths for QSA indexer

- 链接: https://github.com/vllm-project/vllm/pull/54513
- 状态/时间: merged / 2026-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py`, `vllm/models/qwen4_exp/common/qsa_cache.py`, `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa.py` 等 6 个文件；关联提交 `003e34341a82`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+1105/-523，可读 patch 2064 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` added +617/-0 (617 lines); hunks: -0,0 +1,617; symbols: _qsa_mqa_paged_uniform_kernel, _qsa_mqa_paged_prefill_kernel, _expand_qsa_indices_kernel, _decode_tiles_per_program，涉及 `_qsa_mqa_paged_uniform_kernel, _qsa_mqa_paged_prefill_kernel, _expand_qsa_indices_kernel`；`vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +1/-405 (406 lines); hunks: -1,193 +1,13; -588,227 +408,6 @@ def _compress_qsa_groups_kernel(; symbols: _qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel, _compress_qsa_groups_kernel，涉及 `_qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel`；`tests/models/qwen4_exp/test_qsa_reference.py` modified +252/-83 (335 lines); hunks: -14,6 +14,7; -67,8 +68,13 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; symbols: test_qsa_mtp_index_share_updates_cache_but_skips_selection, _expand_qsa_indices_reference, _qsa_select_paged_tokens_reference, _qsa_select_paged_reference，涉及 `test_qsa_mtp_index_share_updates_cache_but_skips_selection, _expand_qsa_indices_reference, _qsa_select_paged_tokens_reference`；`vllm/models/qwen4_exp/common/qsa_cache.py` modified +86/-20 (106 lines); hunks: -11,8 +11,6; -34,7 +32,10; symbols: _build_qsa_metadata_kernel, build_qsa_metadata_triton，涉及 `_build_qsa_metadata_kernel, build_qsa_metadata_triton`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` added +617/-0 (617 lines); hunks: -0,0 +1,617; symbols: _qsa_mqa_paged_uniform_kernel, _qsa_mqa_paged_prefill_kernel, _expand_qsa_indices_kernel, _decode_tiles_per_program
  - `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +1/-405 (406 lines); hunks: -1,193 +1,13; -588,227 +408,6 @@ def _compress_qsa_groups_kernel(; symbols: _qsa_mqa_paged_kernel, _expand_qsa_indices_kernel, _qsa_sparse_paged_gqa_splitk_kernel, _compress_qsa_groups_kernel
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +252/-83 (335 lines); hunks: -14,6 +14,7; -67,8 +68,13 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; symbols: test_qsa_mtp_index_share_updates_cache_but_skips_selection, _expand_qsa_indices_reference, _qsa_select_paged_tokens_reference, _qsa_select_paged_reference
  - `vllm/models/qwen4_exp/common/qsa_cache.py` modified +86/-20 (106 lines); hunks: -11,8 +11,6; -34,7 +32,10; symbols: _build_qsa_metadata_kernel, build_qsa_metadata_triton
  - `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +79/-15 (94 lines); hunks: -2,8 +2,6; -185,6 +183,22 @@ def _metadata(; symbols: _metadata, forward
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
@@ -0,0 +1,617 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Triton kernels for Qwen4Exp QSA index selection."""
+import torch
+import vllm.envs as envs
+from vllm.model_executor.warmup.jit_warmup_triton_helper import (
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -1,193 +1,13 @@
-"""Triton kernels for the Qwen4Exp weight-free QSA path."""
+"""Triton kernels for Qwen4Exp QSA sparse attention and cache updates."""
-import math
-from vllm.platforms import current_platform
-_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
-_TOPK_WORKSPACE_BYTES = 1024 * 1024
diff -- tests/models/qwen4_exp/test_qsa_reference.py
@@ -14,6 +14,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` added +617/-0; `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +1/-405; `vllm/models/qwen4_exp/common/qsa_cache.py` modified +86/-20; `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +79/-15; `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py` added +66/-0
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` modified +252/-83
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54722 - [Qwen4] validate FP8 PLE weight scale after loading

- 链接: https://github.com/vllm-project/vllm/pull/54722
- 状态/时间: merged / 2026-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py`；关联提交 `1e300895acec`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+26/-5，可读 patch 84 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/qwen4_exp/test_ple.py` modified +19/-4 (23 lines); hunks: -72,7 +72,7 @@ def _make_fp8_ngram_embedding_for_load_test() -> Qwen4ExpNGram...; -156,11 +156,17 @@ def test_ngram_embedding_loads_fp8_shards_and_global_scale...; symbols: _make_fp8_ngram_embedding_for_load_test, test_ngram_embedding_loads_fp8_shards_and_global_scale, _make_fp8_embedding_layer, test_ple_fp8_embedding_dequantizes_in_ple_layer，涉及 `_make_fp8_ngram_embedding_for_load_test, test_ngram_embedding_loads_fp8_shards_and_global_scale, _make_fp8_embedding_layer`；`vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +7/-1 (8 lines); hunks: -106,10 +106,16 @@ def create_weights(; symbols: create_weights, process_weights_after_loading, apply，涉及 `create_weights, process_weights_after_loading, apply`。
- 代码 diff 细节:
  - `tests/models/qwen4_exp/test_ple.py` modified +19/-4 (23 lines); hunks: -72,7 +72,7 @@ def _make_fp8_ngram_embedding_for_load_test() -> Qwen4ExpNGram...; -156,11 +156,17 @@ def test_ngram_embedding_loads_fp8_shards_and_global_scale...; symbols: _make_fp8_ngram_embedding_for_load_test, test_ngram_embedding_loads_fp8_shards_and_global_scale, _make_fp8_embedding_layer, test_ple_fp8_embedding_dequantizes_in_ple_layer
  - `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +7/-1 (8 lines); hunks: -106,10 +106,16 @@ def create_weights(; symbols: create_weights, process_weights_after_loading, apply
- 关键代码摘录:

```diff
diff -- tests/models/qwen4_exp/test_ple.py
@@ -72,7 +72,7 @@ def _make_fp8_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
-        nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False),
+        nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False),
@@ -156,11 +156,17 @@ def test_ngram_embedding_loads_fp8_shards_and_global_scale() -> None:
-    assert torch.equal(module.ngram_embedding.weight_scale, weight_scale)
+    assert module.ngram_embedding.weight_scale.dtype == torch.float32
+    torch.testing.assert_close(
diff -- vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -106,10 +106,16 @@ def create_weights(
-            scale_dtype=torch.bfloat16,
+            scale_dtype=torch.float32,
+    def process_weights_after_loading(self, layer: nn.Module) -> None:
+        """Reject FP8 PLE checkpoints without a global scale."""
+        sentinel = torch.finfo(torch.float32).min
+        if torch.any(layer.weight_scale == sentinel):
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +19/-4
  - runtime: `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +7/-1
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_ple.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54517 - [Qwen3.8-Flash-Next] Fuse Qwen4Exp PLE kernels

- 链接: https://github.com/vllm-project/vllm/pull/54517
- 状态/时间: merged / 2026-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/model.py`, `vllm/models/qwen4_exp/nvidia/mtp.py`, `vllm/models/qwen4_exp/nvidia/ops/ple.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py`；关联提交 `f870b9297685`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+1836/-594，可读 patch 2752 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/qwen4_exp/test_ple.py` modified +983/-65 (1048 lines); hunks: -1,7 +1,9; -17,13 +19,16; symbols: _make_ngram_embedding_for_load_test, test_ple_fp8_embedding_respects_checkpoint_shard_exclusions, test_ple_ngram_ids_custom_op_uses_current_request_layout, RuntimeNGramEmbedding，涉及 `_make_ngram_embedding_for_load_test, test_ple_fp8_embedding_respects_checkpoint_shard_exclusions, test_ple_ngram_ids_custom_op_uses_current_request_layout`；`vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +173/-516 (689 lines); hunks: -2,7 +2,6; -11,7 +10,7; symbols: Qwen4ExpPLEGroupedNorm, __init__, _shift_precompute, compute_ngram_ids，涉及 `Qwen4ExpPLEGroupedNorm, __init__, _shift_precompute`；`vllm/models/qwen4_exp/nvidia/ops/ple.py` added +669/-0 (669 lines); hunks: -0,0 +1,669; symbols: _ple_ngram_ids_kernel, _ple_ngram_ids, ple_ngram_ids, _ple_gate_kernel，涉及 `_ple_ngram_ids_kernel, _ple_ngram_ids, ple_ngram_ids`；`vllm/models/qwen4_exp/nvidia/model.py` modified +9/-5 (14 lines); hunks: -141,9 +141,9 @@ def _remap_qsa_cache_scale_name(; -153,6 +153,8 @@ def _remap_qsa_cache_scale_name(; symbols: _remap_qsa_cache_scale_name, update_physical_experts_metadata, Qwen4ExpModel, __init__，涉及 `_remap_qsa_cache_scale_name, update_physical_experts_metadata, Qwen4ExpModel`。
- 代码 diff 细节:
  - `tests/models/qwen4_exp/test_ple.py` modified +983/-65 (1048 lines); hunks: -1,7 +1,9; -17,13 +19,16; symbols: _make_ngram_embedding_for_load_test, test_ple_fp8_embedding_respects_checkpoint_shard_exclusions, test_ple_ngram_ids_custom_op_uses_current_request_layout, RuntimeNGramEmbedding
  - `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +173/-516 (689 lines); hunks: -2,7 +2,6; -11,7 +10,7; symbols: Qwen4ExpPLEGroupedNorm, __init__, _shift_precompute, compute_ngram_ids
  - `vllm/models/qwen4_exp/nvidia/ops/ple.py` added +669/-0 (669 lines); hunks: -0,0 +1,669; symbols: _ple_ngram_ids_kernel, _ple_ngram_ids, ple_ngram_ids, _ple_gate_kernel
  - `vllm/models/qwen4_exp/nvidia/model.py` modified +9/-5 (14 lines); hunks: -141,9 +141,9 @@ def _remap_qsa_cache_scale_name(; -153,6 +153,8 @@ def _remap_qsa_cache_scale_name(; symbols: _remap_qsa_cache_scale_name, update_physical_experts_metadata, Qwen4ExpModel, __init__
  - `vllm/models/qwen4_exp/nvidia/mtp.py` modified +2/-2 (4 lines); hunks: -52,7 +52,7; -157,7 +157,7 @@ def _make_draft_vllm_config(; symbols: _make_draft_vllm_config, Qwen4ExpMultiTokenPredictor, __init__
- 关键代码摘录:

```diff
diff -- tests/models/qwen4_exp/test_ple.py
@@ -1,7 +1,9 @@
+from dataclasses import dataclass
+from itertools import accumulate
@@ -17,13 +19,16 @@
-from vllm.models.qwen4_exp.nvidia import ple_layer as ple_layer_module
+from vllm.v1.attention.backends.short_conv_attn import (
+    PleShortConvAttentionMetadata,
diff -- vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -2,7 +2,6 @@
-import math
@@ -11,7 +10,7 @@
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import MergedColumnParallelLinear
@@ -42,9 +41,9 @@
-from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
diff -- vllm/models/qwen4_exp/nvidia/ops/ple.py
@@ -0,0 +1,669 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +983/-65
  - runtime: `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +173/-516; `vllm/models/qwen4_exp/nvidia/ops/ple.py` added +669/-0; `vllm/models/qwen4_exp/nvidia/model.py` modified +9/-5; `vllm/models/qwen4_exp/nvidia/mtp.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_ple.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54915 - [Qwen3.8-Flash-Next] Compact indexer logits workspace to improve prefill efficiency

- 链接: https://github.com/vllm-project/vllm/pull/54915
- 状态/时间: merged / 2026-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/common/qsa_cache.py`, `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`；关联提交 `a5c9179e731e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+46/-16，可读 patch 239 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +23/-13 (36 lines); hunks: -97,7 +97,7 @@ def _qsa_mqa_paged_uniform_kernel(; -123,16 +123,16 @@ def _qsa_mqa_paged_prefill_kernel(; symbols: _qsa_mqa_paged_uniform_kernel, _qsa_mqa_paged_prefill_kernel，涉及 `_qsa_mqa_paged_uniform_kernel, _qsa_mqa_paged_prefill_kernel`；`tests/models/qwen4_exp/test_qsa_reference.py` modified +20/-3 (23 lines); hunks: -244,6 +244,7 @@ def test_qsa_side_metadata_marks_cudagraph_padding_inert() -...; -297,6 +298,7 @@ def test_qsa_circular_buffer_metadata_keeps_only_each_reques...; symbols: test_qsa_side_metadata_marks_cudagraph_padding_inert, test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix, test_qsa_compressed_metadata_keeps_dummy_slots_inert, test_qsa_decode_selection_correctness，涉及 `test_qsa_side_metadata_marks_cudagraph_padding_inert, test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix, test_qsa_compressed_metadata_keeps_dummy_slots_inert`；`vllm/models/qwen4_exp/common/qsa_cache.py` modified +2/-0 (2 lines); hunks: -591,6 +591,7 @@ class QSAForwardMetadata(AttentionMetadata):; -717,6 +718,7 @@ def build(; symbols: QSAForwardMetadata, build，涉及 `QSAForwardMetadata, build`；`vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +1/-0 (1 lines); hunks: -405,6 +405,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +23/-13 (36 lines); hunks: -97,7 +97,7 @@ def _qsa_mqa_paged_uniform_kernel(; -123,16 +123,16 @@ def _qsa_mqa_paged_prefill_kernel(; symbols: _qsa_mqa_paged_uniform_kernel, _qsa_mqa_paged_prefill_kernel
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +20/-3 (23 lines); hunks: -244,6 +244,7 @@ def test_qsa_side_metadata_marks_cudagraph_padding_inert() -...; -297,6 +298,7 @@ def test_qsa_circular_buffer_metadata_keeps_only_each_reques...; symbols: test_qsa_side_metadata_marks_cudagraph_padding_inert, test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix, test_qsa_compressed_metadata_keeps_dummy_slots_inert, test_qsa_decode_selection_correctness
  - `vllm/models/qwen4_exp/common/qsa_cache.py` modified +2/-0 (2 lines); hunks: -591,6 +591,7 @@ class QSAForwardMetadata(AttentionMetadata):; -717,6 +718,7 @@ def build(; symbols: QSAForwardMetadata, build
  - `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +1/-0 (1 lines); hunks: -405,6 +405,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
@@ -97,7 +97,7 @@ def _qsa_mqa_paged_uniform_kernel(
-        score = tl.sum(scores, axis=2) / HEAD_DIM**0.5
+        score = tl.sum(scores, axis=2)
@@ -123,16 +123,16 @@ def _qsa_mqa_paged_prefill_kernel(
+    page_table_width,
-    PAGE_TABLE_WIDTH: tl.constexpr,
-    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
diff -- tests/models/qwen4_exp/test_qsa_reference.py
@@ -244,6 +244,7 @@ def test_qsa_side_metadata_marks_cudagraph_padding_inert() -> None:
+        max_seq_len=68,
@@ -297,6 +298,7 @@ def test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix() -> None:
+        max_seq_len=11,
@@ -428,6 +430,7 @@ def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
+        max_seq_len=12,
@@ -673,13 +676,24 @@ def test_qsa_decode_selection_correctness(
diff -- vllm/models/qwen4_exp/common/qsa_cache.py
@@ -591,6 +591,7 @@ class QSAForwardMetadata(AttentionMetadata):
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +23/-13; `vllm/models/qwen4_exp/common/qsa_cache.py` modified +2/-0; `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +1/-0
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` modified +20/-3
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54687 - [Kernel] Reuse Qwen4Exp HC combine-norm for MTP input

- 链接: https://github.com/vllm-project/vllm/pull/54687
- 状态/时间: merged / 2026-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_hc_ops.py`, `vllm/models/qwen4_exp/nvidia/hyperconnection.py`, `vllm/models/qwen4_exp/nvidia/model.py`, `vllm/models/qwen4_exp/nvidia/mtp.py`, `vllm/models/qwen4_exp/nvidia/ops/hc.py`；关联提交 `fd4a1512628a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+93/-40，可读 patch 326 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/hc.py` modified +45/-28 (73 lines); hunks: -215,14 +215,17 @@ def _hc_combine_kernel(; -232,17 +235,20 @@ def _hc_combine_kernel(; symbols: _hc_combine_kernel, _hc_combine, _hc_combine_norm_kernel，涉及 `_hc_combine_kernel, _hc_combine, _hc_combine_norm_kernel`；`tests/models/qwen4_exp/test_hc_ops.py` modified +33/-0 (33 lines); hunks: -68,6 +68,18 @@ def test_hc_combine() -> None:; -92,3 +104,24 @@ def test_hc_combine_norm() -> None:; symbols: test_hc_combine, test_hc_combine_unit_injection, test_hc_combine_norm, test_hc_combine_norm_unit_injection，涉及 `test_hc_combine, test_hc_combine_unit_injection, test_hc_combine_norm`；`vllm/models/qwen4_exp/nvidia/hyperconnection.py` modified +8/-6 (14 lines); hunks: -52,9 +52,10 @@ class GatedResidual(nn.Module):; -153,13 +154,14 @@ def combine_and_mix(; symbols: GatedResidual, combine_and_mix, combine，涉及 `GatedResidual, combine_and_mix, combine`；`vllm/models/qwen4_exp/nvidia/mtp.py` modified +3/-4 (7 lines); hunks: -280,6 +280,7 @@ def forward(; -300,10 +301,8 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/hc.py` modified +45/-28 (73 lines); hunks: -215,14 +215,17 @@ def _hc_combine_kernel(; -232,17 +235,20 @@ def _hc_combine_kernel(; symbols: _hc_combine_kernel, _hc_combine, _hc_combine_norm_kernel
  - `tests/models/qwen4_exp/test_hc_ops.py` modified +33/-0 (33 lines); hunks: -68,6 +68,18 @@ def test_hc_combine() -> None:; -92,3 +104,24 @@ def test_hc_combine_norm() -> None:; symbols: test_hc_combine, test_hc_combine_unit_injection, test_hc_combine_norm, test_hc_combine_norm_unit_injection
  - `vllm/models/qwen4_exp/nvidia/hyperconnection.py` modified +8/-6 (14 lines); hunks: -52,9 +52,10 @@ class GatedResidual(nn.Module):; -153,13 +154,14 @@ def combine_and_mix(; symbols: GatedResidual, combine_and_mix, combine
  - `vllm/models/qwen4_exp/nvidia/mtp.py` modified +3/-4 (7 lines); hunks: -280,6 +280,7 @@ def forward(; -300,10 +301,8 @@ def forward(; symbols: forward
  - `vllm/models/qwen4_exp/nvidia/model.py` modified +4/-2 (6 lines); hunks: -284,11 +284,13 @@ def forward(; -304,7 +306,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/hc.py
@@ -215,14 +215,17 @@ def _hc_combine_kernel(
-    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
+    if inj_ptr is not None:
+        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
-    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
-    out = res.to(tl.float32) + block.to(tl.float32)[None, :] * inj[:, None]
+    if inj_ptr is not None:
diff -- tests/models/qwen4_exp/test_hc_ops.py
@@ -68,6 +68,18 @@ def test_hc_combine() -> None:
+def test_hc_combine_unit_injection() -> None:
+    torch.manual_seed(0)
+    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
+    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
+    actual = hc_combine(residual, block_output, None, HC)
+    expected = residual.unflatten(-1, (HC, HIDDEN_SIZE))
diff -- vllm/models/qwen4_exp/nvidia/hyperconnection.py
@@ -52,9 +52,10 @@ class GatedResidual(nn.Module):
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/hc.py` modified +45/-28; `vllm/models/qwen4_exp/nvidia/hyperconnection.py` modified +8/-6; `vllm/models/qwen4_exp/nvidia/mtp.py` modified +3/-4; `vllm/models/qwen4_exp/nvidia/model.py` modified +4/-2
  - tests: `tests/models/qwen4_exp/test_hc_ops.py` modified +33/-0
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_hc_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54873 - [Qwen3.8-Flash-Next] Improve QSA sparse GQA for prefill and short-ctx decode

- 链接: https://github.com/vllm-project/vllm/pull/54873
- 状态/时间: merged / 2026-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py`, `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` 等 6 个文件；关联提交 `31a8a2666227`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+366/-107，可读 patch 826 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +202/-44 (246 lines); hunks: -6,10 +6,14; -52,6 +56,12 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; symbols: _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel，涉及 `_qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel`；`tests/models/qwen4_exp/test_qsa_reference.py` modified +65/-26 (91 lines); hunks: -754,7 +754,10 @@ def test_qsa_block_expansion_correctness() -> None:; -771,30 +774,47 @@ def test_qsa_block_expansion_correctness() -> None:; symbols: test_qsa_block_expansion_correctness, test_qsa_sparse_paged_attention_correctness，涉及 `test_qsa_block_expansion_correctness, test_qsa_sparse_paged_attention_correctness`；`vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py` modified +47/-24 (71 lines); hunks: -5,6 +5,8; -15,37 +17,41; symbols: qwen4_exp_qsa_triton_warmup, block_table_for，涉及 `qwen4_exp_qsa_triton_warmup, block_table_for`；`vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +24/-4 (28 lines); hunks: -167,8 +167,19 @@ def __init__(; -211,19 +222,28 @@ def forward(; symbols: __init__, output_width, packed_output_width, _metadata，涉及 `__init__, output_width, packed_output_width`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +202/-44 (246 lines); hunks: -6,10 +6,14; -52,6 +56,12 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; symbols: _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +65/-26 (91 lines); hunks: -754,7 +754,10 @@ def test_qsa_block_expansion_correctness() -> None:; -771,30 +774,47 @@ def test_qsa_block_expansion_correctness() -> None:; symbols: test_qsa_block_expansion_correctness, test_qsa_sparse_paged_attention_correctness
  - `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py` modified +47/-24 (71 lines); hunks: -5,6 +5,8; -15,37 +17,41; symbols: qwen4_exp_qsa_triton_warmup, block_table_for
  - `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +24/-4 (28 lines); hunks: -167,8 +167,19 @@ def __init__(; -211,19 +222,28 @@ def forward(; symbols: __init__, output_width, packed_output_width, _metadata
  - `vllm/models/qwen4_exp/nvidia/qsa.py` modified +15/-8 (23 lines); hunks: -30,7 +30,6; -124,6 +123,7 @@ def forward_qsa(; symbols: forward_qsa, __init__
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -6,10 +6,14 @@
+from vllm.model_executor.warmup.jit_warmup_triton_helper import (
+    TritonWarmupTensor,
+    triton_scalar_specialization_rep,
+)
-@triton.jit
+@triton.jit(do_not_specialize=["num_rows", "num_requests"])
diff -- tests/models/qwen4_exp/test_qsa_reference.py
@@ -754,7 +754,10 @@ def test_qsa_block_expansion_correctness() -> None:
-    actual = torch.empty((2, 11), device="cuda", dtype=torch.int32)
+    # Packed layout: one trailing column per row holds the valid-entry count
+    # (never a token index). Row 0: 1 visible block + 2 tail; row 1: 2 blocks
+    # + 3 tail.
+    actual = torch.empty((2, 12), device="cuda", dtype=torch.int32)
@@ -771,30 +774,47 @@ def test_qsa_block_expansion_correctness() -> None:
diff -- vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py
@@ -5,6 +5,8 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +202/-44; `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py` modified +47/-24; `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +24/-4; `vllm/models/qwen4_exp/nvidia/qsa.py` modified +15/-8; `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +13/-1
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` modified +65/-26
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #55375 - [Bugfix][Qwen4Exp] fix state index strides in fused PLE conv

- 链接: https://github.com/vllm-project/vllm/pull/55375
- 状态/时间: merged / 2026-09-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/ops/ple.py`；关联提交 `28e605fb3300`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+31/-3，可读 patch 113 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/qwen4_exp/test_ple.py` modified +23/-1 (24 lines); hunks: -1077,6 +1077,8 @@ class _ConvBatchCase:; -1123,9 +1125,18 @@ def _make_conv_metadata(; symbols: _ConvBatchCase, _make_conv_metadata, test_fused_conv_correctness，涉及 `_ConvBatchCase, _make_conv_metadata, test_fused_conv_correctness`；`vllm/models/qwen4_exp/nvidia/ops/ple.py` modified +8/-2 (10 lines); hunks: -330,6 +330,7 @@ def _ple_conv_kernel(; -386,7 +387,7 @@ def _ple_conv_kernel(; symbols: _ple_conv_kernel, _ple_conv_writeback_kernel, ple_conv，涉及 `_ple_conv_kernel, _ple_conv_writeback_kernel, ple_conv`。
- 代码 diff 细节:
  - `tests/models/qwen4_exp/test_ple.py` modified +23/-1 (24 lines); hunks: -1077,6 +1077,8 @@ class _ConvBatchCase:; -1123,9 +1125,18 @@ def _make_conv_metadata(; symbols: _ConvBatchCase, _make_conv_metadata, test_fused_conv_correctness
  - `vllm/models/qwen4_exp/nvidia/ops/ple.py` modified +8/-2 (10 lines); hunks: -330,6 +330,7 @@ def _ple_conv_kernel(; -386,7 +387,7 @@ def _ple_conv_kernel(; symbols: _ple_conv_kernel, _ple_conv_writeback_kernel, ple_conv
- 关键代码摘录:

```diff
diff -- tests/models/qwen4_exp/test_ple.py
@@ -1077,6 +1077,8 @@ class _ConvBatchCase:
+    state_index_stride: int = 1
+    include_null_state: bool = True
@@ -1123,9 +1125,18 @@ def _make_conv_metadata(
+    if case.state_index_stride > 1:
+        strided_indices = torch.full(
+            (non_spec_state_indices.numel(), case.state_index_stride),
diff -- vllm/models/qwen4_exp/nvidia/ops/ple.py
@@ -330,6 +330,7 @@ def _ple_conv_kernel(
+    state_idx_stride,
@@ -386,7 +387,7 @@ def _ple_conv_kernel(
-    sid = tl.load(state_idx_ptr + r).to(tl.int64)
+    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
@@ -484,6 +485,7 @@ def _ple_conv_writeback_kernel(
+    state_idx_stride,
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +23/-1
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/ple.py` modified +8/-2
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_ple.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #55272 - [Qwen3.8-Flash-Next] Remove torch.compile for NVIDIA implementation

- 链接: https://github.com/vllm-project/vllm/pull/55272
- 状态/时间: merged / 2026-09-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_config.py`, `tests/models/qwen4_exp/test_ple.py`, `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`, `vllm/models/qwen4_exp/nvidia/model.py` 等 9 个文件；关联提交 `d9105ea8001e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+165/-185，可读 patch 669 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/qsa.py` modified +14/-60 (74 lines); hunks: -9,6 +9,7; -27,10 +28,6; symbols: get_kv_cache_spec, _run_qsa, forward, qwen4_exp_qsa_with_output，涉及 `get_kv_cache_spec, _run_qsa, forward`；`tests/models/qwen4_exp/test_qsa_reference.py` modified +66/-2 (68 lines); hunks: -45,7 +45,6 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; -80,7 +79,7 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; symbols: test_qsa_mtp_index_share_updates_cache_but_skips_selection, test_qsa_compressed_metadata_keeps_dummy_slots_inert, test_qsa_unfused_cache_update_ignores_padded_qk，涉及 `test_qsa_mtp_index_share_updates_cache_but_skips_selection, test_qsa_compressed_metadata_keeps_dummy_slots_inert, test_qsa_unfused_cache_update_ignores_padded_qk`；`vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +5/-61 (66 lines); hunks: -8,6 +8,7; -38,7 +39,6; symbols: __init__, forward, load_weights, _short_conv_dilated_dispatch，涉及 `__init__, forward, load_weights`；`tests/models/qwen4_exp/test_config.py` modified +37/-14 (51 lines); hunks: -147,9 +147,9 @@ def test_qwen4_exp_model_state_prepares_ngram_context() -> N...; -167,34 +167,57 @@ def test_qwen4_exp_model_state_prepares_ngram_context() ->...; symbols: test_qwen4_exp_model_state_prepares_ngram_context, test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs，涉及 `test_qwen4_exp_model_state_prepares_ngram_context, test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/qsa.py` modified +14/-60 (74 lines); hunks: -9,6 +9,7; -27,10 +28,6; symbols: get_kv_cache_spec, _run_qsa, forward, qwen4_exp_qsa_with_output
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +66/-2 (68 lines); hunks: -45,7 +45,6 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; -80,7 +79,7 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; symbols: test_qsa_mtp_index_share_updates_cache_but_skips_selection, test_qsa_compressed_metadata_keeps_dummy_slots_inert, test_qsa_unfused_cache_update_ignores_padded_qk
  - `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +5/-61 (66 lines); hunks: -8,6 +8,7; -38,7 +39,6; symbols: __init__, forward, load_weights, _short_conv_dilated_dispatch
  - `tests/models/qwen4_exp/test_config.py` modified +37/-14 (51 lines); hunks: -147,9 +147,9 @@ def test_qwen4_exp_model_state_prepares_ngram_context() -> N...; -167,34 +167,57 @@ def test_qwen4_exp_model_state_prepares_ngram_context() ->...; symbols: test_qwen4_exp_model_state_prepares_ngram_context, test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs
  - `vllm/models/qwen4_exp/nvidia/mtp.py` modified +0/-19 (19 lines); hunks: -19,7 +19,6; -147,15 +146,6 @@ def _make_draft_vllm_config(; symbols: _make_draft_vllm_config, Qwen4ExpMultiTokenPredictor, load_weights, Qwen4ExpMTP
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/qsa.py
@@ -9,6 +9,7 @@
+from vllm.compilation.breakable_cudagraph import eager_break_during_capture
@@ -27,10 +28,6 @@
-    LayerNameType,
-    _encode_layer_name,
-    _resolve_layer_name,
-    direct_register_custom_op,
diff -- tests/models/qwen4_exp/test_qsa_reference.py
@@ -45,7 +45,6 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(
-        index_qk_proj=lambda hidden: (torch.zeros(2, 2), None),
@@ -80,7 +79,7 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(
-        torch.zeros(2, 4),
+        torch.zeros(2, 2),
@@ -446,6 +445,71 @@ def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
+@requires_qsa_kernels
diff -- vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -8,6 +8,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/qsa.py` modified +14/-60; `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +5/-61; `vllm/models/qwen4_exp/nvidia/mtp.py` modified +0/-19; `vllm/models/qwen4_exp/nvidia/model_state.py` modified +11/-7; `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +6/-8
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` modified +66/-2; `tests/models/qwen4_exp/test_config.py` modified +37/-14; `tests/models/qwen4_exp/test_ple.py` modified +12/-0
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_config.py`, `tests/models/qwen4_exp/test_ple.py`, `tests/models/qwen4_exp/test_qsa_reference.py`, `tests/test_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54890 - [Qwen3.8-Flash-Next] Support FP8 indexer cache for QSA

- 链接: https://github.com/vllm-project/vllm/pull/54890
- 状态/时间: merged / 2026-09-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/common/qsa_cache.py`, `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`；关联提交 `94e26dd3dd7d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+142/-24，可读 patch 369 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/qwen4_exp/test_qsa_reference.py` modified +61/-6 (67 lines); hunks: -48,6 +48,7 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; -484,6 +485,7 @@ def test_qsa_unfused_cache_update_ignores_padded_qk() -> None:; symbols: test_qsa_mtp_index_share_updates_cache_but_skips_selection, test_qsa_unfused_cache_update_ignores_padded_qk, make_buffers, test_qsa_decode_selection_correctness，涉及 `test_qsa_mtp_index_share_updates_cache_but_skips_selection, test_qsa_unfused_cache_update_ignores_padded_qk, make_buffers`；`vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +17/-6 (23 lines); hunks: -200,6 +200,9 @@ def _qsa_mqa_paged_prefill_kernel(; -337,7 +340,7 @@ def warmup_qsa_mqa_paged_decode(; symbols: _qsa_mqa_paged_prefill_kernel, warmup_qsa_mqa_paged_decode, _prefill_logits，涉及 `_qsa_mqa_paged_prefill_kernel, warmup_qsa_mqa_paged_decode, _prefill_logits`；`vllm/models/qwen4_exp/common/qsa_cache.py` modified +15/-5 (20 lines); hunks: -725,10 +725,16 @@ def build(; -790,8 +796,12 @@ def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:; symbols: build, QSAStateBackend, get_name, bind_kv_cache，涉及 `build, QSAStateBackend, get_name`；`vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +18/-1 (19 lines); hunks: -147,6 +147,21 @@ def __init__(; -158,7 +173,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +61/-6 (67 lines); hunks: -48,6 +48,7 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; -484,6 +485,7 @@ def test_qsa_unfused_cache_update_ignores_padded_qk() -> None:; symbols: test_qsa_mtp_index_share_updates_cache_but_skips_selection, test_qsa_unfused_cache_update_ignores_padded_qk, make_buffers, test_qsa_decode_selection_correctness
  - `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +17/-6 (23 lines); hunks: -200,6 +200,9 @@ def _qsa_mqa_paged_prefill_kernel(; -337,7 +340,7 @@ def warmup_qsa_mqa_paged_decode(; symbols: _qsa_mqa_paged_prefill_kernel, warmup_qsa_mqa_paged_decode, _prefill_logits
  - `vllm/models/qwen4_exp/common/qsa_cache.py` modified +15/-5 (20 lines); hunks: -725,10 +725,16 @@ def build(; -790,8 +796,12 @@ def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:; symbols: build, QSAStateBackend, get_name, bind_kv_cache
  - `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +18/-1 (19 lines); hunks: -147,6 +147,21 @@ def __init__(; -158,7 +173,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tests/models/qwen4_exp/test_qsa_reference.py
@@ -48,6 +48,7 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(
+        indexer_dtype=torch.bfloat16,
@@ -484,6 +485,7 @@ def test_qsa_unfused_cache_update_ignores_padded_qk() -> None:
+        indexer_dtype=torch.bfloat16,
@@ -666,13 +668,16 @@ def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tens
+@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
-    decode_query_len: int, num_requests: int
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
@@ -200,6 +200,9 @@ def _qsa_mqa_paged_prefill_kernel(
+        # With an fp8 cache, SM90 lowers this dot to reduced-precision wgmma
+        # accumulation; SM100 tcgen05 accumulates in fp32. The SM90 error
+        # (~1e-4 relative) is far below the e4m3 quantization noise floor.
@@ -337,7 +340,7 @@ def warmup_qsa_mqa_paged_decode(
-            torch.bfloat16,
+            k_cache.dtype,
diff -- vllm/models/qwen4_exp/common/qsa_cache.py
@@ -725,10 +725,16 @@ def build(
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` modified +61/-6
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +17/-6; `vllm/models/qwen4_exp/common/qsa_cache.py` modified +15/-5; `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +18/-1
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_qsa_pre_indexer.py`, `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54371 - [Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism

- 链接: https://github.com/vllm-project/vllm/pull/54371
- 状态/时间: merged / 2026-09-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/model.py`, `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py`；关联提交 `3116c5d06bfe`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+1712/-495，可读 patch 2601 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` added +939/-0 (939 lines); hunks: -0,0 +1,939; symbols: Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight, dequantize，涉及 `Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight`；`vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +30/-467 (497 lines); hunks: -2,10 +2,9; -18,24 +17,6; symbols: Qwen4ExpPLEGroupedNorm, forward, Qwen4ExpPLEFp8EmbeddingMethod, create_weights，涉及 `Qwen4ExpPLEGroupedNorm, forward, Qwen4ExpPLEFp8EmbeddingMethod`；`tests/models/qwen4_exp/test_ple.py` modified +327/-21 (348 lines); hunks: -13,27 +13,50; -171,11 +194,57 @@ def test_ngram_embedding_loads_fp8_shards_and_global_scale...; symbols: _mock_etp_group, _make_ngram_embedding_for_load_test, test_ngram_embedding_loads_fp8_shards_and_global_scale, test_etp_lookup_gathers_and_returns_dp_local_rows，涉及 `_mock_etp_group, _make_ngram_embedding_for_load_test, test_ngram_embedding_loads_fp8_shards_and_global_scale`；`vllm/models/qwen4_exp/nvidia/model.py` modified +37/-0 (37 lines); hunks: -461,6 +461,27 @@ def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:; -487,10 +508,26 @@ def forward(; symbols: get_layer, embed_input_ids, _start_layer_ple_prefetch, forward，涉及 `get_layer, embed_input_ids, _start_layer_ple_prefetch`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` added +939/-0 (939 lines); hunks: -0,0 +1,939; symbols: Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight, dequantize
  - `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +30/-467 (497 lines); hunks: -2,10 +2,9; -18,24 +17,6; symbols: Qwen4ExpPLEGroupedNorm, forward, Qwen4ExpPLEFp8EmbeddingMethod, create_weights
  - `tests/models/qwen4_exp/test_ple.py` modified +327/-21 (348 lines); hunks: -13,27 +13,50; -171,11 +194,57 @@ def test_ngram_embedding_loads_fp8_shards_and_global_scale...; symbols: _mock_etp_group, _make_ngram_embedding_for_load_test, test_ngram_embedding_loads_fp8_shards_and_global_scale, test_etp_lookup_gathers_and_returns_dp_local_rows
  - `vllm/models/qwen4_exp/nvidia/model.py` modified +37/-0 (37 lines); hunks: -461,6 +461,27 @@ def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:; -487,10 +508,26 @@ def forward(; symbols: get_layer, embed_input_ids, _start_layer_ple_prefetch, forward
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ngram_embedding.py
@@ -0,0 +1,939 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Qwen4Exp n-gram embeddings with device and pinned-host storage."""
+from abc import ABC, abstractmethod
+from collections.abc import Iterable
+from typing import ClassVar
diff -- vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -2,10 +2,9 @@
-from collections.abc import Iterable, Sequence
+from collections.abc import Sequence
-import torch.nn.functional as F
@@ -18,24 +17,6 @@
-from vllm.model_executor.layers.quantization.base_config import (
-    QuantizationConfig,
diff -- tests/models/qwen4_exp/test_ple.py
@@ -13,27 +13,50 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` added +939/-0; `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +30/-467; `vllm/models/qwen4_exp/nvidia/model.py` modified +37/-0
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +327/-21
- 验证与风险: diff 自带测试面 `tests/engine/test_arg_utils.py`, `tests/models/qwen4_exp/test_ple.py`, `tests/test_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #55309 - [Qwen3.8-Flash-Next] Fuse PLE residual and QSA output gate

- 链接: https://github.com/vllm-project/vllm/pull/55309
- 状态/时间: merged / 2026-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/nvidia/model.py`, `vllm/models/qwen4_exp/nvidia/ops/ple.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa.py` 等 7 个文件；关联提交 `3f55ad2f073a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+111/-12，可读 patch 448 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +48/-0 (48 lines); hunks: -24,6 +24,7 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; -36,6 +37,8 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; symbols: _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel，涉及 `_qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel`；`vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +26/-4 (30 lines); hunks: -195,6 +195,7 @@ def _short_conv_dilated_dispatch(; -209,6 +210,7 @@ def _short_conv_dilated_dispatch(; symbols: _short_conv_dilated_dispatch，涉及 `_short_conv_dilated_dispatch`；`vllm/models/qwen4_exp/nvidia/ops/ple.py` modified +15/-2 (17 lines); hunks: -322,6 +322,7 @@ def _ple_conv_kernel(; -442,11 +443,21 @@ def _ple_conv_kernel(; symbols: _ple_conv_kernel, ple_conv，涉及 `_ple_conv_kernel, ple_conv`；`tests/models/qwen4_exp/test_ple.py` modified +12/-3 (15 lines); hunks: -1610,13 +1610,20 @@ def test_fused_conv_correctness(; -1630,15 +1637,17 @@ def test_fused_conv_correctness(; symbols: test_fused_conv_correctness，涉及 `test_fused_conv_correctness`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +48/-0 (48 lines); hunks: -24,6 +24,7 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; -36,6 +37,8 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; symbols: _qsa_sparse_paged_gqa_splitk_kernel, _qsa_merge_splitk_kernel
  - `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +26/-4 (30 lines); hunks: -195,6 +195,7 @@ def _short_conv_dilated_dispatch(; -209,6 +210,7 @@ def _short_conv_dilated_dispatch(; symbols: _short_conv_dilated_dispatch
  - `vllm/models/qwen4_exp/nvidia/ops/ple.py` modified +15/-2 (17 lines); hunks: -322,6 +322,7 @@ def _ple_conv_kernel(; -442,11 +443,21 @@ def _ple_conv_kernel(; symbols: _ple_conv_kernel, ple_conv
  - `tests/models/qwen4_exp/test_ple.py` modified +12/-3 (15 lines); hunks: -1610,13 +1610,20 @@ def test_fused_conv_correctness(; -1630,15 +1637,17 @@ def test_fused_conv_correctness(; symbols: test_fused_conv_correctness
  - `vllm/models/qwen4_exp/nvidia/qsa.py` modified +6/-2 (8 lines); hunks: -121,6 +121,7 @@ def forward_qsa(; -157,6 +158,7 @@ def forward_qsa(; symbols: forward_qsa, _run_qsa, forward
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -24,6 +24,7 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
+    output_gate_ptr,
@@ -36,6 +37,8 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
+    stride_output_gate_row,
+    stride_output_gate_head,
@@ -151,6 +154,18 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
+        # Preserve the unfused path's BF16 attention-output rounding before
diff -- vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -195,6 +195,7 @@ def _short_conv_dilated_dispatch(
+        outer_residual: torch.Tensor,
@@ -209,6 +210,7 @@ def _short_conv_dilated_dispatch(
+        outer_residual = outer_residual[: metadata.num_actual_tokens]
@@ -235,6 +237,7 @@ def _short_conv_dilated_dispatch(
+                outer_residual=outer_residual,
@@ -259,11 +262,17 @@ def _short_conv_dilated_dispatch(
diff -- vllm/models/qwen4_exp/nvidia/ops/ple.py
@@ -322,6 +322,7 @@ def _ple_conv_kernel(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +48/-0; `vllm/models/qwen4_exp/nvidia/ple_layer.py` modified +26/-4; `vllm/models/qwen4_exp/nvidia/ops/ple.py` modified +15/-2; `vllm/models/qwen4_exp/nvidia/qsa.py` modified +6/-2; `vllm/models/qwen4_exp/nvidia/model.py` modified +1/-1
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +12/-3; `tests/models/qwen4_exp/test_qsa_reference.py` modified +3/-0
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_ple.py`, `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #55557 - [Model] Qwen4Exp: fp8_e4m3 main KV cache on the QSA path

- 链接: https://github.com/vllm-project/vllm/pull/55557
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `vllm/models/qwen4_exp/nvidia/qsa.py`；关联提交 `dff1bde84dd6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+287/-45，可读 patch 595 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +124/-16 (140 lines); hunks: -4,15 +4,24; -24,6 +33,8 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; symbols: _is_sm120, _qsa_sparse_paged_gqa_splitk_kernel，涉及 `_is_sm120, _qsa_sparse_paged_gqa_splitk_kernel`；`vllm/models/qwen4_exp/nvidia/qsa.py` modified +106/-11 (117 lines); hunks: -24,6 +24,7; -64,7 +65,38 @@ class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):; symbols: Qwen4ExpQSAFlashAttentionBackend, supports_kv_cache_dtype, supports_combination, get_name，涉及 `Qwen4ExpQSAFlashAttentionBackend, supports_kv_cache_dtype, supports_combination`；`tests/models/qwen4_exp/test_qsa_reference.py` modified +57/-18 (75 lines); hunks: -203,7 +203,14 @@ def _qsa_sparse_paged_attention_reference(; -215,13 +222,15 @@ def _qsa_sparse_paged_attention_reference(; symbols: _qsa_sparse_paged_attention_reference, test_qsa_block_expansion_correctness, test_qsa_sparse_paged_attention_correctness，涉及 `_qsa_sparse_paged_attention_reference, test_qsa_block_expansion_correctness, test_qsa_sparse_paged_attention_correctness`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +124/-16 (140 lines); hunks: -4,15 +4,24; -24,6 +33,8 @@ def _qsa_sparse_paged_gqa_splitk_kernel(; symbols: _is_sm120, _qsa_sparse_paged_gqa_splitk_kernel
  - `vllm/models/qwen4_exp/nvidia/qsa.py` modified +106/-11 (117 lines); hunks: -24,6 +24,7; -64,7 +65,38 @@ class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):; symbols: Qwen4ExpQSAFlashAttentionBackend, supports_kv_cache_dtype, supports_combination, get_name
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +57/-18 (75 lines); hunks: -203,7 +203,14 @@ def _qsa_sparse_paged_attention_reference(; -215,13 +222,15 @@ def _qsa_sparse_paged_attention_reference(; symbols: _qsa_sparse_paged_attention_reference, test_qsa_block_expansion_correctness, test_qsa_sparse_paged_attention_correctness
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -4,15 +4,24 @@
+from functools import lru_cache
+from vllm.platforms import current_platform
+@lru_cache(maxsize=1)
+def _is_sm120() -> bool:
+    """True on sm_120 (RTX PRO 6000 Blackwell): selects the sm_120 tuning table."""
+    return current_platform.get_device_capability() == (12, 0)
diff -- vllm/models/qwen4_exp/nvidia/qsa.py
@@ -24,6 +24,7 @@
+from vllm.platforms.interface import DeviceCapability
@@ -64,7 +65,38 @@ class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
-    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]
+    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
+    # with the layer's per-tensor scales and dequantized on load inside the QSA
+    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
diff -- tests/models/qwen4_exp/test_qsa_reference.py
@@ -203,7 +203,14 @@ def _qsa_sparse_paged_attention_reference(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +124/-16; `vllm/models/qwen4_exp/nvidia/qsa.py` modified +106/-11
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` modified +57/-18
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57273 - [Perf][Model] Qwen4Exp QSA: sm_90 tuning table for _select_config

- 链接: https://github.com/vllm-project/vllm/pull/57273
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/qwen4_exp/nvidia/ops/qsa.py`；关联提交 `7bbce752b85f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+39/-1，可读 patch 67 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +39/-1 (40 lines); hunks: -22,6 +22,12 @@ def _is_sm120() -> bool:; -509,6 +515,33 @@ def _select_sm120_config(; symbols: _is_sm120, _is_sm90, _qsa_sparse_paged_gqa_splitk_kernel, _select_sm120_config，涉及 `_is_sm120, _is_sm90, _qsa_sparse_paged_gqa_splitk_kernel`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +39/-1 (40 lines); hunks: -22,6 +22,12 @@ def _is_sm120() -> bool:; -509,6 +515,33 @@ def _select_sm120_config(; symbols: _is_sm120, _is_sm90, _qsa_sparse_paged_gqa_splitk_kernel, _select_sm120_config
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -22,6 +22,12 @@ def _is_sm120() -> bool:
+@lru_cache(maxsize=1)
+def _is_sm90() -> bool:
+    """True on sm_90 (H100/H200/H20): selects the sm_90 tuning table."""
+    return current_platform.get_device_capability() == (9, 0)
@@ -509,6 +515,33 @@ def _select_sm120_config(
+def _select_sm90_config(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa.py` modified +39/-1
- 验证与风险: runtime 路径改动集中在 `vllm/models/qwen4_exp/nvidia/ops/qsa.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58489 - [Bugfix][Qwen4Exp] Keep pinned PLE prefetch ids out of the CUDA graph pool

- 链接: https://github.com/vllm-project/vllm/pull/58489
- 状态/时间: merged / 2026-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`；关联提交 `48d8880d098c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+42/-0，可读 patch 70 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/qwen4_exp/test_ple.py` modified +35/-0 (35 lines); hunks: -35,6 +35,7; -885,6 +886,40 @@ def test_fused_ngram_ids_correctness(; symbols: test_fused_ngram_ids_correctness, test_ngram_prefetch_ids_outlive_start_prefetch, _short_conv_dilated_decode_pytorch，涉及 `test_fused_ngram_ids_correctness, test_ngram_prefetch_ids_outlive_start_prefetch, _short_conv_dilated_decode_pytorch`；`vllm/models/qwen4_exp/nvidia/ngram_embedding.py` modified +7/-0 (7 lines); hunks: -716,6 +716,12 @@ def __init__(; -864,6 +870,7 @@ def start_prefetch(; symbols: __init__, start_prefetch，涉及 `__init__, start_prefetch`。
- 代码 diff 细节:
  - `tests/models/qwen4_exp/test_ple.py` modified +35/-0 (35 lines); hunks: -35,6 +35,7; -885,6 +886,40 @@ def test_fused_ngram_ids_correctness(; symbols: test_fused_ngram_ids_correctness, test_ngram_prefetch_ids_outlive_start_prefetch, _short_conv_dilated_decode_pytorch
  - `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` modified +7/-0 (7 lines); hunks: -716,6 +716,12 @@ def __init__(; -864,6 +870,7 @@ def start_prefetch(; symbols: __init__, start_prefetch
- 关键代码摘录:

```diff
diff -- tests/models/qwen4_exp/test_ple.py
@@ -35,6 +35,7 @@
+from vllm.utils.torch_utils import weak_ref_tensor
@@ -885,6 +886,40 @@ def test_fused_ngram_ids_correctness(
+@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused PLE needs CUDA")
+def test_ngram_prefetch_ids_outlive_start_prefetch() -> None:
+    """Eager breaks hand the side-stream lookup weak refs, so the ids must stay
+    valid after start_prefetch returns rather than be reused from the pool."""
diff -- vllm/models/qwen4_exp/nvidia/ngram_embedding.py
@@ -716,6 +716,12 @@ def __init__(
+        if self.ngram_embedding.supports_prefetch:
+            # The side-stream lookup outlives eager-break args, whose
+            # graph-pool storage later segments may reuse.
+            self._prefetch_ids = torch.empty(
+                max_total_tokens, self.ngram_heads, dtype=torch.long
+            )
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +35/-0
  - runtime: `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` modified +7/-0
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_ple.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57497 - [Qwen4Exp][ROCm] PLE n-gram table CPU offload

- 链接: https://github.com/vllm-project/vllm/pull/57497
- 状态/时间: merged / 2026-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/amd/ple_layer.py`, `vllm/models/qwen4_exp/common/ngram_embedding.py`, `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`；关联提交 `267eee54bf23`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+887/-529，可读 patch 1568 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/common/ngram_embedding.py` added +542/-0 (542 lines); hunks: -0,0 +1,542; symbols: Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight, dequantize，涉及 `Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight`；`vllm/models/qwen4_exp/nvidia/ngram_embedding.py` modified +17/-509 (526 lines); hunks: -2,534 +2,42; symbols: Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight, dequantize，涉及 `Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight`；`tests/models/qwen4_exp/test_ple.py` modified +214/-1 (215 lines); hunks: -13,19 +13,24; -1715,3 +1720,211 @@ def grouped_norm(inputs: torch.Tensor, weight: torch.Ten...; symbols: grouped_norm, _build_amd_ngram_embedding, test_amd_fp8_embedding_loads_checkpoint_shards_and_global_scale, test_amd_pinned_embedding_output_written_under_compile，涉及 `grouped_norm, _build_amd_ngram_embedding, test_amd_fp8_embedding_loads_checkpoint_shards_and_global_scale`；`vllm/models/qwen4_exp/amd/ple_layer.py` modified +90/-4 (94 lines); hunks: -11,13 +11,17; -30,7 +34,13; symbols: Qwen4ExpPLEGroupedNorm, __init__, forward，涉及 `Qwen4ExpPLEGroupedNorm, __init__, forward`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/common/ngram_embedding.py` added +542/-0 (542 lines); hunks: -0,0 +1,542; symbols: Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight, dequantize
  - `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` modified +17/-509 (526 lines); hunks: -2,534 +2,42; symbols: Qwen4ExpPLEEmbedding, __init__, allocate_embedding_weight, dequantize
  - `tests/models/qwen4_exp/test_ple.py` modified +214/-1 (215 lines); hunks: -13,19 +13,24; -1715,3 +1720,211 @@ def grouped_norm(inputs: torch.Tensor, weight: torch.Ten...; symbols: grouped_norm, _build_amd_ngram_embedding, test_amd_fp8_embedding_loads_checkpoint_shards_and_global_scale, test_amd_pinned_embedding_output_written_under_compile
  - `vllm/models/qwen4_exp/amd/ple_layer.py` modified +90/-4 (94 lines); hunks: -11,13 +11,17; -30,7 +34,13; symbols: Qwen4ExpPLEGroupedNorm, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/common/ngram_embedding.py
@@ -0,0 +1,542 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.
+Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
+n-gram embedding table can be kept in pinned host memory and looked up through
+Unified Virtual Addressing on any CUDA-alike platform.
diff -- vllm/models/qwen4_exp/nvidia/ngram_embedding.py
@@ -2,534 +2,42 @@
-from abc import ABC, abstractmethod
-from typing import ClassVar
-import torch.nn.functional as F
-from vllm.compilation.breakable_cudagraph import eager_break_during_capture
-from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
-from vllm.forward_context import DPMetadata, get_forward_context
diff -- tests/models/qwen4_exp/test_ple.py
@@ -13,19 +13,24 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/common/ngram_embedding.py` added +542/-0; `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` modified +17/-509; `vllm/models/qwen4_exp/amd/ple_layer.py` modified +90/-4
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +214/-1
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_ple.py`, `tests/test_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57105 - [Qwen3.8-Flash-Next] Avoid memory fragmentation in QSA indexer logits workspace

- 链接: https://github.com/vllm-project/vllm/pull/57105
- 状态/时间: merged / 2026-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`；关联提交 `7a877ae34dcf`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+13/-2，可读 patch 47 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +13/-2 (15 lines); hunks: -383,6 +383,7 @@ def _prefill_logits(; -392,10 +393,14 @@ def _prefill_logits(; symbols: _prefill_logits, qsa_select_paged_prefill，涉及 `_prefill_logits, qsa_select_paged_prefill`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +13/-2 (15 lines); hunks: -383,6 +383,7 @@ def _prefill_logits(; -392,10 +393,14 @@ def _prefill_logits(; symbols: _prefill_logits, qsa_select_paged_prefill
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
@@ -383,6 +383,7 @@ def _prefill_logits(
+    logits_workspace: torch.Tensor,
@@ -392,10 +393,14 @@ def _prefill_logits(
+    assert logits_workspace.is_contiguous()
+    assert logits_workspace.numel() >= num_queries * logits_width
-    logits = torch.empty(
-        (num_queries, logits_width), dtype=torch.float32, device=q.device
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` modified +13/-2
- 验证与风险: runtime 路径改动集中在 `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58961 - [Bugfix][Qwen4Exp] Release the profiling KV cache held by QSA key views

- 链接: https://github.com/vllm-project/vllm/pull/58961
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/common/qsa_cache.py`；关联提交 `a2be4d3cc7b9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+34/-10，可读 patch 79 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/qwen4_exp/test_qsa_reference.py` modified +22/-1 (23 lines); hunks: -17,6 +17,7; -365,7 +366,9 @@ def test_qsa_circular_buffer_survives_one_speculative_step(c...; symbols: test_qsa_circular_buffer_survives_one_speculative_step, _qsa_key_cache, test_qsa_state_caches_adapt_the_unified_logical_layout，涉及 `test_qsa_circular_buffer_survives_one_speculative_step, _qsa_key_cache, test_qsa_state_caches_adapt_the_unified_logical_layout`；`vllm/models/qwen4_exp/common/qsa_cache.py` modified +12/-9 (21 lines); hunks: -825,15 +825,18 @@ def __init__(self, *, cache_rope_positions: bool = False,...; symbols: __init__, bind_kv_cache, key_cache, rope_position_cache，涉及 `__init__, bind_kv_cache, key_cache`。
- 代码 diff 细节:
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +22/-1 (23 lines); hunks: -17,6 +17,7; -365,7 +366,9 @@ def test_qsa_circular_buffer_survives_one_speculative_step(c...; symbols: test_qsa_circular_buffer_survives_one_speculative_step, _qsa_key_cache, test_qsa_state_caches_adapt_the_unified_logical_layout
  - `vllm/models/qwen4_exp/common/qsa_cache.py` modified +12/-9 (21 lines); hunks: -825,15 +825,18 @@ def __init__(self, *, cache_rope_positions: bool = False,...; symbols: __init__, bind_kv_cache, key_cache, rope_position_cache
- 关键代码摘录:

```diff
diff -- tests/models/qwen4_exp/test_qsa_reference.py
@@ -17,6 +17,7 @@
+from vllm.v1.worker.utils import clear_layer_kv_caches
@@ -365,7 +366,9 @@ def test_qsa_circular_buffer_survives_one_speculative_step(chunk_start: int) ->
-def _qsa_key_cache(block_size: int, compress_ratio: int) -> qsa_cache.QSAKeyStateCache:
+def _qsa_key_cache(
+    block_size: int, compress_ratio: int, **kwargs
+) -> qsa_cache.QSAKeyStateCache:
diff -- vllm/models/qwen4_exp/common/qsa_cache.py
@@ -825,15 +825,18 @@ def __init__(self, *, cache_rope_positions: bool = False, **kwargs) -> None:
-    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
-        super().bind_kv_cache(kv_cache)
-        qsa_cache = self.kv_cache
-        self.key_cache = qsa_cache[..., : self.key_head_size]
-        if self.cache_rope_positions:
-            position_tail = qsa_cache[..., self.rope_position_offset :]
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/qwen4_exp/test_qsa_reference.py` modified +22/-1
  - runtime: `vllm/models/qwen4_exp/common/qsa_cache.py` modified +12/-9
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58957 - [Perf][Qwen4Exp] Fuse HC down projection and SiLU on NVIDIA

- 链接: https://github.com/vllm-project/vllm/pull/58957
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_hc_ops.py`, `vllm/models/qwen4_exp/nvidia/hyperconnection.py`, `vllm/models/qwen4_exp/nvidia/model.py`, `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/__init__.py`, `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py` 等 7 个文件；关联提交 `d6cce94fd422`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+1175/-23，可读 patch 1291 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py` added +536/-0 (536 lines); hunks: -0,0 +1,536; symbols: set_block_rank, st_shared_remote_f32, _sigmoid_f32, HcDownSiluMma，涉及 `set_block_rank, st_shared_remote_f32, _sigmoid_f32`；`vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py` added +334/-0 (334 lines); hunks: -0,0 +1,334; symbols: _sigmoid_f32, HcDownSiluFma, docstring, __init__，涉及 `_sigmoid_f32, HcDownSiluFma, docstring`；`vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: HcDownSiluGemm, __init__, dispatch, compile，涉及 `HcDownSiluGemm, __init__, dispatch`；`vllm/models/qwen4_exp/nvidia/hyperconnection.py` modified +38/-23 (61 lines); hunks: -25,16 +25,19; -58,9 +61,9 @@ class GatedResidual(nn.Module):; symbols: GatedResidual, __init__, _down_and_inject，涉及 `GatedResidual, __init__, _down_and_inject`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py` added +536/-0 (536 lines); hunks: -0,0 +1,536; symbols: set_block_rank, st_shared_remote_f32, _sigmoid_f32, HcDownSiluMma
  - `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py` added +334/-0 (334 lines); hunks: -0,0 +1,334; symbols: _sigmoid_f32, HcDownSiluFma, docstring, __init__
  - `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: HcDownSiluGemm, __init__, dispatch, compile
  - `vllm/models/qwen4_exp/nvidia/hyperconnection.py` modified +38/-23 (61 lines); hunks: -25,16 +25,19; -58,9 +61,9 @@ class GatedResidual(nn.Module):; symbols: GatedResidual, __init__, _down_and_inject
  - `tests/models/qwen4_exp/test_hc_ops.py` modified +30/-0 (30 lines); hunks: -4,11 +4,13; -22,6 +24,13; symbols: test_grouped_gemma_rmsnorm, test_hc_combine_norm_unit_injection, test_hc_down_silu_fused
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py
@@ -0,0 +1,536 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Adapted from vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_splitk.py.
+Fuses the Qwen4Exp mHC SiLU epilogue (SiLU(x/HC) on columns < rank, passthrough
+elsewhere, with the production bf16 rounding boundary) into the split-K kernel,
+applied after the cluster reduction where the final fp32 result materializes.
diff -- vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py
@@ -0,0 +1,334 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Adapted from vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_dotprod.py.
+Fuses the Qwen4Exp mHC SiLU epilogue (SiLU(x/HC) on columns < rank, passthrough
+elsewhere, with the production bf16 rounding boundary) into the FMA kernel.
+Output is bf16 instead of fp32.
diff -- vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py
@@ -0,0 +1,224 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py` added +536/-0; `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py` added +334/-0; `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py` added +224/-0; `vllm/models/qwen4_exp/nvidia/hyperconnection.py` modified +38/-23; `vllm/models/qwen4_exp/nvidia/model.py` modified +11/-0; `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/__init__.py` added +2/-0
  - tests: `tests/models/qwen4_exp/test_hc_ops.py` modified +30/-0
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_hc_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #51289 - [Model] Extend device-side mm normalization to Qwen3VL/Qwen3.5/Qwen4Next

- 链接: https://github.com/vllm-project/vllm/pull/51289
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/qwen4_exp/amd/model.py`, `vllm/models/qwen4_exp/nvidia/model.py`；关联提交 `491f44adfa42`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+87/-25，可读 patch 321 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/amd/model.py` modified +2/-0 (2 lines); hunks: -14,6 +14,7; -897,6 +898,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`；`vllm/models/qwen4_exp/nvidia/model.py` modified +2/-0 (2 lines); hunks: -13,6 +13,7; -939,6 +940,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/amd/model.py` modified +2/-0 (2 lines); hunks: -14,6 +14,7; -897,6 +898,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
  - `vllm/models/qwen4_exp/nvidia/model.py` modified +2/-0 (2 lines); hunks: -13,6 +13,7; -939,6 +940,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/amd/model.py
@@ -14,6 +14,7 @@
+from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
@@ -897,6 +898,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
+                    input_norm=build_mm_input_norm(self.model_config),
diff -- vllm/models/qwen4_exp/nvidia/model.py
@@ -13,6 +13,7 @@
+from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
@@ -939,6 +940,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
+                    input_norm=build_mm_input_norm(self.model_config),
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/amd/model.py` modified +2/-0; `vllm/models/qwen4_exp/nvidia/model.py` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation_ppl_test/ppl_utils.py`, `tests/models/multimodal/generation_ppl_test/test_qwen.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #59214 - [Perf][Qwen4Exp] Add SM100 low-latency decode GEMM plans

- 链接: https://github.com/vllm-project/vllm/pull/59214
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`；关联提交 `4ed438b79cea`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+99/-0，可读 patch 134 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +69/-0 (69 lines); hunks: -144,6 +144,73; -155,6 +222,8 @@ def _is_sm90() -> bool:; symbols: _is_sm100, _is_sm103, _is_sm90, _gemm_plans，涉及 `_is_sm100, _is_sm103, _is_sm90`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +69/-0 (69 lines); hunks: -144,6 +144,73; -155,6 +222,8 @@ def _is_sm90() -> bool:; symbols: _is_sm100, _is_sm103, _is_sm90, _gemm_plans
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/low_latency_gemm.py
@@ -144,6 +144,73 @@
+# B200 plans; the winning configs differ from the B300 table.
+QWEN4_EXP_SM100_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
+    # GDN fused B/A projection, TP=4.
+    (24, 2560): {
+        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
+        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=2560),
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +69/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_bf16_skinny_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57097 - [Qwen3.8-Flash-Next] Fuse main QK-norm/RoPE/gate and KV-cache write into the QSA pre-indexer launch

- 链接: https://github.com/vllm-project/vllm/pull/57097
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_qsa_prepare.py`, `tests/models/qwen4_exp/test_qsa_reference.py`, `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`, `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py`, `vllm/models/qwen4_exp/nvidia/qsa.py`；关联提交 `c9578f10ab3e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+325/-47，可读 patch 654 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py` renamed +146/-8 (154 lines); hunks: -1,6 +1,6; -80,6 +80,27 @@ def _norm_rope(; symbols: _norm_rope, _to_dst_dtype, _store_rotated, _qsa_pre_indexer_kernel，涉及 `_norm_rope, _to_dst_dtype, _store_rotated`；`tests/models/qwen4_exp/test_qsa_prepare.py` renamed +87/-6 (93 lines); hunks: -1,6 +1,6; -15,9 +15,7; symbols: _make_block_table, assert_fp8_within_one_ulp, _make_main_inputs, _check_main_outputs，涉及 `_make_block_table, assert_fp8_within_one_ulp, _make_main_inputs`；`vllm/models/qwen4_exp/nvidia/qsa.py` modified +40/-18 (58 lines); hunks: -407,6 +407,12 @@ def __init__(; -446,12 +452,15 @@ def _run_qsa(; symbols: __init__, _run_qsa, forward，涉及 `__init__, _run_qsa, forward`；`vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +36/-10 (46 lines); hunks: -2,7 +2,7; -23,7 +23,10; symbols: apply_qsa_rope, forward，涉及 `apply_qsa_rope, forward`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py` renamed +146/-8 (154 lines); hunks: -1,6 +1,6; -80,6 +80,27 @@ def _norm_rope(; symbols: _norm_rope, _to_dst_dtype, _store_rotated, _qsa_pre_indexer_kernel
  - `tests/models/qwen4_exp/test_qsa_prepare.py` renamed +87/-6 (93 lines); hunks: -1,6 +1,6; -15,9 +15,7; symbols: _make_block_table, assert_fp8_within_one_ulp, _make_main_inputs, _check_main_outputs
  - `vllm/models/qwen4_exp/nvidia/qsa.py` modified +40/-18 (58 lines); hunks: -407,6 +407,12 @@ def __init__(; -446,12 +452,15 @@ def _run_qsa(; symbols: __init__, _run_qsa, forward
  - `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +36/-10 (46 lines); hunks: -2,7 +2,7; -23,7 +23,10; symbols: apply_qsa_rope, forward
  - `tests/models/qwen4_exp/test_qsa_reference.py` modified +15/-4 (19 lines); hunks: -56,16 +56,24 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; -79,11 +87,14 @@ def test_qsa_mtp_index_share_updates_cache_but_skips_selection(; symbols: test_qsa_mtp_index_share_updates_cache_but_skips_selection, test_qsa_unfused_cache_update_ignores_padded_qk
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py
@@ -1,6 +1,6 @@
-"""Fused QSA pre-indexer kernel for Qwen4Exp."""
+"""Fused QSA prepare kernel for Qwen4Exp."""
@@ -80,6 +80,27 @@ def _norm_rope(
+@triton.jit
+def _to_dst_dtype(x, dst, scale):
+    """Round to BF16 like the unfused path, then scale for an FP8 destination."""
diff -- tests/models/qwen4_exp/test_qsa_prepare.py
@@ -1,6 +1,6 @@
-"""Correctness tests for the fused QSA pre-indexer."""
+"""Correctness tests for the fused QSA prepare kernel."""
@@ -15,9 +15,7 @@
-from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import (
-    qsa_pre_indexer,
-)
diff -- vllm/models/qwen4_exp/nvidia/qsa.py
@@ -407,6 +407,12 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py` renamed +146/-8; `vllm/models/qwen4_exp/nvidia/qsa.py` modified +40/-18; `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` modified +36/-10
  - tests: `tests/models/qwen4_exp/test_qsa_prepare.py` renamed +87/-6; `tests/models/qwen4_exp/test_qsa_reference.py` modified +15/-4
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_qsa_prepare.py`, `tests/models/qwen4_exp/test_qsa_reference.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57387 - [Model] Use upstream GLM-5.3 and Qwen4-Exp configs and processor

- 链接: https://github.com/vllm-project/vllm/pull/57387
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_config.py`, `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/amd/indexer_qsa.py`, `vllm/models/qwen4_exp/amd/model.py`, `vllm/models/qwen4_exp/amd/mtp.py` 等 13 个文件；关联提交 `e3cae8d2ac6b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 36 个文件，+365/-2384，可读 patch 3571 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/glm5next/common/model.py` modified +74/-26 (100 lines); hunks: -6,6 +6,7; -83,7 +84,6; symbols: _is_moe, _is_kda_layer, _is_linear_attn, _validate_supported_config，涉及 `_is_moe, _is_kda_layer, _is_linear_attn`；`vllm/models/qwen4_exp/amd/model.py` modified +10/-37 (47 lines); hunks: -7,6 +7,7; -47,7 +48,6; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/models/qwen4_exp/nvidia/model.py` modified +10/-37 (47 lines); hunks: -7,6 +7,7; -46,7 +47,6; symbols: __init__, forward，涉及 `__init__, forward`；`tests/models/qwen4_exp/test_config.py` modified +5/-5 (10 lines); hunks: -7,16 +7,13; -37,7 +34,10 @@ def _text_config(**kwargs) -> Qwen4ExpTextConfig:; symbols: _text_config，涉及 `_text_config`。
- 代码 diff 细节:
  - `vllm/models/glm5next/common/model.py` modified +74/-26 (100 lines); hunks: -6,6 +6,7; -83,7 +84,6; symbols: _is_moe, _is_kda_layer, _is_linear_attn, _validate_supported_config
  - `vllm/models/qwen4_exp/amd/model.py` modified +10/-37 (47 lines); hunks: -7,6 +7,7; -47,7 +48,6; symbols: __init__, forward
  - `vllm/models/qwen4_exp/nvidia/model.py` modified +10/-37 (47 lines); hunks: -7,6 +7,7; -46,7 +47,6; symbols: __init__, forward
  - `tests/models/qwen4_exp/test_config.py` modified +5/-5 (10 lines); hunks: -7,16 +7,13; -37,7 +34,10 @@ def _text_config(**kwargs) -> Qwen4ExpTextConfig:; symbols: _text_config
  - `vllm/models/glm5next/common/mtp.py` modified +4/-4 (8 lines); hunks: -40,7 +40,7 @@ class Glm5NextMultiTokenPredictorLayer(nn.Module):; -114,7 +114,7 @@ def forward(; symbols: Glm5NextMultiTokenPredictorLayer, __init__, forward, Glm5NextMultiTokenPredictor
- 关键代码摘录:

```diff
diff -- vllm/models/glm5next/common/model.py
@@ -6,6 +6,7 @@
+from transformers import Glm5NextTextConfig
@@ -83,7 +84,6 @@
-from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
@@ -95,6 +95,57 @@
+_MHC_TAU = 0.05
+"""mHC routing temperature. A GLM-5.3-Flash trained value that neither the
diff -- vllm/models/qwen4_exp/amd/model.py
@@ -7,6 +7,7 @@
+from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig
@@ -47,7 +48,6 @@
-    Qwen3NextMLP,
@@ -71,13 +71,9 @@
-from vllm.transformers_utils.configs.qwen4_exp import (
-    Qwen4ExpTextConfig,
diff -- vllm/models/qwen4_exp/nvidia/model.py
@@ -7,6 +7,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/glm5next/common/model.py` modified +74/-26; `vllm/models/qwen4_exp/amd/model.py` modified +10/-37; `vllm/models/qwen4_exp/nvidia/model.py` modified +10/-37; `vllm/models/glm5next/common/mtp.py` modified +4/-4; `vllm/models/qwen4_exp/amd/mtp.py` modified +2/-4; `vllm/models/qwen4_exp/nvidia/mtp.py` modified +2/-4
  - tests: `tests/models/qwen4_exp/test_config.py` modified +5/-5
- 验证与风险: diff 自带测试面 `tests/models/glm5next/test_sequence_parallel.py`, `tests/models/multimodal/processing/test_glm5next.py`, `tests/models/qwen4_exp/test_config.py`, `tests/models/qwen4_exp/test_ple.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #59753 - [Perf][Qwen4Exp] Add SM121 TP=1 skinny-GEMM plans

- 链接: https://github.com/vllm-project/vllm/pull/59753
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`；关联提交 `f590eb2448a5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+136/-0，可读 patch 171 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +106/-0 (106 lines); hunks: -207,6 +207,110; -226,6 +330,8 @@ def _gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyG...; symbols: _is_sm121, _is_sm100, _gemm_plans，涉及 `_is_sm121, _is_sm100, _gemm_plans`。
- 代码 diff 细节:
  - `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +106/-0 (106 lines); hunks: -207,6 +207,110; -226,6 +330,8 @@ def _gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyG...; symbols: _is_sm121, _is_sm100, _gemm_plans
- 关键代码摘录:

```diff
diff -- vllm/models/qwen4_exp/nvidia/low_latency_gemm.py
@@ -207,6 +207,110 @@
+# DGX Spark (GB10) plans, TP=1.
+QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
+    # Shared-expert gate.
+    (1, 2560): {
+        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2, static_k=2560),
+        2: SkinnyGemmConfig(2, 64, 1, k_unroll=2, vector_width=4, static_k=2560),
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` modified +106/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_bf16_skinny_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #59990 - [Bugfix] Fix Qwen4Exp PLE embedding rejecting INC (AutoRound) checkpoints

- 链接: https://github.com/vllm-project/vllm/pull/59990
- 状态/时间: merged / 2026-10-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/qwen4_exp/test_ple.py`, `vllm/models/qwen4_exp/common/ngram_embedding.py`；关联提交 `4f52fa35efc7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+25/-0，可读 patch 53 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/qwen4_exp/test_ple.py` modified +19/-0 (19 lines); hunks: -20,6 +20,7; -566,6 +567,24 @@ def test_ple_embedding_respects_modelopt_exclusion() -> None:; symbols: test_ple_embedding_respects_modelopt_exclusion, test_ple_embedding_respects_inc_layer_config, test_ple_embedding_dtype_overrides_modelopt_exclusion，涉及 `test_ple_embedding_respects_modelopt_exclusion, test_ple_embedding_respects_inc_layer_config, test_ple_embedding_dtype_overrides_modelopt_exclusion`；`vllm/models/qwen4_exp/common/ngram_embedding.py` modified +6/-0 (6 lines); hunks: -27,6 +27,7; -190,6 +191,11 @@ def from_quant_config(; symbols: from_quant_config，涉及 `from_quant_config`。
- 代码 diff 细节:
  - `tests/models/qwen4_exp/test_ple.py` modified +19/-0 (19 lines); hunks: -20,6 +20,7; -566,6 +567,24 @@ def test_ple_embedding_respects_modelopt_exclusion() -> None:; symbols: test_ple_embedding_respects_modelopt_exclusion, test_ple_embedding_respects_inc_layer_config, test_ple_embedding_dtype_overrides_modelopt_exclusion
  - `vllm/models/qwen4_exp/common/ngram_embedding.py` modified +6/-0 (6 lines); hunks: -27,6 +27,7; -190,6 +191,11 @@ def from_quant_config(; symbols: from_quant_config
- 关键代码摘录:

```diff
diff -- tests/models/qwen4_exp/test_ple.py
@@ -20,6 +20,7 @@
+from vllm.model_executor.layers.quantization.inc import INCConfig
@@ -566,6 +567,24 @@ def test_ple_embedding_respects_modelopt_exclusion() -> None:
+def test_ple_embedding_respects_inc_layer_config() -> None:
+    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
+    quant_config = INCConfig(
+        weight_bits=4,
diff -- vllm/models/qwen4_exp/common/ngram_embedding.py
@@ -27,6 +27,7 @@
+from vllm.model_executor.layers.quantization.inc import INCConfig
@@ -190,6 +191,11 @@ def from_quant_config(
+        if (
+            isinstance(quant_config, INCConfig)
+            and not quant_config.config_parser.resolve(None, prefix).quantized
+        ):
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/qwen4_exp/test_ple.py` modified +19/-0
  - runtime: `vllm/models/qwen4_exp/common/ngram_embedding.py` modified +6/-0
- 验证与风险: diff 自带测试面 `tests/models/qwen4_exp/test_ple.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
