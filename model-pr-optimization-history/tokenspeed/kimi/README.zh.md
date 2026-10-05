# TokenSpeed Kimi K2/K2.5/K3/Linear/VL 模型 PR 优化历史

## 2026-08-23 源码 head 刷新

已复核 TokenSpeed 上游 main：
`lightseekorg/tokenspeed@2706143a8669d50a8f56466b9d340b86922b8f2d`。
已读完上一 head `d73bf0454422092f306d5575e803a08fd35ac41c`
之后的 2-commit 完整增量。

结果：PR #821 新增 Kimi K3 FlatKV/KDA/MLA 部署契约，并在保留硬件与验证限制的
前提下提升为源码指南；PR #823 只增加 README 新闻链接。本文继续覆盖 DP + EAGLE3
collective-size hang、Kimi incremental DFlash capture，以及 Kimi-K2.7 EAGLE3.1
模型语义。

| 合并日期 | PR | Runtime 信号 |
| --- | --- | --- |
| 2026-07-27 | [#821](https://github.com/lightseekorg/tokenspeed/pull/821) | Kimi K3 部署契约 |
| 2026-07-07 | [#596](https://github.com/lightseekorg/tokenspeed/pull/596) | DP + EAGLE3 mixed-step hang |
| 2026-07-25 | [#795](https://github.com/lightseekorg/tokenspeed/pull/795) | Kimi-K2.7 EAGLE3.1 |
| 2026-07-26 | [#797](https://github.com/lightseekorg/tokenspeed/pull/797) | incremental DFlash capture |

## 2026-06-27 PR 补漏复核

已按 TokenSpeed 上游 `HEAD@d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8` 复核。本文覆盖 Kimi K2.5/K2.x 相关的 merged PR，并采用 SGLang 风格的时间线和逐 PR diff 审计卡。

本轮筛选规则：标题/文件命中 `Kimi`、`kimi_k25`、`K2.5`、`NVFP4`、`MXFP4`、`MXINT4`、`lm_head`、`top_k/top_p`、`InstantTensor`、`OCR`、`FA4`、`vision`、`MLA` 的 merged PR；过滤纯格式化和不影响模型/runtime/CI lane 的基础设施 PR。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/guides/kimi-k3-hopper-pd.md` | [#1634](https://github.com/lightseekorg/tokenspeed/pull/1634) |
| `docs/recipes/kimi-k3-shared-expert-tp.md` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/configs/kimi_k25_config.py` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/configs/kimi_k2_config.py` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/configs/kimi_k3_config.py` | [#822](https://github.com/lightseekorg/tokenspeed/pull/822) |
| `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` | [#924](https://github.com/lightseekorg/tokenspeed/pull/924), [#1031](https://github.com/lightseekorg/tokenspeed/pull/1031) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` | [#995](https://github.com/lightseekorg/tokenspeed/pull/995), [#1147](https://github.com/lightseekorg/tokenspeed/pull/1147), [#1152](https://github.com/lightseekorg/tokenspeed/pull/1152), [#1634](https://github.com/lightseekorg/tokenspeed/pull/1634) |
| `python/tokenspeed/runtime/models/kimi_k25.py` | [#477](https://github.com/lightseekorg/tokenspeed/pull/477), [#797](https://github.com/lightseekorg/tokenspeed/pull/797), [#822](https://github.com/lightseekorg/tokenspeed/pull/822) |
| `python/tokenspeed/runtime/models/kimi_k3.py` | [#822](https://github.com/lightseekorg/tokenspeed/pull/822), [#852](https://github.com/lightseekorg/tokenspeed/pull/852), [#878](https://github.com/lightseekorg/tokenspeed/pull/878), [#906](https://github.com/lightseekorg/tokenspeed/pull/906), [#919](https://github.com/lightseekorg/tokenspeed/pull/919), [#924](https://github.com/lightseekorg/tokenspeed/pull/924), [#926](https://github.com/lightseekorg/tokenspeed/pull/926), [#957](https://github.com/lightseekorg/tokenspeed/pull/957), [#958](https://github.com/lightseekorg/tokenspeed/pull/958), [#959](https://github.com/lightseekorg/tokenspeed/pull/959), [#995](https://github.com/lightseekorg/tokenspeed/pull/995), [#1031](https://github.com/lightseekorg/tokenspeed/pull/1031), ... (46 total) |
| `python/tokenspeed/runtime/models/kimi_k3_comm.py` | [#1062](https://github.com/lightseekorg/tokenspeed/pull/1062), [#1128](https://github.com/lightseekorg/tokenspeed/pull/1128), [#1174](https://github.com/lightseekorg/tokenspeed/pull/1174), [#1300](https://github.com/lightseekorg/tokenspeed/pull/1300), [#1318](https://github.com/lightseekorg/tokenspeed/pull/1318), [#1330](https://github.com/lightseekorg/tokenspeed/pull/1330), [#1358](https://github.com/lightseekorg/tokenspeed/pull/1358), [#1383](https://github.com/lightseekorg/tokenspeed/pull/1383), [#1457](https://github.com/lightseekorg/tokenspeed/pull/1457), [#1471](https://github.com/lightseekorg/tokenspeed/pull/1471), [#1489](https://github.com/lightseekorg/tokenspeed/pull/1489), [#1539](https://github.com/lightseekorg/tokenspeed/pull/1539), ... (19 total) |
| `python/tokenspeed/runtime/models/kimi_k3_deepep.py` | [#1634](https://github.com/lightseekorg/tokenspeed/pull/1634), [#1790](https://github.com/lightseekorg/tokenspeed/pull/1790) |
| `python/tokenspeed/runtime/models/kimi_k3_dspark.py` | [#924](https://github.com/lightseekorg/tokenspeed/pull/924), [#995](https://github.com/lightseekorg/tokenspeed/pull/995), [#1031](https://github.com/lightseekorg/tokenspeed/pull/1031), [#1060](https://github.com/lightseekorg/tokenspeed/pull/1060), [#1634](https://github.com/lightseekorg/tokenspeed/pull/1634) |
| `python/tokenspeed/runtime/models/kimi_k3_nextn.py` | [#822](https://github.com/lightseekorg/tokenspeed/pull/822), [#919](https://github.com/lightseekorg/tokenspeed/pull/919), [#1060](https://github.com/lightseekorg/tokenspeed/pull/1060), [#1102](https://github.com/lightseekorg/tokenspeed/pull/1102), [#1383](https://github.com/lightseekorg/tokenspeed/pull/1383), [#1634](https://github.com/lightseekorg/tokenspeed/pull/1634), [#1790](https://github.com/lightseekorg/tokenspeed/pull/1790) |
| `python/tokenspeed/runtime/models/moonvit.py` | [#822](https://github.com/lightseekorg/tokenspeed/pull/822) |
| `test/agentic_benchmark/kimi_k2.5/sglang/agentic_bench.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/sglang/collect_outputs.py` | [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) |
| `test/agentic_benchmark/kimi_k2.5/sglang/configs/attn_tp4_moe_ep4.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/sglang/configs/attn_tp4_moe_tp4.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md` | [#1340](https://github.com/lightseekorg/tokenspeed/pull/1340) |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/agentic_bench.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/collect_outputs.py` | [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/configs/attn_dp8_moe_ep8.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/configs/attn_dp8_moe_tp8.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/configs/attn_tp4_moe_ep4.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/configs/attn_tp4_moe_tp4.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/configs/attn_tp8_moe_ep8.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/tokenspeed/configs/attn_tp8_moe_tp8.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/trtllm/README.md` | [#1340](https://github.com/lightseekorg/tokenspeed/pull/1340) |
| `test/agentic_benchmark/kimi_k2.5/trtllm/agentic_bench.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/trtllm/collect_outputs.py` | [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) |
| `test/agentic_benchmark/kimi_k2.5/trtllm/configs/attn_dp8_moe_ep8.yaml` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/trtllm/configs/attn_dp8_moe_tp8.yaml` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/trtllm/configs/attn_tp4_moe_ep4.yaml` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/trtllm/configs/attn_tp4_moe_tp4.yaml` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/trtllm/configs/attn_tp8_moe_ep8.yaml` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/trtllm/configs/attn_tp8_moe_tp8.yaml` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/vllm/agentic_bench.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/vllm/collect_outputs.py` | [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) |
| `test/agentic_benchmark/kimi_k2.5/vllm/configs/attn_tp4_moe_ep4.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k2.5/vllm/configs/attn_tp4_moe_tp4.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` | [#1187](https://github.com/lightseekorg/tokenspeed/pull/1187), [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326), [#1853](https://github.com/lightseekorg/tokenspeed/pull/1853) |
| `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm` | [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326), [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456), [#1853](https://github.com/lightseekorg/tokenspeed/pull/1853) |
| `test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py` | [#1187](https://github.com/lightseekorg/tokenspeed/pull/1187), [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) |
| `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh` | [#1853](https://github.com/lightseekorg/tokenspeed/pull/1853) |
| `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` | [#1187](https://github.com/lightseekorg/tokenspeed/pull/1187), [#1242](https://github.com/lightseekorg/tokenspeed/pull/1242), [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) |
| `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` | [#1187](https://github.com/lightseekorg/tokenspeed/pull/1187), [#1242](https://github.com/lightseekorg/tokenspeed/pull/1242), [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) |
| `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/README.md` | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) |
| `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/collect_outputs.py` | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) |
| `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/configs/attn_dp16_moe_ep16.sh` | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) |
| `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm` | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) |
| `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/duplicate_agentic_dataset.py` | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) |
| `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm` | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) |
| `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/pd_client.py` | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) |
| `test/ci/deepswe/install_kimi_code.sh` | 无直接 PR 号提交 |
| `test/ci/deepswe/kimi_code_environment.py` | 无直接 PR 号提交 |
| `test/ci/deepswe/kimi_code_pier_agent.py` | 无直接 PR 号提交 |
| `test/ci/deepswe/test_kimi_code.py` | 无直接 PR 号提交 |
| `test/ci/eval/kimi-k2.5-mxfp4-eagle3-evalscope-aime25-amd.yaml` | 无直接 PR 号提交 |
| `test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml` | [#806](https://github.com/lightseekorg/tokenspeed/pull/806) |
| `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` | [#806](https://github.com/lightseekorg/tokenspeed/pull/806), [#1684](https://github.com/lightseekorg/tokenspeed/pull/1684) |
| `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml` | [#806](https://github.com/lightseekorg/tokenspeed/pull/806) |
| `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gsm8k.yaml` | [#806](https://github.com/lightseekorg/tokenspeed/pull/806) |
| `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-mmlu.yaml` | [#806](https://github.com/lightseekorg/tokenspeed/pull/806) |
| `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-ocr-bench.yaml` | [#806](https://github.com/lightseekorg/tokenspeed/pull/806) |
| `test/ci/eval/kimi-k3-deepswe-b300.yaml` | [#1306](https://github.com/lightseekorg/tokenspeed/pull/1306), [#1315](https://github.com/lightseekorg/tokenspeed/pull/1315) |
| `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` | [#1394](https://github.com/lightseekorg/tokenspeed/pull/1394), [#1771](https://github.com/lightseekorg/tokenspeed/pull/1771), [#1790](https://github.com/lightseekorg/tokenspeed/pull/1790) |
| `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml` | [#1165](https://github.com/lightseekorg/tokenspeed/pull/1165), [#1280](https://github.com/lightseekorg/tokenspeed/pull/1280) |
| `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` | [#1165](https://github.com/lightseekorg/tokenspeed/pull/1165), [#1280](https://github.com/lightseekorg/tokenspeed/pull/1280) |
| `test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml` | [#1225](https://github.com/lightseekorg/tokenspeed/pull/1225) |
| `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` | [#1131](https://github.com/lightseekorg/tokenspeed/pull/1131), [#1155](https://github.com/lightseekorg/tokenspeed/pull/1155), [#1225](https://github.com/lightseekorg/tokenspeed/pull/1225), [#1726](https://github.com/lightseekorg/tokenspeed/pull/1726) |
| `test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` | [#1283](https://github.com/lightseekorg/tokenspeed/pull/1283) |
| `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` | [#843](https://github.com/lightseekorg/tokenspeed/pull/843), [#1121](https://github.com/lightseekorg/tokenspeed/pull/1121), [#1379](https://github.com/lightseekorg/tokenspeed/pull/1379) |
| `test/ci/eval/kimi-k3-nvfp4-dflash2-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` | 无直接 PR 号提交 |
| `test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml` | [#1635](https://github.com/lightseekorg/tokenspeed/pull/1635) |
| `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` | [#1159](https://github.com/lightseekorg/tokenspeed/pull/1159), [#1225](https://github.com/lightseekorg/tokenspeed/pull/1225), [#1379](https://github.com/lightseekorg/tokenspeed/pull/1379) |
| `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` | [#1159](https://github.com/lightseekorg/tokenspeed/pull/1159), [#1225](https://github.com/lightseekorg/tokenspeed/pull/1225), [#1635](https://github.com/lightseekorg/tokenspeed/pull/1635), [#1726](https://github.com/lightseekorg/tokenspeed/pull/1726) |
| `test/ci/perf/kimi-k2.5-nvfp4-evalscope-agentic.yaml` | [#806](https://github.com/lightseekorg/tokenspeed/pull/806) |
| `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` | [#1740](https://github.com/lightseekorg/tokenspeed/pull/1740), [#1948](https://github.com/lightseekorg/tokenspeed/pull/1948), [#1992](https://github.com/lightseekorg/tokenspeed/pull/1992) |
| `test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml` | [#1283](https://github.com/lightseekorg/tokenspeed/pull/1283) |
| `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` | [#843](https://github.com/lightseekorg/tokenspeed/pull/843), [#921](https://github.com/lightseekorg/tokenspeed/pull/921), [#935](https://github.com/lightseekorg/tokenspeed/pull/935), [#1109](https://github.com/lightseekorg/tokenspeed/pull/1109), [#1203](https://github.com/lightseekorg/tokenspeed/pull/1203), [#1289](https://github.com/lightseekorg/tokenspeed/pull/1289), [#1379](https://github.com/lightseekorg/tokenspeed/pull/1379) |
| ... | 40 more files omitted from table; all were used for git tracing. |

## PR 覆盖总览

- git 追溯 PR 数: 110
- 原文档显式引用补充 PR 数: 12
- 当前文档总 PR 数: 122
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-08 | [#29](https://github.com/lightseekorg/tokenspeed/pull/29) | merged | Add Kimi K2.5 agentic perf CI task | perf YAML, CI pipeline |
| 2026-05-13 | [#126](https://github.com/lightseekorg/tokenspeed/pull/126) | merged | perf(K2.5): Optimize lm_head | `logits_processor.py`, CUDA `lm_head_gemm` |
| 2026-05-20 | [#184](https://github.com/lightseekorg/tokenspeed/pull/184) | merged | perf(K2.5): optimize top_k_renorm_prob + top_p_renorm_prob | fused sampling CUDA, backend/server args |
| 2026-05-28 | [#253](https://github.com/lightseekorg/tokenspeed/pull/253) | merged | ci(eval): add Kimi-K2.5-NVFP4 ocr_bench task | OCR eval YAML |
| 2026-06-14 | [#444](https://github.com/lightseekorg/tokenspeed/pull/444) | merged | feat(moe): add trtllm mxint4 MoE path for Kimi-K2.x | MXINT4 weights and FlashInfer TRT-LLM MoE op |
| 2026-06-15 | [#418](https://github.com/lightseekorg/tokenspeed/pull/418) | merged | Add InstantTensor weight loader | loader, weight utils, `kimi_k25.py`, docs/CI |
| 2026-06-16 | [#454](https://github.com/lightseekorg/tokenspeed/pull/454) | merged | [AMD] Support Kimi K2.5 MXFP4 serving | MXFP4 layers, dense path, MLA backend, Kimi model |
| 2026-06-19 | [#477](https://github.com/lightseekorg/tokenspeed/pull/477) | merged | perf(kernel): Optimize Kimi Vision FA4 QKV + RoPE | Kimi model, mm attention, packed complex rotary |
| 2026-06-19 | [#482](https://github.com/lightseekorg/tokenspeed/pull/482) | merged | ci: use FA4 mm attention for Kimi OCR eval | OCR eval YAML |
| 2026-06-23 | [#354](https://github.com/lightseekorg/tokenspeed/pull/354) | merged | feat(video) Generalize multimodal runtime support and add Qwen3.5 video | `python/tokenspeed/runtime/multimodal/encoder_cudagraph.py`, `python/tokenspeed/runtime/execution/model_executor.py`, `python/tokenspeed/runtime/multimodal/embedder.py` |
| 2026-06-26 | [#476](https://github.com/lightseekorg/tokenspeed/pull/476) | merged | Add AMD Kimi MXFP4 CI job | AMD eval YAML, MLA metadata unit test |
| 2026-07-07 | [#596](https://github.com/lightseekorg/tokenspeed/pull/596) | DP + EAGLE3 mixed-step hang |
| 2026-07-25 | [#795](https://github.com/lightseekorg/tokenspeed/pull/795) | Kimi-K2.7 EAGLE3.1 |
| 2026-07-26 | [#797](https://github.com/lightseekorg/tokenspeed/pull/797) | incremental DFlash capture |
| 2026-07-26 | [#806](https://github.com/lightseekorg/tokenspeed/pull/806) | merged | fix(ci): extend Kimi engine startup timeout | `test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml` |
| 2026-07-27 | [#821](https://github.com/lightseekorg/tokenspeed/pull/821) | Kimi K3 部署契约 |
| 2026-07-27 | [#822](https://github.com/lightseekorg/tokenspeed/pull/822) | merged | feat(kimi-k3): integrate Kimi K3 support | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/moonvit.py`, `python/tokenspeed/runtime/models/kimi_k25.py` |
| 2026-07-30 | [#847](https://github.com/lightseekorg/tokenspeed/pull/847) | merged | fix(kimi3): correct AMD KDA safe gate | `tokenspeed-kernel/test/kimi3_reference.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/kda.py` |
| 2026-07-30 | [#843](https://github.com/lightseekorg/tokenspeed/pull/843) | merged | ci(kimi-k3): add 8-GPU TP8/EP8 AIME26 eval and 4k/1k perf tasks | `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` |
| 2026-07-30 | [#852](https://github.com/lightseekorg/tokenspeed/pull/852) | merged | refactor(kimi-k3): replace tokenspeed-situ sidecar with flashinfer native SiTU MoE | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-03 | [#906](https://github.com/lightseekorg/tokenspeed/pull/906) | merged | Kimi K3 support dummy weight | `python/tokenspeed/runtime/models/kimi_k3.py` |
| 2026-08-03 | [#909](https://github.com/lightseekorg/tokenspeed/pull/909) | merged | fix(k3): derive Kimi-K3 KDA page packing from the MLA plane size | `test/runtime/test_kimi_k3_cache_spec.py`, `python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` |
| 2026-08-03 | [#876](https://github.com/lightseekorg/tokenspeed/pull/876) | merged | chore(kimi-k3): serve on GB200 -- LCM packing, layer derivation, kda la… | `test/runtime/test_kimi_k3_config.py`, `python/tokenspeed/runtime/multimodal/shm_transport.py`, `python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` |
| 2026-08-04 | [#921](https://github.com/lightseekorg/tokenspeed/pull/921) | merged | perf(kimi-k3): warm up the 4k benchmark shape | `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` |
| 2026-08-04 | [#935](https://github.com/lightseekorg/tokenspeed/pull/935) | merged | fix(ci): stabilize Kimi K3 perf metrics | `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` |
| 2026-08-05 | [#926](https://github.com/lightseekorg/tokenspeed/pull/926) | merged | perf(kimi-k3): router cublas dispatch, grouped MoE-join reduce | `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` |
| 2026-08-06 | [#958](https://github.com/lightseekorg/tokenspeed/pull/958) | merged | fix(k3): fix eagle3 for kimi k3 | `test/runtime/models/test_kimi_k3_eagle3_e2e.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_eagle3.py` |
| 2026-08-07 | [#924](https://github.com/lightseekorg/tokenspeed/pull/924) | merged | feat(dspark): complete Kimi K3 draft execution | `python/tokenspeed/runtime/models/kimi_k3_dspark.py`, `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py`, `python/tokenspeed/runtime/models/kimi_k3.py` |
| 2026-08-07 | [#919](https://github.com/lightseekorg/tokenspeed/pull/919) | merged | perf(kimi3): fuse semantic MLA decode stages | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-08 | [#878](https://github.com/lightseekorg/tokenspeed/pull/878) | merged | perf: Optimize Kimi K3 MoE input projections and low-token decode | `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/kimi3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` |
| 2026-08-08 | [#995](https://github.com/lightseekorg/tokenspeed/pull/995) | merged | feat(K3): Support K3 On H200 | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py` |
| 2026-08-11 | [#1012](https://github.com/lightseekorg/tokenspeed/pull/1012) | merged | fix(k3): restore DSpark cache view on unified arena | `test/runtime/test_kimi_k3_cache_pool.py`, `test/runtime/test_kimi_k3_cache_spec.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/mla.py` |
| 2026-08-11 | [#1056](https://github.com/lightseekorg/tokenspeed/pull/1056) | merged | fix(k3): advance attnres launch and join | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/models/test_kimi_k3_attnres_hoist.py` |
| 2026-08-12 | [#1038](https://github.com/lightseekorg/tokenspeed/pull/1038) | merged | perf(kimi3): fuse MoE norm projection and collectives | `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-12 | [#959](https://github.com/lightseekorg/tokenspeed/pull/959) | merged | perf(kimi3): fuse AttnRes projection and collectives | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-13 | [#1060](https://github.com/lightseekorg/tokenspeed/pull/1060) | merged | feat(kimi-k3): support cross-DP EP token gather | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py` |
| 2026-08-13 | [#1084](https://github.com/lightseekorg/tokenspeed/pull/1084) | merged | perf(kimi-k3): keep small AMD decode batches on one stream | `python/tokenspeed/runtime/models/kimi_k3.py` |
| 2026-08-14 | [#1089](https://github.com/lightseekorg/tokenspeed/pull/1089) | merged | perf(kimi3): fuse batched AttnRes graph on gfx950 | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py` |
| 2026-08-14 | [#957](https://github.com/lightseekorg/tokenspeed/pull/957) | merged | perf(k3): latent moe multicast tail | `python/tokenspeed/runtime/models/kimi_k3.py` |
| 2026-08-14 | [#1086](https://github.com/lightseekorg/tokenspeed/pull/1086) | merged | perf(kimi3): fuse KDA decode core | `python/tokenspeed/runtime/models/kimi_k3.py` |
| 2026-08-15 | [#1102](https://github.com/lightseekorg/tokenspeed/pull/1102) | merged | perf(k3): fold the latent-tail projection into the reduction epilogue and pool its symmetric buffers | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/layers/test_kimi_k3_addmm_fold.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py` |
| 2026-08-15 | [#1109](https://github.com/lightseekorg/tokenspeed/pull/1109) | merged | ci(kimi-k3): refresh MI35x perf baseline | `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` |
| 2026-08-17 | [#1062](https://github.com/lightseekorg/tokenspeed/pull/1062) | merged | perf(kimi3): fold the MoE finalize into the multicast latent tail (+ extract the K3 comm layer) | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-18 | [#1121](https://github.com/lightseekorg/tokenspeed/pull/1121) | merged | ci(kimi-k3): exercise DSpark in the AIME26 gate | `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py` |
| 2026-08-18 | [#1131](https://github.com/lightseekorg/tokenspeed/pull/1131) | merged | ci: run Kimi K3 on two GB300 Slurm nodes | `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` |
| 2026-08-19 | [#1128](https://github.com/lightseekorg/tokenspeed/pull/1128) | merged | Support nvidia/Kimi-K3-NVFP4: ModelOpt FP8_PB_WO attention + NVFP4 SiTU MoE | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` |
| 2026-08-19 | [#1129](https://github.com/lightseekorg/tokenspeed/pull/1129) | merged | perf(kimi3): avoid packed KDA QKV copies | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`, `test/runtime/test_kimi_k3_attn_res.py` |
| 2026-08-19 | [#1147](https://github.com/lightseekorg/tokenspeed/pull/1147) | merged | perf(cache): sparsify and budget Kimi-K3 state cache | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py`, `test/runtime/test_kimi_k3_cache_spec.py` |
| 2026-08-19 | [#1155](https://github.com/lightseekorg/tokenspeed/pull/1155) | merged | ci: load GB300 Kimi weights from local RAID | `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` |
| 2026-08-20 | [#1159](https://github.com/lightseekorg/tokenspeed/pull/1159) | merged | ci: add GB300 two-node AIME26 gates for K3-NVFP4 and K3-NVFP4+DSpark | `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` |
| 2026-08-20 | [#1157](https://github.com/lightseekorg/tokenspeed/pull/1157) | merged | perf(k3): route decode GEMV per shape | `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` |
| 2026-08-20 | [#1179](https://github.com/lightseekorg/tokenspeed/pull/1179) | merged | perf(k3): ll bf16 router GEMM | `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` |
| 2026-08-21 | [#1184](https://github.com/lightseekorg/tokenspeed/pull/1184) | merged | perf(k3): route a whole verify window through the packed top-k | `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py` |
| 2026-08-21 | [#1165](https://github.com/lightseekorg/tokenspeed/pull/1165) | merged | ci: add manual K3 DSpark vision evals | `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` |
| 2026-08-22 | [#1203](https://github.com/lightseekorg/tokenspeed/pull/1203) | merged | ci: move Kimi K3 benchmarks to MI350 | `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` |
| 2026-08-22 | [#1200](https://github.com/lightseekorg/tokenspeed/pull/1200) | merged | perf(gemm): route the unrouted K3 decode projections | `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` |
| 2026-08-23 | [#1031](https://github.com/lightseekorg/tokenspeed/pull/1031) | merged | feat(kimi-k3): serve DSpark drafts (fc_norm + AttnRes tap) | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py`, `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` |
| 2026-08-23 | [#1187](https://github.com/lightseekorg/tokenspeed/pull/1187) | merged | test: kimi-k3 agentic decode-throughput bench | `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` |
| 2026-08-23 | [#1174](https://github.com/lightseekorg/tokenspeed/pull/1174) | merged | perf(k3): extend latent-tail fusion to M64 with split collectives | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py` |
| 2026-08-24 | [#1225](https://github.com/lightseekorg/tokenspeed/pull/1225) | merged | ci(k3): give the Slurm aime26 gates the budget their context window allow | `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` |
| 2026-08-24 | [#1135](https://github.com/lightseekorg/tokenspeed/pull/1135) | merged | perf(kimi3): tune small-batch latent projection | `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` |
| 2026-08-25 | [#1242](https://github.com/lightseekorg/tokenspeed/pull/1242) | merged | chore(kimi-k3): use TokenSpeed MLA for agentic drafter | `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` |
| 2026-08-27 | [#1152](https://github.com/lightseekorg/tokenspeed/pull/1152) | merged | fix(cache): support attention-DP for Kimi-K3 by deriving the MLA packing from the KDA state size | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py`, `test/runtime/test_kimi_k3_cache_spec.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-27 | [#1263](https://github.com/lightseekorg/tokenspeed/pull/1263) | merged | perf(kimi-k3): select SiTU routing by forward phase | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-27 | [#1140](https://github.com/lightseekorg/tokenspeed/pull/1140) | merged | perf(kimi-k3): tune small-M MoE decode | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-08-27 | [#1270](https://github.com/lightseekorg/tokenspeed/pull/1270) | merged | fix(kimi3): warm the MoE auxiliary stream before graph capture | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py` |
| 2026-08-28 | [#1283](https://github.com/lightseekorg/tokenspeed/pull/1283) | merged | ci(kimi-k3): cover TP8/EP1 on gfx950 with manual trigger | `test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `test/runtime/test_kimi_k3_moe_fork_warmup.py` |
| 2026-08-29 | [#1289](https://github.com/lightseekorg/tokenspeed/pull/1289) | merged | ci: fix kimi k3 perf tokenizer cache | `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` |
| 2026-08-29 | [#1280](https://github.com/lightseekorg/tokenspeed/pull/1280) | merged | ci: run Kimi K3 vision evals nightly | `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` |
| 2026-08-29 | [#1300](https://github.com/lightseekorg/tokenspeed/pull/1300) | merged | perf(k3): end the fused MoE tail at its profit edge, not its capacity | `python/tokenspeed/runtime/models/kimi_k3_comm.py` |
| 2026-08-30 | [#1306](https://github.com/lightseekorg/tokenspeed/pull/1306) | merged | ci: add B300 Kimi K3 DeepSWE workflow | `test/ci/eval/kimi-k3-deepswe-b300.yaml` |
| 2026-08-30 | [#1315](https://github.com/lightseekorg/tokenspeed/pull/1315) | merged | ci: trust Kimi K3 tokenizer code | `test/ci/eval/kimi-k3-deepswe-b300.yaml` |
| 2026-08-31 | [#1318](https://github.com/lightseekorg/tokenspeed/pull/1318) | merged | perf(k3): decode gates, GEMV route QA, and FlashInfer 0.6.18 re-tune | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py` |
| 2026-08-31 | [#1330](https://github.com/lightseekorg/tokenspeed/pull/1330) | merged | perf(kimi3): let the MoE tail join its reductions without a lane | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-09-01 | [#1340](https://github.com/lightseekorg/tokenspeed/pull/1340) | merged | docs: fix kimi_k2.5 agentic bench README cd paths | `test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md`, `test/agentic_benchmark/kimi_k2.5/trtllm/README.md` |
| 2026-09-01 | [#1358](https://github.com/lightseekorg/tokenspeed/pull/1358) | merged | perf(kimi-k3): project the router, routed latent and shared gate/up together | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py` |
| 2026-09-03 | [#1375](https://github.com/lightseekorg/tokenspeed/pull/1375) | merged | perf(k3): optimize prefill MoE blocks on gfx950 | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-09-03 | [#1379](https://github.com/lightseekorg/tokenspeed/pull/1379) | merged | ci(kimi-k3): use EAGLE3 for AMD eval and perf gates | `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` |
| 2026-09-03 | [#1394](https://github.com/lightseekorg/tokenspeed/pull/1394) | merged | perf(k3): accelerate TP8/EP1 EAGLE3 verification | `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/moe.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/README.md` |
| 2026-09-04 | [#1326](https://github.com/lightseekorg/tokenspeed/pull/1326) | merged | test: update k3 agentic bench | `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm` |
| 2026-09-08 | [#1383](https://github.com/lightseekorg/tokenspeed/pull/1383) | merged | perf(k3): shard the latent MoE down projection by column | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py` |
| 2026-09-09 | [#1456](https://github.com/lightseekorg/tokenspeed/pull/1456) | merged | test: update k3 agentic bench (disagg part 1) | `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/configs/attn_dp16_moe_ep16.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm`, `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm` |
| 2026-09-10 | [#1458](https://github.com/lightseekorg/tokenspeed/pull/1458) | merged | perf(kimi3): add gfx1250 large-M WMMA projections | `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py` |
| 2026-09-10 | [#1457](https://github.com/lightseekorg/tokenspeed/pull/1457) | merged | feat(k3): reserve Iris buffers before KV cache sizing | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_comm_arming.py` |
| 2026-09-10 | [#1471](https://github.com/lightseekorg/tokenspeed/pull/1471) | merged | perf(kimi-k3): extend Iris all reduce window for moe | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py` |
| 2026-09-12 | [#1489](https://github.com/lightseekorg/tokenspeed/pull/1489) | merged | perf(k3): serve the attention reduce from the tokenspeed collective | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `test/runtime/test_kimi_k3_attn_res.py` |
| 2026-09-14 | [#1539](https://github.com/lightseekorg/tokenspeed/pull/1539) | merged | fix(kimi-k3): restore EAGLE3 prefill numerics | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py` |
| 2026-09-15 | [#1550](https://github.com/lightseekorg/tokenspeed/pull/1550) | merged | revert(kimi-k3): restore TP8 producer-direct window | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py` |
| 2026-09-15 | [#1563](https://github.com/lightseekorg/tokenspeed/pull/1563) | merged | feat(kimi-k3): support MoE all-to-all for attention DP | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-09-17 | [#1541](https://github.com/lightseekorg/tokenspeed/pull/1541) | merged | perf(k3): implement iris barrier free lamport all reduce for small M | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py` |
| 2026-09-17 | [#1545](https://github.com/lightseekorg/tokenspeed/pull/1545) | merged | perf(kimi-k3): select Iris fused push AR+attnres+rmsnorm through M=16 | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py` |
| 2026-09-18 | [#1635](https://github.com/lightseekorg/tokenspeed/pull/1635) | merged | feat(kimi-k3): add NVFP4 MegaMoE and opt-in autotuning | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py`, `test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml` |
| 2026-09-18 | [#1634](https://github.com/lightseekorg/tokenspeed/pull/1634) | merged | feat(kimi-k3): support Hopper PD with DeepEP and DSpark | `python/tokenspeed/runtime/models/kimi_k3_deepep.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py` |
| 2026-09-18 | [#1638](https://github.com/lightseekorg/tokenspeed/pull/1638) | merged | feat(kimi-k3): fuse sharded latent down proj with NVFP4 quantize | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`, `test/runtime/test_kimi_k3_config.py` |
| 2026-09-21 | [#1684](https://github.com/lightseekorg/tokenspeed/pull/1684) | merged | fix(ci): extend Kimi EAGLE3 AIME25 token budget | `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` |
| 2026-09-22 | [#1726](https://github.com/lightseekorg/tokenspeed/pull/1726) | merged | ci: move GB300 Kimi K3 TP8 evaluations to nightly | `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` |
| 2026-09-24 | [#1740](https://github.com/lightseekorg/tokenspeed/pull/1740) | merged | ci: benchmark Kimi-K3 EAGLE3 with TP8 EP1 at 50K/500 C16 | `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` |
| 2026-09-25 | [#1725](https://github.com/lightseekorg/tokenspeed/pull/1725) | merged | ci(amd-kernel): Add Kimi K3 MoE Kernel Benchmarks | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/moe.json` |
| 2026-09-26 | [#1807](https://github.com/lightseekorg/tokenspeed/pull/1807) | merged | ci(amd-kernel): Add Kimi K3 MLA Kernel Benchmarks | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` |
| 2026-09-27 | [#1795](https://github.com/lightseekorg/tokenspeed/pull/1795) | merged | perf(kimi3): split-K gfx1250 decode GEMMs and widen AttnRes | `tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/gemm/fp16/mm.py`, `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/README.md` |
| 2026-09-28 | [#1828](https://github.com/lightseekorg/tokenspeed/pull/1828) | merged | ci(amd-kernel): Add Kimi K3 a16w16 GEMM benchmark cases | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` |
| 2026-09-28 | [#1768](https://github.com/lightseekorg/tokenspeed/pull/1768) | merged | refactor(kimi3): select packed sigmoid top-k from the registry | `test/runtime/layers/test_kimi_moe_topk_gfx950.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py`, `tokenspeed-kernel/test/nvidia/ops/test_kimi3_sigmoid_topk_multitoken.py` |
| 2026-09-29 | [#1796](https://github.com/lightseekorg/tokenspeed/pull/1796) | merged | perf(kimi3): skip the C16 MoE cat on gfx1250 | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx1250.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py` |
| 2026-09-29 | [#1855](https://github.com/lightseekorg/tokenspeed/pull/1855) | merged | feat(amd): Add K3 support in Gluon MegaMoE | `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py` |
| 2026-09-30 | [#1884](https://github.com/lightseekorg/tokenspeed/pull/1884) | merged | ci(amd-kernel): Add Kimi K3 KDA prefill benchmark cases | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` |
| 2026-09-30 | [#1845](https://github.com/lightseekorg/tokenspeed/pull/1845) | merged | refactor(kimi-k3): refactor and optimize latent MoE tail | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py` |
| 2026-09-30 | [#1771](https://github.com/lightseekorg/tokenspeed/pull/1771) | merged | perf(kimi-k3): split post moe all reduce in prefill and shard the moe tail | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`, `test/runtime/test_kimi_k3_comm_arming.py` |
| 2026-10-01 | [#1790](https://github.com/lightseekorg/tokenspeed/pull/1790) | merged | perf(kimi-k3): shard prefill attention reduction and attnres | `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_deepep.py` |
| 2026-10-01 | [#1853](https://github.com/lightseekorg/tokenspeed/pull/1853) | merged | feat(DCP): Add Kimi K3 DCP support to the CuTe MLA backend | `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh`, `test/runtime/test_kimi_k3_mla.py`, `test/runtime/test_kimi_k3_cudagraph.py` |
| 2026-10-01 | [#1912](https://github.com/lightseekorg/tokenspeed/pull/1912) | merged | Fix KimiLinearMoE native layer state for attention DP | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py` |
| 2026-10-01 | [#1898](https://github.com/lightseekorg/tokenspeed/pull/1898) | merged | feat(amd): serve AMD Quark MXFP4 Kimi K3 with per-layer FP8 attention | `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py` |
| 2026-10-01 | [#1913](https://github.com/lightseekorg/tokenspeed/pull/1913) | merged | feat(gemm): add Gluon gfx950 decode GEMM for Kimi K3 | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` |
| 2026-10-02 | [#1896](https://github.com/lightseekorg/tokenspeed/pull/1896) | merged | ci(amd-kernel): Benchmark Kimi K3 fused KDA decode, verify and replay | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` |
| 2026-10-02 | [#1923](https://github.com/lightseekorg/tokenspeed/pull/1923) | merged | perf(amd): unify k3 attention all reduce selection | `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py` |
| 2026-10-02 | [#1937](https://github.com/lightseekorg/tokenspeed/pull/1937) | merged | ci(amd-kernel): Benchmark Kimi K3 MLA verify on the query axis | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` |
| 2026-10-03 | [#1943](https://github.com/lightseekorg/tokenspeed/pull/1943) | merged | ci(amd-kernel): Update Kimi K3 decode GEMM cases | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` |
| 2026-10-03 | [#1944](https://github.com/lightseekorg/tokenspeed/pull/1944) | merged | ci(amd-kernel): Benchmark Kimi K3 MLA cached extend | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` |
| 2026-10-03 | [#1947](https://github.com/lightseekorg/tokenspeed/pull/1947) | merged | ci(amd-kernel): Benchmark Kimi K3 AttnRes prefill mixing | `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/residual.json` |
| 2026-10-03 | [#1948](https://github.com/lightseekorg/tokenspeed/pull/1948) | merged | ci(amd): Raise Kimi K3 EAGLE3 50K/500 perf reference | `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` |
| 2026-10-03 | [#1946](https://github.com/lightseekorg/tokenspeed/pull/1946) | merged | perf(amd): route Kimi K3 prefill projections to Gluon large-M GEMM | `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` |
| 2026-10-04 | [#1966](https://github.com/lightseekorg/tokenspeed/pull/1966) | merged | perf(amd): Fuse K3 latent up-projection add3 for decode batches | `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` |
| 2026-10-04 | [#1992](https://github.com/lightseekorg/tokenspeed/pull/1992) | merged | fix(ci): drop expert-placement flags from Kimi-K3 AMD job | `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` |

## 逐 PR diff 审计卡

### PR #29 - Add Kimi K2.5 agentic perf CI task

- 链接: https://github.com/lightseekorg/tokenspeed/pull/29
- 状态/时间: merged / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+387/-3，本地 patch 650 行。
- 动机: 把 `nvidia/Kimi-K2.5-NVFP4` 的 agentic workload 变成 perf CI，而不是只靠手工压测。
- 实现要点: 新增 Kimi K2.5 agentic perf YAML，服务端使用 `python3 -m tokenspeed.api_server`，配套 `tokenspeed_mla`、NVFP4、speculative draft 和 EvalScope agentic workload。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--attention-backend tokenspeed_mla
+--quantization nvfp4
```

- 已读文件: `.github/workflows/pr-test.yml`, `test/ci/perf/kimi-k2.5-nvfp4-evalscope-agentic.yaml`, CI pipeline helper
- 验证与风险: SOTA loop 里要把 agentic perf lane 和公共 synthetic workload 分开记录；TokenSpeed 的领先可能来自 workload/state 管理，而不是单个 kernel。

### PR #126 - perf(K2.5): Optimize lm_head

- 链接: https://github.com/lightseekorg/tokenspeed/pull/126
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 6 个文件，+1173/-3，本地 patch 1,246 行。
- 动机: Kimi K2.5 decode 末端 `lm_head` 大 GEMM 显著影响 TPOT；PR 用专用 CUDA kernel 替换一般 `torch.matmul` 路径。
- 实现要点: `LogitsProcessor` 里按 `model_type == "kimi_k2"` gate fused path；新增 `lm_head_gemm.cu`、binding 和 Python wrapper，并保留 shape 不匹配时 fallback。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+self._use_fused_lm_head = getattr(self.config, "model_type", None) == "kimi_k2"
+logits = _lm_head_matmul(hidden_states, lm_head.weight)
```

- 已读文件: `logits_processor.py`, `lm_head_gemm.cu`, `lm_head_gemm_binding.cu`, `lm_head_gemm.py`, setup files
- 验证与风险: 对 SGLang Kimi/Qwen 类模型，`lm_head` 需要单独进 profiler 表；不能只优化 attention/MoE 后就宣布收敛。

### PR #184 - perf(K2.5): optimize top_k_renorm_prob + top_p_renorm_prob

- 链接: https://github.com/lightseekorg/tokenspeed/pull/184
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 8 个文件，+3104/-12，本地 patch 3,580 行。
- 动机: sampling backend 连续调用 `top_k_renorm_prob` 与 deterministic `top_p_renorm_prob`，在高并发 decode 尾部形成重复扫描和多 launch。
- 实现要点: 新增 fused TopK+TopP renorm CUDA 路径，按 `top_k` sentinel 切分分支，接入 `flashinfer_full.py` 和 `server_args.py` 的参数限制。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
-probs = top_k_renorm_prob(probs, top_ks)
-probs = top_p_renorm_prob(probs, top_ps, is_deterministic=True)
+probs = fused_topk_topp_renorm(probs, top_ks, top_ps)
```

- 已读文件: `fused_topk_topp/*`, `flashinfer_full.py`, `server_args.py`, sampling tests
- 验证与风险: 对 SGLang profiler 的采样阶段，若 top-k/top-p 占比高，应把 sampling kernel 当成主优化面；同时检查 `top_k < 128` 这类 kernel contract。

### PR #253 - ci(eval): add Kimi-K2.5-NVFP4 ocr_bench task

- 链接: https://github.com/lightseekorg/tokenspeed/pull/253
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+48/-0，本地 patch 72 行。
- 动机: Kimi K2.5 是强多模态模型，OCR benchmark 需要进入常规 eval lane。
- 实现要点: 新增 `kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`，复用 Kimi NVFP4 server config，只把 EvalScope dataset 切到 `ocr_bench`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--model nvidia/Kimi-K2.5-NVFP4
+--datasets ocr_bench
```

- 已读文件: `test/ci/eval/kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- 验证与风险: 多模态 SOTA loop 要保留 OCR lane；text-only benchmark 不能代表 Kimi K2.5 全部优化收益。

### PR #444 - feat(moe): add trtllm mxint4 MoE path for Kimi-K2.x

- 链接: https://github.com/lightseekorg/tokenspeed/pull/444
- 状态/时间: merged / 2026-06-14
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 8 个文件，+469/-6，本地 patch 581 行。
- 动机: Kimi K2.x 需要 INT4 W4A16 group-32 MoE path，现有 NVFP4/MXFP4 path 不能覆盖 MXINT4 checkpoint。
- 实现要点: 新增 `create_mxint4_weight_pair`、quant config 识别和 `flashinfer_trtllm_mxint4` MoE process/apply op。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+from tokenspeed.runtime.layers.moe.weights.mxint4 import create_mxint4_weight_pair
+name="flashinfer_trtllm_mxint4_moe_apply"
```

- 已读文件: `expert.py`, `weights/mxint4.py`, quantization configs, `trtllm_mxint4.py`
- 验证与风险: SGLang 做 Kimi K2.x 量化路径对比时，要把 weight dtype、group size、activation dtype 和 FlashInfer TRT-LLM op 名称全部记录进结果表。

### PR #418 - Add InstantTensor weight loader

- 链接: https://github.com/lightseekorg/tokenspeed/pull/418
- 状态/时间: merged / 2026-06-15
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 25 个文件，+468/-60，本地 patch 1,373 行。
- 动机: Kimi K2.5 大模型启动和权重加载成本高，需要 `--load-format instanttensor` 这样的专用 loader。
- 实现要点: 新增 loader/weight utils 分支，接入 server args、Kimi model 权重路径和 CI eval 文档。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--load-format instanttensor
+        elif self.load_config.load_format == LoadFormat.INSTANTTENSOR:
+            weights_iterator = instanttensor_weights_iterator(hf_weights_files)
```

- 已读文件: `runtime/model_loader/loader.py`, `weight_utils.py`, `runtime/models/kimi_k25.py`, `runtime/utils/server_args.py`, docs and eval configs
- 验证与风险: 公平 benchmark 要记录冷启动/热启动边界；如果比较 steady-state TPOT，InstantTensor 不应混入吞吐结论。

### PR #454 - [AMD] Support Kimi K2.5 MXFP4 serving

- 链接: https://github.com/lightseekorg/tokenspeed/pull/454
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 33 个文件，+1924/-142，本地 patch 3,856 行。
- 动机: AMD 平台需要 Kimi K2.5 MXFP4 serving，包括 attention、dense、MoE、quantization 和模型加载路径。
- 实现要点: 新增 MXFP4 quantization/layer/dense 支持，调整 MLA backend、Kimi model 和 AMD 相关测试。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--quantization mxfp4
+model_type == "kimi_k25"
```

- 已读文件: MXFP4 layers/quantization, dense kernels, attention backends, `runtime/models/kimi_k25.py`, tests
- 验证与风险: 这是跨硬件路径，不应把 AMD MXFP4 结论直接外推到 NVIDIA NVFP4；SOTA loop 应按 GPU/backend 分 lane。

### PR #477 - perf(kernel): Optimize Kimi Vision FA4 QKV + RoPE

- 链接: https://github.com/lightseekorg/tokenspeed/pull/477
- 状态/时间: merged / 2026-06-19
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 3 个文件，+195/-7，本地 patch 304 行。
- 动机: Kimi vision FA4 path 的 complex RoPE 和 packed QKV 拆分存在额外 layout 搬运。
- 实现要点: 新增 `packed_qkv_complex_rotary` Triton kernel，在 multimodal encoder attention 中走 packed complex-RoPE fast path。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+        if use_packed_qkv_complex_rotary:
+            q, k, v = packed_qkv_complex_rotary(
+def packed_qkv_complex_rotary(
```

- 已读文件: `mm_encoder_attention.py`, `runtime/models/kimi_k25.py`, `qkv_rotary.py`
- 验证与风险: SGLang Kimi/OCR VLM 优化需要关注 FA4 attention 前的 QKV/RoPE layout，而不是只看 FA4 主 kernel。

### PR #482 - ci: use FA4 mm attention for Kimi OCR eval

- 链接: https://github.com/lightseekorg/tokenspeed/pull/482
- 状态/时间: merged / 2026-06-19
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+1/-0，本地 patch 22 行。
- 动机: OCR eval 应覆盖新的 FA4 multimodal attention path，否则 #477 的 kernel 优化没有固定回归 lane。
- 实现要点: 在 Kimi OCR eval YAML 增加 `--mm-attention-backend fa4`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--mm-attention-backend fa4
```

- 已读文件: `test/ci/eval/kimi-k2.5-nvfp4-evalscope-ocr-bench.yaml`
- 验证与风险: 多模态 benchmark 需要显式记录 `mm-attention-backend`，否则同一个 Kimi checkpoint 会跑出不同 kernel 形态。

### PR #354 - feat(video) Generalize multimodal runtime support and add Qwen3.5 video

- 链接: https://github.com/lightseekorg/tokenspeed/pull/354
- 状态/时间: merged / 2026-06-23
- 反查来源: 保留自原 history/skill 显式引用
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+982/-266，可读 patch 1880 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/multimodal/encoder_cudagraph.py` modified +425/-195 (620 lines); hunks: -18,54 +18,123; -74,7 +143,7 @@ def num_items(self) -> int:; symbols: BudgetGraphMetadata, EncoderCudaGraphBatch, input_tensors, encoder_output_tokens，涉及 `BudgetGraphMetadata, EncoderCudaGraphBatch, input_tensors`；`python/tokenspeed/runtime/execution/model_executor.py` modified +197/-9 (206 lines); hunks: -72,6 +72,7; -407,27 +408,60 @@ def __init__(; symbols: _draft_idle_global_num_tokens_for_step, __init__, _make_mrope_decode_deltas_cpu, capturable_grammar，涉及 `_draft_idle_global_num_tokens_for_step, __init__, _make_mrope_decode_deltas_cpu`；`python/tokenspeed/runtime/multimodal/embedder.py` modified +141/-12 (153 lines); hunks: -52,6 +52,8; -65,9 +67,14; symbols: EncoderSpec, pad_input_tokens, EncodePlan, __bool__，涉及 `EncoderSpec, pad_input_tokens, EncodePlan`；`python/tokenspeed/runtime/models/qwen3_5.py` modified +70/-13 (83 lines); hunks: -95,7 +95,10; -106,6 +109,8; symbols: __init__, get_image_feature, get_video_feature, pre_encode，涉及 `__init__, get_image_feature, get_video_feature`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/multimodal/encoder_cudagraph.py` modified +425/-195 (620 lines); hunks: -18,54 +18,123; -74,7 +143,7 @@ def num_items(self) -> int:; symbols: BudgetGraphMetadata, EncoderCudaGraphBatch, input_tensors, encoder_output_tokens
  - `python/tokenspeed/runtime/execution/model_executor.py` modified +197/-9 (206 lines); hunks: -72,6 +72,7; -407,27 +408,60 @@ def __init__(; symbols: _draft_idle_global_num_tokens_for_step, __init__, _make_mrope_decode_deltas_cpu, capturable_grammar
  - `python/tokenspeed/runtime/multimodal/embedder.py` modified +141/-12 (153 lines); hunks: -52,6 +52,8; -65,9 +67,14; symbols: EncoderSpec, pad_input_tokens, EncodePlan, __bool__
  - `python/tokenspeed/runtime/models/qwen3_5.py` modified +70/-13 (83 lines); hunks: -95,7 +95,10; -106,6 +109,8; symbols: __init__, get_image_feature, get_video_feature, pre_encode
  - `python/tokenspeed/runtime/multimodal/shm_transport.py` modified +60/-15 (75 lines); hunks: -20,19 +20,28; -77,6 +86,7 @@ def consume(self) -> torch.Tensor:; symbols: ShmTensorHandle, consume, release, sync_shm_features
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/multimodal/encoder_cudagraph.py
@@ -18,54 +18,123 @@
-"""Budget-bucketed CUDA graph capture/replay for vision encoders.
-Vision-encoder analogue of the LM :class:`CudaGraphWrapper`. Capture-safety invariants
-(violating any of these is silent numerical corruption):
-  * each budget graph gets its own private pool -- a shared pool collides the
-    vision-TP custom-AR IPC buffer registrations across budgets;
-  * ``max_seqlen`` is baked at the per-budget worst case (single image filling
diff -- python/tokenspeed/runtime/execution/model_executor.py
@@ -72,6 +72,7 @@
+LOG_MM_TIMING = envs.TOKENSPEED_LOG_MM_TIMING.get()
@@ -407,27 +408,60 @@ def __init__(
-        # Encoder CUDA graph: install the model-built wrapper by overriding
-        # ``image_encoder``. Vision-encoder analogue of ``forward_step``'s
-        # ``CudaGraphWrapper``.
-        self.encoder_graph_wrapper = None
diff -- python/tokenspeed/runtime/multimodal/embedder.py
@@ -52,6 +52,8 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/multimodal/encoder_cudagraph.py` modified +425/-195; `python/tokenspeed/runtime/execution/model_executor.py` modified +197/-9; `python/tokenspeed/runtime/multimodal/embedder.py` modified +141/-12; `python/tokenspeed/runtime/models/qwen3_5.py` modified +70/-13; `python/tokenspeed/runtime/multimodal/shm_transport.py` modified +60/-15; `python/tokenspeed/runtime/models/kimi_k25.py` modified +25/-16
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/engine/generation_output_processor.py`, `python/tokenspeed/runtime/engine/input_processor.py`, `python/tokenspeed/runtime/execution/model_executor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #476 - Add AMD Kimi MXFP4 CI job

- 链接: https://github.com/lightseekorg/tokenspeed/pull/476
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 3 个文件，+138/-4，本地 patch 181 行。
- 动机: #454 接入 AMD MXFP4 后，需要持续验证 AIME25 eval 和 MLA metadata 兼容性。
- 实现要点: 新增 `kimi-k2.5-mxfp4-evalscope-aime25-amd.yaml`，并增加 `MLAAttnBackend` metadata unit test。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+--model amd/Kimi-K2.5-MXFP4
+--quantization mxfp4
```

- 已读文件: `mla.py`, `kimi-k2.5-mxfp4-evalscope-aime25-amd.yaml`, `test_mla_verify_metadata.py`
- 验证与风险: 这条 lane 是 AMD 特化；SGLang 竞品比较时要在结果里标出硬件和 quantization，不要把它当成通用 Kimi lane。


### PR #596 - 修复 Kimi DP EAGLE3 mixed-step hang

- 链接: https://github.com/lightseekorg/tokenspeed/pull/596
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 184 行 diff，6 个文件，+27/-30。
- 动机: 同一 scheduler step 中 DP rank 混合 EXTEND 与 DECODE 时，active 与 idle rank 可能为 EAGLE3 首次 catch-up collective 计算出不同 row 数，最终 hang。
- 实现要点: 把首步激活缩减定义为 active EAGLE 与 idle replay 共享的 draft-model capability，为 Kimi/DeepSeek、Llama、Qwen3.5 draft model 标记该能力，并禁止 fused `lm_head_gemm` 对 zero-token 发起 kernel。
- 代码 diff 细节: 用 `draft_first_step_reduce_for_catchup` 替换硬编码 class check，让 collective sizing 跟随模型行为而不是模型列表。
- 关键代码摘录:

```diff
+def draft_model_reduces_first_step_catchup(draft_model) -> bool:
+    return bool(getattr(draft_model, "draft_first_step_reduce_for_catchup", False))
+draft_first_step_reduce = step_idx == 0 and (
+    all_decode_or_idle or draft_reduces_first_step_catchup)
```

- 已读文件: runtime：`execution/drafter/eagle.py`、`execution/model_executor.py`、`models/{deepseek_v3,llama_eagle3,qwen3_5_nextn}.py`、`lm_head_gemm.py`；没有新增独立测试文件。
- 验证与风险: mixed forward mode 下所有 rank 必须得到相同 collective row count；PR 报告修复后 DP8 + EAGLE3 AIME25 以 28/30 完成，zero-row fused lm-head routing 必须保持 no-op。

### PR #795 - 为 Kimi-K2.7 Code 支持 EAGLE3.1

- 链接: https://github.com/lightseekorg/tokenspeed/pull/795
- 状态/时间: merged / 2026-07-25
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 49 行 diff，1 个文件，+24/-0。
- 动机: Kimi-K2.7 EAGLE3.1 MLA speculator 定义了逐 FC 输入的归一化与可选 normalized auxiliary output，而共享 DeepSeek-style drafter 尚未实现。
- 实现要点: `fc_norm` 开启时为每个拼接 FC 输入 chunk 建立 RMSNorm，在 projection 前分别归一化，并用 `norm_output` 控制 auxiliary hidden-state 输出。
- 代码 diff 细节: 所有变化都在 `Eagle3MlaModel` 内受 config gate 控制，不改变旧 EAGLE checkpoint。
- 关键代码摘录:

```diff
+if self.fc_norm is not None:
+    chunks = hidden_states.chunk(self.num_fc_input_dim, dim=-1)
+    hidden_states = torch.cat(
+        [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)], dim=-1)
```

- 已读文件: runtime：`python/tokenspeed/runtime/models/deepseek_v3.py`；验证证据：`nvidia/Kimi-K2.7-Code-NVFP4` 的 PR benchmark 与 launch recipe。
- 验证与风险: `fc_norm`/`norm_output` 必须与 checkpoint config 绑定；PR 报告 4xGB200 1-3-4 配置各类别 1.36x-1.91x，但不能外推到不同 acceptance length 或 serving shape。

### PR #797 - 支持 Kimi incremental DFlash capture

- 链接: https://github.com/lightseekorg/tokenspeed/pull/797
- 状态/时间: merged / 2026-07-26
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 166 行 diff，4 个文件，+61/-7。
- 动机: Kimi 把 DFlash capture 委托给 DeepSeek-style language model 时丢失了 executor 所需的 incremental projection callback 与 slot buffer，导致开启 incremental projection 后无法启动。
- 实现要点: 让 `KimiK25ForConditionalGeneration.set_dflash_layers_to_capture` 透传 callback 与 slot buffer，保存 layer-to-slot map，每层完成时把 hidden state 拷入相应 slot 并立即调用 incremental projection callback。
- 代码 diff 细节: 模型用 `_dflash_incr_active` 控制生命周期；CI 还让 Slurm server startup timeout 与 readiness 对齐，避免长时间 Kimi 启动被另一套超时提前终止。
- 关键代码摘录:

```diff
+self.model._dflash_capture_idx_map = {
+    layer_idx: i for i, layer_idx in enumerate(sorted(self.model.layers_to_capture))
+}
+self.model._dflash_incremental_callback(capture_idx, num_tokens)
```

- 已读文件: runtime：`models/deepseek_v3.py`、`models/kimi_k25.py`；测试/CI：`test/ci_system/{pipeline,test_pipeline}.py`。
- 验证与风险: callback 顺序、slot 容量、CUDA stream 生命周期和 `_dflash_incr_active` reset 必须与 executor 一致；PR 记录了 syntax/pre-commit 检查和 live B200 验证 lane。

### PR #806 - fix(ci): extend Kimi engine startup timeout

- 链接: https://github.com/lightseekorg/tokenspeed/pull/806
- 状态/时间: merged / 2026-07-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gsm8k.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-mmlu.yaml` 等 7 个文件；关联提交 `ebdee770fd2f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+14/-0，可读 patch 112 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml` modified +2/-0 (2 lines); hunks: -12,6 +12,7 @@ env:; -32,6 +33,7 @@ server:；`test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` modified +2/-0 (2 lines); hunks: -13,6 +13,7 @@ env:; -30,6 +31,7 @@ server:；`test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml` modified +2/-0 (2 lines); hunks: -12,6 +12,7 @@ env:; -29,6 +30,7 @@ server:；`test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gsm8k.yaml` modified +2/-0 (2 lines); hunks: -12,6 +12,7 @@ env:; -29,6 +30,7 @@ server:。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml` modified +2/-0 (2 lines); hunks: -12,6 +12,7 @@ env:; -32,6 +33,7 @@ server:
  - `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` modified +2/-0 (2 lines); hunks: -13,6 +13,7 @@ env:; -30,6 +31,7 @@ server:
  - `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml` modified +2/-0 (2 lines); hunks: -12,6 +12,7 @@ env:; -29,6 +30,7 @@ server:
  - `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gsm8k.yaml` modified +2/-0 (2 lines); hunks: -12,6 +12,7 @@ env:; -29,6 +30,7 @@ server:
  - `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-mmlu.yaml` modified +2/-0 (2 lines); hunks: -12,6 +12,7 @@ env:; -29,6 +30,7 @@ server:
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml
@@ -12,6 +12,7 @@ env:
+  # Temporary NFS cold-load headroom; revert after the startup bottleneck is fixed.
@@ -32,6 +33,7 @@ server:
+    --engine-startup-timeout 2400
diff -- test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml
@@ -13,6 +13,7 @@ env:
+  # Temporary NFS cold-load headroom; revert after the startup bottleneck is fixed.
@@ -30,6 +31,7 @@ server:
+    --engine-startup-timeout 2400
diff -- test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml
@@ -12,6 +12,7 @@ env:
+  # Temporary NFS cold-load headroom; revert after the startup bottleneck is fixed.
@@ -29,6 +30,7 @@ server:
+    --engine-startup-timeout 2400
diff -- test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gsm8k.yaml
@@ -12,6 +12,7 @@ env:
+  # Temporary NFS cold-load headroom; revert after the startup bottleneck is fixed.
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml` modified +2/-0; `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` modified +2/-0; `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml` modified +2/-0; `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gsm8k.yaml` modified +2/-0; `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-mmlu.yaml` modified +2/-0; `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-ocr-bench.yaml` modified +2/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k2.5-nvfp4-dflash-evalscope-aime25.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gpqa-diamond.yaml`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-gsm8k.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #821 - 新增 Kimi K3 部署 recipe

- 链接: https://github.com/lightseekorg/tokenspeed/pull/821
- 状态/时间: merged / 2026-07-27
- 反查来源: 最终上游提交
  `55a8390007e5ace17919d76e5cfaef0c68c79e25`；已读完 2-commit 增量和
  recipe 完整 diff。
- 代码 diff 已读范围: 1 file，+117/-0。
- 动机: 明确 Kimi K3 的 runtime/packaging 约束，避免把 K2.5 flags 当成可互换配置。
- 实现要点: 要求 FlatKV，记录 vendor-neutral KDA dispatch 与 NVIDIA FLA-derived /
  AMD native state layout、MLA backend 选择，以及 NVIDIA B300、AMD gfx950 分离命令。
- 代码 diff 细节: recipe 新增 FlatKV build/preflight、flattened checkpoint、
  writable module cache、NVIDIA `tokenspeed-situ` + Triton fallback 和 AMD Gluon 路径。
- 关键代码摘录:

```diff
+- K3 is FlatKV-only. Build the `tokenspeed_scheduler` extension with
+  `-DTOKENSPEED_FLAT_KVCACHE=ON`
+tokenspeed serve moonshotai/Kimi-K3 \
+  --kv-cache-dtype fp8 \
+  --tensor-parallel-size 8
```

- 已读文件: `docs/recipes/models.md`；后续 README commit 只链接 K3 公告。
- 验证与风险: NVIDIA 依赖 B300/CUDA 13 `tokenspeed-situ` sidecar，其他平台回退
  Triton；checkpoint 需 flattened、remote-code cache 需可写，默认 FP8 KV scale
  可能影响精度。这不是跨框架性能实测。

### PR #822 - feat(kimi-k3): integrate Kimi K3 support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/822
- 状态/时间: merged / 2026-07-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/kimi_k3_config.py`, `python/tokenspeed/runtime/models/kimi_k25.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py`, `python/tokenspeed/runtime/models/moonvit.py` 等 14 个文件；关联提交 `0f6867606914`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 216 个文件，+33902/-1318，可读 patch 39415 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` added +2109/-0 (2109 lines); hunks: -0,0 +1,2109; symbols: KimiK3Vision, load_weight, KimiLinearMLP, __init__，涉及 `KimiK3Vision, load_weight, KimiLinearMLP`；`python/tokenspeed/runtime/models/moonvit.py` added +978/-0 (978 lines); hunks: -0,0 +1,978; symbols: ModelSlimConfig, MoonViTConfig, from_config, MLP2，涉及 `ModelSlimConfig, MoonViTConfig, from_config`；`python/tokenspeed/runtime/models/kimi_k25.py` modified +55/-803 (858 lines); hunks: -22,719 +22,43; -743,27 +67,13 @@ def __init__(; symbols: ModelSlimConfig, QuarkConfig, MLP2, __init__，涉及 `ModelSlimConfig, QuarkConfig, MLP2`；`python/tokenspeed/runtime/models/kimi_k3_nextn.py` added +498/-0 (498 lines); hunks: -0,0 +1,498; symbols: KimiK3DraftAttentionMLA, forward, KimiK3DraftDecoderLayer, __init__，涉及 `KimiK3DraftAttentionMLA, forward, KimiK3DraftDecoderLayer`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` added +2109/-0 (2109 lines); hunks: -0,0 +1,2109; symbols: KimiK3Vision, load_weight, KimiLinearMLP, __init__
  - `python/tokenspeed/runtime/models/moonvit.py` added +978/-0 (978 lines); hunks: -0,0 +1,978; symbols: ModelSlimConfig, MoonViTConfig, from_config, MLP2
  - `python/tokenspeed/runtime/models/kimi_k25.py` modified +55/-803 (858 lines); hunks: -22,719 +22,43; -743,27 +67,13 @@ def __init__(; symbols: ModelSlimConfig, QuarkConfig, MLP2, __init__
  - `python/tokenspeed/runtime/models/kimi_k3_nextn.py` added +498/-0 (498 lines); hunks: -0,0 +1,498; symbols: KimiK3DraftAttentionMLA, forward, KimiK3DraftDecoderLayer, __init__
  - `python/tokenspeed/runtime/configs/kimi_k3_config.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: KimiK3VisionConfig, __init__, KimiLinearConfig, is_kda_layer
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -0,0 +1,2109 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/moonvit.py
@@ -0,0 +1,978 @@
+# SPDX-License-Identifier: MIT AND Apache-2.0
+# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
+# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
+#
+# Copyright (c) 2026 LightSeek Foundation
+#
diff -- python/tokenspeed/runtime/models/kimi_k25.py
@@ -22,719 +22,43 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` added +2109/-0; `python/tokenspeed/runtime/models/moonvit.py` added +978/-0; `python/tokenspeed/runtime/models/kimi_k25.py` modified +55/-803; `python/tokenspeed/runtime/models/kimi_k3_nextn.py` added +498/-0; `python/tokenspeed/runtime/configs/kimi_k3_config.py` added +421/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` added +824/-0
  - tests: `test/runtime/models/test_kimi_k3_vlm.py` added +258/-0; `test/runtime/layers/test_kimi_moe_topk_gfx950.py` added +195/-0
- 验证与风险: diff 自带测试面 `test/cli/test_serve_smg_unit.py`, `test/runtime/cache/test_mla_kv_buffer.py`, `test/runtime/conftest.py`, `test/runtime/distributed/test_auto_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #847 - fix(kimi3): correct AMD KDA safe gate

- 链接: https://github.com/lightseekorg/tokenspeed/pull/847
- 状态/时间: merged / 2026-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/test/kimi3_reference.py`；关联提交 `478fdf559f10`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+146/-6，可读 patch 190 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/test/kimi3_reference.py` modified +68/-0 (68 lines); hunks: -37,6 +37,74 @@ def situ_and_mul(; symbols: situ_and_mul, kda_gate, kda_recurrent，涉及 `situ_and_mul, kda_gate, kda_recurrent`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/kda.py` modified +11/-6 (17 lines); hunks: -64,11 +64,12 @@ def _kda_prepare_gate_beta_kernel(; -411,10 +412,14 @@ def _kda_recurrent_decode_kernel(; symbols: _kda_prepare_gate_beta_kernel, _kda_recurrent_decode_kernel，涉及 `_kda_prepare_gate_beta_kernel, _kda_recurrent_decode_kernel`。
- 代码 diff 细节:
  - `tokenspeed-kernel/test/kimi3_reference.py` modified +68/-0 (68 lines); hunks: -37,6 +37,74 @@ def situ_and_mul(; symbols: situ_and_mul, kda_gate, kda_recurrent
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/kda.py` modified +11/-6 (17 lines); hunks: -64,11 +64,12 @@ def _kda_prepare_gate_beta_kernel(; -411,10 +412,14 @@ def _kda_recurrent_decode_kernel(; symbols: _kda_prepare_gate_beta_kernel, _kda_recurrent_decode_kernel
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/test/kimi3_reference.py
@@ -37,6 +37,74 @@ def situ_and_mul(
+def kda_gate(
+    raw_g: torch.Tensor,
+    a_log: torch.Tensor,
+    dt_bias: torch.Tensor,
+    *,
+    lower_bound: float | None = -5.0,
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/kda.py
@@ -64,11 +64,12 @@ def _kda_prepare_gate_beta_kernel(
-    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
-    g = -tl.exp(a) * softplus
-        g = tl.maximum(g, LOWER_BOUND)
+        g = LOWER_BOUND * tl.sigmoid(tl.exp(a) * x)
+    else:
+        softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
```

- 提取文件（未人工审阅）:
  - tests: `tokenspeed-kernel/test/kimi3_reference.py` modified +68/-0
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/kda.py` modified +11/-6
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/kimi3_reference.py`, `tokenspeed-kernel/test/ops/test_kda_recurrent.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #843 - ci(kimi-k3): add 8-GPU TP8/EP8 AIME26 eval and 4k/1k perf tasks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/843
- 状态/时间: merged / 2026-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；关联提交 `b866d96816f4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+248/-12，可读 patch 307 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` added +94/-0 (94 lines); hunks: -0,0 +1,94；`test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` added +62/-0 (62 lines); hunks: -0,0 +1,62。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` added +94/-0 (94 lines); hunks: -0,0 +1,94
  - `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` added +62/-0 (62 lines); hunks: -0,0 +1,62
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml
@@ -0,0 +1,94 @@
+api_version: ci.tokenspeed.io/v1
+name: perf-kimi-k3-mxfp4-tp8ep8-random-4k-1k-mi35x
+type: perf
+triggers:
+  - per-commit
+  - manual
diff -- test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml
@@ -0,0 +1,62 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-kimi-k3-mxfp4-tp8ep8-aime26-amd
+type: eval
+triggers:
+  - per-commit
+  - manual
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` added +94/-0; `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` added +62/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/ci_system/test_random_benchmark_perf_csv.py`, `test/random_benchmark/tokenspeed/collect_outputs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #852 - refactor(kimi-k3): replace tokenspeed-situ sidecar with flashinfer native SiTU MoE

- 链接: https://github.com/lightseekorg/tokenspeed/pull/852
- 状态/时间: merged / 2026-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `a8a6843ab287`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 20 个文件，+1264/-193，可读 patch 1854 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +30/-43 (73 lines); hunks: -31,7 +31,7; -82,6 +82,10; symbols: _apply_attn_res, _situ_sidecar_unavailable_reason, _situ_betas, KimiLinearMoE，涉及 `_apply_attn_res, _situ_sidecar_unavailable_reason, _situ_betas`；`test/runtime/test_kimi_k3_config.py` modified +1/-1 (2 lines); hunks: -338,7 +338,7 @@ def __init__(self, **kwargs):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +30/-43 (73 lines); hunks: -31,7 +31,7; -82,6 +82,10; symbols: _apply_attn_res, _situ_sidecar_unavailable_reason, _situ_betas, KimiLinearMoE
  - `test/runtime/test_kimi_k3_config.py` modified +1/-1 (2 lines); hunks: -338,7 +338,7 @@ def __init__(self, **kwargs):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -31,7 +31,7 @@
-* ``KimiLinearMoE`` — sigmoid/noaux_tc router + Latent MoE + sidecar-backed
+* ``KimiLinearMoE`` — sigmoid/noaux_tc router + Latent MoE + flashinfer's
@@ -82,6 +82,10 @@
+from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
+    situ_moe_unavailable_reason,
+)
diff -- test/runtime/test_kimi_k3_config.py
@@ -338,7 +338,7 @@ def __init__(self, **kwargs):
-                    use_sidecar=False,
+                    use_trtllm=False,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +30/-43
  - tests: `test/runtime/test_kimi_k3_config.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci_system/flashinfer_jit_cache_installer.py`, `test/ci_system/install_deps.sh`, `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #906 - Kimi K3 support dummy weight

- 链接: https://github.com/lightseekorg/tokenspeed/pull/906
- 状态/时间: merged / 2026-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`；关联提交 `418ec9d26dc3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+5/-0，可读 patch 12 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +5/-0 (5 lines); hunks: -2046,6 +2046,11 @@ def forward(; symbols: forward, post_load_weights, load_weights，涉及 `forward, post_load_weights, load_weights`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +5/-0 (5 lines); hunks: -2046,6 +2046,11 @@ def forward(; symbols: forward, post_load_weights, load_weights
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -2046,6 +2046,11 @@ def forward(
+    def post_load_weights(self) -> None:
+        """Prepare text-model derived weights for loaders that skip checkpoints."""
+        if self.language_model is not None:
+            self.language_model.post_load_weights()
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/kimi_k3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #909 - fix(k3): derive Kimi-K3 KDA page packing from the MLA plane size

- 链接: https://github.com/lightseekorg/tokenspeed/pull/909
- 状态/时间: merged / 2026-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/test_kimi_k3_cache_spec.py`；关联提交 `5b3e90e54776`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+45/-8，可读 patch 87 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/test_kimi_k3_cache_spec.py` modified +24/-0 (24 lines); hunks: -58,6 +58,30 @@ def test_lcm_reference_geometry_is_exact() -> None:; symbols: test_lcm_reference_geometry_is_exact, test_lcm_geometry_packs_two_kda_pages_at_tp16, test_lcm_parent_demand_uses_per_group_packing，涉及 `test_lcm_reference_geometry_is_exact, test_lcm_geometry_packs_two_kda_pages_at_tp16, test_lcm_parent_demand_uses_per_group_packing`；`python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` modified +21/-8 (29 lines); hunks: -41,6 +41,7; -179,27 +180,39 @@ def plan_kimi_k3_lcm_cache(; symbols: _require_non_negative_int, plan_kimi_k3_lcm_cache，涉及 `_require_non_negative_int, plan_kimi_k3_lcm_cache`。
- 代码 diff 细节:
  - `test/runtime/test_kimi_k3_cache_spec.py` modified +24/-0 (24 lines); hunks: -58,6 +58,30 @@ def test_lcm_reference_geometry_is_exact() -> None:; symbols: test_lcm_reference_geometry_is_exact, test_lcm_geometry_packs_two_kda_pages_at_tp16, test_lcm_parent_demand_uses_per_group_packing
  - `python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` modified +21/-8 (29 lines); hunks: -41,6 +41,7; -179,27 +180,39 @@ def plan_kimi_k3_lcm_cache(; symbols: _require_non_negative_int, plan_kimi_k3_lcm_cache
- 关键代码摘录:

```diff
diff -- test/runtime/test_kimi_k3_cache_spec.py
@@ -58,6 +58,30 @@ def test_lcm_reference_geometry_is_exact() -> None:
+def test_lcm_geometry_packs_two_kda_pages_at_tp16() -> None:
+    """KDA state halves at TP16; two pages pack per MLA-sized plane."""
+    plan = plan_kimi_k3_lcm_cache(
+        KimiLinearConfig(),
+        flat_kvcache_enabled=True,
+        tp_size=16,
diff -- python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py
@@ -41,6 +41,7 @@
+_KIMI_K3_MLA_PACKING = 12
@@ -179,27 +180,39 @@ def plan_kimi_k3_lcm_cache(
+    # The MLA latent history is TP-invariant while the KDA state shards by
+    # TP, so pack as many KDA pages per MLA-sized plane as fit to keep the
+    # planner's padding fraction bounded at any TP.
+    mla_plane_bytes = _KIMI_K3_MLA_PACKING * next(
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/test_kimi_k3_cache_spec.py` modified +24/-0
  - runtime: `python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` modified +21/-8
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_cache_spec.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #876 - chore(kimi-k3): serve on GB200 -- LCM packing, layer derivation, kda la…

- 链接: https://github.com/lightseekorg/tokenspeed/pull/876
- 状态/时间: merged / 2026-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/test_kimi_k3_config.py`；关联提交 `ffa91254ec54`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+195/-23，可读 patch 344 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/test_kimi_k3_config.py` modified +80/-0 (80 lines); hunks: -478,3 +478,83 @@ def load_weights(self, weights):; symbols: load_weights, KimiK3LcmPlanTests, _plan, test_linear_packing_scales_with_attn_tp，涉及 `load_weights, KimiK3LcmPlanTests, _plan`；`python/tokenspeed/runtime/multimodal/shm_transport.py` modified +68/-2 (70 lines); hunks: -59,6 +59,9 @@ class ShmTensorHandle(msgspec.Struct, eq=False, dict=True):; -82,10 +85,38 @@ def attach(self) -> None:; symbols: ShmTensorHandle, publish, attach, try_attach，涉及 `ShmTensorHandle, publish, attach`；`python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` modified +38/-17 (55 lines); hunks: -64,49 +64,64 @@ def _one_based_layers(value: object, name: str, num_layers:...; -215,8 +230,14 @@ def plan_kimi_k3_lcm_cache(; symbols: _one_based_layers, kimi_k3_layer_group_ids, plan_kimi_k3_lcm_cache，涉及 `_one_based_layers, kimi_k3_layer_group_ids, plan_kimi_k3_lcm_cache`；`python/tokenspeed/runtime/distributed/process_group_manager.py` modified +6/-1 (7 lines); hunks: -46,6 +46,7 @@ def _make_all_groups(group: Group) -> list[Group]:; -55,18 +56,22 @@ def init_distributed(; symbols: _make_all_groups, ProcessGroupManager, __init__, init_distributed，涉及 `_make_all_groups, ProcessGroupManager, __init__`。
- 代码 diff 细节:
  - `test/runtime/test_kimi_k3_config.py` modified +80/-0 (80 lines); hunks: -478,3 +478,83 @@ def load_weights(self, weights):; symbols: load_weights, KimiK3LcmPlanTests, _plan, test_linear_packing_scales_with_attn_tp
  - `python/tokenspeed/runtime/multimodal/shm_transport.py` modified +68/-2 (70 lines); hunks: -59,6 +59,9 @@ class ShmTensorHandle(msgspec.Struct, eq=False, dict=True):; -82,10 +85,38 @@ def attach(self) -> None:; symbols: ShmTensorHandle, publish, attach, try_attach
  - `python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` modified +38/-17 (55 lines); hunks: -64,49 +64,64 @@ def _one_based_layers(value: object, name: str, num_layers:...; -215,8 +230,14 @@ def plan_kimi_k3_lcm_cache(; symbols: _one_based_layers, kimi_k3_layer_group_ids, plan_kimi_k3_lcm_cache
  - `python/tokenspeed/runtime/distributed/process_group_manager.py` modified +6/-1 (7 lines); hunks: -46,6 +46,7 @@ def _make_all_groups(group: Group) -> list[Group]:; -55,18 +56,22 @@ def init_distributed(; symbols: _make_all_groups, ProcessGroupManager, __init__, init_distributed
  - `python/tokenspeed/runtime/utils/server_args.py` modified +1/-3 (4 lines); hunks: -606,9 +606,7 @@ def resolve_parallelism(self):; symbols: resolve_parallelism
- 关键代码摘录:

```diff
diff -- test/runtime/test_kimi_k3_config.py
@@ -478,3 +478,83 @@ def load_weights(self, weights):
+class KimiK3LcmPlanTests(unittest.TestCase):
+    """LCM planning across attention-TP widths and reduced-layer variants."""
+    @staticmethod
+    def _plan(cfg, tp):
+        from tokenspeed.runtime.configs.kimi_k3_cache_spec import (
+            plan_kimi_k3_lcm_cache,
diff -- python/tokenspeed/runtime/multimodal/shm_transport.py
@@ -59,6 +59,9 @@ class ShmTensorHandle(msgspec.Struct, eq=False, dict=True):
+    # Payload received over the CPU group when the producer's POSIX segment
+    # lives on another host; also non-wire.
+    _remote = None
@@ -82,10 +85,38 @@ def attach(self) -> None:
+    def try_attach(self) -> bool:
+        """Attach if the segment exists on this host; False when the
diff -- python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py
@@ -64,49 +64,64 @@ def _one_based_layers(value: object, name: str, num_layers: int) -> tuple[int, .
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/test_kimi_k3_config.py` modified +80/-0
  - runtime: `python/tokenspeed/runtime/multimodal/shm_transport.py` modified +68/-2; `python/tokenspeed/runtime/configs/kimi_k3_cache_spec.py` modified +38/-17; `python/tokenspeed/runtime/distributed/process_group_manager.py` modified +6/-1; `python/tokenspeed/runtime/utils/server_args.py` modified +1/-3; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/__init__.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #921 - perf(kimi-k3): warm up the 4k benchmark shape

- 链接: https://github.com/lightseekorg/tokenspeed/pull/921
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；关联提交 `7b1cd098d939`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +1/-1 (2 lines); hunks: -71,7 +71,7 @@ perf:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +1/-1 (2 lines); hunks: -71,7 +71,7 @@ perf:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml
@@ -71,7 +71,7 @@ perf:
-    --warmup-num 0
+    --warmup-num 1
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #935 - fix(ci): stabilize Kimi K3 perf metrics

- 链接: https://github.com/lightseekorg/tokenspeed/pull/935
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；关联提交 `5ccbf29882cb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+55/-12，可读 patch 95 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +3/-1 (4 lines); hunks: -46,7 +46,9 @@ server:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +3/-1 (4 lines); hunks: -46,7 +46,9 @@ server:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml
@@ -46,7 +46,9 @@ server:
-    - python3 -m uv venv --seed --clear /tmp/evalscope-perf && python3 -m uv pip install --python /tmp/evalscope-perf/bin/python 'evalscope[perf]'
+    # Keep the load generator aligned with the version used to establish the
+    # perf_reference below; EvalScope 1.10 changed its metric semantics.
+    - python3 -m uv venv --seed --clear /tmp/evalscope-perf && python3 -m uv pip install --python /tmp/evalscope-perf/bin/python 'evalscope[perf]==1.9.1'
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +3/-1
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/ci_system/test_random_benchmark_perf_csv.py`, `test/random_benchmark/tokenspeed/collect_outputs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #926 - perf(kimi-k3): router cublas dispatch, grouped MoE-join reduce

- 链接: https://github.com/lightseekorg/tokenspeed/pull/926
- 状态/时间: merged / 2026-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；关联提交 `99de35c34a43`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+248/-28，可读 patch 433 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +5/-9 (14 lines); hunks: -115,7 +115,7; -1184,14 +1184,10 @@ def forward(; symbols: forward，涉及 `forward`；`tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +50/-0 (50 lines); hunks: -58,6 +58,56 @@ def test_kimi3_router_projection_falls_back_for_noncanonical_...; symbols: test_kimi3_router_projection_falls_back_for_noncanonical_shape, test_kimi3_router_projection_auto_splits_on_token_count, solution_for, test_kimi3_router_projection_cublas_requires_out_dtype_support，涉及 `test_kimi3_router_projection_falls_back_for_noncanonical_shape, test_kimi3_router_projection_auto_splits_on_token_count, solution_for`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +46/-3 (49 lines); hunks: -12,6 +12,7; -715,6 +716,25 @@ def kimi3_qkvfab_projection(; symbols: kimi3_qkvfab_projection, _mm_out_dtype_supported, kimi3_router_projection，涉及 `kimi3_qkvfab_projection, _mm_out_dtype_supported, kimi3_router_projection`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +5/-9 (14 lines); hunks: -115,7 +115,7; -1184,14 +1184,10 @@ def forward(; symbols: forward
  - `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +50/-0 (50 lines); hunks: -58,6 +58,56 @@ def test_kimi3_router_projection_falls_back_for_noncanonical_...; symbols: test_kimi3_router_projection_falls_back_for_noncanonical_shape, test_kimi3_router_projection_auto_splits_on_token_count, solution_for, test_kimi3_router_projection_cublas_requires_out_dtype_support
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +46/-3 (49 lines); hunks: -12,6 +12,7; -715,6 +716,25 @@ def kimi3_qkvfab_projection(; symbols: kimi3_qkvfab_projection, _mm_out_dtype_supported, kimi3_router_projection
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -115,7 +115,7 @@
-    kimi3_reduce_fused_moe,
+    kimi3_join_reduce_moe,
@@ -1184,14 +1184,10 @@ def forward(
-            # Post-join: one [T, latent+hidden] all-reduce covers both
-            # partials, element-wise identical to the two separate reduces.
-            if lane is not None and routed_out.data_ptr() == lane.data_ptr():
diff -- tokenspeed-kernel/test/test_kimi_prefill_ops.py
@@ -58,6 +58,56 @@ def test_kimi3_router_projection_falls_back_for_noncanonical_shape() -> None:
+def test_kimi3_router_projection_auto_splits_on_token_count() -> None:
+    """auto keeps the CUDA kernel at small M and switches to cublas above it.
+    The CUDA kernel's per-thread token loop runs on CUDA cores, so its time
+    grows linearly with M while the tensor-core GEMM stays flat; the dispatch
+    threshold is where they cross. Solution selection is observed by mocking
+    the two terminal paths -- no GPU needed.
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -12,6 +12,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +5/-9; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +46/-3
  - tests: `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +50/-0
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_moe.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #958 - fix(k3): fix eagle3 for kimi k3

- 链接: https://github.com/lightseekorg/tokenspeed/pull/958
- 状态/时间: merged / 2026-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/models/test_kimi_k3_eagle3_e2e.py`, `test/runtime/test_kimi_k3_eagle3.py`；关联提交 `2b504c85740b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+647/-28，可读 patch 828 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_kimi_k3_eagle3_e2e.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: TestKimiK3Eagle3E2E, _wait_for_ready, test_k3_eagle3_accepts_nonzero_drafts，涉及 `TestKimiK3Eagle3E2E, _wait_for_ready, test_k3_eagle3_accepts_nonzero_drafts`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +39/-2 (41 lines); hunks: -1644,6 +1644,8 @@ def get_layer(idx: int, prefix: str):; -1674,10 +1676,17 @@ def forward(; symbols: get_layer, get_input_embeddings, forward, KimiLinearForCausalLM，涉及 `get_layer, get_input_embeddings, forward`；`test/runtime/test_kimi_k3_eagle3.py` added +123/-0 (123 lines); hunks: -0,0 +1,123; symbols: _post_layer_attnres_reference, test_capture_tensor_matches_post_layer_attnres_reference, FakeLayer, __init__，涉及 `_post_layer_attnres_reference, test_capture_tensor_matches_post_layer_attnres_reference, FakeLayer`。
- 代码 diff 细节:
  - `test/runtime/models/test_kimi_k3_eagle3_e2e.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: TestKimiK3Eagle3E2E, _wait_for_ready, test_k3_eagle3_accepts_nonzero_drafts
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +39/-2 (41 lines); hunks: -1644,6 +1644,8 @@ def get_layer(idx: int, prefix: str):; -1674,10 +1676,17 @@ def forward(; symbols: get_layer, get_input_embeddings, forward, KimiLinearForCausalLM
  - `test/runtime/test_kimi_k3_eagle3.py` added +123/-0 (123 lines); hunks: -0,0 +1,123; symbols: _post_layer_attnres_reference, test_capture_tensor_matches_post_layer_attnres_reference, FakeLayer, __init__
- 关键代码摘录:

```diff
diff -- test/runtime/models/test_kimi_k3_eagle3_e2e.py
@@ -0,0 +1,183 @@
+"""Opt-in TP16 Kimi-K3 + EAGLE3 acceptance smoke.
+This is intentionally opt-in because it needs the real K3 target and a
+compatible serving-format Eagle3 draft.  It validates the production K3 chat
+path and fails if speculative decoding never accepts a draft token.
+Run on a 16-GPU PP1 node:
+    KIMI_K3_EAGLE3_E2E=1 \
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1644,6 +1644,8 @@ def get_layer(idx: int, prefix: str):
+        # One-based completed-layer ids; see set_eagle3_layers_to_capture.
+        self.eagle3_layers_to_capture: tuple[int, ...] = ()
@@ -1674,10 +1676,17 @@ def forward(
-        for layer in self.layers:
+        aux_hidden_states = [] if self.eagle3_layers_to_capture else None
+        for layer_idx, layer in enumerate(self.layers):
diff -- test/runtime/test_kimi_k3_eagle3.py
@@ -0,0 +1,123 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_kimi_k3_eagle3_e2e.py` added +183/-0; `test/runtime/test_kimi_k3_eagle3.py` added +123/-0
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +39/-2
- 验证与风险: diff 自带测试面 `test/runtime/models/test_eagle3_mla_draft_config.py`, `test/runtime/models/test_kimi_k3_eagle3_e2e.py`, `test/runtime/test_draft_page_table_units.py`, `test/runtime/test_kimi_k3_eagle3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #924 - feat(dspark): complete Kimi K3 draft execution

- 链接: https://github.com/lightseekorg/tokenspeed/pull/924
- 状态/时间: merged / 2026-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py`, `test/runtime/test_kimi_k3_cudagraph.py`, `test/runtime/test_kimi_k3_dspark_capture.py` 等 7 个文件；关联提交 `2f06d82f0584`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 36 个文件，+3983/-112，可读 patch 4836 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_dspark.py` added +514/-0 (514 lines); hunks: -0,0 +1,514; symbols: K3DSparkAttention, _attn, project_latent_kv, apply_latent_rope，涉及 `K3DSparkAttention, _attn, project_latent_kv`；`python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` added +289/-0 (289 lines); hunks: -0,0 +1,289; symbols: KimiK3DSparkConfig, __init__, instead, name，涉及 `KimiK3DSparkConfig, __init__, instead`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +94/-5 (99 lines); hunks: -1647,9 +1647,34 @@ def get_layer(idx: int, prefix: str):; -1675,17 +1700,33 @@ def forward(; symbols: get_layer, get_input_embeddings, _dspark_capture_stream, forward，涉及 `get_layer, get_input_embeddings, _dspark_capture_stream`；`test/runtime/test_kimi_k3_dspark_model.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: _per_layer_keys, make_config, test_published_checkpoint_config_validates, test_manifest_has_the_published_tensor_count，涉及 `_per_layer_keys, make_config, test_published_checkpoint_config_validates`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_dspark.py` added +514/-0 (514 lines); hunks: -0,0 +1,514; symbols: K3DSparkAttention, _attn, project_latent_kv, apply_latent_rope
  - `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` added +289/-0 (289 lines); hunks: -0,0 +1,289; symbols: KimiK3DSparkConfig, __init__, instead, name
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +94/-5 (99 lines); hunks: -1647,9 +1647,34 @@ def get_layer(idx: int, prefix: str):; -1675,17 +1700,33 @@ def forward(; symbols: get_layer, get_input_embeddings, _dspark_capture_stream, forward
  - `test/runtime/test_kimi_k3_dspark_model.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: _per_layer_keys, make_config, test_published_checkpoint_config_validates, test_manifest_has_the_published_tensor_count
  - `test/runtime/test_kimi_k3_dspark_capture.py` added +131/-0 (131 lines); hunks: -0,0 +1,131; symbols: _make_model, _CausalLM, __init__, _bind_setter
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_dspark.py
@@ -0,0 +1,514 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py
@@ -0,0 +1,289 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1647,9 +1647,34 @@ def get_layer(idx: int, prefix: str):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_dspark.py` added +514/-0; `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` added +289/-0; `python/tokenspeed/runtime/models/kimi_k3.py` modified +94/-5
  - tests: `test/runtime/test_kimi_k3_dspark_model.py` added +345/-0; `test/runtime/test_kimi_k3_dspark_capture.py` added +131/-0; `test/runtime/test_kimi_k3_cudagraph.py` modified +2/-0; `test/runtime/test_kimi_k3_eagle3.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-dspark-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `test/ci/perf/kimi-k3-dspark-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/runtime/kda_paged_prefill_nan_demo.py`, `test/runtime/test_draft_page_table_units.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #919 - perf(kimi3): fuse semantic MLA decode stages

- 链接: https://github.com/lightseekorg/tokenspeed/pull/919
- 状态/时间: merged / 2026-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `3e31dca8034b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 25 个文件，+2328/-177，可读 patch 3092 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +39/-10 (49 lines); hunks: -69,6 +69,7; -379,8 +380,15 @@ def _project_q_latent_gated(; symbols: _project_q_latent_gated, forward，涉及 `_project_q_latent_gated, forward`；`python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +10/-2 (12 lines); hunks: -83,15 +83,23 @@ def forward(; symbols: forward，涉及 `forward`；`test/runtime/test_kimi_k3_config.py` modified +78/-11 (89 lines); hunks: -148,6 +148,49 @@ def test_mamba2_cache_params_respects_tp(self):; -364,7 +407,7 @@ def __init__(self, **kwargs):; symbols: test_mamba2_cache_params_respects_tp, KimiK3RegistrationTests, test_mla_mixed_batch_slices_decode_gate_to_live_rows, test_shared_projection_preserves_direct_write_output，涉及 `test_mamba2_cache_params_respects_tp, KimiK3RegistrationTests, test_mla_mixed_batch_slices_decode_gate_to_live_rows`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +39/-10 (49 lines); hunks: -69,6 +69,7; -379,8 +380,15 @@ def _project_q_latent_gated(; symbols: _project_q_latent_gated, forward
  - `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +10/-2 (12 lines); hunks: -83,15 +83,23 @@ def forward(; symbols: forward
  - `test/runtime/test_kimi_k3_config.py` modified +78/-11 (89 lines); hunks: -148,6 +148,49 @@ def test_mamba2_cache_params_respects_tp(self):; -364,7 +407,7 @@ def __init__(self, **kwargs):; symbols: test_mamba2_cache_params_respects_tp, KimiK3RegistrationTests, test_mla_mixed_batch_slices_decode_gate_to_live_rows, test_shared_projection_preserves_direct_write_output
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -69,6 +69,7 @@
+from tokenspeed_kernel.ops.attention import mla_normalize_project_query
@@ -379,8 +380,15 @@ def _project_q_latent_gated(
-    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
-        """Project MLA Q, latent KV, and the local output gate in one GEMM."""
+    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
+        """Project MLA Q, latent KV, and the local output gate in one GEMM.
diff -- python/tokenspeed/runtime/models/kimi_k3_nextn.py
@@ -83,15 +83,23 @@ def forward(
-            q, latent_cache, gate = self._project_q_latent_gated(
+            q, latent_cache, gate, absorbed_query = self._project_q_latent_gated(
-        attn_output = self._attn(positions, q, latent_cache, ctx, out_cache_loc)
+            absorbed_query = None
+        attn_output = self._attn(
+            positions,
diff -- test/runtime/test_kimi_k3_config.py
@@ -148,6 +148,49 @@ def test_mamba2_cache_params_respects_tp(self):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +39/-10; `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +10/-2
  - tests: `test/runtime/test_kimi_k3_config.py` modified +78/-11
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel-amd/test/ops/gemm/test_fp16.py`, `tokenspeed-kernel/test/ops/test_attention_mla.py`, `tokenspeed-kernel/test/ops/test_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #878 - perf: Optimize Kimi K3 MoE input projections and low-token decode

- 链接: https://github.com/lightseekorg/tokenspeed/pull/878
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`；关联提交 `145d62ddbde2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 33 个文件，+2826/-191，可读 patch 3810 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +149/-9 (158 lines); hunks: -83,6 +83,11; -254,18 +259,14 @@ def __init__(; symbols: __init__, forward, pack_input_projection_weights，涉及 `__init__, forward, pack_input_projection_weights`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/kimi3.py` removed +0/-34 (34 lines); hunks: -1,34 +0,0; symbols: kimi3_native_moe_available，涉及 `kimi3_native_moe_available`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +27/-5 (32 lines); hunks: -411,9 +411,10 @@ def kimi3_latent_projection_add3(; -441,7 +442,7 @@ def kimi3_latent_projection_add3(; symbols: kimi3_latent_projection_add3，涉及 `kimi3_latent_projection_add3`；`test/runtime/test_kimi_k3_config.py` modified +8/-0 (8 lines); hunks: -328,16 +328,24 @@ def test_ep_kimi_moe_combines_shared_and_routed_reductions...; symbols: test_ep_kimi_moe_combines_shared_and_routed_reductions, FakeLinear, __init__, FakeExperts，涉及 `test_ep_kimi_moe_combines_shared_and_routed_reductions, FakeLinear, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +149/-9 (158 lines); hunks: -83,6 +83,11; -254,18 +259,14 @@ def __init__(; symbols: __init__, forward, pack_input_projection_weights
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/kimi3.py` removed +0/-34 (34 lines); hunks: -1,34 +0,0; symbols: kimi3_native_moe_available
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +27/-5 (32 lines); hunks: -411,9 +411,10 @@ def kimi3_latent_projection_add3(; -441,7 +442,7 @@ def kimi3_latent_projection_add3(; symbols: kimi3_latent_projection_add3
  - `test/runtime/test_kimi_k3_config.py` modified +8/-0 (8 lines); hunks: -328,16 +328,24 @@ def test_ep_kimi_moe_combines_shared_and_routed_reductions...; symbols: test_ep_kimi_moe_combines_shared_and_routed_reductions, FakeLinear, __init__, FakeExperts
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -83,6 +83,11 @@
+from tokenspeed_kernel.ops.moe import (
+    latent_moe_decode_pipeline_available,
+    latent_moe_expert_shared,
+    latent_moe_input_projections,
+)
@@ -254,18 +259,14 @@ def __init__(
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/kimi3.py
@@ -1,34 +0,0 @@
-# Copyright (c) 2026 LightSeek Foundation
-#
-# Permission is hereby granted, free of charge, to any person obtaining a copy
-# of this software and associated documentation files (the "Software"), to deal
-# in the Software without restriction, including without limitation the rights
-# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -411,9 +411,10 @@ def kimi3_latent_projection_add3(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +149/-9; `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/kimi3.py` removed +0/-34; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +27/-5
  - tests: `test/runtime/test_kimi_k3_config.py` modified +8/-0
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel-amd/test/ops/moe/test_latent_input.py`, `tokenspeed-kernel/test/ops/moe/test_gluon_mxfp4_situ_gfx950.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #995 - feat(K3): Support K3 On H200

- 链接: https://github.com/lightseekorg/tokenspeed/pull/995
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py`, `test/runtime/test_kimi_k3_cudagraph.py`；关联提交 `88be7a3a70e6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 36 个文件，+5975/-181，可读 patch 6586 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +17/-11 (28 lines); hunks: -965,12 +965,14 @@ def __init__(; -989,6 +991,10 @@ def __init__(; symbols: __init__, _routed_experts，涉及 `__init__, _routed_experts`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +13/-2 (15 lines); hunks: -188,8 +188,16 @@ def build_kimi_k3_cache_fields(; -268,11 +276,14 @@ def solve_kimi_k3_cache_layout(; symbols: build_kimi_k3_cache_fields, solve_kimi_k3_cache_layout，涉及 `build_kimi_k3_cache_fields, solve_kimi_k3_cache_layout`；`python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +9/-0 (9 lines); hunks: -51,6 +51,7; -199,6 +200,14 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_kimi_k3_cudagraph.py` modified +0/-1 (1 lines); hunks: -138,7 +138,6 @@ def _bare_amd_mla_backend(; symbols: _bare_amd_mla_backend，涉及 `_bare_amd_mla_backend`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +17/-11 (28 lines); hunks: -965,12 +965,14 @@ def __init__(; -989,6 +991,10 @@ def __init__(; symbols: __init__, _routed_experts
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +13/-2 (15 lines); hunks: -188,8 +188,16 @@ def build_kimi_k3_cache_fields(; -268,11 +276,14 @@ def solve_kimi_k3_cache_layout(; symbols: build_kimi_k3_cache_fields, solve_kimi_k3_cache_layout
  - `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +9/-0 (9 lines); hunks: -51,6 +51,7; -199,6 +200,14 @@ def __init__(; symbols: __init__
  - `test/runtime/test_kimi_k3_cudagraph.py` modified +0/-1 (1 lines); hunks: -138,7 +138,6 @@ def _bare_amd_mla_backend(; symbols: _bare_amd_mla_backend
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -965,12 +965,14 @@ def __init__(
-        if not self.execution_plan.use_native:
+        self.use_marlin_situ_moe = self.execution_plan.use_marlin
+        if not self.execution_plan.use_native and not self.use_marlin_situ_moe:
-                    "Kimi-K3 MXFP4 SiTU MoE requires the native backend or the "
-                    "FlashInfer TRT-LLM backend; no portable SiTU Triton fallback "
-                    f"exists (selected MoE backend: {moe_backend.value!r})."
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py
@@ -188,8 +188,16 @@ def build_kimi_k3_cache_fields(
-    if mla_cache_dtype != torch.float8_e4m3fn:
-        raise ValueError("Kimi-K3 cache requires mla_cache_dtype=torch.float8_e4m3fn")
+    # fp8_e4m3 is the memory-lean default (matches the Blackwell tokenspeed_mla
+    # kernels). bf16 is the Hopper path: FlashMLA has no SM90 dense-fp8 MLA
+    # kernel, so on SM90 the MLA layers run bf16 (flashinfer ragged prefill +
+    # bf16 FlashMLA decode), mirroring how vLLM/sglang serve K3 on Hopper.
diff -- python/tokenspeed/runtime/models/kimi_k3_dspark.py
@@ -51,6 +51,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +17/-11; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +13/-2; `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +9/-0
  - tests: `test/runtime/test_kimi_k3_cudagraph.py` modified +0/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_cudagraph_per_group.py`, `test/runtime/test_kimi_k3_cudagraph.py`, `test/runtime/test_mla_block_decode.py`, `test/runtime/test_trtllm_mla_block_decode.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1012 - fix(k3): restore DSpark cache view on unified arena

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1012
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/test_kimi_k3_cache_pool.py`, `test/runtime/test_kimi_k3_cache_spec.py`；关联提交 `bc6ff9e45967`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+227/-34，可读 patch 394 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/test_kimi_k3_cache_pool.py` modified +148/-0 (148 lines); hunks: -115,3 +115,151 @@ def test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing...; symbols: test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing, test_kimi_k3_bf16_draft_uses_typed_view_over_fp8_target_arena，涉及 `test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing, test_kimi_k3_bf16_draft_uses_typed_view_over_fp8_target_arena`；`test/runtime/test_kimi_k3_cache_spec.py` modified +23/-15 (38 lines); hunks: -115,8 +115,8 @@ def test_lcm_parent_demand_uses_per_group_packing() -> None:; -126,10 +126,10 @@ def test_k3_merged_solve_with_draft_shares_page_ids():; symbols: test_lcm_parent_demand_uses_per_group_packing, test_k3_merged_solve_with_draft_shares_page_ids, test_k3_binding_utilization_baseline_and_draft_widening，涉及 `test_lcm_parent_demand_uses_per_group_packing, test_k3_merged_solve_with_draft_shares_page_ids, test_k3_binding_utilization_baseline_and_draft_widening`；`python/tokenspeed/runtime/layers/attention/kv_cache/mla.py` modified +20/-8 (28 lines); hunks: -67,6 +67,8 @@ def __init__(; -78,6 +80,8 @@ def __init__(; symbols: __init__, _field_layer_id, _create_buffers，涉及 `__init__, _field_layer_id, _create_buffers`；`python/tokenspeed/runtime/layers/attention/registry.py` modified +14/-5 (19 lines); hunks: -81,11 +81,20 @@ def _resolve_heterogeneous_draft_family(; symbols: _resolve_heterogeneous_draft_family，涉及 `_resolve_heterogeneous_draft_family`。
- 代码 diff 细节:
  - `test/runtime/test_kimi_k3_cache_pool.py` modified +148/-0 (148 lines); hunks: -115,3 +115,151 @@ def test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing...; symbols: test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing, test_kimi_k3_bf16_draft_uses_typed_view_over_fp8_target_arena
  - `test/runtime/test_kimi_k3_cache_spec.py` modified +23/-15 (38 lines); hunks: -115,8 +115,8 @@ def test_lcm_parent_demand_uses_per_group_packing() -> None:; -126,10 +126,10 @@ def test_k3_merged_solve_with_draft_shares_page_ids():; symbols: test_lcm_parent_demand_uses_per_group_packing, test_k3_merged_solve_with_draft_shares_page_ids, test_k3_binding_utilization_baseline_and_draft_widening
  - `python/tokenspeed/runtime/layers/attention/kv_cache/mla.py` modified +20/-8 (28 lines); hunks: -67,6 +67,8 @@ def __init__(; -78,6 +80,8 @@ def __init__(; symbols: __init__, _field_layer_id, _create_buffers
  - `python/tokenspeed/runtime/layers/attention/registry.py` modified +14/-5 (19 lines); hunks: -81,11 +81,20 @@ def _resolve_heterogeneous_draft_family(; symbols: _resolve_heterogeneous_draft_family
  - `python/tokenspeed/runtime/layers/attention/kv_cache/factory.py` modified +7/-2 (9 lines); hunks: -26,9 +26,12 @@ def create_cache_pool(; -203,6 +206,8 @@ def create_cache_pool(; symbols: create_cache_pool
- 关键代码摘录:

```diff
diff -- test/runtime/test_kimi_k3_cache_pool.py
@@ -115,3 +115,151 @@ def test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing() -> None:
+def test_kimi_k3_bf16_draft_uses_typed_view_over_fp8_target_arena() -> None:
+    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
+    from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
+    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
+    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
+        mla_cache_fields,
diff -- test/runtime/test_kimi_k3_cache_spec.py
@@ -115,8 +115,8 @@ def test_lcm_parent_demand_uses_per_group_packing() -> None:
-    """One big model: a draft MLA layer joins the K3 solve as continuation
-    layer 93 in the full_attention group — same packing, same page-id
+    """One big model: five BF16 draft MLA layers join the K3 solve as
+    continuation layers 93-97 in the full_attention group — same packing/page-id
@@ -126,10 +126,10 @@ def test_k3_merged_solve_with_draft_shares_page_ids():
-        layer_group_ids=("full_attention",),
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/mla.py
@@ -67,6 +67,8 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/test_kimi_k3_cache_pool.py` modified +148/-0; `test/runtime/test_kimi_k3_cache_spec.py` modified +23/-15
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/mla.py` modified +20/-8; `python/tokenspeed/runtime/layers/attention/registry.py` modified +14/-5; `python/tokenspeed/runtime/layers/attention/kv_cache/factory.py` modified +7/-2; `python/tokenspeed/runtime/layers/attention/configs/mla.py` modified +3/-2
- 验证与风险: diff 自带测试面 `test/runtime/test_cache_setup.py`, `test/runtime/test_kimi_k3_cache_pool.py`, `test/runtime/test_kimi_k3_cache_spec.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1056 - fix(k3): advance attnres launch and join

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1056
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/models/test_kimi_k3_attnres_hoist.py`；关联提交 `4c02a1aaa450`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+115/-16，可读 patch 200 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +43/-16 (59 lines); hunks: -869,6 +869,13 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; -877,8 +884,8 @@ def _attnres_scratch(; symbols: forward, _attnres_mlp_slot, _attnres_scratch, __init__，涉及 `forward, _attnres_mlp_slot, _attnres_scratch`；`test/runtime/models/test_kimi_k3_attnres_hoist.py` added +55/-0 (55 lines); hunks: -0,0 +1,55; symbols: _mlp_blocks, TestAttnResMlpHoist, test_adjacent_layers_use_different_mlp_slots, test_mlp_slots_never_take_the_attn_side_slot，涉及 `_mlp_blocks, TestAttnResMlpHoist, test_adjacent_layers_use_different_mlp_slots`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +43/-16 (59 lines); hunks: -869,6 +869,13 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; -877,8 +884,8 @@ def _attnres_scratch(; symbols: forward, _attnres_mlp_slot, _attnres_scratch, __init__
  - `test/runtime/models/test_kimi_k3_attnres_hoist.py` added +55/-0 (55 lines); hunks: -0,0 +1,55; symbols: _mlp_blocks, TestAttnResMlpHoist, test_adjacent_layers_use_different_mlp_slots, test_mlp_slots_never_take_the_attn_side_slot
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -869,6 +869,13 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+def _attnres_mlp_slot(layer_id: int) -> int:
+    """Slot (0 or 2) for this layer's mlp-side partial; alternates so a layer
+    never reads the buffer its own aux branch is writing for the next one."""
+    return 2 * (layer_id % 2)
@@ -877,8 +884,8 @@ def _attnres_scratch(
-    slot 0 = the layer's mlp-side mix; slot 1 = the next layer's attn-side mix
diff -- test/runtime/models/test_kimi_k3_attnres_hoist.py
@@ -0,0 +1,55 @@
+"""Slot invariants for the hoisted AttnRes mlp-side partial.
+Layer L's mlp-side partial is computed on layer L-1's aux sweep, so it lands a
+layer before the all-reduce that reads it. That is safe only while a layer's
+own slot differs from the one its aux branch writes for the next layer, and
+only where the previous layer sweeps the same block range.
+Usage:
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +43/-16
  - tests: `test/runtime/models/test_kimi_k3_attnres_hoist.py` added +55/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_kimi_k3_attnres_hoist.py`, `tokenspeed-kernel/test/ops/activation/test_attnres_split.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1038 - perf(kimi3): fuse MoE norm projection and collectives

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1038
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`；关联提交 `9ed75f5aff38`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 24 个文件，+1723/-622，可读 patch 3313 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +10/-26 (36 lines); hunks: -57,7 +57,6; -85,7 +84,6; symbols: __init__, pack_input_projection_weights, _latent_input_projections, _forward_fused_decode_pipeline，涉及 `__init__, pack_input_projection_weights, _latent_input_projections`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +44/-0 (44 lines); hunks: -401,6 +401,8 @@ def kimi3_latent_projection_add3(; -410,6 +412,10 @@ def kimi3_latent_projection_add3(; symbols: kimi3_latent_projection_add3，涉及 `kimi3_latent_projection_add3`；`test/runtime/test_kimi_k3_config.py` modified +4/-3 (7 lines); hunks: -411,9 +411,10 @@ def __init__(self, **kwargs):; symbols: __init__, test_mla_gate_projection_uses_api_selected_layout，涉及 `__init__, test_mla_gate_projection_uses_api_selected_layout`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +10/-26 (36 lines); hunks: -57,7 +57,6; -85,7 +84,6; symbols: __init__, pack_input_projection_weights, _latent_input_projections, _forward_fused_decode_pipeline
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +44/-0 (44 lines); hunks: -401,6 +401,8 @@ def kimi3_latent_projection_add3(; -410,6 +412,10 @@ def kimi3_latent_projection_add3(; symbols: kimi3_latent_projection_add3
  - `test/runtime/test_kimi_k3_config.py` modified +4/-3 (7 lines); hunks: -411,9 +411,10 @@ def __init__(self, **kwargs):; symbols: __init__, test_mla_gate_projection_uses_api_selected_layout
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -57,7 +57,6 @@
-from functools import partial
@@ -85,7 +84,6 @@
-    latent_moe_expert_shared,
@@ -98,7 +96,6 @@
-    all_reduce_two,
@@ -121,6 +118,7 @@
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -401,6 +401,8 @@ def kimi3_latent_projection_add3(
+    norm_weight: torch.Tensor | None = None,
+    eps: float | None = None,
@@ -410,6 +412,10 @@ def kimi3_latent_projection_add3(
+        norm_weight: Optional contiguous BF16 RMSNorm weight shaped ``[K]``.
+            When provided, RMSNorm is applied to ``hidden_states`` before the
+            projection.
diff -- test/runtime/test_kimi_k3_config.py
@@ -411,9 +411,10 @@ def __init__(self, **kwargs):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +10/-26; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +44/-0
  - tests: `test/runtime/test_kimi_k3_config.py` modified +4/-3
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_auto_backend.py`, `test/runtime/distributed/test_comm_ops.py`, `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #959 - perf(kimi3): fuse AttnRes projection and collectives

- 链接: https://github.com/lightseekorg/tokenspeed/pull/959
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `b30fccee161c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 18 个文件，+1766/-39，可读 patch 2203 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +195/-23 (218 lines); hunks: -78,6 +78,8; -369,6 +371,7 @@ def _project_q_latent_gated(; symbols: _project_q_latent_gated, can_fuse_attnres_partials, forward，涉及 `_project_q_latent_gated, can_fuse_attnres_partials, forward`；`test/runtime/test_kimi_k3_attn_res.py` modified +20/-4 (24 lines); hunks: -17,6 +17,7; -155,26 +156,41 @@ def _manual_rmsnorm(x: torch.Tensor, weight: torch.Tensor,...; symbols: _manual_rmsnorm, AttnResOutNormTests, test_torch_fallback_out_norm_matches_separate, test_output_eps_is_ignored_without_output_norm，涉及 `_manual_rmsnorm, AttnResOutNormTests, test_torch_fallback_out_norm_matches_separate`；`test/runtime/test_kimi_k3_config.py` modified +8/-0 (8 lines); hunks: -497,6 +497,14 @@ def forward(self, value):; symbols: forward, test_ungated_mla_does_not_select_attnres_projection_fusion, test_config_registry_maps_model_type，涉及 `forward, test_ungated_mla_does_not_select_attnres_projection_fusion, test_config_registry_maps_model_type`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +195/-23 (218 lines); hunks: -78,6 +78,8; -369,6 +371,7 @@ def _project_q_latent_gated(; symbols: _project_q_latent_gated, can_fuse_attnres_partials, forward
  - `test/runtime/test_kimi_k3_attn_res.py` modified +20/-4 (24 lines); hunks: -17,6 +17,7; -155,26 +156,41 @@ def _manual_rmsnorm(x: torch.Tensor, weight: torch.Tensor,...; symbols: _manual_rmsnorm, AttnResOutNormTests, test_torch_fallback_out_norm_matches_separate, test_output_eps_is_ignored_without_output_norm
  - `test/runtime/test_kimi_k3_config.py` modified +8/-0 (8 lines); hunks: -497,6 +497,14 @@ def forward(self, value):; symbols: forward, test_ungated_mla_does_not_select_attnres_projection_fusion, test_config_registry_maps_model_type
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -78,6 +78,8 @@
+    linear_attnres_partials,
+    linear_attnres_partials_available,
@@ -369,6 +371,7 @@ def _project_q_latent_gated(
+        attnres_partial_args: tuple | None = None,
@@ -382,6 +385,29 @@ def _project_q_latent_gated(
+            if attnres_partial_args is not None:
diff -- test/runtime/test_kimi_k3_attn_res.py
@@ -17,6 +17,7 @@
+from tokenspeed_kernel.ops.attn_res import attn_res_fwd  # noqa: E402
@@ -155,26 +156,41 @@ def _manual_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
-    def test_torch_fallback_out_norm_matches_separate(self):
+    def test_output_eps_is_ignored_without_output_norm(self):
+        prefix_sum, block_residual, proj, norm = _make_inputs(11, seed=4)
+        kwargs = {
diff -- test/runtime/test_kimi_k3_config.py
@@ -497,6 +497,14 @@ def forward(self, value):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +195/-23
  - tests: `test/runtime/test_kimi_k3_attn_res.py` modified +20/-4; `test/runtime/test_kimi_k3_config.py` modified +8/-0
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_comm_ops.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/test/ops/test_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1060 - feat(kimi-k3): support cross-DP EP token gather

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1060
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py`, `test/runtime/test_kimi_k3_config.py`, `test/runtime/test_kimi_k3_dspark_model.py`；关联提交 `ae3dd2ebbe59`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+170/-24，可读 patch 307 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +91/-23 (114 lines); hunks: -100,6 +100,7; -1078,15 +1079,20 @@ def __init__(; symbols: __init__, _forward_fused_decode_pipeline, _gather_dp_tokens, forward，涉及 `__init__, _forward_fused_decode_pipeline, _gather_dp_tokens`；`python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +8/-0 (8 lines); hunks: -408,6 +408,14 @@ def forward(; symbols: forward，涉及 `forward`；`python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +1/-0 (1 lines); hunks: -214,6 +214,7 @@ def forward(; symbols: forward，涉及 `forward`；`test/runtime/test_kimi_k3_config.py` modified +50/-1 (51 lines); hunks: -355,6 +355,14 @@ def __init__(self, **kwargs):; -365,7 +373,7 @@ def __init__(self, **kwargs):; symbols: __init__, test_cross_dp_ep_gather_uses_dp_group_and_returns_local_offset, gather，涉及 `__init__, test_cross_dp_ep_gather_uses_dp_group_and_returns_local_offset, gather`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +91/-23 (114 lines); hunks: -100,6 +100,7; -1078,15 +1079,20 @@ def __init__(; symbols: __init__, _forward_fused_decode_pipeline, _gather_dp_tokens, forward
  - `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +8/-0 (8 lines); hunks: -408,6 +408,14 @@ def forward(; symbols: forward
  - `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +1/-0 (1 lines); hunks: -214,6 +214,7 @@ def forward(; symbols: forward
  - `test/runtime/test_kimi_k3_config.py` modified +50/-1 (51 lines); hunks: -355,6 +355,14 @@ def __init__(self, **kwargs):; -365,7 +373,7 @@ def __init__(self, **kwargs):; symbols: __init__, test_cross_dp_ep_gather_uses_dp_group_and_returns_local_offset, gather
  - `test/runtime/test_kimi_k3_dspark_model.py` modified +19/-0 (19 lines); hunks: -22,6 +22,7; -270,6 +271,24 @@ def test_no_inactive_features_reported_without_a_confidence...; symbols: test_no_inactive_features_reported_without_a_confidence_head, test_idle_forward_with_no_rows_skips_dense_draft_layers, test_final_norm_reduces_the_last_row_parallel_mlp_output
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -100,6 +100,7 @@
+    token_all_gather,
@@ -1078,15 +1079,20 @@ def __init__(
-        if not self.execution_plan.use_native:
-            # Both the TRT-LLM and Marlin SiTU paths currently require a
-            # replicated-token all-reduce topology (attn TP == MoE TP*EP);
-            # Attn-DP/MoE-EP RSAG is not wired for either.
diff -- python/tokenspeed/runtime/models/kimi_k3_dspark.py
@@ -408,6 +408,14 @@ def forward(
+        # A DP-idle rank has no draft rows. K3 DSPARK's draft is dense and all
+        # of its collectives are scoped to the local attention TP group, whose
+        # peers are idle together, so there is no cross-DP collective to join.
+        # In particular, FlashInfer's SiLU kernel cannot launch with M=0.
+        if hidden_states.shape[0] == 0:
+            return LogitsProcessorOutput(
diff -- python/tokenspeed/runtime/models/kimi_k3_nextn.py
@@ -214,6 +214,7 @@ def forward(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +91/-23; `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +8/-0; `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +1/-0
  - tests: `test/runtime/test_kimi_k3_config.py` modified +50/-1; `test/runtime/test_kimi_k3_dspark_model.py` modified +19/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_config.py`, `test/runtime/test_kimi_k3_dspark_model.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1084 - perf(kimi-k3): keep small AMD decode batches on one stream

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1084
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`；关联提交 `4d7886bcf9d7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+9/-1，可读 patch 31 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +9/-1 (10 lines); hunks: -92,6 +92,7; -958,6 +959,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, _attnres_mlp_slot，涉及 `forward, _attnres_mlp_slot`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +9/-1 (10 lines); hunks: -92,6 +92,7; -958,6 +959,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, _attnres_mlp_slot
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -92,6 +92,7 @@
+from tokenspeed_kernel.platform import current_platform
@@ -958,6 +959,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+# Paired MI350 measurements show that stream scheduling costs more than the
+# available attention/AttnRes overlap through M=16. Preserve NVIDIA's policy.
+ATTNRES_STREAM_FORK_THRESHOLD = 16 if current_platform().is_amd else 0
@@ -1827,7 +1831,11 @@ def forward(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/kimi_k3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1089 - perf(kimi3): fuse batched AttnRes graph on gfx950

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1089
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py`；关联提交 `41b1e6c6f543`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+1187/-113，可读 patch 1605 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +154/-3 (157 lines); hunks: -69,7 +69,7; -539,6 +539,9 @@ def _apply_attn_res(; symbols: _apply_attn_res, _mix_into_attention, _fused_attnres_graph_available, _forward_fused_attnres_graph，涉及 `_apply_attn_res, _mix_into_attention, _fused_attnres_graph_available`；`test/runtime/test_kimi_k3_attn_res.py` modified +330/-1 (331 lines); hunks: -1,3 +1,23; -9,6 +29,8; symbols: test_model_wiring_slices_valid_blocks, test_model_wiring_keeps_full_storage_only_for_snapshot_writes, test_zero_valid_blocks_is_identity, test_delta_update_and_block_write_match_reference，涉及 `test_model_wiring_slices_valid_blocks, test_model_wiring_keeps_full_storage_only_for_snapshot_writes, test_zero_valid_blocks_is_identity`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +154/-3 (157 lines); hunks: -69,7 +69,7; -539,6 +539,9 @@ def _apply_attn_res(; symbols: _apply_attn_res, _mix_into_attention, _fused_attnres_graph_available, _forward_fused_attnres_graph
  - `test/runtime/test_kimi_k3_attn_res.py` modified +330/-1 (331 lines); hunks: -1,3 +1,23; -9,6 +29,8; symbols: test_model_wiring_slices_valid_blocks, test_model_wiring_keeps_full_storage_only_for_snapshot_writes, test_zero_valid_blocks_is_identity, test_delta_update_and_block_write_match_reference
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -69,7 +69,7 @@
-from tokenspeed_kernel.ops.attn_res import attn_res_fwd
+from tokenspeed_kernel.ops.attn_res import attn_res_fwd, attn_res_fwd_available
@@ -539,6 +539,9 @@ def _apply_attn_res(
+    *,
+    delta: torch.Tensor | None = None,
+    block_write_idx: int = -1,
diff -- test/runtime/test_kimi_k3_attn_res.py
@@ -1,3 +1,23 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +154/-3
  - tests: `test/runtime/test_kimi_k3_attn_res.py` modified +330/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_attn_res.py`, `tokenspeed-kernel/test/ops/test_kimi3_prefill_gluon_gfx950.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #957 - perf(k3): latent moe multicast tail

- 链接: https://github.com/lightseekorg/tokenspeed/pull/957
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`；关联提交 `3c8fb5cb349f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+5684/-63，可读 patch 5968 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +471/-52 (523 lines); hunks: -71,8 +71,14; -104,7 +110,10; symbols: _situ_betas, _shard_k3_up_projection, __init__，涉及 `_situ_betas, _shard_k3_up_projection, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +471/-52 (523 lines); hunks: -71,8 +71,14; -104,7 +110,10; symbols: _situ_betas, _shard_k3_up_projection, __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -71,8 +71,14 @@
+from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
+from tokenspeed_kernel.ops.communication.multimem import (
+    multimem_all_reduce_staged,
+    multimem_available,
+    multimem_prealloc,
+    multimem_stage,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +471/-52
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_k3_moe_tail_equivalence.py`, `test/runtime/test_k3_moe_tail_tier.py`, `tokenspeed-kernel/test/ops/communication/test_multimem_distributed.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1086 - perf(kimi3): fuse KDA decode core

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1086
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`；关联提交 `3c16d939afdf`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+891/-40，可读 patch 1148 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +16/-9 (25 lines); hunks: -918,6 +918,7 @@ def forward(; -943,19 +944,25 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +16/-9 (25 lines); hunks: -918,6 +918,7 @@ def forward(; -943,19 +944,25 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -918,6 +918,7 @@ def forward(
+        fuse_decode_output_norm = ctx.forward_mode.is_decode() and num_tokens == ctx.bs
@@ -943,19 +944,25 @@ def forward(
+            output_gate=out_gate if fuse_decode_output_norm else None,
+            norm_weight=self.o_norm.weight if fuse_decode_output_norm else None,
+            norm_eps=self.o_norm.variance_epsilon if fuse_decode_output_norm else None,
-        # Per-head gated RMSNorm + sigmoid output gate in one kernel.
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +16/-9
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_kda_recurrent.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1102 - perf(k3): fold the latent-tail projection into the reduction epilogue and pool its symmetric buffers

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1102
- 状态/时间: merged / 2026-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py`, `test/runtime/layers/test_kimi_k3_addmm_fold.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `e7c76346f385`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+344/-108，可读 patch 694 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +29/-25 (54 lines); hunks: -1062,8 +1062,9 @@ def __init__(; -1286,13 +1287,17 @@ def __init__(; symbols: __init__, _initialize_tail_capabilities, _tail_multimem_ar_sharded，涉及 `__init__, _initialize_tail_capabilities, _tail_multimem_ar_sharded`；`test/runtime/layers/test_kimi_k3_addmm_fold.py` added +48/-0 (48 lines); hunks: -0,0 +1,48; symbols: test_narrowed_staging_addmm_matches_unfolded_expression，涉及 `test_narrowed_staging_addmm_matches_unfolded_expression`；`python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +14/-13 (27 lines); hunks: -120,6 +120,7 @@ def __init__(; -146,17 +147,15 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_kimi_k3_config.py` modified +2/-1 (3 lines); hunks: -412,8 +412,9 @@ def __init__(self, **kwargs):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +29/-25 (54 lines); hunks: -1062,8 +1062,9 @@ def __init__(; -1286,13 +1287,17 @@ def __init__(; symbols: __init__, _initialize_tail_capabilities, _tail_multimem_ar_sharded
  - `test/runtime/layers/test_kimi_k3_addmm_fold.py` added +48/-0 (48 lines); hunks: -0,0 +1,48; symbols: test_narrowed_staging_addmm_matches_unfolded_expression
  - `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +14/-13 (27 lines); hunks: -120,6 +120,7 @@ def __init__(; -146,17 +147,15 @@ def __init__(; symbols: __init__
  - `test/runtime/test_kimi_k3_config.py` modified +2/-1 (3 lines); hunks: -412,8 +412,9 @@ def __init__(self, **kwargs):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1062,8 +1062,9 @@ def __init__(
+        layer_index: int,
+        model_scope: str,
-        layer_index: int = -1,
@@ -1286,13 +1287,17 @@ def __init__(
-        self._initialize_tail_capabilities(config, mapping, prefix)
+        self._initialize_tail_capabilities(
diff -- test/runtime/layers/test_kimi_k3_addmm_fold.py
@@ -0,0 +1,48 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/kimi_k3_nextn.py
@@ -120,6 +120,7 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +29/-25; `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +14/-13
  - tests: `test/runtime/layers/test_kimi_k3_addmm_fold.py` added +48/-0; `test/runtime/test_kimi_k3_config.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_kimi_k3_addmm_fold.py`, `test/runtime/test_k3_moe_tail_equivalence.py`, `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/test/ops/moe/test_latent_tail_contract.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1109 - ci(kimi-k3): refresh MI35x perf baseline

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1109
- 状态/时间: merged / 2026-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；关联提交 `231654727fff`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 10 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +3/-2 (5 lines); hunks: -90,6 +90,7 @@ report:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +3/-2 (5 lines); hunks: -90,6 +90,7 @@ report:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml
@@ -90,6 +90,7 @@ report:
-# Rounded down from an 8x MI350X concurrency-1 measurement: 43.05 / 5.26.
+# Rounded down from the median of the three most recent passing runs on the 8x
+# MI35x runner (67.48/8.09, 65.66/7.88, 65.15/7.79): 65.66 / 7.88.
-  1: [42, 5.2]
+  1: [65, 7.8]
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +3/-2
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1062 - perf(kimi3): fold the MoE finalize into the multicast latent tail (+ extract the K3 comm layer)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1062
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `b99ca12fca76`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+1796/-711，可读 patch 3165 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` added +988/-0 (988 lines); hunks: -0,0 +1,988; symbols: K3MoETailTier, select_k3_moe_tail_tier, K3AttnCommState, get，涉及 `K3MoETailTier, select_k3_moe_tail_tier, K3AttnCommState`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +48/-553 (601 lines); hunks: -61,7 +61,6; -70,14 +69,6; symbols: __init__, _initialize_tail_capabilities, pack_input_projection_weights, _routed_experts，涉及 `__init__, _initialize_tail_capabilities, pack_input_projection_weights`；`test/runtime/test_kimi_k3_config.py` modified +2/-5 (7 lines); hunks: -329,6 +329,7 @@ class FakeLinear(torch.nn.Module):; -386,11 +387,7 @@ def __init__(self, **kwargs):; symbols: FakeLinear, __init__, FakeExperts，涉及 `FakeLinear, __init__, FakeExperts`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` added +988/-0 (988 lines); hunks: -0,0 +1,988; symbols: K3MoETailTier, select_k3_moe_tail_tier, K3AttnCommState, get
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +48/-553 (601 lines); hunks: -61,7 +61,6; -70,14 +69,6; symbols: __init__, _initialize_tail_capabilities, pack_input_projection_weights, _routed_experts
  - `test/runtime/test_kimi_k3_config.py` modified +2/-5 (7 lines); hunks: -329,6 +329,7 @@ class FakeLinear(torch.nn.Module):; -386,11 +387,7 @@ def __init__(self, **kwargs):; symbols: FakeLinear, __init__, FakeExperts
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -0,0 +1,988 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -61,7 +61,6 @@
-    add3,
@@ -70,14 +69,6 @@
-from tokenspeed_kernel.ops.communication import allreduce_fusion_lane
-from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
-from tokenspeed_kernel.ops.communication.multimem import (
-    multimem_all_reduce_staged,
diff -- test/runtime/test_kimi_k3_config.py
@@ -329,6 +329,7 @@ class FakeLinear(torch.nn.Module):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` added +988/-0; `python/tokenspeed/runtime/models/kimi_k3.py` modified +48/-553
  - tests: `test/runtime/test_kimi_k3_config.py` modified +2/-5
- 验证与风险: diff 自带测试面 `test/runtime/test_k3_moe_tail_equivalence.py`, `test/runtime/test_k3_moe_tail_tier.py`, `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1121 - ci(kimi-k3): exercise DSpark in the AIME26 gate

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1121
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`；关联提交 `4869afc5927c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+61/-38，可读 patch 158 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` modified +2/-1 (3 lines); hunks: -1,9 +1,10；`tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py` modified +21/-34 (55 lines); hunks: -1947,23 +1947,31 @@ def all_reduce_can_run(state: TritonCommState, tensor: t...; -2015,18 +2023,7 @@ def acquire_symm_outputs(; symbols: all_reduce_can_run, _get_or_create_iris_state, all_reduce, acquire_symm_outputs，涉及 `all_reduce_can_run, _get_or_create_iris_state, all_reduce`。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` modified +2/-1 (3 lines); hunks: -1,9 +1,10
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py` modified +21/-34 (55 lines); hunks: -1947,23 +1947,31 @@ def all_reduce_can_run(state: TritonCommState, tensor: t...; -2015,18 +2023,7 @@ def acquire_symm_outputs(; symbols: all_reduce_can_run, _get_or_create_iris_state, all_reduce, acquire_symm_outputs
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml
@@ -1,9 +1,10 @@
+# Keep the pure target as a manual control; the matching DSpark configuration
+# is the per-commit Kimi-K3 AIME26 gate.
-  - per-commit
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py
@@ -1947,23 +1947,31 @@ def all_reduce_can_run(state: TritonCommState, tensor: torch.Tensor, op=None) ->
+def _get_or_create_iris_state(state: TritonCommState, dtype: torch.dtype):
+    """Return the Iris state sized for this communication backing buffer."""
+    import tokenspeed_kernel.ops.communication.iris as _iris_mod
+    key = (id(state.group), state.max_bytes, dtype)
+    iris_state = _iris_mod.IRIS_AR_STATES.get(key)
+    if iris_state is None:
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` modified +2/-1
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py` modified +21/-34
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-dspark-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `tokenspeed-kernel/test/ops/test_iris_communication.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1131 - ci: run Kimi K3 on two GB300 Slurm nodes

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1131
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`；关联提交 `9f62e8c8d7c2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+650/-48，可读 patch 1027 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +54/-0 (54 lines); hunks: -0,0 +1,54。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +54/-0 (54 lines); hunks: -0,0 +1,54
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -0,0 +1,54 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-kimi-k3-mxfp4-tp8-two-node-aime26-gb300-slurm
+type: eval
+workflow_stage: model-test
+triggers:
+  - per-commit
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +54/-0
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci_system/ci_path_filter.py`, `test/ci_system/pipeline.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1128 - Support nvidia/Kimi-K3-NVFP4: ModelOpt FP8_PB_WO attention + NVFP4 SiTU MoE

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1128
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`；关联提交 `1e4a7946ca84`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 12 个文件，+2495/-93，可读 patch 2989 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +506/-31 (537 lines); hunks: -124,6 +124,10; -358,13 +362,69 @@ def __init__(; symbols: __init__, _split_fused_qkv_a, _project_q_latent_gated，涉及 `__init__, _split_fused_qkv_a, _project_q_latent_gated`；`python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +34/-20 (54 lines); hunks: -483,6 +483,25 @@ class TailPlan:; -503,6 +522,7 @@ def __init__(; symbols: TailPlan, _tail_finalize_top_k, K3MoeTailComm, __init__，涉及 `TailPlan, _tail_finalize_top_k, K3MoeTailComm`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +57/-4 (61 lines); hunks: -18,6 +18,10; -710,30 +714,79 @@ def kimi3_qkvfab_projection(; symbols: kimi3_qkvfab_projection，涉及 `kimi3_qkvfab_projection`；`test/runtime/test_kimi_k3_comm_arming.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar，涉及 `test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +506/-31 (537 lines); hunks: -124,6 +124,10; -358,13 +362,69 @@ def __init__(; symbols: __init__, _split_fused_qkv_a, _project_q_latent_gated
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +34/-20 (54 lines); hunks: -483,6 +483,25 @@ class TailPlan:; -503,6 +522,7 @@ def __init__(; symbols: TailPlan, _tail_finalize_top_k, K3MoeTailComm, __init__
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +57/-4 (61 lines); hunks: -18,6 +18,10; -710,30 +714,79 @@ def kimi3_qkvfab_projection(; symbols: kimi3_qkvfab_projection
  - `test/runtime/test_kimi_k3_comm_arming.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar
  - `test/runtime/test_kimi_k3_config.py` modified +3/-0 (3 lines); hunks: -341,6 +341,9 @@ def __init__(self, *args, **kwargs):; symbols: __init__, FakeSharedExperts
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -124,6 +124,10 @@
+from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
+    preprocess_fp8_pb_wo_weights,
+)
+from tokenspeed.runtime.layers.quantization.utils import block_dequant
@@ -358,13 +362,69 @@ def __init__(
+            fused_prefix = add_prefix("fused_qkv_a_proj_with_mqa", prefix)
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -483,6 +483,25 @@ class TailPlan:
+def _tail_finalize_top_k(
+    top_k: int,
+    execution_plan,
+    experts_supports_deferred_finalize: bool,
+) -> int | None:
+    """Deferred-finalize arming decision for the latent tail (rank-uniform).
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -18,6 +18,10 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +506/-31; `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +34/-20; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +57/-4
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` added +51/-0; `test/runtime/test_kimi_k3_config.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_fp8_pb_wo_loading.py`, `test/runtime/test_kda_fp8_w8a8.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1129 - perf(kimi3): avoid packed KDA QKV copies

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1129
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `94f4d07cf68a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+118/-33，可读 patch 241 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +4/-5 (9 lines); hunks: -982,9 +982,6 @@ def forward(; -1159,8 +1156,6 @@ def _project_qkvfab(; symbols: forward, _project_qkvfab，涉及 `forward, _project_qkvfab`；`test/runtime/test_kimi_k3_config.py` modified +71/-1 (72 lines); hunks: -317,7 +317,77 @@ def test_kda_stacks_qkvfab_projection_weights(self):; symbols: test_kda_stacks_qkvfab_projection_weights, test_kda_compacts_prefill_qkv_before_backend_break, BackendCalled, capture_backend，涉及 `test_kda_stacks_qkvfab_projection_weights, test_kda_compacts_prefill_qkv_before_backend_break, BackendCalled`；`test/runtime/test_kimi_k3_attn_res.py` modified +11/-2 (13 lines); hunks: -668,7 +668,12 @@ def ref(w, rows, rk=rank):; -681,9 +686,13 @@ def test_decode_single_row_slice_is_zero_copy(self):; symbols: ref, test_decode_single_row_slice_is_zero_copy，涉及 `ref, test_decode_single_row_slice_is_zero_copy`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +4/-5 (9 lines); hunks: -982,9 +982,6 @@ def forward(; -1159,8 +1156,6 @@ def _project_qkvfab(; symbols: forward, _project_qkvfab
  - `test/runtime/test_kimi_k3_config.py` modified +71/-1 (72 lines); hunks: -317,7 +317,77 @@ def test_kda_stacks_qkvfab_projection_weights(self):; symbols: test_kda_stacks_qkvfab_projection_weights, test_kda_compacts_prefill_qkv_before_backend_break, BackendCalled, capture_backend
  - `test/runtime/test_kimi_k3_attn_res.py` modified +11/-2 (13 lines); hunks: -668,7 +668,12 @@ def ref(w, rows, rk=rank):; -681,9 +686,13 @@ def test_decode_single_row_slice_is_zero_copy(self):; symbols: ref, test_decode_single_row_slice_is_zero_copy
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -982,9 +982,6 @@ def forward(
-        # No-op at decode (single row); prefill pays a small copy for the conv.
-        if not mixed_qkv.is_contiguous():
-            mixed_qkv = mixed_qkv.contiguous()
@@ -1159,8 +1156,6 @@ def _project_qkvfab(
-        if not mixed_qkv.is_contiguous():
-            mixed_qkv = mixed_qkv.contiguous()
diff -- test/runtime/test_kimi_k3_config.py
@@ -317,7 +317,77 @@ def test_kda_stacks_qkvfab_projection_weights(self):
-        self.assertTrue(mixed_qkv.is_contiguous())
+        self.assertFalse(mixed_qkv.is_contiguous())
+        self.assertEqual(
+            mixed_qkv.untyped_storage().data_ptr(),
+            gate.untyped_storage().data_ptr(),
+        )
diff -- test/runtime/test_kimi_k3_attn_res.py
@@ -668,7 +668,12 @@ def ref(w, rows, rk=rank):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +4/-5
  - tests: `test/runtime/test_kimi_k3_config.py` modified +71/-1; `test/runtime/test_kimi_k3_attn_res.py` modified +11/-2
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/test/ops/test_kda_recurrent.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1147 - perf(cache): sparsify and budget Kimi-K3 state cache

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1147
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py`, `test/runtime/test_kimi_k3_cache_spec.py`；关联提交 `978ed2cfdc87`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 18 个文件，+417/-86，可读 patch 850 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +19/-4 (23 lines); hunks: -291,6 +291,23 @@ def check_layout(self, layout: CacheLayout) -> None:; -328,9 +345,6 @@ def parents_needed(self, layout: CacheLayout, token_capacity...; symbols: check_layout, workspace_bytes, parents_needed，涉及 `check_layout, workspace_bytes, parents_needed`；`test/runtime/test_kimi_k3_cache_spec.py` modified +62/-6 (68 lines); hunks: -79,17 +79,73 @@ def test_lcm_geometry_packs_two_kda_pages_at_tp16() -> None:; symbols: test_lcm_geometry_packs_two_kda_pages_at_tp16, test_speculative_verify_workspace_is_reserved_outside_the_arena, test_non_speculative_kimi_reserves_no_verify_workspace, test_lcm_parent_demand_uses_per_group_packing，涉及 `test_lcm_geometry_packs_two_kda_pages_at_tp16, test_speculative_verify_workspace_is_reserved_outside_the_arena, test_non_speculative_kimi_reserves_no_verify_workspace`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +19/-4 (23 lines); hunks: -291,6 +291,23 @@ def check_layout(self, layout: CacheLayout) -> None:; -328,9 +345,6 @@ def parents_needed(self, layout: CacheLayout, token_capacity...; symbols: check_layout, workspace_bytes, parents_needed
  - `test/runtime/test_kimi_k3_cache_spec.py` modified +62/-6 (68 lines); hunks: -79,17 +79,73 @@ def test_lcm_geometry_packs_two_kda_pages_at_tp16() -> None:; symbols: test_lcm_geometry_packs_two_kda_pages_at_tp16, test_speculative_verify_workspace_is_reserved_outside_the_arena, test_non_speculative_kimi_reserves_no_verify_workspace, test_lcm_parent_demand_uses_per_group_packing
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py
@@ -291,6 +291,23 @@ def check_layout(self, layout: CacheLayout) -> None:
+    # ---- extras ----
+    @override
+    def workspace_bytes(self) -> int:
+        """Dense KDA state rows staged by speculative target verification."""
+        if getattr(self.server_args, "speculative_algorithm", None) is None:
+            return 0
diff -- test/runtime/test_kimi_k3_cache_spec.py
@@ -79,17 +79,73 @@ def test_lcm_geometry_packs_two_kda_pages_at_tp16() -> None:
+def test_speculative_verify_workspace_is_reserved_outside_the_arena() -> None:
+    recipe, _, layout = kimi_tp8_layout(
+        draft_layers=5,
+        max_bs=4,
+        speculative_algorithm="DSPARK",
+        speculative_num_draft_tokens=8,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +19/-4
  - tests: `test/runtime/test_kimi_k3_cache_spec.py` modified +62/-6
- 验证与风险: diff 自带测试面 `test/runtime/conftest.py`, `test/runtime/test_gdn_state_paging.py`, `test/runtime/test_kimi_k3_cache_spec.py`, `test/runtime/test_unified_kv_slab_pool.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1155 - ci: load GB300 Kimi weights from local RAID

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1155
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`；关联提交 `0c56b0f3ac0b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+28/-5，可读 patch 99 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ install:。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ install:
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -17,7 +17,7 @@ install:
-    ts serve moonshotai/Kimi-K3
+    ts serve /models/moonshotai--Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci_system/slurm_submit.py`, `test/ci_system/test_slurm_submit.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1159 - ci: add GB300 two-node AIME26 gates for K3-NVFP4 and K3-NVFP4+DSpark

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1159
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`；关联提交 `652151d1f69f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+202/-8，可读 patch 267 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +71/-0 (71 lines); hunks: -0,0 +1,71；`test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +62/-0 (62 lines); hunks: -0,0 +1,62。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +71/-0 (71 lines); hunks: -0,0 +1,71
  - `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +62/-0 (62 lines); hunks: -0,0 +1,62
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -0,0 +1,71 @@
+api_version: ci.tokenspeed.io/v1
+# NVFP4 target + DSpark speculative decoding, on the same two-node shape as
+# the speculator-free NVFP4 gate. Speculative flags follow the proven AMD
+# DSpark gate (kimi-k3-dspark-mxfp4-tp8ep8-evalscope-aime26-amd.yaml):
+# --disable-prefill-graph because prefill graphs add ~40 capture buckets on
+# top of the decode ones the speculator rides on, and the draft weights plus
diff -- test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -0,0 +1,62 @@
+api_version: ci.tokenspeed.io/v1
+# NVFP4 sibling of the mxfp4 two-node gate: identical serve shape, but the
+# ModelOpt MIXED_PRECISION checkpoint (NVFP4 SiTU experts + FP8_PB_WO
+# attention kept FP8-resident). The pinned checkpoint is loaded from the
+# node-local RAID mounted at /models; the 7200s readiness timeout also covers
+# first-boot trtllm-gen NVFP4 cubin autotuning (~15 min).
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +71/-0; `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` added +62/-0
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci_system/test_dispatch_workflows.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1157 - perf(k3): route decode GEMV per shape

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1157
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`；关联提交 `3d596a856df5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+1528/-17，可读 patch 1631 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +3/-1 (4 lines); hunks: -2411,7 +2411,9 @@ def __init__(; symbols: __init__，涉及 `__init__`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +40/-14 (54 lines); hunks: -420,11 +420,12 @@ def kimi3_latent_projection_add3(; -452,7 +453,13 @@ def kimi3_latent_projection_add3(; symbols: kimi3_latent_projection_add3, kimi3_qkvfab_projection，涉及 `kimi3_latent_projection_add3, kimi3_qkvfab_projection`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +3/-1 (4 lines); hunks: -2411,7 +2411,9 @@ def __init__(; symbols: __init__
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +40/-14 (54 lines); hunks: -420,11 +420,12 @@ def kimi3_latent_projection_add3(; -452,7 +453,13 @@ def kimi3_latent_projection_add3(; symbols: kimi3_latent_projection_add3, kimi3_qkvfab_projection
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -2411,7 +2411,9 @@ def __init__(
-        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
+        alt_stream = (
+            torch.cuda.Stream(priority=-1) if torch.cuda.is_available() else None
+        )
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -420,11 +420,12 @@ def kimi3_latent_projection_add3(
-        solution: ``"auto"`` selects the fused row-CTA GEMV for one-token
-            execution, the fused MFMA epilogue for the tuned M=16 tile, and
-            otherwise composes the registered projection and add kernels.
-            ``"rowcta_gemv"``, ``"gluon_mfma_add3"``, and ``"composed"``
-            force an implementation.
+        solution: ``"auto"`` selects the dual-residual skinny epilogue where
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +3/-1; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +40/-14
- 验证与风险: diff 自带测试面 `test/gemm_tuning/tune_route.py`, `tokenspeed-kernel/test/ops/gemm/test_routed_gemv.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1179 - perf(k3): ll bf16 router GEMM

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1179
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`；关联提交 `1d9eb83fa153`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+1340/-2，可读 patch 1387 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +19/-2 (21 lines); hunks: -866,6 +866,15 @@ def kimi3_qkvfab_projection(; -896,6 +905,8 @@ def kimi3_router_projection(; symbols: kimi3_qkvfab_projection, _ll_bf16_usable, _mm_out_dtype_supported, kimi3_router_projection，涉及 `kimi3_qkvfab_projection, _ll_bf16_usable, _mm_out_dtype_supported`。
- 代码 diff 细节:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +19/-2 (21 lines); hunks: -866,6 +866,15 @@ def kimi3_qkvfab_projection(; -896,6 +905,8 @@ def kimi3_router_projection(; symbols: kimi3_qkvfab_projection, _ll_bf16_usable, _mm_out_dtype_supported, kimi3_router_projection
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -866,6 +866,15 @@ def kimi3_qkvfab_projection(
+def _ll_bf16_usable(hidden_states: torch.Tensor, weight: torch.Tensor, m: int) -> bool:
+    """Whether the vendored CuTe dot-product router GEMM can serve this call."""
+    try:
+        from tokenspeed_kernel.ops.gemm.ll_bf16 import ll_bf16_router_supported
+    except ImportError:
+        return False
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +19/-2
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/gemm/test_ll_bf16_router.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1184 - perf(k3): route a whole verify window through the packed top-k

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1184
- 状态/时间: merged / 2026-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py`；关联提交 `786f8f2038f5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+352/-27，可读 patch 581 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py` modified +22/-9 (31 lines); hunks: -32,6 +32,11 @@ def _kimi3_sigmoid_bias_topk_kernel(; -52,7 +57,9 @@ def _kimi3_sigmoid_bias_topk_kernel(; symbols: _kimi3_sigmoid_bias_topk_kernel, kimi3_sigmoid_bias_topk，涉及 `_kimi3_sigmoid_bias_topk_kernel, kimi3_sigmoid_bias_topk`。
- 代码 diff 细节:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py` modified +22/-9 (31 lines); hunks: -32,6 +32,11 @@ def _kimi3_sigmoid_bias_topk_kernel(; -52,7 +57,9 @@ def _kimi3_sigmoid_bias_topk_kernel(; symbols: _kimi3_sigmoid_bias_topk_kernel, kimi3_sigmoid_bias_topk
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py
@@ -32,6 +32,11 @@ def _kimi3_sigmoid_bias_topk_kernel(
+    # One program per row; every reduction below is row-local.
+    row = tl.program_id(0)
+    logits += row * num_experts
+    topk_ids += row * topk
+    topk_weights += row * topk
@@ -52,7 +57,9 @@ def _kimi3_sigmoid_bias_topk_kernel(
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py` modified +22/-9
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/moe/test_sigmoid_bias_topk_nvidia.py`, `tokenspeed-kernel/test/ops/test_kimi3_sigmoid_topk_multitoken.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1165 - ci: add manual K3 DSpark vision evals

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1165
- 状态/时间: merged / 2026-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml`；关联提交 `eba3f1b0b1da`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+268/-22，可读 patch 439 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml` added +63/-0 (63 lines); hunks: -0,0 +1,63；`test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` added +63/-0 (63 lines); hunks: -0,0 +1,63。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml` added +63/-0 (63 lines); hunks: -0,0 +1,63
  - `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` added +63/-0 (63 lines); hunks: -0,0 +1,63
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml
@@ -0,0 +1,63 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-mmmu-pro-vision-gb300-slurm
+type: eval
+workflow_stage: model-test
+triggers:
+  - slurm
diff -- test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml
@@ -0,0 +1,63 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-ocr-bench-gb300-slurm
+type: eval
+workflow_stage: model-test
+triggers:
+  - slurm
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml` added +63/-0; `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` added +63/-0
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml`, `test/ci_system/pipeline.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1203 - ci: move Kimi K3 benchmarks to MI350

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1203
- 状态/时间: merged / 2026-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；关联提交 `40b25aa07450`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+2/-2，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +1/-1 (2 lines); hunks: -7,7 +7,7 @@ triggers:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +1/-1 (2 lines); hunks: -7,7 +7,7 @@ triggers:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml
@@ -7,7 +7,7 @@ triggers:
-    - amd-mi35x-8gpu-test
+    - amd-mi350-8gpu-bench
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-dspark-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1200 - perf(gemm): route the unrouted K3 decode projections

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1200
- 状态/时间: merged / 2026-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`；关联提交 `f4d8ec6e039c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+259/-72，可读 patch 523 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-1 (2 lines); hunks: -1857,7 +1857,7 @@ def forward(; symbols: forward，涉及 `forward`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +21/-1 (22 lines); hunks: -26,7 +26,11; -304,6 +308,7 @@ def kimi3_latent_projection(; symbols: kimi3_latent_projection, kimi3_shared_situ_projection, kimi3_shared_down_projection，涉及 `kimi3_latent_projection, kimi3_shared_situ_projection, kimi3_shared_down_projection`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-1 (2 lines); hunks: -1857,7 +1857,7 @@ def forward(; symbols: forward
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +21/-1 (22 lines); hunks: -26,7 +26,11; -304,6 +308,7 @@ def kimi3_latent_projection(; symbols: kimi3_latent_projection, kimi3_shared_situ_projection, kimi3_shared_down_projection
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1857,7 +1857,7 @@ def forward(
-            routed_in = decode_gemv(hidden_states, self.routed_expert_down_proj.weight)
+            routed_in, _ = self.routed_expert_down_proj(hidden_states)
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -26,7 +26,11 @@
+from tokenspeed_kernel.ops.gemm.routed_gemv import decode_gemv_routed
@@ -304,6 +308,7 @@ def kimi3_latent_projection(
+    routed = solution == "auto"
@@ -352,6 +357,10 @@ def kimi3_latent_projection(
+    if routed and decode_gemv_routed(hidden_states, weight):
+        from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-1; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +21/-1
- 验证与风险: diff 自带测试面 `test/gemm_tuning/tune_route.py`, `tokenspeed-kernel/test/ops/gemm/test_routed_gemv.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1031 - feat(kimi-k3): serve DSpark drafts (fc_norm + AttnRes tap)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1031
- 状态/时间: merged / 2026-08-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py`；关联提交 `a2daa17d1a14`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+226/-10，可读 patch 368 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +62/-9 (71 lines); hunks: -2020,6 +2020,7 @@ def __init__(; -2093,6 +2094,10 @@ def _fused_attnres_graph_available(; symbols: __init__, _fused_attnres_graph_available, get_layer, _refresh_dflash_capture_fallback，涉及 `__init__, _fused_attnres_graph_available, get_layer`；`python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +38/-0 (38 lines); hunks: -56,6 +56,7; -315,6 +316,17 @@ def __init__(; symbols: __init__, project_target_hidden, _finalize_hidden, post_load_weights，涉及 `__init__, project_target_hidden, _finalize_hidden`；`python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` modified +11/-0 (11 lines); hunks: -42,6 +42,8; -73,6 +75,8 @@ def __init__(; symbols: KimiK3DSparkConfig, __init__, validate_k3_dspark_config，涉及 `KimiK3DSparkConfig, __init__, validate_k3_dspark_config`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +62/-9 (71 lines); hunks: -2020,6 +2020,7 @@ def __init__(; -2093,6 +2094,10 @@ def _fused_attnres_graph_available(; symbols: __init__, _fused_attnres_graph_available, get_layer, _refresh_dflash_capture_fallback
  - `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +38/-0 (38 lines); hunks: -56,6 +56,7; -315,6 +316,17 @@ def __init__(; symbols: __init__, project_target_hidden, _finalize_hidden, post_load_weights
  - `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` modified +11/-0 (11 lines); hunks: -42,6 +42,8; -73,6 +75,8 @@ def __init__(; symbols: KimiK3DSparkConfig, __init__, validate_k3_dspark_config
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -2020,6 +2020,7 @@ def __init__(
+        self._dflash_attnres_capture_fallback = False
@@ -2093,6 +2094,10 @@ def _fused_attnres_graph_available(
+        if getattr(self, "_dflash_attnres_capture_fallback", False):
+            return False
+        # B1 already fuses the post-attention mix into the all-reduce.
@@ -2478,10 +2483,20 @@ def get_layer(idx: int, prefix: str):
diff -- python/tokenspeed/runtime/models/kimi_k3_dspark.py
@@ -56,6 +56,7 @@
+from tokenspeed.runtime.layers.segmented_rmsnorm import segmented_rmsnorm
@@ -315,6 +316,17 @@ def __init__(
+        self.fc_norm = (
+            nn.ModuleList(
+                [
+                    RMSNorm(int(config.target_hidden_size), eps=eps)
diff -- python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py
@@ -42,6 +42,8 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +62/-9; `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +38/-0; `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/configs/kimi_k3_dspark_config.py`, `python/tokenspeed/runtime/execution/drafter/dflash.py`, `python/tokenspeed/runtime/layers/segmented_rmsnorm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1187 - test: kimi-k3 agentic decode-throughput bench

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1187
- 状态/时间: merged / 2026-08-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh`；关联提交 `0b61a2de6c20`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+421/-0，可读 patch 427 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` added +25/-0 (25 lines); hunks: -0,0 +1,25；`test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` added +25/-0 (25 lines); hunks: -0,0 +1,25；`test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` added +169/-0 (169 lines); hunks: -0,0 +1,169；`test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: num_gpus_from_config, collect, main，涉及 `num_gpus_from_config, collect, main`。
- 代码 diff 细节:
  - `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` added +25/-0 (25 lines); hunks: -0,0 +1,25
  - `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` added +25/-0 (25 lines); hunks: -0,0 +1,25
  - `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` added +169/-0 (169 lines); hunks: -0,0 +1,169
  - `test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: num_gpus_from_config, collect, main
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh
@@ -0,0 +1,25 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model nvidia/Kimi-K3-NVFP4 \
+    --attn-tp-size 8 \
+    --ep-size 8 \
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh
@@ -0,0 +1,25 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model nvidia/Kimi-K3-NVFP4 \
+    --attn-tp-size 8 \
+    --moe-tp-size 8 \
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh
@@ -0,0 +1,169 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` added +25/-0; `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` added +25/-0; `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` added +169/-0; `test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py` added +86/-0
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/kimi_k3/tokenspeed/README.md`, `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_dp8_moe_ep8.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1174 - perf(k3): extend latent-tail fusion to M64 with split collectives

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1174
- 状态/时间: merged / 2026-08-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`；关联提交 `103120bf778e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+320/-35，可读 patch 787 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +39/-3 (42 lines); hunks: -59,13 +59,17; -475,12 +479,15 @@ class TailPlan:; symbols: TailPlan, _tail_finalize_top_k, __init__, plan，涉及 `TailPlan, _tail_finalize_top_k, __init__`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +6/-0 (6 lines); hunks: -1844,6 +1844,7 @@ def forward(; -1857,6 +1858,10 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +39/-3 (42 lines); hunks: -59,13 +59,17; -475,12 +479,15 @@ class TailPlan:; symbols: TailPlan, _tail_finalize_top_k, __init__, plan
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +6/-0 (6 lines); hunks: -1844,6 +1844,7 @@ def forward(; -1857,6 +1858,10 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -59,13 +59,17 @@
+from tokenspeed_kernel.platform import current_platform
-from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_cuda_graph_phase
+from tokenspeed.runtime.execution.cuda_graph_wrapper import (
+    get_is_capture_mode,
+    get_is_cuda_graph_phase,
+)
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1844,6 +1844,7 @@ def forward(
+        prepared_shared_shard = None
@@ -1857,6 +1858,10 @@ def forward(
+                if plan.split_shared_rs and fork._active:
+                    prepared_shared_shard = self.comm.reduce_scatter_shared(
+                        shared_partial
+                    )
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +39/-3; `python/tokenspeed/runtime/models/kimi_k3.py` modified +6/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_k3_moe_tail_equivalence.py`, `tokenspeed-kernel/test/ops/moe/test_latent_tail_distributed.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1225 - ci(k3): give the Slurm aime26 gates the budget their context window allow

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1225
- 状态/时间: merged / 2026-08-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`；关联提交 `06a63156a1d4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+6/-6，可读 patch 41 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +3/-3 (6 lines); hunks: -53,10 +53,10 @@ eval:；`test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml` modified +1/-1 (2 lines); hunks: -48,7 +48,7 @@ eval:；`test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -48,7 +48,7 @@ eval:；`test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -61,7 +61,7 @@ eval:。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +3/-3 (6 lines); hunks: -53,10 +53,10 @@ eval:
  - `test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml` modified +1/-1 (2 lines); hunks: -48,7 +48,7 @@ eval:
  - `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -48,7 +48,7 @@ eval:
  - `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -61,7 +61,7 @@ eval:
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -53,10 +53,10 @@ eval:
-    --generation-config '{"do_sample":true,"temperature":1.0,"max_tokens":32768,"extra_body":{"reasoning_effort":"high"}}'
+    --generation-config '{"do_sample":true,"temperature":1.0,"max_tokens":63488,"extra_body":{"reasoning_effort":"high"}}'
-# 65536-token evalscope protocol (tp16, PR #1128); the mxfp4 sibling records
-# 0.9333 on this exact 32768-token CI protocol.
+# 65536-token evalscope protocol (tp16, PR #1128); the mxfp4 sibling recorded
+# 0.9333 under the earlier 32768-token cap this protocol no longer uses.
diff -- test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml
@@ -48,7 +48,7 @@ eval:
-    --generation-config '{"do_sample":true,"temperature":1.0,"max_tokens":32768,"extra_body":{"reasoning_effort":"high"}}'
+    --generation-config '{"do_sample":true,"temperature":1.0,"max_tokens":63488,"extra_body":{"reasoning_effort":"high"}}'
diff -- test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -48,7 +48,7 @@ eval:
-    --generation-config '{"do_sample":true,"temperature":1.0,"max_tokens":32768,"extra_body":{"reasoning_effort":"high"}}'
+    --generation-config '{"do_sample":true,"temperature":1.0,"max_tokens":63488,"extra_body":{"reasoning_effort":"high"}}'
diff -- test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -61,7 +61,7 @@ eval:
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +3/-3; `test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml` modified +1/-1; `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1; `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-mxfp4-tp16-four-node-evalscope-aime26-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1135 - perf(kimi3): tune small-batch latent projection

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1135
- 状态/时间: merged / 2026-08-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`；关联提交 `4b3e51426500`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+58/-7，可读 patch 149 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +38/-6 (44 lines); hunks: -55,6 +55,10 @@ def _use_gluon_mediumm(m: int, k: int, n: int) -> bool:; -280,10 +284,11 @@ def kimi3_latent_projection(; symbols: _use_gluon_mediumm, _use_gluon_smallm, _use_gluon_largem, kimi3_latent_projection，涉及 `_use_gluon_mediumm, _use_gluon_smallm, _use_gluon_largem`。
- 代码 diff 细节:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +38/-6 (44 lines); hunks: -55,6 +55,10 @@ def _use_gluon_mediumm(m: int, k: int, n: int) -> bool:; -280,10 +284,11 @@ def kimi3_latent_projection(; symbols: _use_gluon_mediumm, _use_gluon_smallm, _use_gluon_largem, kimi3_latent_projection
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -55,6 +55,10 @@ def _use_gluon_mediumm(m: int, k: int, n: int) -> bool:
+def _use_gluon_smallm(m: int, k: int, n: int) -> bool:
+    return m in (2, 4) and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
@@ -280,10 +284,11 @@ def kimi3_latent_projection(
+    ``solution='gluon_smallm'`` forces the split-K small-M gfx950 kernel,
-    ``solution='gluon_largem'`` forces the large-M gfx950 kernel. ``auto`` uses
-    their measured gfx950 crossovers for canonical K3 shapes and retains the
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +38/-6
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_kimi3_projection_gfx950.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1242 - chore(kimi-k3): use TokenSpeed MLA for agentic drafter

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1242
- 状态/时间: merged / 2026-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh`；关联提交 `1d9061892b1b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+3/-3，可读 patch 27 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ exec ts serve \；`test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ exec ts serve \。
- 代码 diff 细节:
  - `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ exec ts serve \
  - `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ exec ts serve \
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh
@@ -17,7 +17,7 @@ exec ts serve \
-    --drafter-attention-backend mla \
+    --drafter-attention-backend tokenspeed_mla \
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh
@@ -17,7 +17,7 @@ exec ts serve \
-    --drafter-attention-backend mla \
+    --drafter-attention-backend tokenspeed_mla \
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` modified +1/-1; `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` modified +1/-1
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_dp8_moe_ep8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1152 - fix(cache): support attention-DP for Kimi-K3 by deriving the MLA packing from the KDA state size

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1152
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py`, `test/runtime/test_kimi_k3_cache_spec.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `f5819a660d21`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+93/-26，可读 patch 196 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +12/-8 (20 lines); hunks: -21,8 +21,9; -57,7 +58,6; symbols: _one_based_layers, fields_for_layer, packing，涉及 `_one_based_layers, fields_for_layer, packing`；`test/runtime/test_kimi_k3_cache_spec.py` modified +62/-8 (70 lines); hunks: -58,27 +58,81 @@ def test_lcm_reference_geometry_is_exact() -> None:; symbols: test_lcm_reference_geometry_is_exact, test_lcm_geometry_packs_two_kda_pages_at_tp16, test_lcm_geometry_shrinks_with_the_kda_state_at_tp16, test_attention_dp_layouts_grow_the_mla_packing，涉及 `test_lcm_reference_geometry_is_exact, test_lcm_geometry_packs_two_kda_pages_at_tp16, test_lcm_geometry_shrinks_with_the_kda_state_at_tp16`；`test/runtime/test_kimi_k3_config.py` modified +17/-9 (26 lines); hunks: -696,21 +696,29 @@ def _plan(cfg, tp):; symbols: _plan, test_linear_packing_scales_with_attn_tp, test_mla_packing_scales_with_attn_tp, test_reduced_layer_variant_plans，涉及 `_plan, test_linear_packing_scales_with_attn_tp, test_mla_packing_scales_with_attn_tp`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +12/-8 (20 lines); hunks: -21,8 +21,9; -57,7 +58,6; symbols: _one_based_layers, fields_for_layer, packing
  - `test/runtime/test_kimi_k3_cache_spec.py` modified +62/-8 (70 lines); hunks: -58,27 +58,81 @@ def test_lcm_reference_geometry_is_exact() -> None:; symbols: test_lcm_reference_geometry_is_exact, test_lcm_geometry_packs_two_kda_pages_at_tp16, test_lcm_geometry_shrinks_with_the_kda_state_at_tp16, test_attention_dp_layouts_grow_the_mla_packing
  - `test/runtime/test_kimi_k3_config.py` modified +17/-9 (26 lines); hunks: -696,21 +696,29 @@ def _plan(cfg, tp):; symbols: _plan, test_linear_packing_scales_with_attn_tp, test_mla_packing_scales_with_attn_tp, test_reduced_layer_variant_plans
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py
@@ -21,8 +21,9 @@
-are dense and pin a 12:1 packing; the state groups pack to match the MLA
-plane's byte width so no parent is wasted.
+are dense and pin the smallest packing whose plane width covers one per-layer
+KDA state (12 at attn tp=8); the state groups pack to match the MLA plane's
+byte width so no parent is wasted.
@@ -57,7 +58,6 @@
diff -- test/runtime/test_kimi_k3_cache_spec.py
@@ -58,27 +58,81 @@ def test_lcm_reference_geometry_is_exact() -> None:
-def test_lcm_geometry_packs_two_kda_pages_at_tp16() -> None:
-    """KDA state halves at TP16; two pages pack per MLA-sized plane."""
+def test_lcm_geometry_shrinks_with_the_kda_state_at_tp16() -> None:
+    """KDA state halves at TP16; the plane and the parent halve with it."""
-        "full_attention": 12,
-        "linear_attention_0": 2,
diff -- test/runtime/test_kimi_k3_config.py
@@ -696,21 +696,29 @@ def _plan(cfg, tp):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +12/-8
  - tests: `test/runtime/test_kimi_k3_cache_spec.py` modified +62/-8; `test/runtime/test_kimi_k3_config.py` modified +17/-9
- 验证与风险: diff 自带测试面 `test/runtime/conftest.py`, `test/runtime/test_kimi_k3_cache_spec.py`, `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1263 - perf(kimi-k3): select SiTU routing by forward phase

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1263
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `0cfd6a30da55`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 12 个文件，+492/-164，可读 patch 1050 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +71/-32 (103 lines); hunks: -1352,7 +1352,8 @@ class KimiLinearMoE(nn.Module):; -1437,24 +1438,9 @@ def __init__(; symbols: KimiLinearMoE, __init__，涉及 `KimiLinearMoE, __init__`；`test/runtime/test_kimi_k3_config.py` modified +33/-0 (33 lines); hunks: -148,6 +148,32 @@ def test_mamba2_cache_params_respects_tp(self):; -391,6 +417,7 @@ def capture_backend(**kwargs):; symbols: test_mamba2_cache_params_respects_tp, KimiK3RegistrationTests, test_hybrid_moe_precomputes_routing_only_for_decode, test_mla_mixed_batch_slices_decode_gate_to_live_rows，涉及 `test_mamba2_cache_params_respects_tp, KimiK3RegistrationTests, test_hybrid_moe_precomputes_routing_only_for_decode`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +71/-32 (103 lines); hunks: -1352,7 +1352,8 @@ class KimiLinearMoE(nn.Module):; -1437,24 +1438,9 @@ def __init__(; symbols: KimiLinearMoE, __init__
  - `test/runtime/test_kimi_k3_config.py` modified +33/-0 (33 lines); hunks: -148,6 +148,32 @@ def test_mamba2_cache_params_respects_tp(self):; -391,6 +417,7 @@ def capture_backend(**kwargs):; symbols: test_mamba2_cache_params_respects_tp, KimiK3RegistrationTests, test_hybrid_moe_precomputes_routing_only_for_decode, test_mla_mixed_batch_slices_decode_gate_to_live_rows
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1352,7 +1352,8 @@ class KimiLinearMoE(nn.Module):
-      platforms use flashinfer's TRTLLM-Gen SiTU MoE.
+      platforms use flashinfer's TRTLLM-Gen SiTU MoE. The selected MoE kernel
+      advertises whether it consumes precomputed TopK or routes from logits.
@@ -1437,24 +1438,9 @@ def __init__(
-        self.topk = TopK(
-            top_k=self.top_k,
diff -- test/runtime/test_kimi_k3_config.py
@@ -148,6 +148,32 @@ def test_mamba2_cache_params_respects_tp(self):
+    def test_hybrid_moe_precomputes_routing_only_for_decode(self):
+        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
+        from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
+        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE
+        layer = KimiLinearMoE.__new__(KimiLinearMoE)
+        torch.nn.Module.__init__(layer)
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +71/-32
  - tests: `test/runtime/test_kimi_k3_config.py` modified +33/-0
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_moe.py`, `test/runtime/layers/test_moe_expert.py`, `test/runtime/layers/test_moe_topk.py`, `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1140 - perf(kimi-k3): tune small-M MoE decode

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1140
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `f17b03efc172`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+348/-79，可读 patch 752 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-1 (2 lines); hunks: -1853,7 +1853,7 @@ def forward(; symbols: forward，涉及 `forward`；`test/runtime/test_kimi_k3_config.py` modified +34/-0 (34 lines); hunks: -592,6 +592,40 @@ def __init__(self, **kwargs):; symbols: __init__, test_native_kimi_moe_zero_tokens_bypass_fused_pipeline, test_cross_dp_ep_gather_uses_dp_group_and_returns_local_offset，涉及 `__init__, test_native_kimi_moe_zero_tokens_bypass_fused_pipeline, test_cross_dp_ep_gather_uses_dp_group_and_returns_local_offset`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-1 (2 lines); hunks: -1853,7 +1853,7 @@ def forward(; symbols: forward
  - `test/runtime/test_kimi_k3_config.py` modified +34/-0 (34 lines); hunks: -592,6 +592,40 @@ def __init__(self, **kwargs):; symbols: __init__, test_native_kimi_moe_zero_tokens_bypass_fused_pipeline, test_cross_dp_ep_gather_uses_dp_group_and_returns_local_offset
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1853,7 +1853,7 @@ def forward(
-            if self._use_fused_decode_pipeline and hidden_states.shape[0] == 1:
+            if self._use_fused_decode_pipeline and 0 < hidden_states.shape[0] <= 4:
diff -- test/runtime/test_kimi_k3_config.py
@@ -592,6 +592,40 @@ def __init__(self, **kwargs):
+    def test_native_kimi_moe_zero_tokens_bypass_fused_pipeline(self):
+        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE
+        hidden_states = torch.empty(0, 64)
+        prefix_sum = torch.empty_like(hidden_states)
+        native_latent_moe = mock.Mock(return_value=prefix_sum)
+        fused_pipeline = mock.Mock(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-1
  - tests: `test/runtime/test_kimi_k3_config.py` modified +34/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/test/ops/moe/test_gluon_mxfp4_amd.py`, `tokenspeed-kernel/test/ops/moe/test_gluon_mxfp4_routing_gfx950.py`, `tokenspeed-kernel/test/ops/test_kimi3_projection_gfx950.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1270 - fix(kimi3): warm the MoE auxiliary stream before graph capture

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1270
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；关联提交 `e478127abd94`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+174/-1，可读 patch 190 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +16/-1 (17 lines); hunks: -102,6 +102,7; -1846,7 +1847,21 @@ def forward(; symbols: forward，涉及 `forward`；`test/runtime/test_kimi_k3_moe_fork_warmup.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: _SpyFork, __init__, scope, branch，涉及 `_SpyFork, __init__, scope`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +16/-1 (17 lines); hunks: -102,6 +102,7; -1846,7 +1847,21 @@ def forward(; symbols: forward
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: _SpyFork, __init__, scope, branch
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -102,6 +102,7 @@
+    get_is_cuda_graph_phase,
@@ -1846,7 +1847,21 @@ def forward(
-        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
+        # Enable the fork for the whole graph phase, but only overlap during
+        # capture. The pre-capture warmup runs with capture mode off, so gating
+        # ``enable`` on it left the auxiliary stream completely untouched until
diff -- test/runtime/test_kimi_k3_moe_fork_warmup.py
@@ -0,0 +1,158 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +16/-1
  - tests: `test/runtime/test_kimi_k3_moe_fork_warmup.py` added +158/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_moe_fork_warmup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1283 - ci(kimi-k3): cover TP8/EP1 on gfx950 with manual trigger

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1283
- 状态/时间: merged / 2026-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；关联提交 `189ea2ce7a85`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+231/-1，可读 patch 261 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: of，涉及 `of`；`test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` added +80/-0 (80 lines); hunks: -0,0 +1,80；`test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +21/-1 (22 lines); hunks: -34,18 +34,32; -99,8 +113,14 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _SpyFork, _make_moe，涉及 `_SpyFork, _make_moe`。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: of
  - `test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` added +80/-0 (80 lines); hunks: -0,0 +1,80
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +21/-1 (22 lines); hunks: -34,18 +34,32; -99,8 +113,14 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _SpyFork, _make_moe
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml
@@ -0,0 +1,130 @@
+api_version: ci.tokenspeed.io/v1
+# TP8/EP1 counterpart to the tp8ep8 job. Every other Kimi-K3 AMD job runs
+# EP8, so the tensor-parallel MoE placement -- and with it the gfx950 A8W4
+# SiTU kernel -- had no CI coverage at all.
+#
+# The gap was not theoretical: EP1 deadlocked during HIP graph capture
diff -- test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml
@@ -0,0 +1,80 @@
+api_version: ci.tokenspeed.io/v1
+# Accuracy coverage for the TP8/EP1 placement, which selects the gfx950 A8W4
+# SiTU MoE kernel instead of the A16W4 EP path every other Kimi-K3 AMD job
+# exercises. That kernel quantizes activations to FP8, so it is the config
+# where a numerics regression would show up first -- and it shipped with one:
+# the MoE combine wrote to a freshly allocated tensor instead of the caller's
diff -- test/runtime/test_kimi_k3_moe_fork_warmup.py
@@ -34,18 +34,32 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml` added +130/-0; `test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` added +80/-0; `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +21/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1289 - ci: fix kimi k3 perf tokenizer cache

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1289
- 状态/时间: merged / 2026-08-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；关联提交 `3c9418bac124`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+13/-3，可读 patch 65 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +6/-1 (7 lines); hunks: -49,13 +49,17 @@ perf:; -65,7 +69,8 @@ perf:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +6/-1 (7 lines); hunks: -49,13 +49,17 @@ perf:; -65,7 +69,8 @@ perf:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml
@@ -49,13 +49,17 @@ perf:
+  # EvalScope loads remote tokenizer paths through ModelScope even when the
+  # dataset source is Hugging Face, so materialize the HF tokenizer locally.
+    TOKENIZER_PATH="$OUTPUTS_DIR/tokenizer" &&
+    /tmp/evalscope-perf/bin/python -c 'import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("moonshotai/Kimi-K3", trust_remote_code=True).save_pretrai
@@ -65,7 +69,8 @@ perf:
-    --tokenizer-path moonshotai/Kimi-K3
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +6/-1
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-dspark-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/random_benchmark/tokenspeed/collect_outputs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1280 - ci: run Kimi K3 vision evals nightly

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1280
- 状态/时间: merged / 2026-08-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml`；关联提交 `ef55149fcc1b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+250/-11，可读 patch 335 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-mmmu-pro-vision-gb3...；`test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-ocr-bench-gb300-slurm。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-mmmu-pro-vision-gb3...
  - `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-ocr-bench-gb300-slurm
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml
@@ -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-mmmu-pro-vision-gb300-slurm
-  - slurm
+  - nightly
diff -- test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml
@@ -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-dspark-tp8-two-node-ocr-bench-gb300-slurm
-  - slurm
+  - nightly
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml` modified +1/-1; `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml`, `test/ci_system/pipeline.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1300 - perf(k3): end the fused MoE tail at its profit edge, not its capacity

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1300
- 状态/时间: merged / 2026-08-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3_comm.py`；关联提交 `8f0507fa2da8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+44/-4，可读 patch 90 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +8/-2 (10 lines); hunks: -99,6 +99,9 @@ class K3MoETailTier(IntEnum):; -117,7 +120,8 @@ def select_k3_moe_tail_tier(; symbols: K3MoETailTier, select_k3_moe_tail_tier, plan，涉及 `K3MoETailTier, select_k3_moe_tail_tier, plan`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +8/-2 (10 lines); hunks: -99,6 +99,9 @@ class K3MoETailTier(IntEnum):; -117,7 +120,8 @@ def select_k3_moe_tail_tier(; symbols: K3MoETailTier, select_k3_moe_tail_tier, plan
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -99,6 +99,9 @@ class K3MoETailTier(IntEnum):
+# Measured profit edge of the fused tail; the kernel's own capacity is larger.
+TAIL_FUSION_MAX_TOKENS = 32
@@ -117,7 +120,8 @@ def select_k3_moe_tail_tier(
-        tail_fusion_max_tokens: Fused decode kernel capacity, 0 when absent.
+        tail_fusion_max_tokens: Largest token count the fused tail is both
+            able and worth running at, 0 when absent.
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +8/-2
- 验证与风险: diff 自带测试面 `test/runtime/test_k3_moe_tail_equivalence.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1306 - ci: add B300 Kimi K3 DeepSWE workflow

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1306
- 状态/时间: merged / 2026-08-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-deepswe-b300.yaml`；关联提交 `5afee017455c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+597/-0，可读 patch 610 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-deepswe-b300.yaml` added +55/-0 (55 lines); hunks: -0,0 +1,55。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-deepswe-b300.yaml` added +55/-0 (55 lines); hunks: -0,0 +1,55
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-deepswe-b300.yaml
@@ -0,0 +1,55 @@
+api_version: ci.tokenspeed.io/v1
+name: eval-kimi-k3-deepswe-b300
+type: eval
+workflow_stage: model-test
+triggers:
+  - manual
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-deepswe-b300.yaml` added +55/-0
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/deepswe/kimi_code_agent.py`, `test/ci/deepswe/run_deepswe.sh`, `test/ci/deepswe/summarize.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1315 - ci: trust Kimi K3 tokenizer code

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1315
- 状态/时间: merged / 2026-08-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-deepswe-b300.yaml`；关联提交 `ad9d831e8062`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-deepswe-b300.yaml` modified +1/-0 (1 lines); hunks: -17,6 +17,7 @@ server:。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-deepswe-b300.yaml` modified +1/-0 (1 lines); hunks: -17,6 +17,7 @@ server:
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-deepswe-b300.yaml
@@ -17,6 +17,7 @@ server:
+    --trust-remote-code
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-deepswe-b300.yaml` modified +1/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-deepswe-b300.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1318 - perf(k3): decode gates, GEMV route QA, and FlashInfer 0.6.18 re-tune

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1318
- 状态/时间: merged / 2026-08-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；关联提交 `c0ab49a9e888`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+649/-20，可读 patch 833 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +17/-2 (19 lines); hunks: -114,6 +114,7 @@ def select_k3_moe_tail_tier(; -124,6 +125,8 @@ def select_k3_moe_tail_tier(; symbols: select_k3_moe_tail_tier, __init__, plan，涉及 `select_k3_moe_tail_tier, __init__, plan`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +14/-3 (17 lines); hunks: -1874,12 +1874,23 @@ def forward(; symbols: forward，涉及 `forward`；`test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +2/-1 (3 lines); hunks: -101,7 +101,8 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _make_moe，涉及 `_make_moe`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +17/-2 (19 lines); hunks: -114,6 +114,7 @@ def select_k3_moe_tail_tier(; -124,6 +125,8 @@ def select_k3_moe_tail_tier(; symbols: select_k3_moe_tail_tier, __init__, plan
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +14/-3 (17 lines); hunks: -1874,12 +1874,23 @@ def forward(; symbols: forward
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +2/-1 (3 lines); hunks: -101,7 +101,8 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _make_moe
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -114,6 +114,7 @@ def select_k3_moe_tail_tier(
+    is_decode: bool = False,
@@ -124,6 +125,8 @@ def select_k3_moe_tail_tier(
+        is_decode: Whether this forward is a decode (spec-verify included);
+            rank-uniform and stable between graph capture and replay.
@@ -134,7 +137,12 @@ def select_k3_moe_tail_tier(
-    if multimem_ok and MULTIMEM_AR_MIN_TOKENS <= num_tokens <= MULTIMEM_AR_MAX_TOKENS:
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1874,12 +1874,23 @@ def forward(
-        # TopK, its single small CTA overlaps down_proj from the aux stream,
-        # followed by the shared chain. Kernel routing bypasses that CTA.
+        # TopK runs on the fork branch beside down_proj; routing bypasses it.
-        plan = self.comm.plan(num_tokens, hidden_states)
+        plan = self.comm.plan(
+            num_tokens,
diff -- test/runtime/test_kimi_k3_moe_fork_warmup.py
@@ -101,7 +101,8 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +17/-2; `python/tokenspeed/runtime/models/kimi_k3.py` modified +14/-3
  - tests: `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/gemm_tuning/tune_route.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`, `tokenspeed-kernel/test/ops/gemm/test_routed_gemv.py`, `tokenspeed-kernel/test/thirdparty/test_kda_gate_precompute.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1330 - perf(kimi3): let the MoE tail join its reductions without a lane

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1330
- 状态/时间: merged / 2026-08-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `405acad13163`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+137/-3，可读 patch 249 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +26/-1 (27 lines); hunks: -115,6 +115,7 @@ def select_k3_moe_tail_tier(; -123,7 +124,12 @@ def select_k3_moe_tail_tier(; symbols: select_k3_moe_tail_tier, plan，涉及 `select_k3_moe_tail_tier, plan`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-0 (1 lines); hunks: -1541,6 +1541,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_kimi_k3_config.py` modified +3/-0 (3 lines); hunks: -238,6 +238,7 @@ def __init__(self, *args, **kwargs):; -476,6 +477,7 @@ def __init__(self, **kwargs):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +26/-1 (27 lines); hunks: -115,6 +115,7 @@ def select_k3_moe_tail_tier(; -123,7 +124,12 @@ def select_k3_moe_tail_tier(; symbols: select_k3_moe_tail_tier, plan
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-0 (1 lines); hunks: -1541,6 +1541,7 @@ def __init__(; symbols: __init__
  - `test/runtime/test_kimi_k3_config.py` modified +3/-0 (3 lines); hunks: -238,6 +238,7 @@ def __init__(self, *args, **kwargs):; -476,6 +477,7 @@ def __init__(self, **kwargs):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -115,6 +115,7 @@ def select_k3_moe_tail_tier(
+    join_moe_reduce: bool = False,
@@ -123,7 +124,12 @@ def select_k3_moe_tail_tier(
-        fused_moe_ar: Whether the fused-AR execution plan is armed.
+        fused_moe_ar: Whether the fused-AR execution plan is armed (implies a
+            backend-owned lane, so TRT-LLM only).
+        join_moe_reduce: Whether the routed and shared partials can be reduced
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1541,6 +1541,7 @@ def __init__(
+            shard_up_projection=self._shard_up_projection,
diff -- test/runtime/test_kimi_k3_config.py
@@ -238,6 +238,7 @@ def __init__(self, *args, **kwargs):
+                has_tp_ep=True,
@@ -476,6 +477,7 @@ def __init__(self, **kwargs):
+                has_tp_ep=True,
@@ -544,6 +546,7 @@ def __init__(self, **kwargs):
+                has_tp_ep=True,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +26/-1; `python/tokenspeed/runtime/models/kimi_k3.py` modified +1/-0
  - tests: `test/runtime/test_kimi_k3_config.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_k3_moe_tail_equivalence.py`, `test/runtime/test_k3_moe_tail_tier.py`, `test/runtime/test_kimi_k3_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1340 - docs: fix kimi_k2.5 agentic bench README cd paths

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1340
- 状态/时间: merged / 2026-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md`, `test/agentic_benchmark/kimi_k2.5/trtllm/README.md`；关联提交 `ba1b1c71da93`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+2/-2，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md` modified +1/-1 (2 lines); hunks: -18,7 +18,7 @@ outputs/ / /parallel_ _number_ / # per-run evalscope artifa；`test/agentic_benchmark/kimi_k2.5/trtllm/README.md` modified +1/-1 (2 lines); hunks: -18,7 +18,7 @@ outputs/ / /parallel_ _number_ / # per-run evalscope artifa。
- 代码 diff 细节:
  - `test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md` modified +1/-1 (2 lines); hunks: -18,7 +18,7 @@ outputs/ / /parallel_ _number_ / # per-run evalscope artifa
  - `test/agentic_benchmark/kimi_k2.5/trtllm/README.md` modified +1/-1 (2 lines); hunks: -18,7 +18,7 @@ outputs/ / /parallel_ _number_ / # per-run evalscope artifa
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md
@@ -18,7 +18,7 @@ outputs/<sweep_ts>/<config>/parallel_<P>_number_<N>/  # per-run evalscope artifa
-cd test/agentic_benchmark/tokenspeed
+cd test/agentic_benchmark/kimi_k2.5/tokenspeed
diff -- test/agentic_benchmark/kimi_k2.5/trtllm/README.md
@@ -18,7 +18,7 @@ outputs/<sweep_ts>/<config>/parallel_<P>_number_<N>/  # per-run evalscope artifa
-cd test/agentic_benchmark/trtllm
+cd test/agentic_benchmark/kimi_k2.5/trtllm
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md` modified +1/-1; `test/agentic_benchmark/kimi_k2.5/trtllm/README.md` modified +1/-1
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/kimi_k2.5/tokenspeed/README.md`, `test/agentic_benchmark/kimi_k2.5/trtllm/README.md`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1358 - perf(kimi-k3): project the router, routed latent and shared gate/up together

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1358
- 状态/时间: merged / 2026-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；关联提交 `fe80daf104b5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+287/-35，可读 patch 536 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +72/-7 (79 lines); hunks: -62,7 +62,9; -499,6 +501,9 @@ class TailPlan:; symbols: TailPlan, _tail_finalize_top_k, _acquire_symm_join_outputs, K3MoeTailComm，涉及 `TailPlan, _tail_finalize_top_k, _acquire_symm_join_outputs`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +35/-16 (51 lines); hunks: -1873,10 +1873,6 @@ def forward(; -1892,10 +1888,35 @@ def forward(; symbols: forward，涉及 `forward`；`test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +4/-0 (4 lines); hunks: -96,6 +96,7 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; -122,6 +123,9 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _make_moe，涉及 `_make_moe`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +72/-7 (79 lines); hunks: -62,7 +62,9; -499,6 +501,9 @@ class TailPlan:; symbols: TailPlan, _tail_finalize_top_k, _acquire_symm_join_outputs, K3MoeTailComm
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +35/-16 (51 lines); hunks: -1873,10 +1873,6 @@ def forward(; -1892,10 +1888,35 @@ def forward(; symbols: forward
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +4/-0 (4 lines); hunks: -96,6 +96,7 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; -122,6 +123,9 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _make_moe
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -62,7 +62,9 @@
+    acquire_all_reduce_outputs,
+    can_acquire_all_reduce_outputs,
@@ -499,6 +501,9 @@ class TailPlan:
+        symm_outputs: Producer-direct (routed, shared) views of symmetric
+            memory, or None. When set the producers write into these and the
+            tail reduces the pair in place; ``lane`` is None in that case.
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1873,10 +1873,6 @@ def forward(
-        # Router runs uncontended on main (3us; on aux it starves to 14us
-        # under concurrent GEMMs). When the selected experts need precomputed
-        # TopK runs on the fork branch beside down_proj; routing bypasses it.
-        router_logits = self.gate(hidden_states)
@@ -1892,10 +1888,35 @@ def forward(
-        if plan.lane is not None:
diff -- test/runtime/test_kimi_k3_moe_fork_warmup.py
@@ -96,6 +96,7 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +72/-7; `python/tokenspeed/runtime/models/kimi_k3.py` modified +35/-16
  - tests: `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_k3_moe_tail_equivalence.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1375 - perf(k3): optimize prefill MoE blocks on gfx950

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1375
- 状态/时间: merged / 2026-09-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `f02a74b5f06f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+2440/-754，可读 patch 4662 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +3/-1 (4 lines); hunks: -1471,7 +1471,8 @@ def __init__(; -1606,6 +1607,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_kimi_k3_config.py` modified +1/-0 (1 lines); hunks: -440,6 +440,7 @@ def __init__(self, *args, **kwargs):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +3/-1 (4 lines); hunks: -1471,7 +1471,8 @@ def __init__(; -1606,6 +1607,7 @@ def __init__(; symbols: __init__
  - `test/runtime/test_kimi_k3_config.py` modified +1/-0 (1 lines); hunks: -440,6 +440,7 @@ def __init__(self, *args, **kwargs):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1471,7 +1471,8 @@ def __init__(
-            # Native gfx950 and Hopper Marlin both run A16W4 (bf16 activations).
+            # Native gfx950 accepts bf16 model activations; the selected kernel
+            # may quantize them internally. Hopper Marlin runs A16W4.
@@ -1606,6 +1607,7 @@ def __init__(
+                linear_clamp=self.experts.activation_situ_linear_beta,
diff -- test/runtime/test_kimi_k3_config.py
@@ -440,6 +440,7 @@ def __init__(self, *args, **kwargs):
+                self.activation_situ_linear_beta = kwargs["activation_situ_linear_beta"]
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +3/-1
  - tests: `test/runtime/test_kimi_k3_config.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_config.py`, `tokenspeed-kernel/test/ops/moe/test_gluon_mxfp4_situ_gfx950.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1379 - ci(kimi-k3): use EAGLE3 for AMD eval and perf gates

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1379
- 状态/时间: merged / 2026-09-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；关联提交 `2903e0d53064`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+84/-59，可读 patch 257 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +3/-4 (7 lines); hunks: -1,9 +1,8；`test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +2/-1 (3 lines); hunks: -1,9 +1,10；`test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` modified +1/-1 (2 lines); hunks: -1,5 +1,5。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +3/-4 (7 lines); hunks: -1,9 +1,8
  - `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +2/-1 (3 lines); hunks: -1,9 +1,10
  - `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` modified +1/-1 (2 lines); hunks: -1,5 +1,5
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -1,9 +1,8 @@
-# the speculator-free NVFP4 gate. Speculative flags follow the proven AMD
-# DSpark gate (kimi-k3-dspark-mxfp4-tp8ep8-evalscope-aime26-amd.yaml), except
-# the drafter backend: that gate's `mla` is the portable choice, and CuteDSL
-# is faster where it exists.
+# the speculator-free NVFP4 gate. This NVIDIA job intentionally retains DSpark
+# even though the AMD AIME26 gate now uses EAGLE3; CuteDSL remains the faster
diff -- test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml
@@ -1,9 +1,10 @@
+# Keep the pure target as a manual control; the matching EAGLE3 configuration
+# is the per-commit Kimi-K3 performance gate.
-  - per-commit
diff -- test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml
@@ -1,5 +1,5 @@
-# Keep the pure target as a manual control; the matching DSpark configuration
+# Keep the pure target as a manual control; the matching EAGLE3 configuration
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +3/-4; `test/ci/perf/kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml` modified +2/-1; `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml`, `test/ci/eval/kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1394 - perf(k3): accelerate TP8/EP1 EAGLE3 verification

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1394
- 状态/时间: merged / 2026-09-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`；关联提交 `5e19bea72385`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+21/-13，可读 patch 85 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` renamed +5/-4 (9 lines); hunks: -1,7 +1,8; -30,7 +31,7 @@ server:；`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/moe.py` modified +5/-3 (8 lines); hunks: -1191,9 +1191,11 @@ def _maybe_gluon_package_mxfp4_prefill(; symbols: _maybe_gluon_package_mxfp4_prefill，涉及 `_maybe_gluon_package_mxfp4_prefill`；`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/README.md` modified +4/-0 (4 lines); hunks: -18,6 +18,10 @@ The fused dispatch policy (`fused/moe.py`) is the top of the...。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` renamed +5/-4 (9 lines); hunks: -1,7 +1,8; -30,7 +31,7 @@ server:
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/moe.py` modified +5/-3 (8 lines); hunks: -1191,9 +1191,11 @@ def _maybe_gluon_package_mxfp4_prefill(; symbols: _maybe_gluon_package_mxfp4_prefill
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/README.md` modified +4/-0 (4 lines); hunks: -18,6 +18,10 @@ The fused dispatch policy (`fused/moe.py`) is the top of the...
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml
@@ -1,7 +1,8 @@
-# Kimi-K3 + EAGLE3 is the per-commit AIME26 gate. The matching speculator-free
-# configuration remains available as a manual control for direct comparison.
-name: eval-kimi-k3-eagle3-mxfp4-tp8ep8-aime26-amd
+# Kimi-K3 + EAGLE3 TP8/EP1 is the per-commit AIME26 gate. Its small target-MoE
+# verification batches use the optimized A8W4 SiTU path; the matching
+# speculator-free configuration remains available as a manual control.
diff -- tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/moe.py
@@ -1191,9 +1191,11 @@ def _maybe_gluon_package_mxfp4_prefill(
-        # EP ranks own only a fraction of each token's routes. Combining those
-        # sparse local contributions atomically avoids the full top-k scratch.
-        force_reduce = global_num_experts == n_experts and expert_start == 0
+        # EP ranks own only a fraction of each token's routes. For TP, keep the
+        # faster atomic path within the graph-captured EAGLE3 decode window and
+        # preserve deterministic FP32 reduction for larger batches.
diff -- tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/README.md
@@ -18,6 +18,10 @@ The fused dispatch policy (`fused/moe.py`) is the top of the funnel: it calls
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` renamed +5/-4
  - runtime: `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/moe.py` modified +5/-3; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/README.md` modified +4/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `test/ci_system/test_eval_configs.py`, `tokenspeed-kernel/test/ops/moe/test_gluon_mxfp4_situ_gfx950.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1326 - test: update k3 agentic bench

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1326
- 状态/时间: merged / 2026-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/kimi_k2.5/sglang/collect_outputs.py`, `test/agentic_benchmark/kimi_k2.5/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/kimi_k2.5/trtllm/collect_outputs.py`, `test/agentic_benchmark/kimi_k2.5/vllm/collect_outputs.py`, `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` 等 9 个文件；关联提交 `7b715917c253`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+509/-184，可读 patch 970 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` modified +10/-5 (15 lines); hunks: -8,18 +8,23 @@ exec ts serve \；`test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` modified +10/-5 (15 lines); hunks: -8,18 +8,23 @@ exec ts serve \；`test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm` added +193/-0 (193 lines); hunks: -0,0 +1,193；`test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` modified +9/-38 (47 lines); hunks: -4,23 +4,12 @@ set -euo pipefail; -31,24 +20,10 @@ python3 -c "import evalscope" 2>/dev/null || \。
- 代码 diff 细节:
  - `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` modified +10/-5 (15 lines); hunks: -8,18 +8,23 @@ exec ts serve \
  - `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` modified +10/-5 (15 lines); hunks: -8,18 +8,23 @@ exec ts serve \
  - `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm` added +193/-0 (193 lines); hunks: -0,0 +1,193
  - `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` modified +9/-38 (47 lines); hunks: -4,23 +4,12 @@ set -euo pipefail; -31,24 +20,10 @@ python3 -c "import evalscope" 2>/dev/null || \
  - `test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py` modified +30/-6 (36 lines); hunks: -5,6 +5,7; -25,6 +26,33 @@ def num_gpus_from_config(config: str) -> int:; symbols: num_gpus_from_config, decoded_tok_per_iter, collect
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh
@@ -8,18 +8,23 @@ exec ts serve \
-    --gpu-memory-utilization 0.92 \
+    --gpu-memory-utilization 0.9 \
+    --disable-cuda-graph-padding \
-    --speculative-algorithm DSPARK \
-    --speculative-draft-model-path Inferact/Kimi-K3-DSpark \
-    --speculative-num-draft-tokens 8 \
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh
@@ -8,18 +8,23 @@ exec ts serve \
-    --gpu-memory-utilization 0.92 \
+    --gpu-memory-utilization 0.9 \
+    --disable-cuda-graph-padding \
-    --speculative-algorithm DSPARK \
-    --speculative-draft-model-path Inferact/Kimi-K3-DSpark \
-    --speculative-num-draft-tokens 8 \
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm
@@ -0,0 +1,193 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_ep8.sh` modified +10/-5; `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh` modified +10/-5; `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm` added +193/-0; `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` modified +9/-38; `test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py` modified +30/-6; `test/agentic_benchmark/glm5.2/tokenspeed/collect_outputs.py` modified +29/-1
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/glm5.2/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/glm5.2/trtllm/collect_outputs.py`, `test/agentic_benchmark/inkling/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/kimi_k2.5/sglang/collect_outputs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1383 - perf(k3): shard the latent MoE down projection by column

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1383
- 状态/时间: merged / 2026-09-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py`, `test/runtime/test_kimi_k3_config.py`；关联提交 `4a60ccf97c15`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 25 个文件，+5514/-97，可读 patch 6298 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +86/-10 (96 lines); hunks: -88,6 +88,7; -116,6 +117,7; symbols: _assemble_fp8_fused_qkv_a, _shard_k3_up_projection, _k3_local_moe_blocks, _shard_k3_latent_projection，涉及 `_assemble_fp8_fused_qkv_a, _shard_k3_up_projection, _k3_local_moe_blocks`；`python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +1/-1 (2 lines); hunks: -610,7 +610,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +2/-0 (2 lines); hunks: -149,6 +149,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_kimi_k3_config.py` modified +542/-1 (543 lines); hunks: -6,13 +6,15; -144,7 +146,464 @@ def test_kda_state_shapes_respect_tp(self):; symbols: test_kda_state_shapes_respect_tp, _linear_calls_by_prefix, _model_dtype, KimiK3RegistrationTests，涉及 `test_kda_state_shapes_respect_tp, _linear_calls_by_prefix, _model_dtype`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +86/-10 (96 lines); hunks: -88,6 +88,7; -116,6 +117,7; symbols: _assemble_fp8_fused_qkv_a, _shard_k3_up_projection, _k3_local_moe_blocks, _shard_k3_latent_projection
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +1/-1 (2 lines); hunks: -610,7 +610,7 @@ def __init__(; symbols: __init__
  - `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +2/-0 (2 lines); hunks: -149,6 +149,8 @@ def __init__(; symbols: __init__
  - `test/runtime/test_kimi_k3_config.py` modified +542/-1 (543 lines); hunks: -6,13 +6,15; -144,7 +146,464 @@ def test_kda_state_shapes_respect_tp(self):; symbols: test_kda_state_shapes_respect_tp, _linear_calls_by_prefix, _model_dtype, KimiK3RegistrationTests
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -88,6 +88,7 @@
+from tokenspeed_kernel.ops.moe.latent_down import KimiK3LatentDownOp
@@ -116,6 +117,7 @@
+    DOWN_MAILBOX_MAX_TOKENS,
@@ -779,10 +781,42 @@ def _assemble_fp8_fused_qkv_a(
-def _shard_k3_up_projection(mapping: Mapping, hidden_size: int) -> bool:
-    """Whether to column-shard K3's routed up projection on NVIDIA."""
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -610,7 +610,7 @@ def __init__(
-        # _shard_k3_up_projection held), so comm and module cannot disagree.
+        # _shard_k3_latent_projection held), so comm and module cannot disagree.
diff -- python/tokenspeed/runtime/models/kimi_k3_nextn.py
@@ -149,6 +149,8 @@ def __init__(
+            # One block, re-entered every draft step: nothing rotates.
+            moe_block_count=1,
diff -- test/runtime/test_kimi_k3_config.py
@@ -6,13 +6,15 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +86/-10; `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +1/-1; `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +2/-0
  - tests: `test/runtime/test_kimi_k3_config.py` modified +542/-1
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_comm_ops.py`, `test/runtime/layers/test_latent_down_op.py`, `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_k3_moe_tail_equivalence.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1456 - test: update k3 agentic bench (disagg part 1)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1456
- 状态/时间: merged / 2026-09-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm`, `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/README.md`, `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/collect_outputs.py`, `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/configs/attn_dp16_moe_ep16.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm` 等 8 个文件；关联提交 `de0af866f552`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 12 个文件，+736/-602，可读 patch 1516 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/configs/attn_dp16_moe_ep16.sh` added +36/-0 (36 lines); hunks: -0,0 +1,36；`test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm` added +208/-0 (208 lines); hunks: -0,0 +1,208；`test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm` added +204/-0 (204 lines); hunks: -0,0 +1,204；`test/agentic_benchmark/kimi_k3/tokenspeed_disagg/README.md` added +160/-0 (160 lines); hunks: -0,0 +1,160。
- 代码 diff 细节:
  - `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/configs/attn_dp16_moe_ep16.sh` added +36/-0 (36 lines); hunks: -0,0 +1,36
  - `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm` added +208/-0 (208 lines); hunks: -0,0 +1,208
  - `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm` added +204/-0 (204 lines); hunks: -0,0 +1,204
  - `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/README.md` added +160/-0 (160 lines); hunks: -0,0 +1,160
  - `test/agentic_benchmark/kimi_k3/tokenspeed/pd_sim/README.md` removed +0/-154 (154 lines); hunks: -1,154 +0,0
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/kimi_k3/tokenspeed_disagg/configs/attn_dp16_moe_ep16.sh
@@ -0,0 +1,36 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model nvidia/Kimi-K3-NVFP4 \
+    --data-parallel-size 16 \
+    --ep-size 16 \
diff -- test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm
@@ -0,0 +1,208 @@
+#!/usr/bin/bash
+# Decode-node simulation as a Slurm sweep: dataset prep, the server and
+# pd_client run as job steps (attention-DP 16 spans 4 nodes of 4 GPUs).
+#
+#   sbatch -N 4 --gres=gpu:4 --time=12:00:00 [-A <account> -p <partition>] \
+#       --export=ALL,CONTAINER_IMAGE=<sqsh-or-ref> \
diff -- test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm
@@ -0,0 +1,204 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/configs/attn_dp16_moe_ep16.sh` added +36/-0; `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm` added +208/-0; `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm` added +204/-0; `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/README.md` added +160/-0; `test/agentic_benchmark/kimi_k3/tokenspeed/pd_sim/README.md` removed +0/-154; `test/agentic_benchmark/kimi_k3/tokenspeed_disagg/pd_client.py` renamed +59/-63
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/.gitignore`, `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm`, `test/agentic_benchmark/kimi_k3/tokenspeed/pd_sim/README.md`, `test/agentic_benchmark/kimi_k3/tokenspeed/pd_sim/d_bench.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1458 - perf(kimi3): add gfx1250 large-M WMMA projections

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1458
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；关联提交 `3fd6ade50d14`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+1717/-23，可读 patch 1952 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +383/-17 (400 lines); hunks: -16,8 +16,19; -26,7 +37,6; symbols: use_gluon_largem_gfx1250, _use_gluon_largem, _try_gluon_largem_gfx1250, _kimi3_projection_gemv_kernel，涉及 `use_gluon_largem_gfx1250, _use_gluon_largem, _try_gluon_largem_gfx1250`；`tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +3/-3 (6 lines); hunks: -70,7 +70,7 @@ def test_kimi3_router_projection_auto_splits_on_token_count()...; -134,7 +134,7 @@ def test_kimi3_mla_projection_owns_schedule_selection() -> N...; symbols: test_kimi3_router_projection_auto_splits_on_token_count, solution_for, test_kimi3_mla_projection_owns_schedule_selection, test_kimi3_mla_projection_preserves_non_cdna_prefill_schedule，涉及 `test_kimi3_router_projection_auto_splits_on_token_count, solution_for, test_kimi3_mla_projection_owns_schedule_selection`。
- 代码 diff 细节:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +383/-17 (400 lines); hunks: -16,8 +16,19; -26,7 +37,6; symbols: use_gluon_largem_gfx1250, _use_gluon_largem, _try_gluon_largem_gfx1250, _kimi3_projection_gemv_kernel
  - `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +3/-3 (6 lines); hunks: -70,7 +70,7 @@ def test_kimi3_router_projection_auto_splits_on_token_count()...; -134,7 +134,7 @@ def test_kimi3_mla_projection_owns_schedule_selection() -> N...; symbols: test_kimi3_router_projection_auto_splits_on_token_count, solution_for, test_kimi3_mla_projection_owns_schedule_selection, test_kimi3_mla_projection_preserves_non_cdna_prefill_schedule
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -16,8 +16,19 @@
+from tokenspeed_kernel.ops.gemm.routed_gemv import decode_gemv_routed
+try:
+    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
+        use_gluon_largem_gfx1250,
+    )
+except ImportError:
diff -- tokenspeed-kernel/test/test_kimi_prefill_ops.py
@@ -70,7 +70,7 @@ def test_kimi3_router_projection_auto_splits_on_token_count() -> None:
-    platform = SimpleNamespace(is_cdna4=False, is_hopper_plus=True)
+    platform = SimpleNamespace(is_cdna4=False, is_cdna5=False, is_hopper_plus=True)
@@ -134,7 +134,7 @@ def test_kimi3_mla_projection_owns_schedule_selection() -> None:
-        return_value=SimpleNamespace(is_cdna4=True),
+        return_value=SimpleNamespace(is_cdna4=True, is_cdna5=False),
@@ -152,7 +152,7 @@ def test_kimi3_mla_projection_preserves_non_cdna_prefill_schedule() -> None:
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +383/-17
  - tests: `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_kimi3_projection_gfx1250.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1457 - feat(k3): reserve Iris buffers before KV cache sizing

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1457
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `97541ba647e7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+604/-8，可读 patch 881 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +59/-0 (59 lines); hunks: -65,6 +65,7; -79,6 +80,8; symbols: K3MoETailTier, select_k3_moe_tail_tier, prepare_k3_all_reduce_buffers, K3AttnCommState，涉及 `K3MoETailTier, select_k3_moe_tail_tier, prepare_k3_all_reduce_buffers`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +19/-0 (19 lines); hunks: -147,6 +147,7; -2844,6 +2845,19 @@ class KimiLinearForCausalLM(BaseCausalLM):; symbols: KimiLinearForCausalLM, prepare_communication_runtime, set_eagle3_layers_to_capture, get_input_embeddings，涉及 `KimiLinearForCausalLM, prepare_communication_runtime, set_eagle3_layers_to_capture`；`test/runtime/test_kimi_k3_comm_arming.py` modified +115/-0 (115 lines); hunks: -31,6 +31,9; -49,3 +52,115 @@ def test_arming_requires_fused_moe_ar():; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group，涉及 `test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_handles_distinct_groups`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +59/-0 (59 lines); hunks: -65,6 +65,7; -79,6 +80,8; symbols: K3MoETailTier, select_k3_moe_tail_tier, prepare_k3_all_reduce_buffers, K3AttnCommState
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +19/-0 (19 lines); hunks: -147,6 +147,7; -2844,6 +2845,19 @@ class KimiLinearForCausalLM(BaseCausalLM):; symbols: KimiLinearForCausalLM, prepare_communication_runtime, set_eagle3_layers_to_capture, get_input_embeddings
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +115/-0 (115 lines); hunks: -31,6 +31,9; -49,3 +52,115 @@ def test_arming_requires_fused_moe_ar():; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -65,6 +65,7 @@
+    prepare_all_reduce_buffers,
@@ -79,6 +80,8 @@
+_IRIS_MAX_TOKENS = 8192
@@ -173,6 +176,62 @@ def select_k3_moe_tail_tier(
+def prepare_k3_all_reduce_buffers(
+    *,
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -147,6 +147,7 @@
+    prepare_k3_all_reduce_buffers,
@@ -2844,6 +2845,19 @@ class KimiLinearForCausalLM(BaseCausalLM):
+    def prepare_communication_runtime(self, max_num_tokens: int) -> bool:
+        routed_hidden_size = (
+            self.config.routed_expert_hidden_size
+            if self.config.routed_expert_hidden_size is not None
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -31,6 +31,9 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +59/-0; `python/tokenspeed/runtime/models/kimi_k3.py` modified +19/-0
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` modified +115/-0
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_auto_backend.py`, `test/runtime/test_device_handle.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `tokenspeed-kernel/test/ops/test_communcation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1471 - perf(kimi-k3): extend Iris all reduce window for moe

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1471
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `4ce763928104`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+107/-21，可读 patch 245 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +11/-2 (13 lines); hunks: -81,6 +81,7; -200,13 +201,21 @@ def prepare_k3_all_reduce_buffers(; symbols: K3MoETailTier, prepare_k3_all_reduce_buffers，涉及 `K3MoETailTier, prepare_k3_all_reduce_buffers`；`test/runtime/test_kimi_k3_comm_arming.py` modified +35/-2 (37 lines); hunks: -123,7 +123,7 @@ def test_iris_preparation_handles_distinct_groups(monkeypatch):; -158,7 +158,40 @@ def test_iris_preparation_handles_moe_only_group(monkeypatch):; symbols: test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group, test_iris_preparation_keeps_baseline_window_for_equal_tp4，涉及 `test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group, test_iris_preparation_keeps_baseline_window_for_equal_tp4`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +11/-2 (13 lines); hunks: -81,6 +81,7; -200,13 +201,21 @@ def prepare_k3_all_reduce_buffers(; symbols: K3MoETailTier, prepare_k3_all_reduce_buffers
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +35/-2 (37 lines); hunks: -123,7 +123,7 @@ def test_iris_preparation_handles_distinct_groups(monkeypatch):; -158,7 +158,40 @@ def test_iris_preparation_handles_moe_only_group(monkeypatch):; symbols: test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group, test_iris_preparation_keeps_baseline_window_for_equal_tp4
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -81,6 +81,7 @@
+_IRIS_BASELINE_PRODUCER_DIRECT_MAX_TOKENS = 48
@@ -200,13 +201,21 @@ def prepare_k3_all_reduce_buffers(
+    expand_moe_window = (
+        groups_are_equal and mapping.attn.tp_size == 8 and mapping.moe.tp_ep_size == 8
+    )
+    producer_direct_max_tokens = (
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -123,7 +123,7 @@ def test_iris_preparation_handles_distinct_groups(monkeypatch):
-            producer_direct_max_numel=8192 * (7168 + 3584),
+            producer_direct_max_numel=48 * (7168 + 3584),
@@ -158,7 +158,40 @@ def test_iris_preparation_handles_moe_only_group(monkeypatch):
-        producer_direct_max_numel=8192 * (7168 + 3584),
+        producer_direct_max_numel=48 * (7168 + 3584),
+        attnres_max_numel=0,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +11/-2
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` modified +35/-2
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_auto_backend.py`, `test/runtime/test_kimi_k3_comm_arming.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1489 - perf(k3): serve the attention reduce from the tokenspeed collective

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1489
- 状态/时间: merged / 2026-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `92b062a7332b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+975/-71，可读 patch 1331 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +91/-2 (93 lines); hunks: -34,6 +34,8; -57,7 +59,10; symbols: attn_ar_eligible, K3MoETailTier, __init__, K3MoeTailCommState，涉及 `attn_ar_eligible, K3MoETailTier, __init__`；`test/runtime/test_kimi_k3_comm_arming.py` modified +180/-1 (181 lines); hunks: -30,12 +30,30; -54,6 +72,7 @@ def test_arming_requires_fused_moe_ar():; symbols: test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_handles_distinct_groups，涉及 `test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups`；`test/runtime/test_kimi_k3_attn_res.py` modified +2/-0 (2 lines); hunks: -84,6 +84,7 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):; -135,6 +136,7 @@ def test_unsupported_iris_reduce_defers_attnres_combine(self):; symbols: test_batched_iris_reduce_consumes_attnres_combine, test_unsupported_iris_reduce_defers_attnres_combine，涉及 `test_batched_iris_reduce_consumes_attnres_combine, test_unsupported_iris_reduce_defers_attnres_combine`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +91/-2 (93 lines); hunks: -34,6 +34,8; -57,7 +59,10; symbols: attn_ar_eligible, K3MoETailTier, __init__, K3MoeTailCommState
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +180/-1 (181 lines); hunks: -30,12 +30,30; -54,6 +72,7 @@ def test_arming_requires_fused_moe_ar():; symbols: test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_handles_distinct_groups
  - `test/runtime/test_kimi_k3_attn_res.py` modified +2/-0 (2 lines); hunks: -84,6 +84,7 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):; -135,6 +136,7 @@ def test_unsupported_iris_reduce_defers_attnres_combine(self):; symbols: test_batched_iris_reduce_consumes_attnres_combine, test_unsupported_iris_reduce_defers_attnres_combine
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -34,6 +34,8 @@
+attention reduce        ``1 <= M <= ATTN_AR_MAX_TOKENS`` (tokenspeed
+                        CuteDSL collective, attn TP group)
@@ -57,7 +59,10 @@
+    attn_reduce_shape_supported,
+    build_attn_reduce_collective,
+    multicast_backend_available,
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -30,12 +30,30 @@
+import os
+import sys
+from importlib.util import find_spec
+import pytest
-from tokenspeed.runtime.models.kimi_k3_comm import _tail_finalize_top_k
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
diff -- test/runtime/test_kimi_k3_attn_res.py
@@ -84,6 +84,7 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +91/-2
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` modified +180/-1; `test/runtime/test_kimi_k3_attn_res.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `tokenspeed-kernel/test/nvidia/ops/moe/test_latent_tail_attn_contract.py`, `tokenspeed-kernel/test/nvidia/ops/moe/test_latent_tail_distributed.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1539 - fix(kimi-k3): restore EAGLE3 prefill numerics

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1539
- 状态/时间: merged / 2026-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `97090e201bfa`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+16/-17，可读 patch 110 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-11 (20 lines); hunks: -86,7 +86,7; -221,21 +221,20 @@ def prepare_k3_all_reduce_buffers(; symbols: prepare_k3_all_reduce_buffers，涉及 `prepare_k3_all_reduce_buffers`；`test/runtime/test_kimi_k3_comm_arming.py` modified +5/-5 (10 lines); hunks: -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():; -98,7 +98,7 @@ def test_iris_preparation_deduplicates_equal_groups(monkeypatch):; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_caps_producer_direct_for_equal_groups, test_iris_preparation_handles_distinct_groups，涉及 `test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_caps_producer_direct_for_equal_groups`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-11 (20 lines); hunks: -86,7 +86,7; -221,21 +221,20 @@ def prepare_k3_all_reduce_buffers(; symbols: prepare_k3_all_reduce_buffers
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +5/-5 (10 lines); hunks: -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():; -98,7 +98,7 @@ def test_iris_preparation_deduplicates_equal_groups(monkeypatch):; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_deduplicates_equal_groups, test_iris_preparation_caps_producer_direct_for_equal_groups, test_iris_preparation_handles_distinct_groups
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -86,7 +86,7 @@
-_IRIS_BASELINE_PRODUCER_DIRECT_MAX_TOKENS = 48
+_IRIS_PRODUCER_DIRECT_MAX_BYTES = 1024 * 1024
@@ -221,21 +221,20 @@ def prepare_k3_all_reduce_buffers(
-    expand_moe_window = (
-        groups_are_equal and mapping.attn.tp_size == 8 and mapping.moe.tp_ep_size == 8
-    )
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():
-def test_iris_preparation_deduplicates_equal_groups(monkeypatch):
+def test_iris_preparation_caps_producer_direct_for_equal_groups(monkeypatch):
@@ -98,7 +98,7 @@ def test_iris_preparation_deduplicates_equal_groups(monkeypatch):
-        producer_direct_max_numel=8192 * (7168 + 3584),
+        producer_direct_max_numel=512 * 1024,
@@ -143,7 +143,7 @@ def test_iris_preparation_handles_distinct_groups(monkeypatch):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-11
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` modified +5/-5
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml`, `test/ci_system/test_eval_configs.py`, `test/runtime/test_kimi_k3_comm_arming.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1550 - revert(kimi-k3): restore TP8 producer-direct window

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1550
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `87ff3db8781d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+19/-14，可读 patch 96 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +14/-9 (23 lines); hunks: -86,7 +86,7; -221,20 +221,24 @@ def prepare_k3_all_reduce_buffers(; symbols: prepare_k3_all_reduce_buffers，涉及 `prepare_k3_all_reduce_buffers`；`test/runtime/test_kimi_k3_comm_arming.py` modified +5/-5 (10 lines); hunks: -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():; -98,7 +98,7 @@ def test_iris_preparation_caps_producer_direct_for_equal_group...; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_caps_producer_direct_for_equal_groups, test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups，涉及 `test_arming_requires_fused_moe_ar, test_iris_preparation_caps_producer_direct_for_equal_groups, test_iris_preparation_uses_full_window_for_equal_tp8_groups`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +14/-9 (23 lines); hunks: -86,7 +86,7; -221,20 +221,24 @@ def prepare_k3_all_reduce_buffers(; symbols: prepare_k3_all_reduce_buffers
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +5/-5 (10 lines); hunks: -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():; -98,7 +98,7 @@ def test_iris_preparation_caps_producer_direct_for_equal_group...; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_caps_producer_direct_for_equal_groups, test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -86,7 +86,7 @@
-_IRIS_PRODUCER_DIRECT_MAX_BYTES = 1024 * 1024
+_IRIS_BASELINE_PRODUCER_DIRECT_MAX_TOKENS = 48
@@ -221,20 +221,24 @@ def prepare_k3_all_reduce_buffers(
-    # Iris and RCCL sum in different orders. Using Iris for K3 prefill changed
-    # the greedy EAGLE3 trajectory enough to reduce acceptance and end-to-end
-    # output throughput, despite making this collective faster in isolation.
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():
-def test_iris_preparation_caps_producer_direct_for_equal_groups(monkeypatch):
+def test_iris_preparation_uses_full_window_for_equal_tp8_groups(monkeypatch):
@@ -98,7 +98,7 @@ def test_iris_preparation_caps_producer_direct_for_equal_groups(monkeypatch):
-        producer_direct_max_numel=512 * 1024,
+        producer_direct_max_numel=8192 * (7168 + 3584),
@@ -143,7 +143,7 @@ def test_iris_preparation_handles_distinct_groups(monkeypatch):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +14/-9
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` modified +5/-5
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_comm_arming.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1563 - feat(kimi-k3): support MoE all-to-all for attention DP

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1563
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；关联提交 `5ee35faa6f9b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 31 个文件，+1646/-441，可读 patch 2952 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +241/-119 (360 lines); hunks: -60,6 +60,7; -68,6 +69,7; symbols: __init__, _project_q_latent_gated, _k3_local_moe_blocks, _shard_k3_latent_projection，涉及 `__init__, _project_q_latent_gated, _k3_local_moe_blocks`；`test/runtime/test_kimi_k3_moe_attn_dp.py` added +487/-0 (487 lines); hunks: -0,0 +1,487; symbols: test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts, __init__，涉及 `test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts`；`test/runtime/test_kimi_k3_config.py` modified +7/-48 (55 lines); hunks: -257,9 +257,6 @@ def __init__(self, **kwargs):; -574,7 +571,10 @@ def test_shard_predicate_requires_a_divisible_multi_rank_nv...; symbols: __init__, test_shard_predicate_requires_a_divisible_multi_rank_nvidia_group, mapping_for, test_hybrid_moe_precomputes_routing_only_for_decode，涉及 `__init__, test_shard_predicate_requires_a_divisible_multi_rank_nvidia_group, mapping_for`；`test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +1/-1 (2 lines); hunks: -110,7 +110,7 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _make_moe，涉及 `_make_moe`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +241/-119 (360 lines); hunks: -60,6 +60,7; -68,6 +69,7; symbols: __init__, _project_q_latent_gated, _k3_local_moe_blocks, _shard_k3_latent_projection
  - `test/runtime/test_kimi_k3_moe_attn_dp.py` added +487/-0 (487 lines); hunks: -0,0 +1,487; symbols: test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts, __init__
  - `test/runtime/test_kimi_k3_config.py` modified +7/-48 (55 lines); hunks: -257,9 +257,6 @@ def __init__(self, **kwargs):; -574,7 +571,10 @@ def test_shard_predicate_requires_a_divisible_multi_rank_nv...; symbols: __init__, test_shard_predicate_requires_a_divisible_multi_rank_nvidia_group, mapping_for, test_hybrid_moe_precomputes_routing_only_for_decode
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +1/-1 (2 lines); hunks: -110,7 +110,7 @@ def _make_moe(fork: _SpyFork) -> SimpleNamespace:; symbols: _make_moe
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -60,6 +60,7 @@
+import torch.nn.functional as F
@@ -68,6 +69,7 @@
+from tokenspeed_kernel.ops.communication.flashinfer import get_flashinfer_moe_alltoall
@@ -84,10 +86,8 @@
-from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
-    situ_moe_unavailable_reason,
diff -- test/runtime/test_kimi_k3_moe_attn_dp.py
@@ -0,0 +1,487 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- test/runtime/test_kimi_k3_config.py
@@ -257,9 +257,6 @@ def __init__(self, **kwargs):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +241/-119
  - tests: `test/runtime/test_kimi_k3_moe_attn_dp.py` added +487/-0; `test/runtime/test_kimi_k3_config.py` modified +7/-48; `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/runtime/execution/test_breakable_cuda_graph.py`, `test/runtime/layers/test_latent_moe.py`, `test/runtime/layers/test_moe_expert.py`, `test/runtime/layers/test_moe_loader_ep.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1541 - perf(k3): implement iris barrier free lamport all reduce for small M

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1541
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `b8d84fcf1c7c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+859/-9，可读 patch 1367 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-0 (9 lines); hunks: -221,6 +221,13 @@ def prepare_k3_all_reduce_buffers(; -244,6 +251,7 @@ def prepare_k3_all_reduce_buffers(; symbols: prepare_k3_all_reduce_buffers，涉及 `prepare_k3_all_reduce_buffers`；`test/runtime/test_kimi_k3_comm_arming.py` modified +59/-1 (60 lines); hunks: -79,7 +79,7 @@ def test_iris_preparation_uses_full_window_for_equal_tp8_group...; -101,6 +101,7 @@ def test_iris_preparation_uses_full_window_for_equal_tp8_gro...; symbols: test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group，涉及 `test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-0 (9 lines); hunks: -221,6 +221,13 @@ def prepare_k3_all_reduce_buffers(; -244,6 +251,7 @@ def prepare_k3_all_reduce_buffers(; symbols: prepare_k3_all_reduce_buffers
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +59/-1 (60 lines); hunks: -79,7 +79,7 @@ def test_iris_preparation_uses_full_window_for_equal_tp8_group...; -101,6 +101,7 @@ def test_iris_preparation_uses_full_window_for_equal_tp8_gro...; symbols: test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups, test_iris_preparation_handles_moe_only_group
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -221,6 +221,13 @@ def prepare_k3_all_reduce_buffers(
+    # The Lamport crossover was measured with attention TP8 and MoE TP8.
+    enable_lamport = (
+        groups_are_equal
+        and mapping.attn.tp_size == 8
+        and mapping.moe.tp_size == 8
+        and mapping.moe.ep_size == 1
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -79,7 +79,7 @@ def test_iris_preparation_uses_full_window_for_equal_tp8_groups(monkeypatch):
-        moe=SimpleNamespace(tp_ep_size=8, tp_ep_group=group),
+        moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_size=8, tp_ep_group=group),
@@ -101,6 +101,7 @@ def test_iris_preparation_uses_full_window_for_equal_tp8_groups(monkeypatch):
+        enable_lamport=True,
@@ -137,6 +138,7 @@ def test_iris_preparation_handles_distinct_groups(monkeypatch):
+            enable_lamport=False,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-0
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` modified +59/-1
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_auto_backend.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `tokenspeed-kernel/test/amd/ops/test_iris_communication.py`, `tokenspeed-kernel/test/amd/ops/test_iris_lamport.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1545 - perf(kimi-k3): select Iris fused push AR+attnres+rmsnorm through M=16

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1545
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `102d4c14d1f8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+295/-29，可读 patch 440 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +37/-18 (55 lines); hunks: -534,6 +534,34 @@ def __init__(self, state: K3AttnCommState) -> None:; -621,16 +649,18 @@ def attn_reduce(; symbols: __init__, fused_attnres_reduce_available, attn_reduce，涉及 `__init__, fused_attnres_reduce_available, attn_reduce`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +28/-5 (33 lines); hunks: -2393,6 +2393,29 @@ def _fused_attnres_graph_available(; -2591,11 +2614,11 @@ def forward(; symbols: _fused_attnres_graph_available, forward，涉及 `_fused_attnres_graph_available, forward`；`test/runtime/test_kimi_k3_attn_res.py` modified +218/-3 (221 lines); hunks: -29,6 +29,7; -128,7 +129,7 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):; symbols: test_batched_iris_reduce_consumes_attnres_combine, test_unsupported_iris_reduce_defers_attnres_combine, test_fused_attention_window_defers_attnres_combine，涉及 `test_batched_iris_reduce_consumes_attnres_combine, test_unsupported_iris_reduce_defers_attnres_combine, test_fused_attention_window_defers_attnres_combine`；`test/runtime/test_kimi_k3_comm_arming.py` modified +1/-1 (2 lines); hunks: -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_caps_attnres_for_equal_tp8_groups，涉及 `test_arming_requires_fused_moe_ar, test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_caps_attnres_for_equal_tp8_groups`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +37/-18 (55 lines); hunks: -534,6 +534,34 @@ def __init__(self, state: K3AttnCommState) -> None:; -621,16 +649,18 @@ def attn_reduce(; symbols: __init__, fused_attnres_reduce_available, attn_reduce
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +28/-5 (33 lines); hunks: -2393,6 +2393,29 @@ def _fused_attnres_graph_available(; -2591,11 +2614,11 @@ def forward(; symbols: _fused_attnres_graph_available, forward
  - `test/runtime/test_kimi_k3_attn_res.py` modified +218/-3 (221 lines); hunks: -29,6 +29,7; -128,7 +129,7 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):; symbols: test_batched_iris_reduce_consumes_attnres_combine, test_unsupported_iris_reduce_defers_attnres_combine, test_fused_attention_window_defers_attnres_combine
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +1/-1 (2 lines); hunks: -73,7 +73,7 @@ def test_arming_requires_fused_moe_ar():; symbols: test_arming_requires_fused_moe_ar, test_iris_preparation_uses_full_window_for_equal_tp8_groups, test_iris_preparation_caps_attnres_for_equal_tp8_groups
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -534,6 +534,34 @@ def __init__(self, state: K3AttnCommState) -> None:
+    def fused_attnres_reduce_available(
+        self,
+        partial: torch.Tensor,
+        residual: torch.Tensor,
+        combine: tuple,
+        score_weight: torch.Tensor | None,
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -2393,6 +2393,29 @@ def _fused_attnres_graph_available(
+        num_tokens = hidden_states.shape[0]
+        if (
+            not self.is_block_write_layer
+            and hidden_states.is_cuda
+            and self.prev_valid_blocks > 0
+            and self._mlp_wp is not None
diff -- test/runtime/test_kimi_k3_attn_res.py
@@ -29,6 +29,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +37/-18; `python/tokenspeed/runtime/models/kimi_k3.py` modified +28/-5
  - tests: `test/runtime/test_kimi_k3_attn_res.py` modified +218/-3; `test/runtime/test_kimi_k3_comm_arming.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_auto_backend.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1635 - feat(kimi-k3): add NVFP4 MegaMoE and opt-in autotuning

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1635
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/runtime/test_kimi_k3_config.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py`；关联提交 `b41ea7d762ac`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 47 个文件，+18232/-98，可读 patch 13184 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +33/-12 (45 lines); hunks: -1426,6 +1426,7 @@ def __init__(; -1437,6 +1438,18 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_kimi_k3_moe_attn_dp.py` modified +79/-7 (86 lines); hunks: -61,8 +61,9 @@ def test_attn_dp_rejects_partial_world_layout_before_backend_s...; -73,11 +74,14 @@ def __init__(self, **kwargs):; symbols: test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts, __init__，涉及 `test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts`；`test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml` added +72/-0 (72 lines); hunks: -0,0 +1,72；`test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +16/-7 (23 lines); hunks: -24,15 +24,24 @@ server:; -53,7 +62,7 @@ eval:。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +33/-12 (45 lines); hunks: -1426,6 +1426,7 @@ def __init__(; -1437,6 +1438,18 @@ def __init__(; symbols: __init__
  - `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +79/-7 (86 lines); hunks: -61,8 +61,9 @@ def test_attn_dp_rejects_partial_world_layout_before_backend_s...; -73,11 +74,14 @@ def __init__(self, **kwargs):; symbols: test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts, __init__
  - `test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml` added +72/-0 (72 lines); hunks: -0,0 +1,72
  - `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +16/-7 (23 lines); hunks: -24,15 +24,24 @@ server:; -53,7 +62,7 @@ eval:
  - `test/runtime/test_kimi_k3_config.py` modified +8/-0 (8 lines); hunks: -305,6 +305,7 @@ def test_the_packed_input_projection_declines_a_narrowed_wei...; -328,6 +329,7 @@ def test_the_shard_is_wired_on_the_plan_that_absorbs_it(self):; symbols: test_the_packed_input_projection_declines_a_narrowed_weight, test_the_shard_is_wired_on_the_plan_that_absorbs_it, test_the_column_group_needs_a_divisible_latent, test_the_shard_stays_off_the_plan_that_cannot_absorb_it
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1426,6 +1426,7 @@ def __init__(
+        moe_backend = get_moe_backend()
@@ -1437,6 +1438,18 @@ def __init__(
+        self.execution_plan = Kimi3MoEExecutionPlan.build(
+            mapping,
+            moe_backend,
+            alt_stream,
diff -- test/runtime/test_kimi_k3_moe_attn_dp.py
@@ -61,8 +61,9 @@ def test_attn_dp_rejects_partial_world_layout_before_backend_setup(
+@pytest.mark.parametrize("mega_moe", [False, True])
-    monkeypatch, backend: str, fabric_available: bool
+    monkeypatch, backend: str, fabric_available: bool, mega_moe: bool
@@ -73,11 +74,14 @@ def __init__(self, **kwargs):
-        kimi_k3, "get_moe_backend", lambda: SimpleNamespace(value="flashinfer_trtllm")
+        kimi_k3,
diff -- test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml
@@ -0,0 +1,72 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +33/-12
  - tests: `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +79/-7; `test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml` added +72/-0; `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +16/-7; `test/runtime/test_kimi_k3_config.py` modified +8/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci_system/test_dispatch_workflows.py`, `test/ci_system/test_eval_configs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1634 - feat(kimi-k3): support Hopper PD with DeepEP and DSpark

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1634
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/guides/kimi-k3-hopper-pd.md`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_deepep.py`, `python/tokenspeed/runtime/models/kimi_k3_dspark.py` 等 12 个文件；关联提交 `f5b45743f06c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 89 个文件，+4254/-712，可读 patch 7073 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_deepep.py` added +262/-0 (262 lines); hunks: -0,0 +1,262; symbols: KimiLinearMoEDeepEP, __init__, pack_input_projection_weights, forward，涉及 `KimiLinearMoEDeepEP, __init__, pack_input_projection_weights`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +152/-52 (204 lines); hunks: -227,35 +227,24 @@ class KimiLinearMLP(nn.Module):; -1428,13 +1417,10 @@ def __init__(; symbols: KimiLinearMLP, __init__, _reduce_shared，涉及 `KimiLinearMLP, __init__, _reduce_shared`；`python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +169/-21 (190 lines); hunks: -30,8 +30,9; -50,25 +51,44; symbols: _context_tap_owner_layer, K3DSparkAttention, _norm_with_allreduce, K3DSparkModel，涉及 `_context_tap_owner_layer, K3DSparkAttention, _norm_with_allreduce`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +6/-1 (7 lines); hunks: -319,7 +319,12 @@ def replay_kda(self) -> bool:; symbols: replay_kda, workspace_bytes，涉及 `replay_kda, workspace_bytes`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_deepep.py` added +262/-0 (262 lines); hunks: -0,0 +1,262; symbols: KimiLinearMoEDeepEP, __init__, pack_input_projection_weights, forward
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +152/-52 (204 lines); hunks: -227,35 +227,24 @@ class KimiLinearMLP(nn.Module):; -1428,13 +1417,10 @@ def __init__(; symbols: KimiLinearMLP, __init__, _reduce_shared
  - `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +169/-21 (190 lines); hunks: -30,8 +30,9; -50,25 +51,44; symbols: _context_tap_owner_layer, K3DSparkAttention, _norm_with_allreduce, K3DSparkModel
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +6/-1 (7 lines); hunks: -319,7 +319,12 @@ def replay_kda(self) -> bool:; symbols: replay_kda, workspace_bytes
  - `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +2/-2 (4 lines); hunks: -54,7 +54,7; -144,7 +144,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_deepep.py
@@ -0,0 +1,262 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -227,35 +227,24 @@ class KimiLinearMLP(nn.Module):
+    Unsharded callers use tp_size=1 and tp_group=None.
-        mapping: Mapping,
-        quant_config: QuantizationConfig | None = None,
-        prefix: str = "",
-        reduce_results: bool = True,
-        is_shared_expert: bool = False,
diff -- python/tokenspeed/runtime/models/kimi_k3_dspark.py
@@ -30,8 +30,9 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_deepep.py` added +262/-0; `python/tokenspeed/runtime/models/kimi_k3.py` modified +152/-52; `python/tokenspeed/runtime/models/kimi_k3_dspark.py` modified +169/-21; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/kimi_k3.py` modified +6/-1; `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +2/-2
  - docs: `docs/guides/kimi-k3-hopper-pd.md` added +321/-0
  - tests: `test/runtime/test_kimi_k3_deepep.py` added +173/-0; `test/runtime/test_kimi_k3_dspark_capture.py` modified +102/-0
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_cache_pd_executors.py`, `test/runtime/distributed/test_cache_pd_layerwise.py`, `test/runtime/distributed/test_cache_pd_manifest.py`, `test/runtime/distributed/test_pd_transfer_plan.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1638 - feat(kimi-k3): fuse sharded latent down proj with NVFP4 quantize

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1638
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_config.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；关联提交 `ffa16b13fae8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+802/-10，可读 patch 929 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +12/-0 (12 lines); hunks: -1830,6 +1830,18 @@ def _latent_input_projections(; symbols: _latent_input_projections, process_weights_after_loading, _routed_experts，涉及 `_latent_input_projections, process_weights_after_loading, _routed_experts`；`test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +61/-0 (61 lines); hunks: -184,5 +184,66 @@ def test_eager_serving_leaves_the_fork_disabled():; symbols: test_eager_serving_leaves_the_fork_disabled, test_moe_passes_projection_payload_to_experts, test_nvfp4_projection_setup_uses_processed_expert_scale, process_weights，涉及 `test_eager_serving_leaves_the_fork_disabled, test_moe_passes_projection_payload_to_experts, test_nvfp4_projection_setup_uses_processed_expert_scale`；`test/runtime/test_kimi_k3_config.py` modified +2/-2 (4 lines); hunks: -199,7 +199,7 @@ def __init__(self, *args, **kwargs):; -908,7 +908,7 @@ def __init__(self, *args, **kwargs):; symbols: __init__, FakeSharedExperts，涉及 `__init__, FakeSharedExperts`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +12/-0 (12 lines); hunks: -1830,6 +1830,18 @@ def _latent_input_projections(; symbols: _latent_input_projections, process_weights_after_loading, _routed_experts
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +61/-0 (61 lines); hunks: -184,5 +184,66 @@ def test_eager_serving_leaves_the_fork_disabled():; symbols: test_eager_serving_leaves_the_fork_disabled, test_moe_passes_projection_payload_to_experts, test_nvfp4_projection_setup_uses_processed_expert_scale, process_weights
  - `test/runtime/test_kimi_k3_config.py` modified +2/-2 (4 lines); hunks: -199,7 +199,7 @@ def __init__(self, *args, **kwargs):; -908,7 +908,7 @@ def __init__(self, *args, **kwargs):; symbols: __init__, FakeSharedExperts
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1830,6 +1830,18 @@ def _latent_input_projections(
+    def process_weights_after_loading(self, module) -> None:
+        """Configure the latent projection from the processed expert input scale."""
+        if (
+            self.experts.plan["weight_dtype"] == "nvfp4"
+            and self.experts.plan["solution"] == "flashinfer_trtllm"
+        ):
diff -- test/runtime/test_kimi_k3_moe_fork_warmup.py
@@ -184,5 +184,66 @@ def test_eager_serving_leaves_the_fork_disabled():
+@pytest.mark.parametrize("prequantized", [False, True])
+def test_moe_passes_projection_payload_to_experts(prequantized):
+    hidden = torch.zeros(8, 4)
+    payload = (
+        (torch.empty(8, 2, dtype=torch.uint8), torch.empty(8, 1))
+        if prequantized
diff -- test/runtime/test_kimi_k3_config.py
@@ -199,7 +199,7 @@ def __init__(self, *args, **kwargs):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +12/-0
  - tests: `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +61/-0; `test/runtime/test_kimi_k3_config.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_down_op.py`, `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_kimi_k3_config.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1684 - fix(ci): extend Kimi EAGLE3 AIME25 token budget

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1684
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml`；关联提交 `3456ff9f370d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+15/-3，可读 patch 52 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` modified +4/-2 (6 lines); hunks: -15,12 +15,13 @@ install:; -44,6 +45,7 @@ eval:。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` modified +4/-2 (6 lines); hunks: -15,12 +15,13 @@ install:; -44,6 +45,7 @@ eval:
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml
@@ -15,12 +15,13 @@ install:
+  # Leave 6 positions below the model's 262144 limit for EAGLE3 overlap overshoot.
-    --max-model-len 80000
+    --max-model-len 262138
@@ -44,6 +45,7 @@ eval:
+    --work-dir .ci-artifacts/published/evalscope-results
@@ -52,7 +54,7 @@ eval:
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml` modified +4/-2
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/kimi-k2.5-nvfp4-eagle3-evalscope-aime25.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1726 - ci: move GB300 Kimi K3 TP8 evaluations to nightly

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1726
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`；关联提交 `ae04eca27d85`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+2/-2，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-tp8-two-node-aime26-gb300-slurm；`test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -8,7 +8,7 @@ name: eval-kimi-k3-nvfp4-tp8-two-node-aime26-gb300-slurm。
- 代码 diff 细节:
  - `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-tp8-two-node-aime26-gb300-slurm
  - `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1 (2 lines); hunks: -8,7 +8,7 @@ name: eval-kimi-k3-nvfp4-tp8-two-node-aime26-gb300-slurm
- 关键代码摘录:

```diff
diff -- test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -3,7 +3,7 @@ name: eval-kimi-k3-mxfp4-tp8-two-node-aime26-gb300-slurm
-  - per-commit
+  - nightly
diff -- test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml
@@ -8,7 +8,7 @@ name: eval-kimi-k3-nvfp4-tp8-two-node-aime26-gb300-slurm
-  - per-commit
+  - nightly
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1; `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`, `test/ci/eval/kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1740 - ci: benchmark Kimi-K3 EAGLE3 with TP8 EP1 at 50K/500 C16

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1740
- 状态/时间: merged / 2026-09-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml`；关联提交 `fa9b11054dd5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+68/-42，可读 patch 200 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` renamed +36/-28 (64 lines); hunks: -1,8 +1,9; -16,8 +17,8 @@ env:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` renamed +36/-28 (64 lines); hunks: -1,8 +1,9; -16,8 +17,8 @@ env:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml
@@ -1,8 +1,9 @@
-# Kimi-K3 + EAGLE3 speculative decoding. Deliberately a separate job from
-# perf-kimi-k3-mxfp4-tp8ep8-random-4k-1k-mi35x, which tracks pure-target K3
-# performance and must stay speculator-free (Issue #879).
-name: perf-kimi-k3-eagle3-mxfp4-tp8ep8-random-4k-1k-mi35x
+# Kimi-K3 + EAGLE3 long-context throughput at the production-shaped batch.
+# TP8/EP1 reduces prefill latency enough to improve end-to-end latency at
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` renamed +36/-28
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml`, `test/ci_system/test_eval_configs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1725 - ci(amd-kernel): Add Kimi K3 MoE Kernel Benchmarks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1725
- 状态/时间: merged / 2026-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/moe.json`；关联提交 `bf9c64054c8c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+1480/-49，可读 patch 1701 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/moe.json` added +487/-0 (487 lines); hunks: -0,0 +1,487。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/moe.json` added +487/-0 (487 lines); hunks: -0,0 +1,487
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/moe.json
@@ -0,0 +1,487 @@
+{
+  "schema_version": 1,
+  "common_parameters": {
+    "model_profile": "kimi_k3_tp8",
+    "num_experts": 896
+  },
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/moe.json` added +487/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_harness.py`, `tokenspeed-kernel/test/test_benchmark_moe_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1807 - ci(amd-kernel): Add Kimi K3 MLA Kernel Benchmarks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1807
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json`；关联提交 `8bb4a47a2d66`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+1396/-1，可读 patch 1421 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` added +306/-0 (306 lines); hunks: -0,0 +1,306。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` added +306/-0 (306 lines); hunks: -0,0 +1,306
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json
@@ -0,0 +1,306 @@
+{
+  "schema_version": 1,
+  "common_parameters": {
+    "model_profile": "kimi_k3_tp8",
+    "local_heads": 12,
+    "q_lora_rank": 1536,
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` added +306/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_mla_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1795 - perf(kimi3): split-K gfx1250 decode GEMMs and widen AttnRes

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1795
- 状态/时间: merged / 2026-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py`；关联提交 `c7d10753490b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+493/-60，可读 patch 727 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py` modified +60/-1 (61 lines); hunks: -24,7 +24,7; -537,3 +537,62 @@ def test_kimi_topk_prefill_ties_choose_smaller_expert_id()...; symbols: test_kimi_topk_prefill_ties_choose_smaller_expert_id, test_attn_res_warmed_launch_variants_reuse_compilation, project，涉及 `test_kimi_topk_prefill_ties_choose_smaller_expert_id, test_attn_res_warmed_launch_variants_reuse_compilation, project`；`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/gemm/fp16/mm.py` modified +233/-38 (271 lines); hunks: -52,7 +52,26 @@ def use_gluon_largem_gfx1250(m: int, k: int, n: int) -> bool:; -63,24 +82,28 @@ def _wmma_tdm_dense_m16_kernel(; symbols: use_gluon_largem_gfx1250, _wmma_tdm_dense_m16_launch_metadata, _wmma_tdm_dense_m16_kernel，涉及 `use_gluon_largem_gfx1250, _wmma_tdm_dense_m16_launch_metadata, _wmma_tdm_dense_m16_kernel`；`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/README.md` modified +43/-0 (43 lines); hunks: -138,6 +138,49 @@ one-wave direct path when extra partitions or fusion do not...；`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/kda/attn_res.py` modified +23/-15 (38 lines); hunks: -225,8 +225,29 @@ def attn_res_rmsnorm_gfx1250(; -235,20 +256,7 @@ def attn_res_rmsnorm_gfx1250(; symbols: attn_res_rmsnorm_gfx1250，涉及 `attn_res_rmsnorm_gfx1250`。
- 代码 diff 细节:
  - `tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py` modified +60/-1 (61 lines); hunks: -24,7 +24,7; -537,3 +537,62 @@ def test_kimi_topk_prefill_ties_choose_smaller_expert_id()...; symbols: test_kimi_topk_prefill_ties_choose_smaller_expert_id, test_attn_res_warmed_launch_variants_reuse_compilation, project
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/gemm/fp16/mm.py` modified +233/-38 (271 lines); hunks: -52,7 +52,26 @@ def use_gluon_largem_gfx1250(m: int, k: int, n: int) -> bool:; -63,24 +82,28 @@ def _wmma_tdm_dense_m16_kernel(; symbols: use_gluon_largem_gfx1250, _wmma_tdm_dense_m16_launch_metadata, _wmma_tdm_dense_m16_kernel
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/README.md` modified +43/-0 (43 lines); hunks: -138,6 +138,49 @@ one-wave direct path when extra partitions or fusion do not...
  - `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/kda/attn_res.py` modified +23/-15 (38 lines); hunks: -225,8 +225,29 @@ def attn_res_rmsnorm_gfx1250(; -235,20 +256,7 @@ def attn_res_rmsnorm_gfx1250(; symbols: attn_res_rmsnorm_gfx1250
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/__init__.py` modified +8/-2 (10 lines); hunks: -497,9 +497,15 @@ def select_for_layout(; symbols: select_for_layout
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py
@@ -24,7 +24,7 @@
-from utils import is_cdna4, is_cdna5
+from utils import assert_no_triton_compile, is_cdna4, is_cdna5
@@ -537,3 +537,62 @@ def test_kimi_topk_prefill_ties_choose_smaller_expert_id() -> None:
+@pytest.mark.skipif(not is_cdna5(), reason="gfx1250 launch variants")
+def test_attn_res_warmed_launch_variants_reuse_compilation():
+    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.attn_res import (
diff -- tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/gemm/fp16/mm.py
@@ -52,7 +52,26 @@ def use_gluon_largem_gfx1250(m: int, k: int, n: int) -> bool:
-@gluon.jit
+def _wmma_tdm_dense_m16_launch_metadata(grid, kernel, args):
+    """Report dense WMMA work and BF16 or split-K FP32 partial traffic."""
+    m = args["ACTUAL_M"]
+    n = grid[0] * args["BLOCK_N"]
+    k = args["K"]
diff -- tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/README.md
@@ -138,6 +138,49 @@ one-wave direct path when extra partitions or fusion do not pay for their
```

- 提取文件（未人工审阅）:
  - tests: `tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py` modified +60/-1
  - runtime: `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/gemm/fp16/mm.py` modified +233/-38; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/README.md` modified +43/-0; `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/kda/attn_res.py` modified +23/-15; `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/__init__.py` modified +8/-2; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/triton_gemv.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml`, `tokenspeed-kernel/test/amd/ops/test_dense_bf16_gfx1250.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1828 - ci(amd-kernel): Add Kimi K3 a16w16 GEMM benchmark cases

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1828
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json`；关联提交 `94c9aa9907d4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+329/-45，可读 patch 485 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` added +34/-0 (34 lines); hunks: -0,0 +1,34。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` added +34/-0 (34 lines); hunks: -0,0 +1,34
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json
@@ -0,0 +1,34 @@
+{
+  "schema_version": 1,
+  "cases": [
+    {
+      "id": "kimi_k3.gemm.mm/gluon_mm_a16w16_prefill_gfx950/tp8-kv_b-n3072-k512-bfloat16",
+      "comparison_epoch": 1,
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` added +34/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/amd/ops/gemm/test_gluon_a16w16_gfx950.py`, `tokenspeed-kernel/test/test_benchmark_gemm_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1768 - refactor(kimi3): select packed sigmoid top-k from the registry

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1768
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/layers/test_kimi_moe_topk_gfx950.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py`, `tokenspeed-kernel/test/nvidia/ops/test_kimi3_sigmoid_topk_multitoken.py`；关联提交 `066edafac9e2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+388/-125，可读 patch 701 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/layers/test_kimi_moe_topk_gfx950.py` modified +30/-11 (41 lines); hunks: -1,17 +1,31; -33,10 +47,11 @@ def _make_topk(correction_bias: torch.Tensor):; symbols: _make_topk, test_generic_topk_uses_k3_decode_or_small_m_gluon_route, test_generic_topk_uses_k3_decode_or_prefill_gluon_route, spy_route，涉及 `_make_topk, test_generic_topk_uses_k3_decode_or_small_m_gluon_route, test_generic_topk_uses_k3_decode_or_prefill_gluon_route`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py` modified +218/-1 (219 lines); hunks: -6,7 +6,15; -186,4 +194,213 @@ def kimi3_sigmoid_bias_topk(; symbols: kimi3_sigmoid_bias_topk, _invoke_packed, triton_kimi3_packed_sigmoid_bias_topk_nvidia, triton_kimi3_packed_sigmoid_bias_topk_nvidia_mapped，涉及 `kimi3_sigmoid_bias_topk, _invoke_packed, triton_kimi3_packed_sigmoid_bias_topk_nvidia`；`tokenspeed-kernel/test/nvidia/ops/test_kimi3_sigmoid_topk_multitoken.py` modified +108/-7 (115 lines); hunks: -50,6 +50,9; -115,13 +118,13 @@ def test_dispatcher_sends_a_verify_window_to_the_packed_ke...; symbols: test_dispatcher_sends_a_verify_window_to_the_packed_kernel, spy, test_dispatcher_hands_rows_past_the_cap_to_the_grouped_kernel, grouped_spy，涉及 `test_dispatcher_sends_a_verify_window_to_the_packed_kernel, spy, test_dispatcher_hands_rows_past_the_cap_to_the_grouped_kernel`；`tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py` modified +0/-49 (49 lines); hunks: -33,55 +33,6 @@ def _sigmoid_topk(; symbols: _sigmoid_topk, test_kimi3_sigmoid_bias_topk_is_exact_and_captures，涉及 `_sigmoid_topk, test_kimi3_sigmoid_bias_topk_is_exact_and_captures`。
- 代码 diff 细节:
  - `test/runtime/layers/test_kimi_moe_topk_gfx950.py` modified +30/-11 (41 lines); hunks: -1,17 +1,31; -33,10 +47,11 @@ def _make_topk(correction_bias: torch.Tensor):; symbols: _make_topk, test_generic_topk_uses_k3_decode_or_small_m_gluon_route, test_generic_topk_uses_k3_decode_or_prefill_gluon_route, spy_route
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py` modified +218/-1 (219 lines); hunks: -6,7 +6,15; -186,4 +194,213 @@ def kimi3_sigmoid_bias_topk(; symbols: kimi3_sigmoid_bias_topk, _invoke_packed, triton_kimi3_packed_sigmoid_bias_topk_nvidia, triton_kimi3_packed_sigmoid_bias_topk_nvidia_mapped
  - `tokenspeed-kernel/test/nvidia/ops/test_kimi3_sigmoid_topk_multitoken.py` modified +108/-7 (115 lines); hunks: -50,6 +50,9; -115,13 +118,13 @@ def test_dispatcher_sends_a_verify_window_to_the_packed_ke...; symbols: test_dispatcher_sends_a_verify_window_to_the_packed_kernel, spy, test_dispatcher_hands_rows_past_the_cap_to_the_grouped_kernel, grouped_spy
  - `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py` modified +0/-49 (49 lines); hunks: -33,55 +33,6 @@ def _sigmoid_topk(; symbols: _sigmoid_topk, test_kimi3_sigmoid_bias_topk_is_exact_and_captures
- 关键代码摘录:

```diff
diff -- test/runtime/layers/test_kimi_moe_topk_gfx950.py
@@ -1,17 +1,31 @@
-"""Generic TopK's gfx950 E896/top-k16 decode routing integration."""
+"""Generic TopK's gfx950 E896/top-k16 decode and batched routing integration."""
+import os
+import sys
-if not current_platform().is_cdna4:
-    pytest.skip("AMD CDNA4 is required for Kimi routing tests", allow_module_level=True)
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py
@@ -6,7 +6,15 @@
-from tokenspeed_kernel.platform import Platform, current_platform
+from tokenspeed_kernel.platform import (
+    ArchVersion,
+    CapabilityRequirement,
+    Platform,
+    current_platform,
diff -- tokenspeed-kernel/test/nvidia/ops/test_kimi3_sigmoid_topk_multitoken.py
@@ -50,6 +50,9 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/layers/test_kimi_moe_topk_gfx950.py` modified +30/-11; `tokenspeed-kernel/test/nvidia/ops/test_kimi3_sigmoid_topk_multitoken.py` modified +108/-7; `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py` modified +0/-49
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py` modified +218/-1
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_kimi_moe_topk_gfx950.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py`, `tokenspeed-kernel/test/nvidia/ops/test_kimi3_sigmoid_topk_multitoken.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1796 - perf(kimi3): skip the C16 MoE cat on gfx1250

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1796
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx1250.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；关联提交 `a6962f516f26`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+302/-8，可读 patch 430 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +54/-0 (54 lines); hunks: -67,6 +67,7; -722,6 +723,50 @@ def _tail_finalize_top_k(; symbols: _tail_finalize_top_k, _packed_moe_join_lane, _acquire_symm_join_outputs, plan，涉及 `_tail_finalize_top_k, _packed_moe_join_lane, _acquire_symm_join_outputs`；`tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx1250.py` modified +60/-0 (60 lines); hunks: -2,6 +2,8; -452,3 +454,61 @@ def test_kimi3_gfx1250_large_m_linear_o_proj_shape_matches(...; symbols: test_kimi3_gfx1250_large_m_linear_o_proj_shape_matches, test_kimi3_shared_down_strided_output_contract, test_kimi3_shared_down_strided_inputs_use_torch，涉及 `test_kimi3_gfx1250_large_m_linear_o_proj_shape_matches, test_kimi3_shared_down_strided_output_contract, test_kimi3_shared_down_strided_inputs_use_torch`；`tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +54/-0 (54 lines); hunks: -5,6 +5,7; -162,3 +163,56 @@ def test_kimi3_mla_projection_preserves_non_cdna_prefill_sc...; symbols: test_kimi3_mla_projection_preserves_non_cdna_prefill_schedule, test_kimi3_shared_down_preserves_strided_cpu_output, test_kimi3_shared_down_rejects_unsupported_strided_selector, test_kimi3_shared_down_validates_strided_output，涉及 `test_kimi3_mla_projection_preserves_non_cdna_prefill_schedule, test_kimi3_shared_down_preserves_strided_cpu_output, test_kimi3_shared_down_rejects_unsupported_strided_selector`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +46/-4 (50 lines); hunks: -22,12 +22,16; -980,25 +984,63 @@ def kimi3_shared_down_projection(; symbols: use_gluon_largem_gfx1250, use_gluon_wmma_dense_gfx1250, kimi3_shared_down_projection，涉及 `use_gluon_largem_gfx1250, use_gluon_wmma_dense_gfx1250, kimi3_shared_down_projection`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +54/-0 (54 lines); hunks: -67,6 +67,7; -722,6 +723,50 @@ def _tail_finalize_top_k(; symbols: _tail_finalize_top_k, _packed_moe_join_lane, _acquire_symm_join_outputs, plan
  - `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx1250.py` modified +60/-0 (60 lines); hunks: -2,6 +2,8; -452,3 +454,61 @@ def test_kimi3_gfx1250_large_m_linear_o_proj_shape_matches(...; symbols: test_kimi3_gfx1250_large_m_linear_o_proj_shape_matches, test_kimi3_shared_down_strided_output_contract, test_kimi3_shared_down_strided_inputs_use_torch
  - `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +54/-0 (54 lines); hunks: -5,6 +5,7; -162,3 +163,56 @@ def test_kimi3_mla_projection_preserves_non_cdna_prefill_sc...; symbols: test_kimi3_mla_projection_preserves_non_cdna_prefill_schedule, test_kimi3_shared_down_preserves_strided_cpu_output, test_kimi3_shared_down_rejects_unsupported_strided_selector, test_kimi3_shared_down_validates_strided_output
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +46/-4 (50 lines); hunks: -22,12 +22,16; -980,25 +984,63 @@ def kimi3_shared_down_projection(; symbols: use_gluon_largem_gfx1250, use_gluon_wmma_dense_gfx1250, kimi3_shared_down_projection
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -67,6 +67,7 @@
+    COMM_ONESHOT_MAX_BYTES,
@@ -722,6 +723,50 @@ def _tail_finalize_top_k(
+# One buffer per (rows, width, dtype, device). A captured graph holds the
+# pointer, so a later token count must not replace an earlier buffer.
+_PACKED_MOE_JOIN_LANES: dict[tuple, torch.Tensor] = {}
+# Matches the CDNA5 dense WMMA row ceiling. Past this the shared down
diff -- tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx1250.py
@@ -2,6 +2,8 @@
+from unittest import mock
@@ -452,3 +454,61 @@ def test_kimi3_gfx1250_large_m_linear_o_proj_shape_matches() -> None:
+@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
+@pytest.mark.parametrize("solution", ["auto", "torch"])
+@pytest.mark.parametrize("rows,width", [(2, 64), (17, 16)])
+def test_kimi3_shared_down_strided_output_contract(
diff -- tokenspeed-kernel/test/test_kimi_prefill_ops.py
@@ -5,6 +5,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +54/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +46/-4
  - tests: `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx1250.py` modified +60/-0; `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +54/-0
- 验证与风险: diff 自带测试面 `test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml`, `test/runtime/layers/test_latent_moe.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx1250.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1855 - feat(amd): Add K3 support in Gluon MegaMoE

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1855
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py`；关联提交 `67aff9941611`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+355/-63，可读 patch 776 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +17/-5 (22 lines); hunks: -1491,11 +1491,17 @@ def __init__(; -1586,9 +1592,15 @@ def __init__(; symbols: __init__，涉及 `__init__`；`tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py` modified +47/-0 (47 lines); hunks: -33,6 +33,53 @@ def _sigmoid_topk(; symbols: _sigmoid_topk, test_kimi3_sigmoid_bias_topk_matches_torch_and_captures，涉及 `_sigmoid_topk, test_kimi3_sigmoid_bias_topk_matches_torch_and_captures`；`test/runtime/test_kimi_k3_moe_attn_dp.py` modified +25/-11 (36 lines); hunks: -37,7 +37,7; -66,24 +66,34 @@ def test_attn_dp_rejects_partial_world_layout_before_backend...; symbols: test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts, __init__，涉及 `test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +17/-5 (22 lines); hunks: -1491,11 +1491,17 @@ def __init__(; -1586,9 +1592,15 @@ def __init__(; symbols: __init__
  - `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py` modified +47/-0 (47 lines); hunks: -33,6 +33,53 @@ def _sigmoid_topk(; symbols: _sigmoid_topk, test_kimi3_sigmoid_bias_topk_matches_torch_and_captures
  - `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +25/-11 (36 lines); hunks: -37,7 +37,7; -66,24 +66,34 @@ def test_attn_dp_rejects_partial_world_layout_before_backend...; symbols: test_attn_dp_rejects_partial_world_layout_before_backend_setup, test_attn_dp_replicates_dense_weights_and_selects_transport, Experts, __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1491,11 +1491,17 @@ def __init__(
+        fused_all2all_backend = (
+            All2AllBackend.GLUON_PETIT
+            if moe_backend.is_gluon_petit()
+            else All2AllBackend.NONE
+        )
-            mapping.attn.dp_size <= 1 or all2all_backend is not All2AllBackend.NONE
diff -- tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py
@@ -33,6 +33,53 @@ def _sigmoid_topk(
+@pytest.mark.skipif(not is_cdna4(), reason="gfx950 decode routing is CDNA4")
+@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64, 128, 256, 512, 1024])
+@pytest.mark.parametrize("normalize", [False, True])
+@pytest.mark.parametrize("scale", [1.0, 2.5])
+def test_kimi3_sigmoid_bias_topk_matches_torch_and_captures(
+    rows: int,
diff -- test/runtime/test_kimi_k3_moe_attn_dp.py
@@ -37,7 +37,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +17/-5
  - tests: `tokenspeed-kernel/test/amd/ops/test_kimi3_sigmoid_topk_amd.py` modified +47/-0; `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +25/-11
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_gluon_petit_backend.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py`, `tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1884 - ci(amd-kernel): Add Kimi K3 KDA prefill benchmark cases

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1884
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json`；关联提交 `0fbc6fa9e38e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+140/-35，可读 patch 281 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` added +74/-0 (74 lines); hunks: -0,0 +1,74。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` added +74/-0 (74 lines); hunks: -0,0 +1,74
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json
@@ -0,0 +1,74 @@
+{
+  "schema_version": 1,
+  "common_parameters": {
+    "model_profile": "kimi_k3_tp8",
+    "heads": 12,
+    "key_dim": 128,
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` added +74/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_kda_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1845 - refactor(kimi-k3): refactor and optimize latent MoE tail

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1845
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `test/runtime/test_kimi_k3_config.py` 等 7 个文件；关联提交 `2a8dbc6b4c80`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 28 个文件，+3330/-3474，可读 patch 7810 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +225/-1052 (1277 lines); hunks: -18,76 +18,52; -105,225 +81,52 @@ def attn_ar_eligible(; symbols: attn_ar_eligible, K3MoETailTier, select_k3_moe_tail_tier, prepare_k3_all_reduce_buffers，涉及 `attn_ar_eligible, K3MoETailTier, select_k3_moe_tail_tier`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +254/-114 (368 lines); hunks: -70,6 +70,7; -97,8 +98,12; symbols: _attnres_scratch, prepare_k3_all_reduce_buffers, _amd_moe_join_lane, KimiLinearMoE，涉及 `_attnres_scratch, prepare_k3_all_reduce_buffers, _amd_moe_join_lane`；`test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +82/-88 (170 lines); hunks: -18,35 +18,17; -67,50 +49,69 @@ class _SpyFork:; symbols: _SpyFork, __init__, scope, branch，涉及 `_SpyFork, __init__, scope`；`test/runtime/test_kimi_k3_comm_arming.py` modified +61/-105 (166 lines); hunks: -18,15 +18,7; -51,30 +43,13; symbols: test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar, test_iris_preparation_caps_attnres_for_equal_tp8_groups，涉及 `test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar, test_iris_preparation_caps_attnres_for_equal_tp8_groups`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +225/-1052 (1277 lines); hunks: -18,76 +18,52; -105,225 +81,52 @@ def attn_ar_eligible(; symbols: attn_ar_eligible, K3MoETailTier, select_k3_moe_tail_tier, prepare_k3_all_reduce_buffers
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +254/-114 (368 lines); hunks: -70,6 +70,7; -97,8 +98,12; symbols: _attnres_scratch, prepare_k3_all_reduce_buffers, _amd_moe_join_lane, KimiLinearMoE
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +82/-88 (170 lines); hunks: -18,35 +18,17; -67,50 +49,69 @@ class _SpyFork:; symbols: _SpyFork, __init__, scope, branch
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +61/-105 (166 lines); hunks: -18,15 +18,7; -51,30 +43,13; symbols: test_arming_requires_experts_capability_bit, test_arming_requires_fused_moe_ar, test_iris_preparation_caps_attnres_for_equal_tp8_groups
  - `test/runtime/test_kimi_k3_attn_res.py` modified +21/-27 (48 lines); hunks: -83,15 +83,13 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):; -135,15 +133,13 @@ def test_fused_attention_window_defers_attnres_combine(self):; symbols: test_batched_iris_reduce_consumes_attnres_combine, test_fused_attention_window_defers_attnres_combine, test_fused_attention_reduce_window
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -18,76 +18,52 @@
-"""Kimi-K3 communication layer: capability negotiation and fused-reduction
-routing for the sites where K3's AttnRes/latent-lane semantics bypass the
-generic ``CommManager`` (decision D4).
+"""Kimi-K3 communication routing and collective workspace ownership.
-Layering: this module owns *which backend runs where* (votes, workspace
-lifecycle, M-window routing); the kernels themselves stay behind
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -70,6 +70,7 @@
+from tokenspeed_kernel.ops.communication import allreduce_fusion_lane
@@ -97,8 +98,12 @@
+    COMM_ONESHOT_MAX_BYTES,
+    acquire_all_reduce_outputs,
+    can_acquire_all_reduce_outputs,
+    prepare_all_reduce_buffers,
diff -- test/runtime/test_kimi_k3_moe_fork_warmup.py
@@ -18,35 +18,17 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +225/-1052; `python/tokenspeed/runtime/models/kimi_k3.py` modified +254/-114
  - tests: `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +82/-88; `test/runtime/test_kimi_k3_comm_arming.py` modified +61/-105; `test/runtime/test_kimi_k3_attn_res.py` modified +21/-27; `test/runtime/test_kimi_k3_config.py` modified +5/-7; `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +1/-4
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_latent_moe.py`, `test/runtime/test_k3_moe_tail_equivalence.py`, `test/runtime/test_k3_moe_tail_tier.py`, `test/runtime/test_kimi_k3_attn_res.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1771 - perf(kimi-k3): split post moe all reduce in prefill and shard the moe tail

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1771
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `test/runtime/test_kimi_k3_comm_arming.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；关联提交 `f46a87e4ce70`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+1248/-65，可读 patch 2053 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +49/-4 (53 lines); hunks: -116,6 +116,7; -1445,6 +1446,7 @@ def _attnres_scratch(; symbols: _attnres_scratch, prepare_k3_all_reduce_buffers，涉及 `_attnres_scratch, prepare_k3_all_reduce_buffers`；`test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +194/-0 (194 lines); hunks: -239,5 +239,199 @@ def process_weights(module):; symbols: process_weights, _make_amd_moe, test_row_sharded_moe_tail_selection_and_fallback, run_tail，涉及 `process_weights, _make_amd_moe, test_row_sharded_moe_tail_selection_and_fallback`；`test/runtime/test_kimi_k3_comm_arming.py` modified +20/-4 (24 lines); hunks: -48,11 +48,19; -68,15 +76,16 @@ def test_iris_preparation_caps_attnres_for_equal_tp8_groups(...; symbols: test_iris_preparation_caps_attnres_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups，涉及 `test_iris_preparation_caps_attnres_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups`；`test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` modified +2/-0 (2 lines); hunks: -15,6 +15,8 @@ env:。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +49/-4 (53 lines); hunks: -116,6 +116,7; -1445,6 +1446,7 @@ def _attnres_scratch(; symbols: _attnres_scratch, prepare_k3_all_reduce_buffers
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +194/-0 (194 lines); hunks: -239,5 +239,199 @@ def process_weights(module):; symbols: process_weights, _make_amd_moe, test_row_sharded_moe_tail_selection_and_fallback, run_tail
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +20/-4 (24 lines); hunks: -48,11 +48,19; -68,15 +76,16 @@ def test_iris_preparation_caps_attnres_for_equal_tp8_groups(...; symbols: test_iris_preparation_caps_attnres_for_equal_tp8_groups, test_iris_preparation_handles_distinct_groups
  - `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` modified +2/-0 (2 lines); hunks: -15,6 +15,8 @@ env:
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -116,6 +116,7 @@
+    _get_process_group,
@@ -1445,6 +1446,7 @@ def _attnres_scratch(
+_IRIS_MOE_ROW_SHARD_MIN_TOKENS = 40
@@ -1472,12 +1474,13 @@ def prepare_k3_all_reduce_buffers(
-    enable_lamport = (
+    tp8_moe = (
diff -- test/runtime/test_kimi_k3_moe_fork_warmup.py
@@ -239,5 +239,199 @@ def process_weights(module):
+def _make_amd_moe(fork, *, routed, shared, projection, norm, mapping):
+    moe = _make_moe(fork, num_tokens=routed.shape[0])
+    moe.mapping = mapping
+    moe.execution_plan = SimpleNamespace(use_native=True)
+    moe.comm = None
+    moe.routed_hidden = routed.shape[1]
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -48,11 +48,19 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +49/-4
  - tests: `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +194/-0; `test/runtime/test_kimi_k3_comm_arming.py` modified +20/-4; `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` modified +2/-0
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `test/runtime/distributed/test_auto_backend.py`, `test/runtime/test_kimi_k3_comm_arming.py`, `test/runtime/test_kimi_k3_moe_fork_warmup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1790 - perf(kimi-k3): shard prefill attention reduction and attnres

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1790
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `python/tokenspeed/runtime/models/kimi_k3_deepep.py`, `python/tokenspeed/runtime/models/kimi_k3_nextn.py`, `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml` 等 11 个文件；关联提交 `9e003aebfaba`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 18 个文件，+2064/-130，可读 patch 2802 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +105/-0 (105 lines); hunks: -54,7 +54,9; -65,6 +67,10; symbols: __init__, acquire_prefill_projection_output, prefill_reduce_for_attnres, prefill_mix_for_moe，涉及 `__init__, acquire_prefill_projection_output, prefill_reduce_for_attnres`；`python/tokenspeed/runtime/models/kimi_k3.py` modified +88/-16 (104 lines); hunks: -80,6 +80,7; -613,6 +614,8 @@ def forward(; symbols: forward, _forward_amd，涉及 `forward, _forward_amd`；`python/tokenspeed/runtime/models/kimi_k3_deepep.py` modified +4/-0 (4 lines); hunks: -190,7 +190,11 @@ def forward(; symbols: forward，涉及 `forward`；`python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +1/-0 (1 lines); hunks: -214,6 +214,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +105/-0 (105 lines); hunks: -54,7 +54,9; -65,6 +67,10; symbols: __init__, acquire_prefill_projection_output, prefill_reduce_for_attnres, prefill_mix_for_moe
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +88/-16 (104 lines); hunks: -80,6 +80,7; -613,6 +614,8 @@ def forward(; symbols: forward, _forward_amd
  - `python/tokenspeed/runtime/models/kimi_k3_deepep.py` modified +4/-0 (4 lines); hunks: -190,7 +190,11 @@ def forward(; symbols: forward
  - `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +1/-0 (1 lines); hunks: -214,6 +214,7 @@ def forward(; symbols: forward
  - `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +133/-2 (135 lines); hunks: -154,6 +154,7 @@ def _run(*, graph_phase: bool, capture_mode: bool, num_token...; -205,6 +206,7 @@ def test_moe_passes_projection_payload_to_experts(prequantiz...; symbols: _run, test_moe_passes_projection_payload_to_experts, reduce_partials, test_row_sharded_moe_tail_skips_iris_import_on_other_platform
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -54,7 +54,9 @@
+    acquire_all_reduce_outputs,
+    can_acquire_all_reduce_outputs,
@@ -65,6 +67,10 @@
+_IRIS_MAX_TOKENS = 8192
+_IRIS_ATTN_PRODUCER_DIRECT_MIN_TOKENS = 16
+_IRIS_ATTN_SHARDED_PREFIX_MIN_TOKENS = 56
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -80,6 +80,7 @@
+    mm,
@@ -613,6 +614,8 @@ def forward(
+        *,
+        projection_out: torch.Tensor | None,
@@ -651,6 +654,9 @@ def forward(
+        if projection_out is not None:
diff -- python/tokenspeed/runtime/models/kimi_k3_deepep.py
@@ -190,7 +190,11 @@ def forward(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +105/-0; `python/tokenspeed/runtime/models/kimi_k3.py` modified +88/-16; `python/tokenspeed/runtime/models/kimi_k3_deepep.py` modified +4/-0; `python/tokenspeed/runtime/models/kimi_k3_nextn.py` modified +1/-0
  - tests: `test/runtime/test_kimi_k3_moe_fork_warmup.py` modified +133/-2; `test/runtime/test_kimi_k3_comm_arming.py` modified +131/-0; `test/runtime/test_kimi_k3_attn_res.py` modified +99/-6; `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +8/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml`, `test/runtime/distributed/test_auto_backend.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1853 - feat(DCP): Add Kimi K3 DCP support to the CuTe MLA backend

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1853
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh`, `test/runtime/test_kimi_k3_cache_spec.py`, `test/runtime/test_kimi_k3_cudagraph.py` 等 6 个文件；关联提交 `5905f205c30e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 29 个文件，+2187/-104，可读 patch 3345 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh` added +33/-0 (33 lines); hunks: -0,0 +1,33；`test/runtime/test_kimi_k3_mla.py` modified +88/-23 (111 lines); hunks: -34,6 +34,9; -222,8 +225,15 @@ def test_pool_write_location_oracle_page_id_times_p_plus_of...; symbols: test_pool_write_location_oracle_page_id_times_p_plus_offset, cache_dtype, cuda_env，涉及 `test_pool_write_location_oracle_page_id_times_p_plus_offset, cache_dtype, cuda_env`；`test/runtime/test_kimi_k3_cudagraph.py` modified +9/-8 (17 lines); hunks: -63,6 +63,8 @@ def _bare_mla_backend(; -342,10 +344,8 @@ def test_block_decode_stays_off_for_target_and_single_token...; symbols: _bare_mla_backend, test_block_decode_stays_off_for_target_and_single_token_drafts, test_block_decode_hands_the_kernel_one_query_per_block_row, test_block_decode_hands_the_kernel_a_noncausal_query_block，涉及 `_bare_mla_backend, test_block_decode_stays_off_for_target_and_single_token_drafts, test_block_decode_hands_the_kernel_one_query_per_block_row`；`test/runtime/test_kimi_k3_cache_spec.py` modified +15/-0 (15 lines); hunks: -134,9 +134,13 @@ def test_bf16_mla_cache_reuses_the_same_packing_rule() -> N...; -148,6 +152,17 @@ def test_speculative_verify_workspace_is_reserved_outside_t...; symbols: test_bf16_mla_cache_reuses_the_same_packing_rule, test_speculative_verify_workspace_is_reserved_outside_the_arena，涉及 `test_bf16_mla_cache_reuses_the_same_packing_rule, test_speculative_verify_workspace_is_reserved_outside_the_arena`。
- 代码 diff 细节:
  - `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh` added +33/-0 (33 lines); hunks: -0,0 +1,33
  - `test/runtime/test_kimi_k3_mla.py` modified +88/-23 (111 lines); hunks: -34,6 +34,9; -222,8 +225,15 @@ def test_pool_write_location_oracle_page_id_times_p_plus_of...; symbols: test_pool_write_location_oracle_page_id_times_p_plus_offset, cache_dtype, cuda_env
  - `test/runtime/test_kimi_k3_cudagraph.py` modified +9/-8 (17 lines); hunks: -63,6 +63,8 @@ def _bare_mla_backend(; -342,10 +344,8 @@ def test_block_decode_stays_off_for_target_and_single_token...; symbols: _bare_mla_backend, test_block_decode_stays_off_for_target_and_single_token_drafts, test_block_decode_hands_the_kernel_one_query_per_block_row, test_block_decode_hands_the_kernel_a_noncausal_query_block
  - `test/runtime/test_kimi_k3_cache_spec.py` modified +15/-0 (15 lines); hunks: -134,9 +134,13 @@ def test_bf16_mla_cache_reuses_the_same_packing_rule() -> N...; -148,6 +152,17 @@ def test_speculative_verify_workspace_is_reserved_outside_t...; symbols: test_bf16_mla_cache_reuses_the_same_packing_rule, test_speculative_verify_workspace_is_reserved_outside_the_arena
  - `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` modified +1/-0 (1 lines); hunks: -23,6 +23,7 @@ pip install "evalscope[perf] @ git+https://github.com/modelsco...
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh
@@ -0,0 +1,33 @@
+#!/usr/bin/bash
+set -euo pipefail
+# DCP requires KVStore disabled; disable it in the DCP1 baseline too when
+# comparing DCP overhead.
+exec ts serve \
+    --model nvidia/Kimi-K3-NVFP4 \
diff -- test/runtime/test_kimi_k3_mla.py
@@ -34,6 +34,9 @@
+from test.runtime.conftest import (
+    kimi_tp8_layout,
+)
@@ -222,8 +225,15 @@ def test_pool_write_location_oracle_page_id_times_p_plus_offset() -> None:
+@pytest.fixture(
+    scope="module", params=[torch.float8_e4m3fn, torch.bfloat16], ids=["fp8", "bf16"]
diff -- test/runtime/test_kimi_k3_cudagraph.py
@@ -63,6 +63,8 @@ def _bare_mla_backend(
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh` added +33/-0; `test/runtime/test_kimi_k3_mla.py` modified +88/-23; `test/runtime/test_kimi_k3_cudagraph.py` modified +9/-8; `test/runtime/test_kimi_k3_cache_spec.py` modified +15/-0; `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh` modified +1/-0; `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm` modified +1/-0
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/paged/tokenspeed_mla.py` modified +212/-15; `python/tokenspeed/runtime/layers/attention/dcp/metadata.py` modified +54/-0
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/kimi_k3/tokenspeed/agentic_bench.slurm`, `test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_dcp4_moe_tp8.sh`, `test/runtime/distributed/run_cutedsl_mla_dcp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1912 - Fix KimiLinearMoE native layer state for attention DP

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1912
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_moe_attn_dp.py`；关联提交 `8bf3ef59b6de`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+3/-1，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +2/-0 (2 lines); hunks: -1606,6 +1606,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_kimi_k3_moe_attn_dp.py` modified +1/-1 (2 lines); hunks: -190,7 +190,7 @@ def __init__(self, **kwargs):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +2/-0 (2 lines); hunks: -1606,6 +1606,8 @@ def __init__(; symbols: __init__
  - `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +1/-1 (2 lines); hunks: -190,7 +190,7 @@ def __init__(self, **kwargs):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -1606,6 +1606,8 @@ def __init__(
+        # Decoder layers inspect this even when attention DP skips native setup.
+        self.native_latent_moe: LatentMoELayer | None = None
diff -- test/runtime/test_kimi_k3_moe_attn_dp.py
@@ -190,7 +190,7 @@ def __init__(self, **kwargs):
-    assert not hasattr(layer, "native_latent_moe")
+    assert layer.native_latent_moe is None
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +2/-0
  - tests: `test/runtime/test_kimi_k3_moe_attn_dp.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_moe_attn_dp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1898 - feat(amd): serve AMD Quark MXFP4 Kimi K3 with per-layer FP8 attention

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1898
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `test/runtime/test_kimi_k3_attn_res.py`；关联提交 `22251686ff26`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+709/-46，可读 patch 1104 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +99/-14 (113 lines); hunks: -56,6 +56,7; -114,6 +115,7; symbols: _situ_betas, _dense_mlp_quant_config, KimiKDAMergedProj, __init__，涉及 `_situ_betas, _dense_mlp_quant_config, KimiKDAMergedProj`；`test/runtime/test_kimi_k3_attn_res.py` modified +8/-1 (9 lines); hunks: -1056,6 +1056,7 @@ def test_loader_layout_and_forward_parity(self):; -1088,7 +1089,13 @@ def ref(w, rows, rk=rank):; symbols: test_loader_layout_and_forward_parity, ref, test_decode_single_row_slice_is_zero_copy，涉及 `test_loader_layout_and_forward_parity, ref, test_decode_single_row_slice_is_zero_copy`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +99/-14 (113 lines); hunks: -56,6 +56,7; -114,6 +115,7; symbols: _situ_betas, _dense_mlp_quant_config, KimiKDAMergedProj, __init__
  - `test/runtime/test_kimi_k3_attn_res.py` modified +8/-1 (9 lines); hunks: -1056,6 +1056,7 @@ def test_loader_layout_and_forward_parity(self):; -1088,7 +1089,13 @@ def ref(w, rows, rk=rank):; symbols: test_loader_layout_and_forward_parity, ref, test_decode_single_row_slice_is_zero_copy
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -56,6 +56,7 @@
+import re
@@ -114,6 +115,7 @@
+from tokenspeed.runtime.layers.dense.w8a8_fp8 import w8a8_fp8_per_channel_mm
@@ -151,6 +153,10 @@
+from tokenspeed.runtime.layers.quantization.mxfp4 import (
+    Mxfp4Config,
diff -- test/runtime/test_kimi_k3_attn_res.py
@@ -1056,6 +1056,7 @@ def test_loader_layout_and_forward_parity(self):
+                fp8_channel_quant=False,
@@ -1088,7 +1089,13 @@ def ref(w, rows, rk=rank):
-            hidden_size=8, proj=8, num_heads=2, head_dim=4, tp_rank=0, tp_size=1
+            hidden_size=8,
+            proj=8,
+            num_heads=2,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +99/-14
  - tests: `test/runtime/test_kimi_k3_attn_res.py` modified +8/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_kda_fp8_w8a8.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_mxfp4_config.py`, `tokenspeed-kernel/test/ops/test_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1913 - feat(gemm): add Gluon gfx950 decode GEMM for Kimi K3

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1913
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py`；关联提交 `932e1866b160`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+670/-44，可读 patch 997 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +69/-0 (69 lines); hunks: -102,6 +102,75；`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +42/-9 (51 lines); hunks: -58,6 +58,16 @@ def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) -> b...; -94,10 +104,6 @@ def _use_gluon_mediumm(m: int, k: int, n: int) -> bool:; symbols: use_gluon_wmma_dense_gfx1250, supports_gluon_mm_a16w16_decode_gfx950, _use_gluon_mediumm, _use_gluon_smallm，涉及 `use_gluon_wmma_dense_gfx1250, supports_gluon_mm_a16w16_decode_gfx950, _use_gluon_mediumm`；`tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +24/-15 (39 lines); hunks: -5,11 +5,8; -89,19 +86,29 @@ def test_kimi3_latent_projection_dispatch_boundaries(; symbols: test_kimi3_latent_projection_dispatch_boundaries, test_kimi3_latent_projection_smallm_dispatch, test_decode_gemm_routes_measured_k3_shapes, test_kimi3_router_projection_dispatches_all_token_counts，涉及 `test_kimi3_latent_projection_dispatch_boundaries, test_kimi3_latent_projection_smallm_dispatch, test_decode_gemm_routes_measured_k3_shapes`。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +69/-0 (69 lines); hunks: -102,6 +102,75
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +42/-9 (51 lines); hunks: -58,6 +58,16 @@ def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) -> b...; -94,10 +104,6 @@ def _use_gluon_mediumm(m: int, k: int, n: int) -> bool:; symbols: use_gluon_wmma_dense_gfx1250, supports_gluon_mm_a16w16_decode_gfx950, _use_gluon_mediumm, _use_gluon_smallm
  - `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +24/-15 (39 lines); hunks: -5,11 +5,8; -89,19 +86,29 @@ def test_kimi3_latent_projection_dispatch_boundaries(; symbols: test_kimi3_latent_projection_dispatch_boundaries, test_kimi3_latent_projection_smallm_dispatch, test_decode_gemm_routes_measured_k3_shapes, test_kimi3_router_projection_dispatches_all_token_counts
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json
@@ -102,6 +102,75 @@
+    },
+    {
+      "id": "kimi_k3.gemm.decode_gemv/gluon_mm_a16w16_decode_gfx950/tp8-o_proj-n7168-k1536-bfloat16",
+      "comparison_epoch": 1,
+      "measurement_blocks": 50,
+      "definition": {
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -58,6 +58,16 @@ def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) -> bool:
+try:
+    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
+        supports_gluon_mm_a16w16_decode_gfx950,
+    )
+except ImportError:
+    def supports_gluon_mm_a16w16_decode_gfx950(m: int, n: int, k: int) -> bool:
diff -- tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py
@@ -5,11 +5,8 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +69/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +42/-9
  - tests: `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +24/-15
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/amd/ops/gemm/test_gluon_a16w16_gfx950.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1896 - ci(amd-kernel): Benchmark Kimi K3 fused KDA decode, verify and replay

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1896
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json`；关联提交 `9ec647a468b8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+682/-2，可读 patch 736 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` modified +99/-0 (99 lines); hunks: -69,6 +69,105。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` modified +99/-0 (99 lines); hunks: -69,6 +69,105
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json
@@ -69,6 +69,105 @@
+    },
+    {
+      "id": "kimi_k3.attention.kda_fused_paged_decode/decode-bfloat16",
+      "comparison_epoch": 1,
+      "measurement_blocks": 50,
+      "definition": {
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/kda.json` modified +99/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_kda_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1923 - perf(amd): unify k3 attention all reduce selection

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1923
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/kimi_k3.py`, `python/tokenspeed/runtime/models/kimi_k3_comm.py`, `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`；关联提交 `b395a8c825f5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+81/-74，可读 patch 382 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/kimi_k3.py` modified +30/-15 (45 lines); hunks: -2809,6 +2809,8 @@ def _reduce_attn_accumulate(; -2817,7 +2819,11 @@ def _reduce_attn_accumulate(; symbols: _reduce_attn_accumulate, capture_attnres, _forward_fused_attnres_graph，涉及 `_reduce_attn_accumulate, capture_attnres, _forward_fused_attnres_graph`；`python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-11 (20 lines); hunks: -175,21 +175,16 @@ def __init__(self, *, mapping, hidden_size: int) -> None:; -218,7 +213,7 @@ def acquire_prefill_projection_output(; symbols: __init__, acquire_prefill_projection_output, acquire_projection_output, prefill_reduce_for_attnres，涉及 `__init__, acquire_prefill_projection_output, acquire_projection_output`；`test/runtime/test_kimi_k3_comm_arming.py` modified +17/-34 (51 lines); hunks: -309,7 +309,9 @@ def test_the_collective_is_what_serves_an_eligible_reduce():; -322,7 +324,7 @@ def test_the_collective_is_what_serves_an_eligible_reduce():; symbols: test_the_collective_is_what_serves_an_eligible_reduce, test_arming_declines_unsupported_collectives, test_attention_prefill_producer_window, test_attention_producer_window，涉及 `test_the_collective_is_what_serves_an_eligible_reduce, test_arming_declines_unsupported_collectives, test_attention_prefill_producer_window`；`test/runtime/test_kimi_k3_attn_res.py` modified +18/-14 (32 lines); hunks: -116,6 +116,7 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):; -168,6 +169,7 @@ def test_fused_attention_window_defers_attnres_combine(self):; symbols: test_batched_iris_reduce_consumes_attnres_combine, test_fused_attention_window_defers_attnres_combine, mlp, test_prefill_mixer_passes_an_explicit_residual_shard_to_moe，涉及 `test_batched_iris_reduce_consumes_attnres_combine, test_fused_attention_window_defers_attnres_combine, mlp`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/kimi_k3.py` modified +30/-15 (45 lines); hunks: -2809,6 +2809,8 @@ def _reduce_attn_accumulate(; -2817,7 +2819,11 @@ def _reduce_attn_accumulate(; symbols: _reduce_attn_accumulate, capture_attnres, _forward_fused_attnres_graph
  - `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-11 (20 lines); hunks: -175,21 +175,16 @@ def __init__(self, *, mapping, hidden_size: int) -> None:; -218,7 +213,7 @@ def acquire_prefill_projection_output(; symbols: __init__, acquire_prefill_projection_output, acquire_projection_output, prefill_reduce_for_attnres
  - `test/runtime/test_kimi_k3_comm_arming.py` modified +17/-34 (51 lines); hunks: -309,7 +309,9 @@ def test_the_collective_is_what_serves_an_eligible_reduce():; -322,7 +324,7 @@ def test_the_collective_is_what_serves_an_eligible_reduce():; symbols: test_the_collective_is_what_serves_an_eligible_reduce, test_arming_declines_unsupported_collectives, test_attention_prefill_producer_window, test_attention_producer_window
  - `test/runtime/test_kimi_k3_attn_res.py` modified +18/-14 (32 lines); hunks: -116,6 +116,7 @@ def test_batched_iris_reduce_consumes_attnres_combine(self):; -168,6 +169,7 @@ def test_fused_attention_window_defers_attnres_combine(self):; symbols: test_batched_iris_reduce_consumes_attnres_combine, test_fused_attention_window_defers_attnres_combine, mlp, test_prefill_mixer_passes_an_explicit_residual_shard_to_moe
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/kimi_k3.py
@@ -2809,6 +2809,8 @@ def _reduce_attn_accumulate(
+        *,
+        producer_direct: bool,
@@ -2817,7 +2819,11 @@ def _reduce_attn_accumulate(
-            attn_partial, prefix_sum, combine, mlp_wp=self._mlp_wp
+            attn_partial,
+            prefix_sum,
diff -- python/tokenspeed/runtime/models/kimi_k3_comm.py
@@ -175,21 +175,16 @@ def __init__(self, *, mapping, hidden_size: int) -> None:
-    def acquire_prefill_projection_output(
+    def acquire_projection_output(
-        *,
-        is_prefill: bool,
-        sharded_moe_supported: bool,
-            not is_prefill
diff -- test/runtime/test_kimi_k3_comm_arming.py
@@ -309,7 +309,9 @@ def test_the_collective_is_what_serves_an_eligible_reduce():
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/kimi_k3.py` modified +30/-15; `python/tokenspeed/runtime/models/kimi_k3_comm.py` modified +9/-11
  - tests: `test/runtime/test_kimi_k3_comm_arming.py` modified +17/-34; `test/runtime/test_kimi_k3_attn_res.py` modified +18/-14
- 验证与风险: diff 自带测试面 `test/runtime/test_kimi_k3_attn_res.py`, `test/runtime/test_kimi_k3_comm_arming.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1937 - ci(amd-kernel): Benchmark Kimi K3 MLA verify on the query axis

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1937
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json`；关联提交 `3616aa0fbcfb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+125/-71，可读 patch 414 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` modified +22/-46 (68 lines); hunks: -109,23 +109,22; -134,75 +133,52。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` modified +22/-46 (68 lines); hunks: -109,23 +109,22; -134,75 +133,52
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json
@@ -109,23 +109,22 @@
-      "id": "kimi_k3.attention.mla_decode_projected_value/verify-50k-float8",
+      "id": "kimi_k3.attention.mla_decode_with_kvcache/decode-50k-float8",
-        "mode": "mla_decode_projected_value",
+        "mode": "mla_decode_with_kvcache",
-          "requests": 1,
-          "rows_per_request": 4,
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` modified +22/-46
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_mla_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1943 - ci(amd-kernel): Update Kimi K3 decode GEMM cases

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1943
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json`；关联提交 `7bdeb035343b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+115/-0，可读 patch 129 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +115/-0 (115 lines); hunks: -103,6 +103,29; -171,6 +194,98。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +115/-0 (115 lines); hunks: -103,6 +103,29; -171,6 +194,98
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json
@@ -103,6 +103,29 @@
+    {
+      "id": "kimi_k3.gemm.decode_gemv/triton_rowcta_gemv/tp8-lm_head-m1-n20480-k7168-bfloat16",
+      "comparison_epoch": 1,
+      "measurement_blocks": 50,
+      "definition": {
+        "family": "gemm",
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +115/-0
- 验证与风险: runtime 路径改动集中在 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1944 - ci(amd-kernel): Benchmark Kimi K3 MLA cached extend

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1944
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json`；关联提交 `b2dd427d7308`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+199/-0，可读 patch 248 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` modified +48/-0 (48 lines); hunks: -182,6 +182,54。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` modified +48/-0 (48 lines); hunks: -182,6 +182,54
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json
@@ -182,6 +182,54 @@
+    {
+      "id": "kimi_k3.attention.mla_extend_with_kvcache/cached-8k-float8",
+      "comparison_epoch": 1,
+      "measurement_blocks": 30,
+      "definition": {
+        "family": "attention",
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/mla.json` modified +48/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_mla_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1947 - ci(amd-kernel): Benchmark Kimi K3 AttnRes prefill mixing

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1947
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/residual.json`；关联提交 `ab14b32582fe`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+294/-4，可读 patch 349 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/residual.json` added +78/-0 (78 lines); hunks: -0,0 +1,78。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/residual.json` added +78/-0 (78 lines); hunks: -0,0 +1,78
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/residual.json
@@ -0,0 +1,78 @@
+{
+  "schema_version": 1,
+  "common_parameters": {
+    "model_profile": "kimi_k3_tp8",
+    "hidden_size": 7168,
+    "block_slots": 8,
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/residual.json` added +78/-0
- 验证与风险: runtime 路径改动集中在 `tokenspeed-kernel/benchmarks/amd/gfx950.json`, `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/residual.json`, `tokenspeed-kernel/python/tokenspeed_kernel/benchmark/generators/residual.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1948 - ci(amd): Raise Kimi K3 EAGLE3 50K/500 perf reference

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1948
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml`；关联提交 `aeee7be62395`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+7/-5，可读 patch 24 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` modified +6/-4 (10 lines); hunks: -115,8 +115,10 @@ perf:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` modified +6/-4 (10 lines); hunks: -115,8 +115,10 @@ perf:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml
@@ -115,8 +115,10 @@ perf:
-# Values are [Latency (tps/user), Throughput (tps/gpu)]. Rounded down from the
-# median of three consecutive 8x MI350X TP8/EP1 runs. They measured
-# 22.38-24.39 tps/user and 12.33-13.13 tps/GPU at 50K/500, concurrency 16.
+# Values are [Latency (tps/user), Throughput (tps/gpu)]. 8x MI350X runs at
+# #1932, #1923, #1914, #1885, #1944 and #1948 measured (53.79/16.16,
+# 63.86/16.23, 75.53/14.93, 67.11/14.74, 68.21/16.03, 57.01/15.27), with
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` modified +6/-4
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml`, `test/ci_system/test_eval_configs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1946 - perf(amd): route Kimi K3 prefill projections to Gluon large-M GEMM

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1946
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；关联提交 `7da7d6c831b1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+411/-64，可读 patch 774 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +122/-21 (143 lines); hunks: -24,7 +24,9; -68,6 +70,16 @@ def supports_gluon_mm_a16w16_decode_gfx950(m: int, n: int, k:...; symbols: supports_gluon_mm_a16w16_decode_gfx950, supports_gluon_mm_a16w16_prefill_gfx950, _use_gluon_mediumm, _use_gluon_largem，涉及 `supports_gluon_mm_a16w16_decode_gfx950, supports_gluon_mm_a16w16_prefill_gfx950, _use_gluon_mediumm`；`tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +117/-1 (118 lines); hunks: -9,7 +9,7; -30,6 +30,122；`tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +4/-1 (5 lines); hunks: -54,7 +54,10 @@ def test_kimi3_latent_projection_matches_torch(; symbols: test_kimi3_latent_projection_matches_torch，涉及 `test_kimi3_latent_projection_matches_torch`；`tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +3/-1 (4 lines); hunks: -177,7 +177,9 @@ def test_kimi3_shared_down_preserves_strided_cpu_output(dtyp...; symbols: test_kimi3_shared_down_preserves_strided_cpu_output，涉及 `test_kimi3_shared_down_preserves_strided_cpu_output`。
- 代码 diff 细节:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +122/-21 (143 lines); hunks: -24,7 +24,9; -68,6 +70,16 @@ def supports_gluon_mm_a16w16_decode_gfx950(m: int, n: int, k:...; symbols: supports_gluon_mm_a16w16_decode_gfx950, supports_gluon_mm_a16w16_prefill_gfx950, _use_gluon_mediumm, _use_gluon_largem
  - `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +117/-1 (118 lines); hunks: -9,7 +9,7; -30,6 +30,122
  - `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +4/-1 (5 lines); hunks: -54,7 +54,10 @@ def test_kimi3_latent_projection_matches_torch(; symbols: test_kimi3_latent_projection_matches_torch
  - `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +3/-1 (4 lines); hunks: -177,7 +177,9 @@ def test_kimi3_shared_down_preserves_strided_cpu_output(dtyp...; symbols: test_kimi3_shared_down_preserves_strided_cpu_output
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -24,7 +24,9 @@
-a bandwidth-oriented fused Q/K/V/output-gate projection.
+a bandwidth-oriented fused Q/K/V/output-gate projection. On gfx950 the other
+K3 projections take the Gluon large-M kernel for the prefill token counts where
+it beats hipBLASLt.
@@ -68,6 +70,16 @@ def supports_gluon_mm_a16w16_decode_gfx950(m: int, n: int, k: int) -> bool:
+try:
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json
@@ -9,7 +9,7 @@
-          "M": [2816, 3072, 3328, 3584, 3840, 4096],
+          "M": [4608, 5120],
@@ -30,6 +30,122 @@
+    {
+      "id": "kimi_k3.gemm.mm/gluon_mm_a16w16_prefill_gfx950/tp8-kda_qkvfab-n6288-k7168-bfloat16",
+      "comparison_epoch": 1,
diff -- tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py
@@ -54,7 +54,10 @@ def test_kimi3_latent_projection_matches_torch(
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +122/-21; `tokenspeed-kernel/benchmarks/amd/gfx950/kimi_k3/gemm.json` modified +117/-1
  - tests: `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +4/-1; `tokenspeed-kernel/test/test_kimi_prefill_ops.py` modified +3/-1
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/amd/ops/gemm/test_gluon_a16w16_gfx950.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py`, `tokenspeed-kernel/test/test_kimi_prefill_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1966 - perf(amd): Fuse K3 latent up-projection add3 for decode batches

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1966
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py`；关联提交 `eb0474535862`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+158/-51，可读 patch 341 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +43/-27 (70 lines); hunks: -62,13 +62,17 @@ def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) ->...; -813,9 +817,9 @@ def kimi3_latent_projection_add3(; symbols: use_gluon_wmma_dense_gfx1250, supports_gluon_mm_a16w16_decode_gfx950, supports_gluon_mm_a16w16_decode_add3_gfx950, kimi3_latent_projection_add3，涉及 `use_gluon_wmma_dense_gfx1250, supports_gluon_mm_a16w16_decode_gfx950, supports_gluon_mm_a16w16_decode_add3_gfx950`；`tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +2/-2 (4 lines); hunks: -136,7 +136,7 @@ def test_kimi3_latent_projection_writes_out_and_captures(; -164,7 +164,7 @@ def test_kimi3_latent_projection_add3_matches_torch_and_capt...; symbols: test_kimi3_latent_projection_writes_out_and_captures, test_kimi3_latent_projection_add3_matches_torch_and_captures, test_kimi3_rmsnorm_linear_add_matches_composed_and_captures，涉及 `test_kimi3_latent_projection_writes_out_and_captures, test_kimi3_latent_projection_add3_matches_torch_and_captures, test_kimi3_rmsnorm_linear_add_matches_composed_and_captures`。
- 代码 diff 细节:
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +43/-27 (70 lines); hunks: -62,13 +62,17 @@ def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) ->...; -813,9 +817,9 @@ def kimi3_latent_projection_add3(; symbols: use_gluon_wmma_dense_gfx1250, supports_gluon_mm_a16w16_decode_gfx950, supports_gluon_mm_a16w16_decode_add3_gfx950, kimi3_latent_projection_add3
  - `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +2/-2 (4 lines); hunks: -136,7 +136,7 @@ def test_kimi3_latent_projection_writes_out_and_captures(; -164,7 +164,7 @@ def test_kimi3_latent_projection_add3_matches_torch_and_capt...; symbols: test_kimi3_latent_projection_writes_out_and_captures, test_kimi3_latent_projection_add3_matches_torch_and_captures, test_kimi3_rmsnorm_linear_add_matches_composed_and_captures
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py
@@ -62,13 +62,17 @@ def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) -> bool:
+        supports_gluon_mm_a16w16_decode_add3_gfx950,
+    def supports_gluon_mm_a16w16_decode_add3_gfx950(m: int, n: int, k: int) -> bool:
+        return False
@@ -813,9 +817,9 @@ def kimi3_latent_projection_add3(
-            for other one-token execution, the fused MFMA epilogue for the
-            tuned CDNA4 M=16 tile, and otherwise composes the registered projection
diff -- tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py
@@ -136,7 +136,7 @@ def test_kimi3_latent_projection_writes_out_and_captures(
-@pytest.mark.parametrize("num_tokens", [1, 2, 4, 16])
+@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 24, 32, 64])
@@ -164,7 +164,7 @@ def test_kimi3_latent_projection_add3_matches_torch_and_captures(
-@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4])
+@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 16, 17, 32, 64])
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/kimi3.py` modified +43/-27
  - tests: `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/amd/ops/gemm/test_gluon_a16w16_gfx950.py`, `tokenspeed-kernel/test/amd/ops/test_kimi3_projection_gfx950.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1992 - fix(ci): drop expert-placement flags from Kimi-K3 AMD job

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1992
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml`；关联提交 `4cbb277ef738`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+3/-4，可读 patch 21 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` modified +0/-2 (2 lines); hunks: -33,8 +33,6 @@ server:。
- 代码 diff 细节:
  - `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` modified +0/-2 (2 lines); hunks: -33,8 +33,6 @@ server:
- 关键代码摘录:

```diff
diff -- test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml
@@ -33,8 +33,6 @@ server:
-    --init-expert-location trivial
-    --ep-dispatch-algorithm static
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml` modified +0/-2
- 验证与风险: diff 自带测试面 `test/ci/perf/kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-random-50k-500-mi35x.yaml`, `test/ci_system/test_eval_configs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
