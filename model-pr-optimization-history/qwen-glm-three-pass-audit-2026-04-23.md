# Qwen/GLM Three-Pass Completeness Audit - 2026-04-23

Scope: Qwen and GLM model optimization skills/history documents added for `AI-Infra-Auto-Driven-SKILLS`.

Evidence baseline:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Round 1: local SGLang docs, `docs_new`, model files, parsers, configs, and registered/manual tests.
- Round 2: GitHub PR search plus `git log --first-parent --merges` over model/runtime/parser/doc/test paths.
- Round 3: public SGLang/AMD/LMSYS documentation and optimization blog search.

## Public Sources Checked

- SGLang Qwen3-Next usage: `https://docs.sglang.io/docs/basic_usage/qwen3`
- SGLang Qwen3.5 usage: `https://docs.sglang.io/docs/basic_usage/qwen3_5`
- SGLang Qwen3-VL usage: `https://docs.sglang.io/docs/basic_usage/qwen3_vl`
- SGLang GLM-4.5/4.6/4.7 usage: `https://docs.sglang.io/docs/basic_usage/glm45`
- SGLang GLM-4.5V/4.6V usage: `https://docs.sglang.io/docs/basic_usage/glmv`
- SGLang DeepSeek V3.2/GLM-5 usage: `https://docs.sglang.io/docs/basic_usage/deepseek_v32`
- SGLang Cookbook index: `https://docs.sglang.io/cookbook/intro`
- AMD Qwen3.5 day-0 support: `https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html`
- LMSYS GLM-4.5 day-one blog: `https://www.lmsys.org/blog/2025-07-31-glm4-5/`
- LMSYS GLM4-MoE production optimization blog: `https://www.lmsys.org/blog/2026-01-21-novita-glm4/`
- LMSYS HiSparse blog for GLM-5.1/DSA: `https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/`
- LMSYS ModelOpt quantization blog: `https://www.lmsys.org/blog/2025-12-02-modelopt-quantization/`
- LMSYS SGLang Diffusion launch blog: `https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/`
- LMSYS SGLang Diffusion two-month update: `https://www.lmsys.org/blog/2026-01-16-sglang-diffusion/`

## Qwen3 Core Addendum

Early support and MoE runtime PRs that should remain visible in the history:

- `#4693`: initial Qwen3/Qwen3MoE support.
- `#5917`, `#6120`, `#6121`: Qwen3 EP/DeepEP/DP-attention MoE bring-up.
- `#6250`, `#6546`, `#6709`: PP/tied-weight support for Qwen2/Qwen3 and Qwen3 MoE.
- `#6533`, `#6818`, `#6964`, `#7580`, `#8448`: EPLB and expert-distribution path.
- `#6598`, `#6652`, `#8280`, `#9101`: two-batch-overlap, DP LM-head, DP attention padding and throughput improvements.
- `#6820`, `#7222`, `#7723`, `#7966`, `#8036`, `#8421`, `#8450`, `#8658`, `#8751`, `#9338`: fused MoE, FlashInfer, select-expert, and MoE argument evolution.
- `#6990`, `#7312`, `#7634`, `#7681`, `#7740`, `#7745`, `#8987`, `#10975`: embedding, LoRA hidden dim, layer-wise prefill, DP attention, two-stream norm, EAGLE3, hidden-dim helper, and memory-fraction heuristics.
- `#12162`, `#13730`, `#13998`, `#14093`, `#18255`, `#22003`: later shared MoE/attention/quantization topology work.

Open radar additions:

- `#20520`, `#21412`, `#21770`, `#22429`, `#22446`, `#22450`, `#22687`, `#23372`, `#23397`, `#23434`.

Extra test surfaces:

- `test/manual/test_qwen3_235b.py`
- `test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- NPU Qwen3 registered tests under `test/registered/npu/`

## Qwen3-Next Addendum

Initial and hybrid runtime PRs to retain:

- `#10233`: official initial Qwen3-Next support noted by SGLang docs.
- `#10322`, `#10379`, `#10392`, `#10466`, `#10622`: early norm/NPU/MTP+DP/FP8 DeepEP updates.
- `#11487`, `#11969`, `#12508`, `#12525`, `#13081`, `#13708`, `#14607`: hybrid runtime and Qwen3-Next support evolution.
- `#15631`: CuTe DSL GDN decode kernel.
- `#17373`: RadixLinearAttention path.
- `#17627`: NVFP4 quantized model path.
- `#18917`, `#19321`, `#19434`, `#21019`, `#21313`, `#21496`, `#21662`, `#22358`: GDN fusion, loader, DFLASH, and hybrid follow-ups.

Open radar additions:

- `#21684`, `#21698`, `#22876`, `#23075`, `#23273`, `#23474`.

Docs/flag evidence from official SGLang docs:

- `--max-mamba-cache-size`
- `--mamba-ssm-dtype`
- `--mamba-full-memory-ratio`
- `--mamba-scheduler-strategy extra_buffer`
- `--page-size 64`
- EAGLE/NEXTN speculative decoding flags
- `tool-call-parser qwen`, `reasoning-parser qwen3`
- long-context/YaRN caveat when extending beyond the native context length

## Qwen3.5 Addendum

Additional merged PRs:

- `#18926`: block-wise FP8 quantization/model adaptation.
- `#19391`: MTP spec-v2 NVFP4 test path.
- `#20864`, `#21487`, `#21849`, `#22145`, `#22240`, `#22431`, `#22493`, `#22908`, `#22948`: Qwen3.5-adjacent tests, platform enablement, docs, and follow-up fixes found in the second PR sweep.

Open radar additions:

- `#19585`, `#19781`, `#19956`, `#19974`, `#20029`, `#20255`, `#20370`, `#20448`, `#20667`, `#20789`, `#20918`, `#21185`, `#21217`, `#21464`, `#22027`, `#22325`, `#22502`, `#22867`, `#22885`, `#23062`, `#23468`, `#23471`.

Docs/blog evidence:

- Official docs describe hybrid GDN + full-attention architecture, shared experts, DeepStack Vision/Conv3d multimodality, `--attention-backend triton` on AMD, `SGLANG_USE_AITER=1`, and Qwen3 reasoning/tool parsers.
- AMD day-0 article adds the ROCm perspective: GDN via Triton, shared-expert MoE via hipBLASLt/AITER, and native multimodal kernels through MIOpen/PyTorch.

## Qwen3.6 Addendum

Qwen3.6 is currently a docs/config/shared-runtime lane:

- `#23034`: adds Qwen3.6 docs and updates Qwen3.5/GLM-5 links.
- `#23467`: FP8 `modules_to_not_convert` dot-boundary fix affecting Qwen3.6 loading/doc assumptions.
- `#23486`: Qwen3.6-27B dense cookbook docs.
- Keep following Qwen3-Next, Qwen3.5, Qwen VLM, and Qwen3-Coder parser PRs because Qwen3.6 reuses those runtime surfaces.

## Qwen3-Coder Addendum

Parser PRs that need explicit mention:

- `#8357`: XML-like grammar and Qwen3 detector bugfixes.
- `#8371`: streaming update.
- `#8445`: GLM-4.5 follow-up touching shared detector expectations.
- `#12226`: forward unknown tool calls instead of dropping them.
- `#13163`: remove EBNF Composer.
- `#16744`: add `qwen3_coder_detector`.

Runtime/cookbook radar remains tied to Qwen3-Next for Coder-Next MTP/GDN/cache behavior.

## Qwen VLM/Omni/ASR Addendum

Additional PRs found in the second sweep:

- `#12554`, `#12703`, `#15205`, `#16785`, `#19693`, `#21469`, `#21849`, `#22052`, `#22266`, `#22431`, `#22839`, `#22848`, `#23115`, `#23304`, `#23469`.

Extra test/doc surfaces:

- `test/manual/models/test_qwen3_asr.py`
- NPU Qwen3-VL/Qwen3-ASR registered tests under `test/registered/npu/`
- Official Qwen3-VL docs list FP8/BF16 launch paths, `--keep-mm-feature-on-device`, `--mm-max-concurrent-calls`, image and video request examples, and H200/B200 deployment notes.

## Qwen-Image Addendum

Code surfaces found by local search:

- `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/qwen_image.py`
- `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_qwen_image_pipeline.py`
- `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_qwen_image_edit_pipeline.py`
- `python/sglang/multimodal_gen/apps/webui/README.md`

Public blog evidence:

- SGLang Diffusion launch states Qwen-Image and Qwen-Image-Edit support plus OpenAI-compatible API, CLI, and Python interfaces.
- The two-month update adds Qwen-Image-Edit-2511, Qwen-Image-2512, Qwen-Image-Layered, GLM-Image, LoRA API coverage, and reports additional diffusion optimization progress.

## GLM-4.5 Addendum

Early GLM4-MoE support PRs:

- `#8224`, `#8456`, `#8647`, `#8729`, `#8804`, `#8883`, `#9136`, `#9223`, `#9264`, `#10008`, `#11017`, `#11665`, `#11692`, `#11800`, `#11847`, `#12162`, `#12572`, `#12834`, `#12957`, `#14668`.

Parser/tooling PRs:

- `#12456`, `#13989`, `#15333`, `#15753`, `#15754`, `#20543`.

Public docs/blog evidence:

- SGLang docs define GLM-4.5/4.6/4.7 launch, GLM-4.7 parser split, EAGLE flags, and thinking budget via `--enable-custom-logit-processor`.
- GLM-4.5 day-one LMSYS blog documents model shape, 128k context, native function calling, MTP, reasoning/coding/agentic positioning.

## GLM-4.6/4.7 Addendum

Additional merged PRs to keep visible:

- `#13989`, `#14668`, `#19246`, `#20543`, `#21135`, `#21660`, `#22509`, `#22720`, `#22823`.

Production-optimization blog evidence:

- Novita/LMSYS GLM4-MoE production blog ties GLM4.7-style MoE deployments to shared-expert fusion (`#13873`), QKNorm fusion, async transfer, suffix decoding, and EAGLE/speculative decoding configs.

## GLM-5/5.1 Addendum

Missing or under-emphasized PR/doc points:

- `#21405`: IndexCache for GLM-5/DeepSeek V3.2.
- `#21716`, `#22179`, `#22238`, `#22372`, `#19656`: docs/platform/optimization updates around GLM-5/DSA lane.
- `#21487`: GB300/Blackwell CI relevance when Qwen3.5 and GLM-5 share quant/test paths.

Docs/blog evidence:

- Official docs describe GLM-5 as DeepSeek V3.2/DSA-adjacent, with GLM-5 launched by swapping the model path, `glm47` tool parser, `glm45` reasoning parser, NSA backends, PD disaggregation, context parallel modes, FP8 KV caveats, and HiSparse guide linkage.
- HiSparse blog explicitly lists GLM-5.1 as a supported DSA family and shows `--enable-hisparse` plus `--hisparse-config` for high-concurrency long-context decode.

## GLM VLM/OCR Addendum

Additional PRs:

- `#9884`, `#10147`, `#11166`, `#11388`, `#11922`, `#13228`, `#15205`, `#15434`, `#20282`, `#21134`.

Docs evidence:

- Official GLM VLM docs list FP8/BF16 launch examples, `--keep-mm-feature-on-device`, `--mm-attention-backend`, `--mm-max-concurrent-calls`, `--mm-enable-dp-encoder`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, and thinking budget via the same GLM custom logit processor.
- GLM-Glyph should remain in this lane because it shares GLM multimodal/OCR processor and cookbook surfaces.
