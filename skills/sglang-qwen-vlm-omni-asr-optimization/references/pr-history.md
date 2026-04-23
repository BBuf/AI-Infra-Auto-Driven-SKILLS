# Qwen VLM/Omni/ASR PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: Qwen VLM/Omni/ASR model files, multimodal processors, docs/cookbook snippets, VLM tests.
- Searched PR terms: `Qwen3-VL`, `Qwen3 VL`, `Qwen3-Omni`, `Qwen3-ASR`, `Qwen2.5-VL`, `qwen_vl`, `qwen3_vl`, `qwen3_asr`.

## Runtime Surfaces

- `python/sglang/srt/models/qwen2_5_vl.py`
- `python/sglang/srt/models/qwen3_vl.py`
- `python/sglang/srt/models/qwen3_vl_moe.py`
- `python/sglang/srt/models/qwen3_omni_moe.py`
- `python/sglang/srt/models/qwen3_asr.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `python/sglang/srt/multimodal/processors/qwen_audio.py`
- `python/sglang/srt/multimodal/processors/qwen3_asr.py`
- `docs/basic_usage/qwen3_vl.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen2.5-VL.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-VL.mdx`

## Merged/Current-Main PRs

- [#10911](https://github.com/sgl-project/sglang/pull/10911): Qwen3-Omni thinker-only support.
- [#10985](https://github.com/sgl-project/sglang/pull/10985): Qwen3-VL launch fix for MRotaryEmbedding args.
- [#12333](https://github.com/sgl-project/sglang/pull/12333): PP support for Qwen3-VL.
- [#13724](https://github.com/sgl-project/sglang/pull/13724): Qwen3-VL vision model DP.
- [#13736](https://github.com/sgl-project/sglang/pull/13736): replace `repeat_interleave` with `np.repeat` for Qwen-VL.
- [#14292](https://github.com/sgl-project/sglang/pull/14292): position embedding cache for Qwen-VL.
- [#14907](https://github.com/sgl-project/sglang/pull/14907): chunked ViT attention.
- [#15320](https://github.com/sgl-project/sglang/pull/15320): ViT PCG for Qwen3-VL.
- [#16366](https://github.com/sgl-project/sglang/pull/16366): Qwen3-VL video memory optimization.
- [#17624](https://github.com/sgl-project/sglang/pull/17624): DP size greater than 1 for Qwen3-VL.
- [#18024](https://github.com/sgl-project/sglang/pull/18024): Qwen3-VL weight-loading prefix mapping.
- [#18185](https://github.com/sgl-project/sglang/pull/18185): Qwen3-Omni audio encoder optimization.
- [#19003](https://github.com/sgl-project/sglang/pull/19003): FlashInfer CUDNN prefill as ViT backend.
- [#19291](https://github.com/sgl-project/sglang/pull/19291): missing `quant_config` in Qwen3-VL.
- [#19333](https://github.com/sgl-project/sglang/pull/19333): Qwen3-VL visual loading.
- [#20759](https://github.com/sgl-project/sglang/pull/20759): Qwen3-VL hang with `mm-enable-dp-encoder`.
- [#20788](https://github.com/sgl-project/sglang/pull/20788): DP encoder position-embedding layer TP issue.
- [#21458](https://github.com/sgl-project/sglang/pull/21458): AMD Qwen3-VL decode fusion for QK-norm, 3D mRoPE, and KV store.
- [#22038](https://github.com/sgl-project/sglang/pull/22038): chunk-aware ViT encoding cache and lazy transfer.
- [#22073](https://github.com/sgl-project/sglang/pull/22073): Qwen3-ASR support.
- [#22089](https://github.com/sgl-project/sglang/pull/22089): chunk-based streaming ASR.
- [#22230](https://github.com/sgl-project/sglang/pull/22230): EAGLE3 for Qwen3-VL.

## Open PR Radar

- [#12662](https://github.com/sgl-project/sglang/pull/12662): CPU support for Qwen3-VL and Qwen3-Omni.
- [#12261](https://github.com/sgl-project/sglang/pull/12261): Qwen multimodal open support track.
- [#13918](https://github.com/sgl-project/sglang/pull/13918): Qwen VLM follow-up.
- [#14886](https://github.com/sgl-project/sglang/pull/14886): Qwen VLM follow-up.
- [#16491](https://github.com/sgl-project/sglang/pull/16491): Qwen VLM follow-up.
- [#16571](https://github.com/sgl-project/sglang/pull/16571): Qwen VLM follow-up.
- [#16996](https://github.com/sgl-project/sglang/pull/16996): Qwen VLM follow-up.
- [#17202](https://github.com/sgl-project/sglang/pull/17202): Qwen VLM follow-up.
- [#17276](https://github.com/sgl-project/sglang/pull/17276): Qwen VLM follow-up.
- [#18721](https://github.com/sgl-project/sglang/pull/18721): Qwen multimodal follow-up.
- [#18771](https://github.com/sgl-project/sglang/pull/18771): Qwen multimodal follow-up.
- [#19242](https://github.com/sgl-project/sglang/pull/19242): Qwen multimodal follow-up.
- [#19693](https://github.com/sgl-project/sglang/pull/19693): Qwen multimodal follow-up.
- [#20857](https://github.com/sgl-project/sglang/pull/20857): Qwen VLM/audio follow-up.
- [#22052](https://github.com/sgl-project/sglang/pull/22052): Qwen VLM/ASR follow-up.
- [#22848](https://github.com/sgl-project/sglang/pull/22848): Qwen multimodal follow-up.
- [#23220](https://github.com/sgl-project/sglang/pull/23220): Qwen3-VL-MoE encoder-only adaptation.
- [#23469](https://github.com/sgl-project/sglang/pull/23469): NPU Qwen3-ASR adaptation.

## Cookbook Evidence

- sgl-cookbook [#76](https://github.com/sgl-project/sgl-cookbook/pull/76): Qwen3-VL AMD MI300X config generator.
- sgl-cookbook [#84](https://github.com/sgl-project/sgl-cookbook/pull/84): Qwen2.5-VL AMD MI300X deployment guide.
- sgl-cookbook [#102](https://github.com/sgl-project/sgl-cookbook/pull/102): MI355X support for Qwen3-VL.
- sgl-cookbook [#110](https://github.com/sgl-project/sgl-cookbook/pull/110): Qwen2.5-VL MI355X/MI325X AMD support.
- sgl-cookbook [#124](https://github.com/sgl-project/sgl-cookbook/pull/124): Qwen3-VL AMD MI325X support.

## Validation Notes

- Test media processor output before profiling ViT kernels.
- Encoder DP/PP fixes require both DP and no-DP comparisons.
- ASR streaming must validate chunk boundaries and final transcript accumulation.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional PRs from the second sweep: [#12554](https://github.com/sgl-project/sglang/pull/12554), [#12703](https://github.com/sgl-project/sglang/pull/12703), [#15205](https://github.com/sgl-project/sglang/pull/15205), [#16785](https://github.com/sgl-project/sglang/pull/16785), [#19693](https://github.com/sgl-project/sglang/pull/19693), [#21469](https://github.com/sgl-project/sglang/pull/21469), [#21849](https://github.com/sgl-project/sglang/pull/21849), [#22052](https://github.com/sgl-project/sglang/pull/22052), [#22266](https://github.com/sgl-project/sglang/pull/22266), [#22431](https://github.com/sgl-project/sglang/pull/22431), [#22839](https://github.com/sgl-project/sglang/pull/22839), [#22848](https://github.com/sgl-project/sglang/pull/22848), [#23115](https://github.com/sgl-project/sglang/pull/23115), [#23304](https://github.com/sgl-project/sglang/pull/23304), [#23469](https://github.com/sgl-project/sglang/pull/23469).

Extra validation surfaces: `test/manual/models/test_qwen3_asr.py` and Qwen3-VL/Qwen3-ASR NPU registered tests. Official Qwen3-VL docs also require the skill to keep FP8/BF16 launch, `--keep-mm-feature-on-device`, image/video request examples, and `--mm-max-concurrent-calls` in view.
