# GLM VLM/OCR PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: `glm4v.py`, `glm4v_moe.py`, `glm_ocr.py`, `glm_ocr_nextn.py`, GLM multimodal processors, GLM VLM/OCR docs/snippets.
- Searched PR terms: `GLM-4V`, `GLM-4.1V`, `GLM-4.5V`, `GLM-4.6V`, `GLM-OCR`, `GLM-Glyph`, `glm4v`, `glm_ocr`.

## Runtime Surfaces

- `python/sglang/srt/models/glm4v.py`
- `python/sglang/srt/models/glm4v_moe.py`
- `python/sglang/srt/models/glm_ocr.py`
- `python/sglang/srt/models/glm_ocr_nextn.py`
- `python/sglang/srt/multimodal/processors/glm4v.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-Glyph.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx`
- `docs_new/src/snippets/autoregressive/glm-45v-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-46v-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-glyph-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-ocr-deployment.jsx`

## Merged/Current-Main PRs

- [#8798](https://github.com/sgl-project/sglang/pull/8798): GLM-4.1V and GLM-4.5V support.
- [#9059](https://github.com/sgl-project/sglang/pull/9059): GLM-4.1V/4.5V vision transformer dummy-head TP fix.
- [#9245](https://github.com/sgl-project/sglang/pull/9245): GLM-4.5V default attention backend FA3.
- [#9554](https://github.com/sgl-project/sglang/pull/9554): GLM45V torch.compile launch bug.
- [#10228](https://github.com/sgl-project/sglang/pull/10228): `capture_aux_hidden_states` for GLM-4.5V.
- [#12117](https://github.com/sgl-project/sglang/pull/12117): GLM-4-0414 and GLM-4.1V refactor.
- [#14097](https://github.com/sgl-project/sglang/pull/14097): GLM-V vision model DP.
- [#14720](https://github.com/sgl-project/sglang/pull/14720): GLM-4.6V PP support.
- [#14927](https://github.com/sgl-project/sglang/pull/14927): nightly CI for `glm4v_moe`.
- [#14998](https://github.com/sgl-project/sglang/pull/14998): transformers version validation for GLM-4.6V MoE.
- [#17122](https://github.com/sgl-project/sglang/pull/17122): GLM-4V bugfix.
- [#17420](https://github.com/sgl-project/sglang/pull/17420): `get_rope_index` for GLM4V.
- [#17582](https://github.com/sgl-project/sglang/pull/17582): GLM-OCR support.
- [#18885](https://github.com/sgl-project/sglang/pull/18885): GLM-4V processor registration when `glm_ocr` is unavailable.
- [#20033](https://github.com/sgl-project/sglang/pull/20033): replace Conv3D projection with linear for GLM4V.
- [#20463](https://github.com/sgl-project/sglang/pull/20463): GLM-4.6V vision regression in `glm4v_moe`/`glm_ocr`.
- [#20740](https://github.com/sgl-project/sglang/pull/20740): revert GLM-4.6V vision regression.
- [#21134](https://github.com/sgl-project/sglang/pull/21134): GLM-V/OCR transformers 5.x field detection and MTP omission.

## Open PR Radar

- [#9349](https://github.com/sgl-project/sglang/pull/9349): GLM-4.5V FP8.
- [#14662](https://github.com/sgl-project/sglang/pull/14662): GLM4.6V ktransformers.
- [#19728](https://github.com/sgl-project/sglang/pull/19728): ROCm GLM-4.5V-FP8 startup.
- [#22961](https://github.com/sgl-project/sglang/pull/22961): NPU GLM-4.5V.

## Cookbook Evidence

- sgl-cookbook [#95](https://github.com/sgl-project/sgl-cookbook/pull/95): GLM-4.5V AMD MI300X/MI325X/MI355X support.
- sgl-cookbook [#131](https://github.com/sgl-project/sgl-cookbook/pull/131): MI325X support for GLM-4.5V and GLM-4.6V.
- sgl-cookbook [#136](https://github.com/sgl-project/sgl-cookbook/pull/136): GLM-OCR cookbook.

## Validation Notes

- GLM VLM/OCR paths are sensitive to transformers field drift; add compatibility guards instead of assuming one config version.
- OCR validation needs actual OCR examples, not only image captioning.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional PRs: [#9884](https://github.com/sgl-project/sglang/pull/9884), [#10147](https://github.com/sgl-project/sglang/pull/10147), [#11166](https://github.com/sgl-project/sglang/pull/11166), [#11388](https://github.com/sgl-project/sglang/pull/11388), [#11922](https://github.com/sgl-project/sglang/pull/11922), [#13228](https://github.com/sgl-project/sglang/pull/13228), [#15205](https://github.com/sgl-project/sglang/pull/15205), [#15434](https://github.com/sgl-project/sglang/pull/15434), [#20282](https://github.com/sgl-project/sglang/pull/20282), [#21134](https://github.com/sgl-project/sglang/pull/21134).

Official-doc flags to retain: FP8/BF16 GLM-4.5V/4.6V launch, `--keep-mm-feature-on-device`, `--mm-attention-backend`, `--mm-max-concurrent-calls`, `--mm-enable-dp-encoder`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, `SGLANG_VLM_CACHE_SIZE_MB=0`, and GLM thinking budget via the custom logit processor.
