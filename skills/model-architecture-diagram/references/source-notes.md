# Model Architecture Diagram Source Notes

Audited on `2026-04-24`.

## Upstream Diagram Sources

- `datawhalechina/self-llm` at `a7cd4ef135b0` (`master`): broad model deployment/tutorial repository. Most images are screenshots; confirmed architecture-style diagrams include Hunyuan-A13B and Kimi-VL.
- `CalvinXKY/InfraTech` at `0fe7d3a57dce` (`main`): architecture-card repository with original diagrams for DeepSeek V3/V3.2, GLM-5, Kimi K2/K2.5, MiniMax M2.5, Qwen3.5, Qwen3-VL, and Step 3.5 Flash.

The skill stores raw GitHub URLs in `diagram-index.json`; it intentionally does not vendor binary images.

## Local Code Sources To Inspect

When a generated diagram needs higher fidelity, inspect these local paths first:

- SGLang model code: `/Users/bbuf/工作目录/Common/sglang/python/sglang/srt/models`
- SGLang multimodal/diffusion code: `/Users/bbuf/工作目录/Common/sglang/python/sglang/multimodal_gen`
- sgl-cookbook docs: `/Users/bbuf/工作目录/Common/sgl-cookbook/docs`
- sgl-cookbook model YAML: `/Users/bbuf/工作目录/Common/sgl-cookbook/data/models/src`

## sgl-cookbook Coverage Map

Use this map to choose the fallback template before refining from code:

| Cookbook family | Representative docs | Default template | Primary SGLang code surfaces |
| --- | --- | --- | --- |
| DeepSeek V3/R1/V3.1 | `docs/autoregressive/DeepSeek/DeepSeek-V3.md`, `DeepSeek-R1.md`, `DeepSeek-V3_1.md` | `mla_moe` | `deepseek_v2.py`, `deepseek_nextn.py`, `deepseek_common/` |
| DeepSeek V3.2 | `docs/autoregressive/DeepSeek/DeepSeek-V3_2.md` | `dsa_moe` | `deepseek_v2.py`, `nsa_backend.py`, `nsa_indexer.py` |
| DeepSeek OCR/OCR-2 | `docs/autoregressive/DeepSeek/DeepSeek-OCR*.md` | `vlm` | `deepseek_ocr.py`, OCR processors |
| GLM 4.5/4.6/4.7 | `docs/autoregressive/GLM/GLM-4*.md` | `decoder_moe` or `vlm` for V models | `glm4_moe.py`, `glm4v*.py` |
| GLM 5/5.1 | `docs/autoregressive/GLM/GLM-5*.md` | `dsa_moe` | `glm4_moe.py`, `deepseek_v2.py`, NSA files |
| GLM OCR/Glyph | `docs/autoregressive/GLM/GLM-OCR.md`, `Glyph.md` | `vlm` for OCR, `decoder_moe` for Glyph | `glm_ocr.py`, `glm_ocr_nextn.py`, `glm4_moe.py` |
| Qwen3/Qwen3 Coder | `docs/autoregressive/Qwen/Qwen3*.md` | `decoder_moe` | `qwen3.py`, `qwen3_moe.py`, `qwen3_classification.py` |
| Qwen3 Next / Coder Next / Qwen3.5 / Qwen3.6 | Qwen Next/3.5/3.6 docs | `hybrid_delta_moe` | `qwen3_next.py`, `qwen3_5.py`, MTP variants |
| Qwen VLM/Omni/ASR | `Qwen2.5-VL.md`, `Qwen3-VL.md` | `vlm` | `qwen2_5_vl.py`, `qwen3_vl.py`, `qwen3_vl_moe.py`, `qwen3_omni_moe.py`, `qwen3_asr.py` |
| Kimi K2/K2.5/K2.6 | Moonshotai Kimi docs | existing K2/K2.5 diagrams, `moonvit_mla_moe` for K2.6 | `kimi_k25.py`, `kimi_vl.py`, `kimi_vl_moonvit.py` |
| Kimi Linear | `Kimi-Linear.md` | `hybrid_delta_moe` | `kimi_linear.py` |
| MiniMax M2/M2.5/M2.7 | MiniMax docs | existing M2.5 diagram, `decoder_moe` for M2/M2.7 | `minimax_m2.py` |
| Llama 3.x / Llama 4 | Llama docs | `decoder_dense` or `vlm`/`decoder_moe` for Llama 4 | `llama.py`, `llama4.py`, `mllama4.py` |
| Gemma 4 | `docs/autoregressive/Google/Gemma4.md` | `vlm` or `decoder_moe` | `gemma4_*.py` |
| GPT-OSS | `docs/autoregressive/OpenAI/GPT-OSS.md` | `decoder_moe` | `gpt_oss.py` |
| Mistral/Devstral/Ministral | Mistral docs | `decoder_dense` or `vlm` for multimodal variants | `mistral*.py`, `ministral3.py`, `pixtral.py` |
| Nemotron | NVIDIA docs | `decoder_moe` / hybrid | `nemotron_h.py`, `nemotron_h_mtp.py`, `nano_nemotron_vl.py` |
| Ernie 4.5 | Ernie docs | `decoder_moe` / `vlm` | `ernie4.py`, `ernie45_vl.py`, `ernie45_moe_vl.py` |
| InclusionAI Ling/Ring | InclusionAI docs | `hybrid_delta_moe` | local Transformers or related hybrid model path |
| Intern-S1 / InternVL | InternLM/InternVL docs | `vlm` | `interns1.py`, `internvl.py` |
| Jina reranker | Jina docs | `reranker` | `bert.py`, `roberta.py`, pooled-output wrappers |
| LLaDA | `docs/autoregressive/LLaDA/LLaDA-2.1.md` | `llada` | `llada2.py` |
| Step 3.5 / Step3-VL | StepFun docs | `decoder_moe` / `vlm` | `step3p5.py`, `step3_vl*.py` |
| MiMo | Xiaomi docs | `swa_moe_mtp` | `mimo.py`, `mimo_mtp.py`, `mimo_v2_flash.py` |
| FLUX / Qwen-Image / Z-Image | diffusion docs | `diffusion_dit` | `multimodal_gen/configs/pipeline_configs/`, `runtime/models/dits/` |
| MOVA | `docs/diffusion/MOVA/MOVA.md` | `mova_bimodal_dit` | `multimodal_gen/configs/pipeline_configs/`, video/audio DiT runtime |
| Chroma | `docs/autoregressive/FlashLabs/Chroma1.0.md` | `audio_hybrid_serving` | external `Chroma-SGLang` server plus SGLang inner engine |
| Wan 2.1/2.2 | Wan diffusion docs | `diffusion_moe_video` for Wan2.2 | Wan pipeline configs and video DiT runtime |
| S2-Pro omni | `docs/omni/S2-Pro/S2-Pro.md` | `tts_dual_ar` | `sglang-omni` S2 Pro stack |

## Style Notes

Use the visual language of the source repositories:

- Top-down model flow: inputs, embedding/encoder, repeated block, norm/head/output.
- Color-code attention, state-space/linear-attention, MoE/FFN, and modality bridges.
- Show repeated-layer counts as compact labels instead of drawing every layer.
- For VLMs, draw a side branch for vision/audio encoder and projector before the LLM stack.
- For diffusion, draw prompt encoder, latent/noise path, repeated DiT denoiser, scheduler, VAE decode, and output media.
- Keep labels short enough to render in a small chat image preview.
