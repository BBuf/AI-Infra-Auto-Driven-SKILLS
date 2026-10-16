# Pronunciation: English terms read right, in one breath

The user's bar: **every English word in the video is read correctly and fluently** (连贯, no 下划线, no
letter-by-letter where a word is meant). Captions always show the written form; only the text sent to the
voice changes.

## How it works

- `scripts/pronunciation.json` + `scripts/pronounce.py` turn each narration line into a spoken form before
  edge-tts reads it. `edge_tts_engine.py` maps the WordBoundary timings back onto the written tokens, so captions,
  karaoke and scene cues use the script's spelling (`flashmla_sparse_q8`), while the voice says
  `Flash MLA sparse Q8`. The engine log lists every rewrite (`· flashmla_sparse_q8→Flash MLA sparse Q8`).
- `scripts/pron_check.py --asr` reads each English term in a carrier sentence and transcribes it with
  mlx-whisper (large-v3-turbo, on device):
  - **OK**: an unbiased pass writes the term back. Case and spaces are ignored.
  - **OK~**: only a pass primed with the term gets it. It is plausible; listen once on the render.
  - **FAIL**: neither pass gets it. The tool then tries respellings and prints the first that passes.

  Verdicts are cached per voice and spoken form in `$SEV_WS/pron_cache.json`.

## Rules for writing narration

1. **Never put `_` in front of the voice.** snake_case and SCREAMING_CASE are read as 下划线. Writing them in
   the narration is fine: the identifier rule splits them and maps known parts (SGLANG_DSA_FUSE_TOPK →
   SGLang DSA fuse top k). A part that is not a word the voice knows needs a `parts` entry.
2. **CamelCase brand names that run words together get spaces:** FlashInfer → Flash Infer, DeepSeek → Deep Seek,
   TensorRT-LLM → Tensor RT LLM. Unspaced, the voice blurs them ("Flesh and Fur", "Deep Seat").
3. **All-caps words that are English words stay all-caps** (PAGED, EXTEND, DECODE). Xiaoxiao reads them as
   words, and lower-case `ragged` came out worse. Acronyms are spelled by the voice anyway (DSA, KV, MTP).
4. **Acronyms a listener says as letters but the voice says as a word get spaced letters:** VAE → V A E
   (otherwise just "V"), DiT → D I T ("Date"), GEMM → G E M M.
5. **Symbols:** `×` after a number is 倍 (3.7× → 3.7倍, 8×B300 → 8卡B300); `#42565` drops the `#` (otherwise 井号);
   `+5%` / `−12%` → 提升5% / 降低12% (otherwise 正 5%); **a negative number → 负1** — the voice
   silently drops the minus of `-1` ("种子 5、1、-1" was read 5、1、1). The `patterns` rules do this, but prefer writing
   Chinese in the narration in the first place.
6. **Versions:** V2 → V two (二 sounds like R). Model numbers stay as written (GLM-5.3, H200, SM90).
7. **A generic verb the voice can't say:** write it in Chinese instead of respelling it. For example, use 去噪
   for denoise ("Denice" in every spelling). Keep the English term on screen and in code panels.
8. `「」` never goes in narration (captions drop the opening bracket). Use plain words or 引号 on screen only.
9. **No space between two Han characters.** The voice pauses about 0.3 s there; `pronounce.py` drops such spaces
   in the spoken text. Spaces between Han and Latin (`这期 SGLang PR 讲解`) are fine and keep captions readable.
   `pron_check.py --meta` reports any pause > 0.2 s that is not at punctuation.

## Measured on zh-CN-XiaoxiaoNeural (Oct 2026)

| Written | Spoken (lexicon) | Whisper heard the written form as |
| --- | --- | --- |
| top-k | top k | Talk K (in a sentence) → **TopK** |
| -1 | 负1 | 一 (minus dropped) → **负一** |
| FlashInfer | Flash Infer | Flesh and Fur → **Flash Infer** |
| DeepSeek | Deep Seek | Deep Seat → **Deep Seek** |
| TensorRT-LLM | Tensor RT LLM | Tensorite TileLam → **Tensor RTLLM** |
| LoRA | Lora | Lauren → **Laura** |
| Miles | mials | Mouse → **Miles** |
| VAE / DiT / GEMM | V A E / D I T / G E M M | V / Date / (dropped) → **VAE / DIT / GEMM** |
| DRAFT_EXTEND_V2 | draft extend V two | Draft下划线Extend下划线V2 → **Draft Extend V2** |

These read right as written: PR, DSA, MLA, MTP, MoE, KV cache, fused, prefill, decode, batch, token,
kernel, attention, indexer, backend, CUDA graph, FP8/FP4/NVFP4, TP8, H200/B200/GB300, Triton, PyTorch,
torch.compile, vLLM, NVIDIA, Blackwell, Hopper, RoPE, RMSNorm, softmax, Mamba, Ulysses, GLM-5.3, W4AFP8.

**Approved by ear** (`approved` in pronunciation.json — never respelled, reported OK): **SGLang**. The user
compared the voice's own reading with SG Laang / SG Lahng / SG Leng / S G Lang and kept the original
(2026-10-16). Whisper writes it as SGLine / SGLAM; that is an ASR artifact, so ignore it.
**MTP** is approved too: Qwen3-ASR transcribes Xiaoxiao's MTP correctly, while Whisper writes MPP for every
spelling, even Em Tee Pee.

Still **OK~** with no respelling that passes; listen on the render: EAGLE ("Ego"), RAGGED ("Reddit"),
Kimi ("Timmy"). If one sounds wrong, reword the line, e.g. "EAGLE 投机解码" → "投机解码（EAGLE）" on
screen only.

The free Edge endpoint rejects inline SSML (`<phoneme>`, `<sub>`, `<lang>`, even `<break>` → NoAudioReceived),
so respelling is the only lever.

## Adding an entry

When Whisper can't decide (no spelling passes, or the term matters as much as the brand name), render a
short A/B of the candidate spellings in one sentence and let the user pick by ear. Put the result in
`approved` (keep the voice's reading) or `lexicon` (respell).

1. Run `pron_check.py --text "<line>" --asr`. A FAIL prints `try lexicon "X": "..."` when a respelling passes.
2. Add it to `lexicon` (whole token, case-sensitive) or `parts` (inside identifiers, lower-case key).
3. Re-run `pron_check`. Commit the entry with the before/after Whisper text in the table above.
