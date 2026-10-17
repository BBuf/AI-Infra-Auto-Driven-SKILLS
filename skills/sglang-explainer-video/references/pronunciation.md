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
3. **All-caps words that are English words mostly stay all-caps** (PAGED, EXTEND, DECODE): Xiaoxiao reads them
   as words. Exceptions picked by the user's ear have lexicon entries (RAGGED → Ragged, EAGLE → Eagle).
   Acronyms are spelled by the voice anyway (DSA, KV, MTP).
   **A hyphen between two letters becomes a space** in the spoken form (bit-exact → bit exact, top-k → top k):
   the hyphen blurs the second word. Lexicon lookups also fall back to case-insensitive (flashinfer).
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
| top-k | top k | Talk K in a sentence (the carrier alone passes) → **TopK** |
| -1 | 负1 | 一 (minus dropped) → **负一** |
| bit-exact (any letter-letter hyphen) | bit exact | BPZ → **BitExact** |
| 8GB (any number + GB/MB/TB/KB) | 8 G B | 8G → **8GB** |
| SANA | Sana | Sun Eye → **Sanai** |
| epilogue / launch | eppy log / lawnch | Apple Log / Lounge → **epilog / Launch** |
| denoise | 去噪 (Chinese: no spelling passes) | Denice / Denis → **去噪** |
| DeepSelect | Deep Select | DickSelect (in a sentence) → **Deep Select** |
| hmax2 / ctas (identifier part) | h max 2 / CTAs | max2 / Tasked → **Hmax2 / CTS** |
| **Qwen** (also inside Qwen3-8B, Qwen-Image) | Kwen (`patterns`) | **user's ear, 2026-10-17** (Whisper writes QA3 for every spelling) |
| **EAGLE** | Eagle | **user's ear, 2026-10-17** |
| **RAGGED** | Ragged | **user's ear, 2026-10-17** |
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
spelling, even Em Tee Pee. Also approved: **Kimi** (user's ear), **PR** (letters, R as 阿尔, the usual Chinese
reading), **dB** (read 分贝).

Still **OK~** with no respelling that passes; listen on the render: Claude ("Cloud" — every respelling was
worse), Lean ("Lin"), README ("Remi"), TopK ("Talk K"), FLUX.2 (fine in most sentences).
**FAIL with no passing respelling → reword the narration, keep the term on screen** (measured 2026-10-17, video
「改一行，为什么能快几倍？」):

| Written | Whisper heard | Narration says instead |
| --- | --- | --- |
| dstate (Mamba2 state dim) | State | 状态维度 |
| ncu | 看 | Nsight Compute (passes: N-Sight Compute) |
| swizzle | Swizo | 交错排列的内存布局 |

Also OK~ in that video, listen on the render: warp / warps ("World" / "Works"), grid ("Grip"), tile ("他").

In-context checks of long lines are noisy: re-probe a FAIL in a short sentence before changing the lexicon. In
the 13-video re-voice, most FAILs read right in the short sentence. If one sounds wrong, reword the line, e.g. "EAGLE 投机解码" → "投机解码（EAGLE）" on
screen only.

The free Edge endpoint rejects inline SSML (`<phoneme>`, `<sub>`, `<lang>`, even `<break>` → NoAudioReceived),
so respelling is the only lever.

## The lexicon is a living file — every video teaches it something

This file and `scripts/pronunciation.json` are **never finished**. Every new video is also a pronunciation
run. Fold any term that was misread into the skill, so the next video gets it right the first time. That
covers terms found by `pron_check` (FAIL, or an OK~ that sounded wrong) and terms the user flags in review
("这个词读错了", "这里停顿了"). This is a required step of every video, not an optional cleanup
(SKILL.md §4 and §9).

For each misread term, before the video is delivered:

1. **Find the fix.**
   - Use `pron_check.py --text "<the real line>" --asr`, plus a sentence-context probe if the carrier
     passes but the line fails (top-k did).
   - When ASR can't decide, make a short A/B for the user's ear (see below).
2. **Write it into `pronunciation.json`, in the most general form that is still correct:**
   - a `patterns` rule for a class of symbols (`×`, `#`, negative numbers);
   - a `parts` entry for a word inside identifiers;
   - a `lexicon` entry for one token;
   - `approved` when the voice's own reading is right and only Whisper mishears it.
3. **Record the evidence.** Add a row to the "Measured" table above: written → spoken, what was heard before
   and after, and the date or video. If the rule changes how text is cut, add a case to
   `tests/test_explainer_pronounce.py`.
4. **Re-run, then ship the change upstream.**
   - Re-run `pron_check` on the video, then `uvx pytest tests/test_explainer_pronounce.py`.
   - Open a PR to the skill repo with the entries; one small PR per video is fine.
   - The video's own fix stays in the project only until that PR merges.

Keep it healthy:

- **Run `pron_check.py --audit`** when the voice changes, when edge-tts or the Edge endpoint changes, or
  every few videos. Every respelling must still pass.
- **Drop an entry only after its written form also reads right in a real sentence.** A pass in the
  carrier alone is not enough.
- **Never delete measurements from the table.** Mark a superseded row instead.

## Adding an entry

When Whisper can't decide (no spelling passes, or the term matters as much as the brand name), render a
short A/B of the candidate spellings in one sentence and let the user pick by ear. Put the result in
`approved` (keep the voice's reading) or `lexicon` (respell).

1. Run `pron_check.py --text "<line>" --asr`. A FAIL prints `try lexicon "X": "..."` when a respelling passes.
2. Add it to `lexicon` (whole token, case-sensitive) or `parts` (inside identifiers, lower-case key).
3. Re-run `pron_check`. Commit the entry with the before/after Whisper text in the table above.
