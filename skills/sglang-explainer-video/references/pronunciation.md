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
  The carrier reads the term's full spoken form, `patterns` included (24GB → 24 G B), so a pattern-handled
  term is no longer reported as a FAIL because Whisper heard the raw token (it used to print `24GB … heard 24G`).
- `scripts/poly_probe.py` handles what Whisper can't hear: the reading of a Chinese polyphone (行 háng vs xíng).
  It synthesizes several takes and compares each one with unambiguous homophones of each reading (DTW over
  log-mel). It reports the misread rate as written and after the rewrite. See §Chinese polyphones.

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
10. **A comma between two numbers is read as a thousands separator.** In `间距是 2，257 会舍入成 256`, Edge returned
    `2，257` as one word and read it as 两千二百五十七 (3/3 takes; 2-CTA video, flagged by the user). A pattern now
    adds a space after a full-width `，` or `、` between digits (`2， 257`), which splits it into 2 / 257 with a
    0.35 s pause. Captions still show `2，257`. Older projects had the same merge (`4 乘 128，512`, `12、12、8个`).
11. **调 meaning "tune" is read diào when it stands alone.** `可以调 block 大小` came out as diào in 6/6 takes
    (poly_probe). Write a compound, such as 调整 (0/6 wrong), 调优 or 调参. 调用 and 调度 are diào and need nothing.

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
| AIME (AIME25, AIME 2025) | Aime | M → **AIM** (short sentence; in a long line Whisper still writes M — read as "aim", the usual reading) |
| GiB / KiB / MiB (binary units) | G B / K B / M B | **user's ear, 2026-10-18**: say them like GB / KB, not letter by letter (G I B was rejected). Whisper hears GB / KB, so `pron_check` reports KiB as a FAIL against the written form; that is expected (MiB → M B added in the 2-CTA video, 2026-10-21, same FAIL; TiB → T B added in the KV cache video, 2026-10-22, Whisper writes TB) |
| checkpoint | check point | ChatPoint (every carrier, also lower/capitalised) → **checkpoint** (24GB video, 2026-10-19) |
| OOM | O O M | Om (read as a word) → **OOM** |
| DSMEM | D S MEM | SMAM → **D S MEM** (2-CTA video, 2026-10-21) |
| tcgen05 | T C Gen 05 | 看05 / K-05 班班杯 → **T C Gen 05**. Do not say the dotted forms (tcgen05.mma / .commit / .ld / .wait / .dealloc): write 「tcgen05 的 MMA / commit / load / wait / dealloc 指令」 and keep the PTX on screen |
| mbarrier | M barrier | Barrier (the m is dropped) → **M barrier** |
| ptxas | P T X A S | Sys → **P T X A S** |
| ncu | N C U | 看 → **N C U** (supersedes the Nsight Compute reword below) |
| PipelineUmmaAsync | Pipeline U MMA Async | PipelineNum Async → **Pipeline U MMA Async** |
| comfy-kitchen (the comfy_kitchen INT8 library) | Comfy Kitchen | Tichen for a bare kitchen / Kitchen → **Comfy Kitchen**. In narration write the pair "Comfy Kitchen"; the bare word alone still fails |
| fa (the FlashAttention backend name) | F A | In context the voice said "fa" as a syllable (Whisper: Fall / 发 / 法 / Ford; the carrier's "FAR" was a false OK) → **FA** |
| **MoE** (also DeepSeekMoE, MegaMoE, fused_moe) | M O E (lexicon + `parts`) | **user's ear, 2026-10-18**: the voice said "Mo" (the English name *Moe*). Spelled letters come back as MOE |
| VAE / DiT / GEMM | V A E / D I T / G E M M | V / Date / (dropped) → **VAE / DIT / GEMM** |
| DRAFT_EXTEND_V2 | draft extend V two | Draft下划线Extend下划线V2 → **Draft Extend V2** |
| **MLA** (Multi-head Latent Attention) | M L A | MBA in every recorded line. A/B on 「这 24 层 MLA 完全不加位置编码」: bare form heard MBA 3/3, spaced form heard MLA 3/3 → the voice blurs the L (K3 video, 2026-10-19) |
| SiTU (Sigmoid Tanh Unit) | S I T U | C2 / SITU → **SITU** |
| tanh | tan h | 10 → **TangentH / 10H** (OK~; Edge reads "tan" as tangent — accepted) |
| kimi_k3.py | Kimi K3 点 P Y | Timmy K3PY → spelled file name |
| DeepInfer (also inside DEEPINFER_* env names via `parts`) | Deep Infer | Deeping Fur → **DeepInfer** (DeepInfer video, 2026-10-20) |
| deepseek-recipe | Deep Seek recipe | Deep Seed Recipe → **deepseek recipe** |
| DeepEP | Deep E P | DPP, also for "Deep EP" → **Deep EP** |
| DeepGEMM | Deep G E M M | Deep G1 Mbam → **DeepGEMM** |
| CANN (Huawei Ascend toolkit) | C A N N | 看 → **CANN** |
| CED (Causal Encoder-Decoder) | C E D | Set → **CED** |
| xgrammar | X grammar | Squadmire → **X grammar** |
| TileKernels · SamplingParams | Tile Kernels · Sampling Params | CamelCase rule 2 (Telkernos / samplingperms → OK) |
| DSPARK (speculative algorithm flag value, DSV4.1 + K3 sharing video 2026-10-23) | DSpark | Spark (the all-caps D is dropped) → **D Spark** |

These read right as written: PR, DSA, MLA, MTP, KV cache, fused, prefill, decode, batch, token,
kernel, attention, indexer, backend, CUDA graph, FP8/FP4/NVFP4, TP8, H200/B200/GB300, Triton, PyTorch,
torch.compile, vLLM, NVIDIA, Blackwell, Hopper, RoPE, RMSNorm, softmax, Mamba, Ulysses, GLM-5.3, W4AFP8.

**Approved by ear** (`approved` in pronunciation.json — never respelled, reported OK): **SGLang**. The user
compared the voice's own reading with SG Laang / SG Lahng / SG Leng / S G Lang and kept the original
(2026-10-16). Whisper writes it as SGLine / SGLAM; that is an ASR artifact, so ignore it.
**MTP** is approved too: Qwen3-ASR transcribes Xiaoxiao's MTP correctly, while Whisper writes MPP for every
spelling, even Em Tee Pee. Also approved: **Kimi** (user's ear), **PR** (letters, R as 阿尔, the usual Chinese
reading), **dB** (read 分贝).

Still **OK~** with no respelling that passes; listen on the render: Claude ("Cloud" — every respelling was
worse), Lean ("Lin"), README (spoken "read me" since the DeepInfer video; Whisper: Remi → REME), TopK ("Talk K"), FLUX.2 (fine in most sentences).
**FAIL with no passing respelling → reword the narration, keep the term on screen** (measured 2026-10-17, video
「改一行，为什么能快几倍？」):

| Written | Whisper heard | Narration says instead |
| --- | --- | --- |
| dstate (Mamba2 state dim) | State | 状态维度 |
| ncu | 看 | Nsight Compute (passes: N-Sight Compute) — superseded 2026-10-21: the lexicon spelling N C U passes |
| swizzle | Swizo | 交错排列的内存布局 |
| scatter (Beam video, 2026-10-17) | Statter | 「先 gather，算完再分发回去」 |
| key (indexer key, DSV4.1 video 2026-10-18) | T (in every carrier, also 索引 key) | 索引键 / 读键的带宽 |
| int mm (torch._int_mm, 24GB video 2026-10-19) | MBM | PyTorch 自带的 INT8 矩阵乘接口 (torch._int_mm stays on screen) |
| INT32 (accumulator) | 印32 (OK~ in every spelling) | 32 位整数累加 |
| key (as in 128 个 key 通道, K3 video 2026-10-19) | T (again) | k 的 128 个通道 |
| e (as in e 的负 80 次方) | 一 | 累积衰减的对数最多到负 80 (keep e<sup>−80</sup> on screen) |
| tok/s | Talk | 每秒 N 个 token |
| copy on write | on → 嗯 | 写时复制 (the English stays on screen) |
| pybind11 (DeepInfer video, 2026-10-20) | PBind11 | 绑定到 Python |
| asc (the Ascend op backend dir) | VSC | 昇腾版 |
| thinking (模式) | finting | 思考模式 |
| prev_step_state · seq_seed · ZOS · YoCo (identifiers / code names) | previous step state · 6C · GOS · Yuko | 上一步的状态 · 随机种子 · zero-overhead scheduling · YOCO (identifiers stay on screen) |
| R1 (as in V3 和 R1) | 21 / 二 | drop it or say DeepSeek V3 |
| thread (thread block / thread block cluster, 2-CTA video 2026-10-21) | Fred (capitalised Thread: still Fred) | 线程块 / 线程块集群 (the English stays on screen) |
| key (attention key, KV cache video 2026-10-22) | T | say K 和 V (letters read fine); keep query / key / value on screen |
| tree (radix tree) | Country | 前缀树 in the narration; RadixAttention / HiRadixTree on screen |
| phase (mbarrier / pipeline phase) | Face (also capitalised) | 相位 |
| baseline | Bassline | 基线 |
| wrapper (launch wrapper) | rapper | 包装函数 |
| traits (CUTLASS 2SM traits) | Trades | 配置 |
| subtile | Sato | 子块 |
| j (a loop index read alone) | (nothing heard) | 下标 / 下标小于 N |
| key (dequant memo key, DSV4.1 + K3 sharing video 2026-10-23) | T (fourth video in a row) | 缓存键 / 索引键 (key stays on screen) |
| SparsePrefillChunkCache (class name in a sentence) | Sparse Refill Chunk Cache | 每个 chunk 自己的缓存对象 (the class name stays on screen) |
| atol (as in atol 等于 0) | A套 | 误差容忍为 0 |
| supports weight row mapping (function name read as words) | with Role Mapping | 一个专门的检查函数 |
| topk (bare, as in 绕过原来的 topk) | Talk | 原来的 top-k 选择 (top-k → top k passes) |

Also OK~ in that video, listen on the render: warp / warps ("World" / "Works"), grid ("Grip"), tile ("他").

**行 after a K/M/G count** (DSV4.1 + K3 sharing video, 2026-10-23): 「262K 行」「16K 行」 got no 航 respelling because the
line-sense rule only looked behind for a digit. The rule now also accepts `\d[KMG]` and `\d[KMG] ` before 行; 「K 行为」 without a
digit is untouched (test `test_terms_from_the_dsv41_k3_sharing_video`).

**Chinese polyphones across a word boundary** (user's ear, DeepSeek V4.1 video 2026-10-18): 「生成长距离」 was read
「生成 zhǎng 距离」 because the voice segmented the bigram 「成长」 (chéng zhǎng) across the word boundary. `pron_check` cannot
catch this: Whisper writes 长 for either reading (use `poly_probe.py`, below). Before TTS, scan the narration for a polyphone (长 / 行 / 重 / 还 / 得 /
了 / 差 / 便 …) whose neighbour forms a different word (成长, 生长, 银行, 重复 …) and reword so the bigram never forms,
e.g. 「40 层里只有 4 层在生成长距离 KV」 → 「40 层里，长距离 KV 只由 4 层生成」.
Same review: 「decoder 侧重放」 was read cè**zhòng** fàng (侧重 = emphasis) → 「decoder 侧的重放」. Also prefer the dictionary
word over a clipped one: 「重算」 is not a common word and may get zhòng, so write 「重新算」.
Measured in the DeepInfer video (2026-10-20, `poly_probe.py`, 6 takes each): 「就只重算」 was zhòng in 4/6 takes, while
「就只重新计算」 and 「，只重建」 were 0/6. Do not respell 重算 → 崇算 in `patterns`: it fixes 重算 (0/6), but a 重建 later in
the same clause then flips (「重算段只重建」 4/6, 6/6 after the respelling). Reword the narration instead: 重算 → 重新计算,
重算段 → 回放段, 重建 → 重新生成, 重启 → 重新拉起.

**行 meaning "row / line" is respelled automatically** (user's ear, 2026-10-19: 「一行」 was sometimes read yì
**xíng**, as in 一行人). Measured with `scripts/poly_probe.py` (takes vs 航/杭 and 型/形 homophones; 11/11 control
words such as 银行/行业/进行/行为/一行人 judged right):

- **Edge picks the reading per request.** The same sentence, synthesized again, can flip. 「就是这一行在表里的位置」
  came out xíng in 1 of 5 takes in one run and 3 of 6 in the next. A clean listen of one take proves nothing.
- Across 74 「一行」 lines from 13 past videos, 31 (42 %) were read xíng. 「每行」: 4 of 28; 「整行」: 2 of 3; 「长行」: 1 of 1.
- Five lines × 6 takes: **13/30 takes wrong as written, 0/30 after the rule.** 「退回一个 CTA 一行」 was wrong in 6 of 6 takes.

The `patterns` in pronunciation.json send 航 (only ever háng) to the voice; captions keep 行:

| Written | Spoken | Rule |
| --- | --- | --- |
| 一行 · 这一行 · 每一行 · 一行的上下文 | 一航 … | 行 after 一, except after 唯一/统一/万一 |
| 一行行 | 一航航 | so 航行 (háng xíng) never forms |
| 每行 · 整行 · 几行 · 两行 · 多行 · 长行 · 短行 · 逐行 · 按行 · 六行 · 七万八千零八十行 · 第 3 行 | …航 | 行 after a numeral, digit or row quantifier |

Left alone (the voice must say xíng): 行 followed by 为 动 驶 走 人 李 星 政 者 销 军 刑 径 凶 踪 囊 装, by 使 (not 使用),
进 (not 进行), 程 (not 程序), 得通 or 不通. So 统一行为, 唯一行得通, 万一行不通, 一行人, 每行驶 keep 行, while
每行使用, 每行进行, 一行程序 get 航. `tests/test_explainer_pronounce.py` pins both lists.

Not covered on purpose, because both readings are common: 这行 / 那行 (那行吧 = xíng), 此行, 同行, 单行, 上行 / 下行, 各行.
Write 这一行 / 同一行 / 那一行 instead and the rule applies. The guard words also bite the writer: 「按行动态量化」
puts 行 before 动, so the rule leaves it (行动 = xíng dòng) and the voice may read it that way. Write 「逐行做动态量化」
(24GB video, 2026-10-19). One false positive remains: 一行三人 (a party of three)
becomes yì háng. It is very rare in technical narration; reword it if it ever appears.

**Measuring a polyphone** (any character, e.g. 长 常/肠 vs 掌/涨, 重 众/仲 vs 虫/崇):

```bash
$SEV_PY -I $SEV_SKILL/scripts/poly_probe.py --text "就是这一行在表里的位置。" --at 这一行 --char 行 --want 航,杭 --avoid 型,形
#   as written (这一行在): wrong 2/6
#   as spoken (这一航在): wrong 0/6
```

It reports the share of takes that used the avoided reading, as written and after pronunciation.json. It exits
non-zero while the spoken form still misreads. A rule is done when the spoken form is 0/N.

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
