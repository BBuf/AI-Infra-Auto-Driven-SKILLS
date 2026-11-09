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
- `scripts/hang_take.py` listens to the take the video actually uses: every 行 in it, in both directions (a 行
  meant háng read xíng, and a 执行 / 并行 read háng). The respelling lowers the misread rate per request, but it
  doesn't make a given take right, so run it on every recorded take and re-synthesize the WRONG lines.

## Rules for writing narration

1. **Never put `_` in front of the voice.** snake_case and SCREAMING_CASE are read as 下划线. Writing them in
   the narration is fine: the identifier rule splits them and maps known parts (SGLANG_DSA_FUSE_TOPK →
   SGLang DSA fuse top k). A part that is not a word the voice knows needs a `parts` entry. Leading and trailing
   underscores belong to the identifier (`_commit_int8_checkpoint`, `__init__`): before 2026-10-27 the token
   started at the first letter, so the voice said 「下划线 commit int8 checkpoint」 (PR 28185 video).
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
12. **「问题出在 X」 can split as 问题 / 出 / 在 X.** In the Claude Code review video, Edge stretched 出 to 0.34 s and paused
    0.1 s before 在, so the user heard 出 and 在 apart (2026-10-28 review). WordBoundary shows it: 出 and 在 come back as
    separate words with a gap. Reword (「问题的根源是 X」) instead of re-synthesizing; a retake keeps the same split.

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
| mma (lowercase, tensor core 的 mma 指令; tiny GEMM video 2026-11-03) | MMA | 安班杯, also spelled M M A → **M&A / MMV** (OK~) |
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
| SWE (SWE Marathon, SWE-bench) | S W E | SWIT → **SWE** (model-release video, 2026-10-24) |
| BrowseComp | Browse Comp | BraceComp → **BrowsComp** (OK~, closer) |
| rid (request ID, PR 37909 video 2026-10-24) | R I D | read / Reed in 10 of 11 lines (the voice says "reed") → **RID** |
| MegaAttention (DeepSeek Mega kernel, DeepInfer promo 2026-10-25) | Mega Attention | MegaTension (fused) → **Mega** + **Attention** |
| expm1 (softmax-fix paper video 2026-10-26) | exp m 1 | X1 (the letters run together) → **EXP M 1** |
| epsilon (Adam ε, softmax-fix paper video 2026-10-26) | ep-silon | Excelon (OK~, a /ks/ where the p-s belongs) → **EP Silent**, closer to ep-si-lon |
| ack / ACK (and the `ack` part of identifiers) | A C K | at / act / App for every word spelling (PD security blog video, 2026-10-27) → **A C K** |
| ABORT_ACK | Abort A-C-K | AborT CK: the spelled A runs into abort's final t → **Aboard ACK** (OK~; abort/aboard is Whisper's near-homophone) |
| queue (merge queue, Claude Code review video 2026-10-28) | Q | 「接下来Continue」 in the carrier: the voice garbles the vowel run → **Q**; English-mode ASR writes "merge queue" |
| capture (CUDA graph capture, PR 43570 + 47387 video 2026-11-01) | Capture | Catcher in the carrier → **Capture**; in the take 「CUDA graph 照样能 capture 和 replay」 came back as capture |
| wave (SM wave, PR 47684 video 2026-11-01) | WAVE | Web in the carrier (OK~; Wave → Weil, waive → Weil FAIL) → **Wave**; plural waves already reads right |
| 24 只要 (a number right before 只要/只有/只需…, same video, user review) | 24，只要 | Edge takes 只 as a classifier after a number: 二十四只（zhī）要, 6/6 takes (poly_probe, want 纸/指 vs avoid 知/织) → pattern `(?<=\d) ?只(?=[要有需用是剩占能会在])` → 「，只」: 0/6. Also write the comma in the narration, or the caption splits at the spoken pause into a group with no punctuation |
| 一行一行人工 (行 = line, Claude Code review video 2026-10-28, user review) | 一航一航人工 | The 行人 (pedestrian) exception fired on 行人工, so the second 行 stayed xíng (poly_probe 6/6 wrong) → exception is now 人(?!工), plus a 一行一行 → 一航一航 rule (0/6 wrong) |
| SLA (service-level agreement, 2026-11-08; unprimed Whisper on each line) | S L A | slot / SWAT in both lines tried (「通过 SLA 的候选」, 「满足 SLA 的候选」) → **SLA** in both |
| Day-0 (Day-0 support, same date) | Day zero | D-0 / 对零 / Dailing; `Day 0` with a space fails too → **Day Zero**. Write the hyphen in the narration: `Day 0` is two tokens, so the lexicon entry never sees it. pron_check compares the transcript with the written form (`day0`), and Whisper writes `Day Zero` (`dayzero`), so `--meta` reports the respelled take as a FAIL. That is a false positive, like KiB above: listen once and move on |
| fused before top k (`init_dcp_fused_topk`, `DcpFusedTopK`; PR 48895 video, 2026-11-09) | as written | Whisper writes Fuse Top K in the take and in the carrier, so `--meta` reports the identifier as a FAIL. The voice elides the d before t the way a speaker does, and `fused` alone passes. Respellings were worse: fusd → Fast Top K, fyoozd → FilesTopK, fused- → Fuse Top-K. A false positive, like Day-0 above: leave it |

These read right as written: PR, DSA, MLA, MTP, KV cache, fused, prefill, decode, batch, token,
kernel, attention, indexer, backend, CUDA graph, FP8/FP4/NVFP4, TP8, H200/B200/GB300, Triton, PyTorch,
torch.compile, vLLM, NVIDIA, Blackwell, Hopper, RoPE, softmax, Mamba, Ulysses, GLM-5.3, W4AFP8.

**Approved by ear** (`approved` in pronunciation.json — never respelled, reported OK): **SGLang**. The user
compared the voice's own reading with SG Laang / SG Lahng / SG Leng / S G Lang and kept the original
(2026-10-16). Whisper writes it as SGLine / SGLAM; that is an ASR artifact, so ignore it.
**MTP** is approved too: Qwen3-ASR transcribes Xiaoxiao's MTP correctly, while Whisper writes MPP for every
spelling, even Em Tee Pee. Also approved: **Kimi** (user's ear), **PR** (letters, R as 阿尔, the usual Chinese
reading), **dB** (read 分贝), and **WRITE** (RDMA WRITE: Whisper writes the homophone "Right").

Still **OK~** with no respelling that passes; listen on the render: Claude ("Cloud" — every respelling was
worse), Godot (bare form heard "Golded"; "Go dough" was read as Chinese 勾斗, worse), Lean ("Lin"), README (spoken "read me" since the DeepInfer video; Whisper: Remi → REME), TopK ("Talk K"), FLUX.2 (fine in most sentences).
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
| config (model config, PR 28185 video 2026-10-27) | 看Fake (config / Config / con fig / kon fig all fail; configuration passes) | 模型配置 (config.json stays on screen) |
| flush (flush cache, same video) | Flash (in context) | 整个 radix 树被清空时 (flush stays on screen) |
| 1M (as in 128K 到 1M token, PD security blog video 2026-10-27) | EM | 100 万 (keep 1M on screen) |
| diff (Claude Code review video, 2026-10-28) | Dev / Dave's (diff, Diff, dif all fail) | 改动 / 这次的改动 (diff stays on screen) |
| skill (same video; the singular re-measured 2026-11-08) | Scale (skill, Skill, skil all fail). 2026-11-08: speel / Speal / Scale / Steel depending on the sentence, and two takes of the same sentence came back different. The plural **skills** read right in all five sentences tried | the plural skills (这些 skills / 一组 skills), or reword (技能); the singular stays on screen |
| nit / Nit (same video) | Net (knit → Net, nitpick → Netpeak) | 小毛病 (the Nit tag stays on screen) |
| on-call (same video) | Encore (oncall too) | 值班 / 每周轮换的值班 |
| clone (git clone, same video) | Clan | 克隆 |
| tok/s | Talk | 每秒 N 个 token |
| copy on write | on → 嗯 | 写时复制 (the English stays on screen) |
| pybind11 (DeepInfer video, 2026-10-20) | PBind11 | 绑定到 Python |
| an English tagline read as a sentence ("The agent flow system for token maxxing", Humanize 2 video 2026-10-29) | The → 的, maxxing → Maxim | say it in Chinese (它给自己的定位是一句英文标语，大意是…); the English stays on screen |
| pytest (same video) | Ptest | 跑一遍测试 |
| done (as a field name, same video) | 单 | 做完没有 / 做完了 (the done field stays on screen) |
| resume (--resume, same video) | rezoom (the verb reading, but no respelling passed) | 再接着跑 (--resume stays on screen) |
| chasoid (flame_chasoid, same video) | Chaseo at | don't speak it: 一组变体 flow (flame_chasoid stays on screen) |
| epoch (epoch N: distilled tree, same video) | iPad | 压成一个提交 / 第几轮打扫 |
| tree (distilled tree, same video) | 接下来Country in the carrier | 蒸馏过的仓库 / 那棵树 |
| 2 as a version number (Humanize 2 把关系 / Humanize 2 把整个, same video, user review) | 两把 (the following 把 is taken as a classifier, so 2 → 两) | pattern: a capitalised name + space + 2 → 二 (Humanize 二, Llama 二); counts after lowercase words or acronyms (agent 2 个, GPU 2 个) and decimals keep their digit; Whisper writes the fixed take back as "Humanize 2" |
| CLI (Humanize 2 video, user review 2026-10-30) | read as a word | lexicon → C L I (the user wants the three letters); ASR then writes it back as CLI |
| 回调 (callback, same video, user review) | huí tiáo in 2/6 takes (poly_probe, want 掉/钓/吊 vs avoid 条/调皮) | pattern 回调 → 回掉: 0/6 wrong; pick reference homophones with the same tone as each reading, or the probe's margins blur |
| 权重 (weights, three kernel PR videos, user review 2026-11-02) | quán chóng (user's ear; poly_probe 0/6 in two carriers, so the misread is take-dependent) | pattern 权重 → 权众: 0/6 |
| 行宽 (row width, same review) | xíng kuān (user's ear; poly_probe 3/6 then 1/6) | pattern 行宽 → 航宽 (not after 执/运/进/并 …): 0/6 |
| 行距 (row pitch, CUDA perf guide video PR 48380, user review 2026-11-05) | xíng in the delivered take (hang_take margin −0.22) | 距 added to the 行宽 lookahead → 航距; hang_take +1.7 to +5.8 on the new take |
| 字节行 (128 字节行 = a 128-byte cache line, same video) | xíng in the delivered take (hang_take margin −4.99) | pattern (?<=字节)行 → 航 |
| 拆行 / 的行， (row at a clause end, same videos) | xíng in 2–4/6 and 3–5/6 | patterns 拆行 → 拆航, 的行[，。；、] → 的航: 0/6 |
| 行和地址 / 第 row 行 (same videos) | xíng 6/6 and 4/6 | reword: 每一行的宽度或地址 / 编号为 row 的这一行 (the 一行 rule then gives háng, 0/6) |
| 累加 (tiny GEMM video re-voice, user review 2026-11-03) | lèi jiā in 6/6 takes (poly_probe; controls 积累 0/6, 累计 0/6, 太累 6/6 lèi) | pattern 累加 → 垒加: 0/6 |
| 乘 before a Latin letter (64 乘 N, TP 乘 PP, PP4 乘 PCP2; DCP explainer video 2026-11-06, line_asr heard 剩) | shèng in 7/12 and 5/12 takes (poly_probe, want 成/承 vs avoid 剩/圣); 5 乘 64 (before a digit) 0/6 | pattern 乘(?= ?[A-Za-z]) → 成: 0/12 in both carriers |
| 藏在 (就藏在读数据和计算后面, same re-voice) | zàng in 3/6, then 4/12 | reword: 就被读数据和计算掩盖掉了 (also a seg_check cut, below) |
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
| key (u32 sort key, MoE router video PR 47641, 2026-11-02) | P / T in 9 of 11 lines (fifth video in a row) | 键 / 整数键 (key stays on screen) |
| topk inside a function name (fused topk, biased grouped topk, topk softmax pack; same video) | Talk | top k (the identifier stays on screen) |
| 1e30 (as in NaN 换成负 1e30, same video) | 负1乘以10的30次方: Edge expands it itself, so the token never appears and pron_check FAILs; the 「1 乘以」 also sounds clumsy | `patterns`: 1eN → 10 的 N 次方, 1e-N → 10 的负 N 次方, x.ye-N → x.y 乘 10 的负 N 次方 |
| SIMT (tiny GEMM video PR 47683) | SIMP for SIMT, S I M T and S-I-M-T alike (OK~) | no respelling: Whisper merges the letters; keep SIMT |
| N_TILE / tile (K × (4 + N_TILE) / 128, same video) | NTAO for N tile, N Tile, N-tile and NTile; tile alone Tao or 他 | 每个 CTA 的行数 (N_TILE stays in the on-screen formula) |
| r (第 r 份写进第 r 个 CTA, same video) | 二: the letter sounds like 第二份 | 第几份就写进第几个 CTA |
| FFMA (CUDA perf guide video PR 48380, 2026-11-05) | FFMV | lexicon → F F M A (passes: FFMA) |
| DFLASH (speculative algorithm, DCP explainer video 2026-11-06) | Flash (the D is dropped) | lexicon → DFlash: heard D-Flash 2/2 |
| EP2 (expert parallel 2, same video) | EPR | lexicon → EP 2: heard EP2 2/2 |
| fi_a2a (--dcp-comm-backend value, same video) | FiA to A | lexicon → F I A2A: heard FIA2A 2/2 |
| cutedsl_mla (attention backend, same video) | Cutasol MLA | lexicon → cute DSL MLA: heard CuteDSL MLA 2/2 |
| 1P1D (same video) | EPED; the respelling one P one D came back as 万P万D in a carrier | reword: 一台做 prefill，一台做 decode |
| V 预投影 (V pre-projection, same video) | Vu 投影 / Vu 头影 in 4 lines: the letter V runs into 预 | reword: V 的预投影 |
| a number before 时 or 是 (TP8 时 → TP80, 组是 8 张卡 → 18 张卡, DCP 是 2 的幂 → DCP12; same video, line_asr) | the digit merges with the next syllable | reword: 的时候 / 有 8 张卡 / 取 2、4、8 这样的值 |
| single letters in formulas (e 的 s 次方, N r / Z r / o r, 第 j 张卡; same video) | 一 / 二 / 机 / 欧 / dropped | reword: 取指数, 本卡分子 / 本卡分母, 那张卡 (the symbols stay on screen) |
| LDG (same video) | Large (LDG.128 → Large 128) | lexicon → L D G; for LDG.128 / LDG.E.256 say 128 bit 的 LDG / 256 位的 LDG |
| FMA alone (same video) | FMV (no respelling passed) | 乘加 / 乘加合并 / 混合精度乘加 (FMA, fma.rn.f32.bf16 stay on screen; the full PTX name reads fine) |
| lg throttle / mio throttle (same video) | Frooto | LSU 的队列塞满 / LSU 和 MIO 的队列满了 (lg_throttle stays on screen) |
| occupancy (occupancy API, same video) | Ultipancy | 占用率 / CUDA 的 max active clusters 接口 |
| ctarank (%cluster_ctarank, same video) | Turank | 专用寄存器 (%cluster_ctarank on screen) |
| parallel compile (parallel_compile, same video and rmsnorm video PR 47503) | 看PAL | 并行编译 (parallel_compile on screen) |
| spill (in the recorded take, same video) | Steel / Spear | 寄存器溢出 / 溢出 |
| 1.96 GHz (same video) | 1.96千MHz | 每秒 19.6 亿个周期 (keep ≈ 1.96 GHz on screen) |
| r (RMSNorm's 1/rms, rmsnorm video PR 47503) | 二 (again) | 这个倒数 / 乘倒数的结果 (r stays in the on-screen formulas) |
| RMSNorm / rmsnorm (same video, user review 2026-11-05: 「有些单词也没有连起来读」) | Whisper wrote 「每一层有两处，而MSNorm」 for the take (a pause, then the R); rmsnorm → Rim Snorm (OK~) | lexicon → RMS Norm / RMS norm; also don't put it after a clause that ends on a count (两处 RMSNorm → RMSNorm 有两处). The re-voiced line still came back as 而MSNorm once in a whole-soundtrack transcription, but the line alone and the word alone transcribe as RMSNorm, and its R onset sits with the RMS references (+4.5, same as three other RMSNorm lines; 而 references far off). That one is Whisper's context, not the voice |
| PR 47683 (a PR number inside a line, both videos) | 476、83 / P2476、83 in 2 of 3 takes: Edge reads it as a number and breaks it | pattern (?<=PR )(\d{4,6}) → digits in Chinese (fn:digits_zh): PR 四七六八三, no break in 3/3 |
| fma.rn.f32.bf16 (in a sentence, both videos) | FMA、RNF32、BF16 — read in three pieces | 混合精度的乘加指令 (the PTX name stays on screen); the bare name read alone is fine, inside a line it isn't |
| div.full (A 乘以 B 的近似倒数, 和 div.full 一样, PR 48380 video) | DIV 4 in 3/3 takes | 和完整精度的除法一样 |
| L1 carveout (same video) | LE car vault in the line (carveout alone in the carrier is OK) | 给 L1 和 shared memory 挑一个合适的划分 |
| ulp (same video) | 奥布 | 一格 / 最小刻度 / 数格子 (ulp stays on screen) |
| griddepcontrol wait (same video) | Grid App Control | PDL 的等待指令 / 等待指令 wait |
| if constexpr (same video) | F / Constax Bear | 所有分支都在编译期决定 |
| threads per row / rows per CTA (same video) | FREDS / Rose | 每行线程数 / 每个 CTA 的行数 (threads_per_row, rows_per_cta on screen) |
| in bound (in_bound, same video) | 另一部分 | 越界判断 |
| IEEE (same video) | HEE | 浮点标准 |
| e4m3 (FP8 e4m3, same video) | 一4M3 (the lone e again) | 这种 FP8 / 输出的 FP8 (e4m3 stays on screen) |
| NaN (same video) | 9 | 无效的值 |
| wave in a sentence (一个 wave 放得下几行 CTA, same video) | WIM in one take (the carrier passes as Wave) | GPU 一轮能同时放下几行 CTA (wave stays in the kicker) |
| SparsePrefillChunkCache (class name in a sentence) | Sparse Refill Chunk Cache | 每个 chunk 自己的缓存对象 (the class name stays on screen) |
| Three.js (also respelled Three J S) | Free.js / FreeJS | 网页 3D 场景 / 在浏览器里漫游的 3D 房子 (Three.js stays on screen) |
| atol (as in atol 等于 0) | A套 | 误差容忍为 0 |
| supports weight row mapping (function name read as words) | with Role Mapping | 一个专门的检查函数 |
| topk (bare, as in 绕过原来的 topk) | Talk | 原来的 top-k 选择 (top-k → top k passes) |
| Gloo (PyTorch CPU collective backend, PR 37909 video 2026-10-24) | 格尔 / 歌 (glue, Glue, Gluu, Glooo, G loo all fail) | CPU 上的通信组 / 这个后端 (Gloo and gloo::EnforceNotMet stay on screen) |
| MIN (as in 取 MIN, all_reduce op) | main (min: main too; minimum passes) | 取最小值 (MIN / ReduceOp.MIN stay on screen); MAX reads fine |
| wgmma (Hopper warpgroup MMA, PR 42855 video 2026-11-08) | Ugma / Ugman (wgmma and WGMMA alike) | lexicon → WG MMA: heard WGMMA in the carrier; in the take MMA is the usual OK~ (班贝 / M&A) |
| warpgroup / warpgroups (same video) | workgroup (OK~) | lexicon → warp group(s): two words the voice knows; still heard workgroup, accepted |
| cooperative (cooperative launch, same video) | Coopertive | lexicon → co operative: heard Cooperative; also reaches enable_cooperative through the identifier rule |
| Gumbel (Gumbel-max sampling, same video) | Gumbo (Gumbel, Gumble) | lexicon → Gumbell (heard Gumbell); Gumbel-max → Gumbell max (OK~ Gunbel Max) |
| Philox (Philox4x32 RNG, same video) | Felix (OK~) | lexicon → Fylox: heard Fylox |
| dspark (part of SGLANG_DSPARK_* env names, same video) | SGLand Spark (the D is dropped, like DSPARK) | `parts` → DSpark |
| γ (draft length, same video) | read as Gamma in 3/3 lines (Whisper: Gamma, once 干嘛) | keep γ in narration; captions show γ |
| ex2 (ex2.approx, same video) | X2 | 硬件近似的指数 (ex2_approx_ftz stays on screen) |
| issue_fill (ring refill helper, same video) | Essue Fail | 由最后读完的那个 warp 发起 (issue_fill stays on screen) |
| 奇偶位 (mbarrier parity bit, same video) | line_asr heard 其偶位 (qí, not jī) in 2 clauses; seg_check also cut 奇 \| 偶位 | 那一位比特翻转 / 只看这一位; 按轮数的奇偶 → 这一轮用一套，下一轮用另一套 |

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

**长 in 长上下文 and 擅长 is respelled to 常** (user's ear, model-release video 2026-10-24: 「长上下文推理」 was heard
as zhǎng). Measured with `poly_probe.py` (6 takes, 常/肠 vs 掌/涨): 「RL 框架、长上下文推理」 4/6 wrong as written, 「它尤其擅长参数化建模」
4/6 and 6/6 wrong; 0/6 for both after the `patterns` rules 长(?=上下文) → 常 and (?<=擅)长 → 常. Already right as written
(0/6): 不随上下文变长, 长程能力, 长程任务, 长篇讲解.

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
| 多少行 · 切多少行 (PR 47684 video) | …航 | 少 added to the quantifier list: 「每块切多少行」 was xíng in 2/6 and 3/6 takes as written (poly_probe), 0/6 respelled |

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

**The probe has a noise floor of up to about 1/3 in some sentences.** Re-checking the PR 47641 and 47512 videos
(2026-11-03): 「数一数」 scored 4/6, then 0/6 and 0/12; 「连续弹出」 3/6, then 0/12; 「按步长读」 2/6, and its respelling
「按步常读」 — 常 has a single reading — still scored 4/12. Before adding a rule, re-run with `--takes 12` and check
that the respelled form scores 0. A rule lands only when the as-written rate is high and repeats (累加: 6/6, with
积累 / 累计 / 太累 as controls) and the spoken form is 0/N. Comparing a take that is already in a video with fresh
syntheses (the whole line, or just the word's window cut by WordBoundary timings) failed its own controls: a 垒加 take
was judged wrong. 垒 lěi and 累 lèi differ only in tone, which log-mel barely carries (see below), so a 累加 take
cannot be graded. 行 can: háng and xíng differ in both initial and final.

**Grading the recorded take for 行** (`scripts/hang_take.py`). For each 行 it centres a 0.6 s window on the
character (time estimated from the take's WordBoundary words, indices counted without punctuation, which the
words don't carry), and compares it with fresh takes of the same line that say 航/杭 or 型/形 at that spot. Word
windows were noisy, because Edge groups the characters differently per take, so a verdict flipped between runs.
Calibrated 2026-11-05 on 32 control takes with a known reading (16 航, 16 型): 32/32 right at ±0.25, ±0.35 and
±0.5 s. A take synthesized with 型 in a háng spot and one with 航 in 并行 were both reported WRONG; the 执行 in the
same line and a correct 一航 passed. The PR 48380 video, as delivered, had two háng 行 read xíng (字节行 −4.99,
行距 −0.22); after the rules and a re-take, 0 of 47 across both videos, smallest margin +0.77. A margin under
about 1.5 prints as `close`: run it again.

WordBoundary starts lead the audible onset by about 0.08–0.1 s (a word after a comma starts in the silence). For a
window shorter than a syllable or two, such as the first 0.4 s of a term to tell R from 而, start it at the energy
onset (first 10 ms frame above −30 dB after the boundary), in the take and in every reference. Cut at the boundary,
the RMSNorm take above scored −1.14 (而); cut at the onset, +4.46. hang_take's ±0.3 s windows around the character
were not affected.

```bash
$SEV_PY -I $SEV_SKILL/scripts/hang_take.py --script $P/SCRIPT.md --meta $P/audio_meta.json [--lines 24,35]
#   L24  háng  WRONG  margin -4.99  …128 字节行，超过…
#   ✗ hang_take: 1 of 15 行 take the wrong reading
```

**Tone-only polyphones cannot be probed** with poly_probe (use `tone_take.py`, §Tone-only polyphones below). The
probe compares log-mel frames, which carry little pitch, so it tells 行 háng from xíng but not 只 zhǐ from zhī: the control 「它只能存」 (always zhǐ) scored 12/12 wrong against 指,纸 vs 之,织
(PR 28185 re-check, 2026-11-03). For 只 after a number, read the WordBoundary tokens instead (`seg_check` reports
them as `classifier`): the misread take in the PR 43570 video had '24 只' | '要', and takes that read right keep
只 with its verb ('int8' | '只能', 'fp8' | '只有').

In-context checks of long lines are noisy: re-probe a FAIL in a short sentence before changing the lexicon. In
the 13-video re-voice, most FAILs read right in the short sentence. If one sounds wrong, reword the line, e.g. "EAGLE 投机解码" → "投机解码（EAGLE）" on
screen only.

The free Edge endpoint rejects inline SSML (`<phoneme>`, `<sub>`, `<lang>`, even `<break>` → NoAudioReceived),
so respelling is the only lever.

## Tone-only polyphones (tone_take.py)

Some polyphones differ only in tone: 切 qiē (cut) / qiè (一切, 密切). Three things make them hard:
- Respelling cannot fix them. No common character reads qiē alone (苆 is the only one), and the free Edge
  endpoint rejects SSML.
- A retake cannot fix them either. Edge returns the same audio for the same text: eight takes of one sentence
  had pitch slopes within 0.5 st of each other.
- poly_probe, hang_take and Whisper are deaf to tone.

`tone_take.py` measures pitch instead. It needs the tone env once (`bash $SEV_SKILL/scripts/setup_tone.sh`: MFA 3.4.3
from conda-forge, praat-parselmouth, and the mandarin_mfa v3.0.0 models, about 1.5 GB). For each character listed
under `tones` in pronunciation.json, it does three things:
- It synthesizes the line with the character swapped for single-reading controls of each tone.
- It force-aligns the take and the controls with MFA.
- It compares the Praat pitch slope over the character's vowel and coda with the slopes of the controls.

What was measured while building it (2026-11-07, 12 sentences from the DCP explainer video):

| Method | Controls judged right |
| --- | --- |
| MFA alone: the target word gets only its tone-1 and tone-4 pronunciations and the aligner picks one | 58/108 (chance: the GMM features carry no pitch, though the phones are tone-marked, e˥ / e˥˩) |
| Pitch slope in a window placed with Edge's word boundaries | contradictory: the window often lands on the neighbour (不 in 也不切它 falls too) |
| MFA-aligned vowel, Praat slope, one fixed threshold (−1.75 st) | 84 % (79 % leave-one-sentence-out) |
| MFA-aligned vowel only, compared with the same sentence's controls | 85 %; nasal finals (通/痛) hide the fall in ŋ |
| MFA-aligned vowel + coda, same-sentence controls (what tone_take does), 接/揭/歇/包/通 vs 借/谢/窃/卸/报 | **116/120 (97 %)** leave-one-out; with 些 as a tone-1 control it was 89/96, so 些 was dropped |

The four misses were tone-1 controls judged tone 4, with margins of 0.2 to 3.1 st, so a WRONG under 1 st is printed
as weak: listen before rewording. On the DCP video's takes, it flagged:
- 「DCP 也不切它」 as qiè (margin +7.6 st). Earlier, three independent measurements had agreed.
- 「轮流切到各张卡」 in one line as a weak WRONG (+0.7). The same phrase in another line read qiē.

Every 切 right before its object or result (切开, 切成, 切片, 切到, 切出) read qiē. Reword a clause-final 切: 也不切它 →
也不把它切开 / 也不拆分它.

```bash
$SEV_PY -I $SEV_SKILL/scripts/tone_take.py --script $P/SCRIPT.md --meta $P/audio_meta.json [--lines 5] [--chars 切]
#   L5   切  want qiē  heard qiè  WRONG  margin +7.6 st  …DCP 也不切它。…
#   ✗ tone_take: 1 of 8 tone-only polyphones took the wrong tone
```

**Adding a character.**
1. Give its `readings`, the `default` tone, the `words` that take the other tone, and `controls`: at least two
   single-reading characters per tone, with the same final where possible.
2. Run `tone_take.py --calibrate <char> "sentence" …` on 8 to 12 sentences that use it the way the narration does.
3. Add it only if the controls are judged right in at least 90 % of cases. Drop any control that keeps getting
   misjudged; 些 was one.

The slope feature separates a level tone from a falling one (1 vs 4) best. Calibrate a 3-vs-1 pair such as 只 before
trusting it.

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

## Word segmentation (seg_check.py)

The voice segments each line itself, and where it cuts changes the rhythm and sometimes the reading. In the three
kernel PR videos (2026-11-02) the user asked that every line follow Chinese reading. `seg_check.py` flags:

| Written | Voice tokens | Fix |
| --- | --- | --- |
| 值相同时下标小的赢 | 时下 \| 标 (100 ms pause; 时下 = "nowadays") | 值相同的时候，下标小的赢 |
| 下标取反放在低 32 位 | 标取 \| 反放 (88 ms) | 下标按位取反，放在低 32 位 |
| temperature 在 kernel 里直接取 | 里直 \| 接取 | temperature 直接在 kernel 里读取 |
| 得到一个新张量 | 新张 \| 量 | 得到一个新的张量 |
| 把 log2e 和 1 除以 T 合成 | 除 100 ms 以 | 把 log2e 和 T 的倒数合成 |
| 就藏在读数和计算后面 | 在读 \| 数 (在读 = "studying") | 就被读数据和计算掩盖掉了 |
| 在编译期生成好 | 期生 \| 成 | 在编译的时候就生成好 |
| 还能省一次加法依赖 | 加法依 \| 赖 | 还能少做一次加法 |
| token 组尽量少 | 组尽量 \| 少 | token 组，尽量少 |
| 乘 24 只要 27264 个 (PR 43570 video) | 24只 \| 要 (`classifier`: 只 read zhī) | 乘 24，只要 27264 个 |
| 用一棵 radix 树记下算过的 prefix (PR 28185 video) | 树记 \| 下算 (heard 数据下) | 用一棵 radix 树，把算过的 prefix 记下来 |
| 内存放不下整个程序 (Paged Experts video, 2026-11-04) | 内 \| 存放 | 内存装不下整个程序 |
| 上下文限到 16384 (Paged Experts) | 上下 \| 文限 | context length 设成 16384 |
| 才单独列出来不搬 (Paged Experts) | 单 \| 独列 \| 出来 | 才另外列出来，不去搬它 |
| 全部拿来放专家 (Paged Experts) | 拿 \| 来放 | 全部用来放专家 |
| 偏置用 tensor core 上的 int32 点积精确算出来 (PR 42855 video) | 积精 \| 确算 | 偏置用 tensor core 做 int32 点积，结果是精确的 |
| 端到端测试用 Qwen3-4B (same video) | 端到 \| 端测 \| 试用 (heard 端到端侧是用) | 端到端的测试，用的是 Qwen3-4B |
| 直接读进片段寄存器 (same video) | 读 \| 进片 \| 段 | 直接读到片段寄存器里 |
| 哪 7 块走流式，是在 H100 上实测挑的 (same video) | 走流 \| 式, 上实 \| 测 | 流式的 7 块，是根据 H100 上的实测挑出来的 |
| 把 W2 分给所有 SM (same video) | 分 88 ms 给 | 把 W2 分配到所有 SM 上 |
| 为什么按分数线性分桶 (PR 48895 video, 2026-11-09) | 按分 \| 数线性 (分数线 is a cutoff score) | 为什么按分数等距分桶 |
| 窗口大小加标志位 (same video) | 加标 \| 志位 | 窗口大小和标志位 |
| 两到三轮定出分界值 (same video) | 出分 \| 界值 | 两到三轮确定分界值 |
| 改成谓词指令，省掉了编译器生成的分支 (same video) | 成谓词, 编译 \| 器生 \| 成 | 改用谓词指令，省掉了编译出来的分支 |
| 再从全局内存读回这些候选的分数 (same video) | 内 100 ms 存读, in the re-take only: the first take of the same clause cut 内存 \| 读 | 再把这些候选的分数，从全局内存里读回来 |
| metadata 里预先分配好的张量 (same video, re-take) | 里预 88 ms 先 (line_asr: 领域先) | 这个张量，是在 metadata 里提前分配好的 |

Natural cuts (两 | 个数, 第几 | 行, 取 | 出来, 再分 | 给, 7168个) are not reported.
seg_check does report some cuts that leave the reading alone. In the PR 48895 video, 分 | 界值, 融合路 | 径, 本 | 卡槽 | 位 and
上下 | 文长度 came back with no pause and read as written; leave such hits. Run seg_check again after every re-take: a line
re-voiced for a change at its end came back with a new cut (内 | 存读) earlier in the line.

## Listening to whole lines (line_asr.py)

`pron_check` only follows English terms, and `seg_check` only looks at where the voice cuts. `line_asr.py`
transcribes every recorded line with Whisper and diffs it against the narration in pinyin, so it hears the
Chinese as well. In the tiny GEMM re-voice (2026-11-03, the user found the take 「读音不对、AI 味重」), it caught
places where a digit and the character next to it form another number:

| Written | Heard | Fix |
| --- | --- | --- |
| 清空 L2 时是 0.88 | L20 是 (L2 时 is èr shí, the same sound as 二十) | 清空 L2 的话是 0.88 |
| B200 上开 PDL 时耗时是原来的 0.73 | PDL10号时 | B200 上开 PDL，耗时是原来的 0.73 |
| batch 是 1 的时候 | batch 11 的时候 (2 of 3 lines; shì yī vs shí yī) | batch 只有 1 的时候 |
| batch 为 1、163840 的词表 (PR 47512 video) | batch 唯一 (wéi yī is exactly 唯一) | batch 等于 1 / 只有 1 个请求 |
| 块比一个 wave 大时，整除再乘回去 (PR 47684 re-check) | 大十 | 块比一个 wave 大的时候 |
| 缓存容量接近时：原来的单池 (PR 28185 re-check) | 接近十 | 缓存容量差不多的时候 |
| SM 数只查一次 (PR 47684 re-check) | SM 数之差一次 (只 heard as zhī after 数) | SM 的数量查一次就缓存起来 |
| 别让最后一波半空着。 (PR 47684 re-check) | 半空整 (a sentence-final 着 drawn out) | 别让最后一波空着一半。 |
| K 是 63 / K 是 86 / K 是 92 (Paged Experts video, 2026-11-04) | K163 / K186 / K192 (是 after a letter fuses into the number) | K 等于 63 |
| 加载专家 e 的时候，写进内存里专家 e 的那一份 (Paged Experts) | 专家一 (a lone lowercase letter is read as a Chinese number) | 加载某个专家的时候，写进内存里属于它的那一份 |
| 算完再散回 merged (Paged Experts) | 算回 | 算完再放回 merged 里原来的位置 |
| 持续负载下的降频 (CUDA perf guide video PR 48380, 2026-11-05) | 详频 (降 read xiáng) | 持续负载下 GPU 时钟变慢 |
| 把 32 字节加载降成两条 LD.E.128 (rmsnorm video PR 47503) | 详成 (降 read xiáng again) | 拆成两条 |
| 判据是数格子 (same video) | 竖格子 (数 read shù, not shǔ) | 判据是看差了几格 |
| 也要有测量撑着 (PR 48380 video) | 成长 (撑着 blurred into 成长) | 也要拿测量结果来证明 |
| D 是 4096 时 / 输出 FP8 时 (PR 47503 video) | 第四 4096 时 / FP80 (是 after a letter, 时 after a digit) | D 等于 4096 的时候 / 输出 FP8 的时候 |
| 原来的 walk 慢在哪 (PR 42855 video, 2026-11-08) | Walkman在哪 (walk 慢 runs together) | 原来的 walk 为什么慢 |
| 避开 0 和 1，零概率的 token (same video) | 避开0和10概率 (1，零 merged into 10) | 避开了 0 和 1，概率为零的 token |
| 一个记最大比较值，一个记计数 (same video) | 一个G最大比较值 (记 heard as a letter) | 一个存最大比较值，一个存计数 |
| 读这一对数 / 每一对数 (same video) | seg_check: 这一 \| 对数, and 对数 is "logarithm" | 读这两个数 / 每个请求的这两个数 |
| 64 行重新排序 (same video) | 64行中心排序 | 调整了每块 W2 里 64 行的顺序 |
| 而清零的写，排在… (PR 48895 video, 2026-11-09) | 清零的血 (写 used as a noun after 的; 血 also reads xiě) | 而清零这一步，排在… |
| 到顶或到底的那一端不检查 (same video) | 到底或到底 (顶 dǐng heard as 底) | 到了第一名或最后一名的那一端，不检查 |

Rule for the narration: a single digit right after 是 or 为 makes another word (after a letter variable such as K, any
number does: K 是 63 → K163), and so does a 时 that closes a
clause after any word (L2 时, 大时, 接近时 all come out as 十); write 只有 / 等于 / 的时候 instead. Keep
single-letter code variables (e, s, i) out of the narration; name the thing instead (某个专家, 分到的槽位). `line_asr` also prints noise (homophones such as 旧/就, numbers, a letter heard as Chinese
such as y → 外) and cannot hear polyphones (Whisper writes 行 for both readings): read each hit, and use
`poly_probe.py` for polyphones.

```bash
$SEV_PY -I $SEV_SKILL/scripts/line_asr.py --script $P/SCRIPT.md --meta $P/audio_meta.json [--lines 5,41]
```

**Telling a misreading from Whisper noise** (re-check of six published videos, 2026-11-03). Edge returns nearly the
same audio for the same text, so a hit that repeats in fresh takes proves nothing on its own: the voice and Whisper
both repeat. Judge by the sound instead:
- A homophone that makes another word in context (时 / 十, 数只 / 之) is real: reword it and check that the reword
  transcribes clean.
- An extra or swallowed syllable: compare the token's duration with the same word elsewhere. 「吞吐涨了」 was heard
  「吞吐涂涨了」 in every take, but the 吞吐 token was 0.55 s, the same as in lines Whisper heard right (0.53 s), so
  no syllable was added. 「wave 越少」 heard 「位越少」 had a 0.25 s wave token like the others.
- Tone sandhi and Whisper's preference for common words: 「因果卷积」 heard 英国 (果 before 卷 is guó anyway; only
  yīn / yīng differ), 「四舍五入」 heard 四手 (the same idiom elsewhere in the take was heard right), 块数 heard 快速.
  None of these was changed.
