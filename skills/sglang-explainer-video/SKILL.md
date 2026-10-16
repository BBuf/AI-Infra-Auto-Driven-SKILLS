---
name: sglang-explainer-video
description: Turn an SGLang PR (number or URL) or a technical article (Markdown path) into a Chinese explainer video that fully explains every algorithm and code change (length follows the material, no time cap) — natural edge-tts narration, word-synced bilingual 中文/English captions, the unified orange "SGLang PR 讲解 / SGLang 技术解读" cover, SGLang-brand visuals — built on HyperFrames' pr-to-video pipeline and delivered to ~/Desktop. Use when the user sends an sgl-project/sglang PR number or link, or an article file, and asks for a video (做个视频, 讲解视频, PR 视频, 文章转视频).
---

# SGLang explainer video

Input: `sglang PR <N>` (or a github.com/sgl-project/sglang/pull/N URL), or a Markdown article path.
Output: `~/Desktop/SGLang_PR<N>_讲解_中英字幕.mp4` (article: `SGLang_<slug>_技术解读_中英字幕.mp4`) plus a still `…_封面.png` and a Bilibili cover `…_封面_B站.png`, rendered 1920×1080 with a 中文 narration and bilingual captions.

This skill drives HyperFrames' **pr-to-video** workflow (heygen-com/hyperframes, pinned) with a fixed house
style and several fixes that make it work for Chinese. Read `references/house-style.md` before writing any
script — it holds the rules the user already corrected once (terms, captions, cover, credits, PR status).

## Core principle: understand it fully, explain all of it

The video exists so a viewer who has not read the PR / article **understands every algorithm and every
meaningful code change** in it. That decides everything else:

- **No target length.** Length follows the material: a one-line docs PR may need ~1 min; a multi-file kernel
  or scheduler PR, or a long article, may need 5–10+ min. Never cut a mechanism, code path, edge case, number
  or caveat to make the film shorter — add frames instead. Keep each frame to one idea (≲ 60 汉字, ~11 s) so
  visuals keep up; the *number* of frames is unbounded.
- **Understand before you script.** Read all of the source (Step 2) and write `capture/understanding.md`
  first; every item in it must map to at least one frame.
- **Code is the spine, not a footnote.** Open with the value (what the change fixes / speeds up), then walk
  through the algorithm and every changed code path in a sensible order, with a worked example where it helps.
  This overrides pr-to-video's `story-design.md` where it says otherwise (its "Whole-film target ≤ ~3 min",
  "feature 2–4 real diff hunks", "implementation is the footnote"): feature **every hunk that carries logic**
  (group pure renames / formatting / test boilerplate into one frame), split long hunks across frames
  (≤ 12 lines each), and explain why each change is needed, not just what it is.

## 0. Setup (start of every session)

```bash
bash <this-skill>/scripts/setup_env.sh      # first run ~2 min; afterwards seconds (idempotent)
source ~/.cache/sglang-explainer-video/env.sh
```

Re-run it at the start of each session: `env.sh` records this skill's install path (`$SEV_SKILL`,
`$HF_MEDIA_ENGINE`), so it must point at the copy you are using now.

It needs Node ≥ 22 (`brew install node@22`), `ffmpeg`/`ffprobe`, git, curl and Chrome. It creates a venv
(edge-tts, fonttools, and mlx-whisper on Apple Silicon for `pron_check.py`), a pinned HyperFrames checkout with `patches/hyperframes-captions-cjk-bilingual.patch`
applied, the Noto Sans/Serif SC fonts, and `env.sh` (`$SKILL_DIR` = pr-to-video, `hf` = pinned HyperFrames CLI wrapper,
`$HF_MEDIA_ENGINE` = our edge-tts engine, `$SEV_SKILL`, `$SEV_PY`). Nothing is installed globally:
always run the CLI as `hf …`, never `npx hyperframes init` without the env (init would install skills
into `~/.claude/skills`).

Every command below assumes `source ~/.cache/sglang-explainer-video/env.sh` and `P=<project dir>`.

## 1. Project + brief

```bash
P=$SEV_WS/projects/sglang-pr-<N>            # article: $SEV_WS/projects/sglang-article-<slug>
hf init "$P" --non-interactive --example=blank --skill=pr-to-video
```

Write `$P/BRIEF.md` (template in `references/house-style.md` §Brief). Run **autonomous** (`flow: automation`,
`storyboard: no`): do not stop for plan/sketch approval; the only questions worth asking are real choices
(voice change) — otherwise follow the house style. Never ask about length: it follows the material.

## 2. Ingest the source

PR mode — always the **full URL** (`owner/repo#N` fails outside a git checkout):

```bash
(cd $P && node $SKILL_DIR/scripts/fetch-pr.mjs --pr https://github.com/sgl-project/sglang/pull/<N> --out-dir ./capture \
  && node $SKILL_DIR/scripts/ingest.mjs --pr-json ./capture/pr.json --diff ./capture/diff.patch --out-dir ./capture/extracted \
  && node $SKILL_DIR/scripts/fetch-people-avatars.mjs --people ./capture/extracted/people.json)
```

Article mode — copy the article to `$P/capture/extracted/visible-text.txt`, write a minimal
`capture/extracted/tokens.json` (`{"title": "...", "colors": [], "fonts": []}`), and download the article's own
figures into `$P/assets/article/fig<i>.<ext>` (they are the author's images; they become real assets on frames).

**Read everything before writing anything.** `capture/extracted/visible-text.txt` is budget-truncated, so it is
not enough on its own:

- PR mode: the PR body, the **full** `capture/diff.patch`, review comments that changed the design
  (`gh pr view <URL> --comments`), and every touched function **in full at the PR head**
  (`gh api "repos/sgl-project/sglang/contents/<path>?ref=<headRefOid>" --jq .content | base64 -d`) so you can
  explain the code around each hunk, not just the hunk. Follow calls into helpers when the change depends on them.
- Article mode: the whole article, every code block and figure, and any PR / source file it links to that the
  argument depends on.

Then write `$P/capture/understanding.md`: the problem; the algorithm step by step; every changed code path
(`file:function` → what changed and why); the numbers with their conditions (hardware, model, workload);
the caveats the author states. This is the coverage checklist for Step 3.

Then put the SGLang brand on the project (frame.md remix, CJK/bilingual caption skin, logos):

```bash
$SEV_PY -I $SEV_SKILL/scripts/apply_brand.py --project $P
```

## 3. Story + script

Follow `$SKILL_DIR/SKILL.md` Step 3 and its `references/story-design.md` (PR: fix-explainer / feature-reveal /
refactor-walkthrough; article: a value-first explainer — hook → landscape → method → every evidence beat the
article makes → lessons → summary), **with the Core principle above overriding its length and hunk limits**.
Before moving on, tick every item of `capture/understanding.md` against the storyboard; add frames for
anything uncovered. Then apply the house style:

- **Frame 1 is always the cover** `compositions/frames/00-cover.html` (built by `make_cover.py`, not a worker),
  `status: animated`. Frame N's file is `NN-<slug>.html` with `NN = N-1`.
- PR videos end on the **author page** (name/@login/avatar on screen, never spoken; the PR link; no status).
- **Every video's last frame carries the SGLang follow strip** (`assets/brand/follow-strip.html`: 关注 SGLang ·
  GitHub · 博客 · 文档) at y 792–872, so plan that frame's content to end above y≈770. It is on screen only, not
  narrated (house style §Follow strip).
- Chinese narration, English technical terms, one idea per frame (≲ 60 汉字, ~11 s); as many frames as the
  material needs — no whole-film cap. Long films get section kickers (e.g. 「① 问题」「② 算法」「③ 代码」) so
  viewers keep their place.
- `SCRIPT.md` voice line: `zh-CN-XiaoxiaoNeural (edge-tts)`. Numbers only from the PR/article.
- **Every English word must be read right and in one breath** (`references/pronunciation.md`): no `_` reaches
  the voice, CamelCase brand names are spaced, symbols (×, #, +%) are spoken as Chinese. Write the narration
  normally — `pronunciation.json` rewrites only what the voice hears, captions keep the written form — then run
  the check and fix every FAIL (lexicon entry or reword) before TTS:

  ```bash
  $SEV_PY -I $SEV_SKILL/scripts/pron_check.py --script $P/SCRIPT.md --asr
  ```

## 4. Narration (edge-tts, word timings)

```bash
(cd $P && node $SKILL_DIR/scripts/audio.mjs --script ./SCRIPT.md --storyboard ./STORYBOARD.md --hyperframes . \
   --out ./audio_meta.json --voice zh-CN-XiaoxiaoNeural \
 && node $SKILL_DIR/scripts/audio.mjs sync-durations --audio-meta ./audio_meta.json --storyboard ./STORYBOARD.md \
 && cp audio_meta.json audio_meta.built.json)
```

`HF_MEDIA_ENGINE` swaps HyperFrames' TTS for `scripts/edge_tts_engine.py`: per-word timestamps from WordBoundary,
punctuation re-attached, split terms re-joined (FP32, FLUX.2, torch.equal), 2 s of silence after the last line;
the voice reads the spoken forms from `pronounce.py` and the timings map back to the written tokens.
Then check the recorded take — pauses inside a clause and every English term in context:

```bash
$SEV_PY -I $SEV_SKILL/scripts/pron_check.py --meta $P/audio_meta.json --hyperframes $P --asr
```

A FAIL there means re-synthesize that line after a lexicon fix or a reword (subset re-TTS, see troubleshooting).
A frame running long means it holds two ideas — split it; never drop content to shorten the film.
Print the timings for Step 5:

```bash
python3 -c "import json;[print(v['frame'],v['duration_s'],' | '.join(f\"{w['text']}@{w['start']}\" for w in v['words'])) for v in json.load(open('$P/audio_meta.json'))['voices']]"
```

**Keep `audio_meta.built.json`.** edge-tts takes are not bit-identical; frames are cued to this take.

## 5. Visual design (Step 4 of pr-to-video)

Write `## Video direction` + per-frame time-coded Scenes per `$SKILL_DIR/references/visual-design.md`, using the
**real word start times** above. Put the palette/type/font rules from `references/house-style.md` §Visuals into
Video direction. Code beats need a `### Source excerpt` (≤ 12 real lines); article frames that show a real figure
carry `- asset_candidates: assets/article/figN.png — …`.

## 6. Cover

```bash
$SEV_PY -I $SEV_SKILL/scripts/make_cover.py --project $P --series "SGLang PR 讲解" --badge "#<N>" \
  --title-zh "<一句中文问题/主张>" --title-en "<PR title>" --meta "<file> · +A / −D · @<author>" \
  --duration <frame-1 voice s> --cue-title <s> --cue-meta <s> --png $P/cover.png --png-bili $P/cover_bili.png
```

Article: `--series "SGLang 技术解读" --badge "<one keyword>"`. The design is fixed (see house style §Cover).
`--png-bili` renders the same design at 1440×1080 with every element inside the centred 16:9 band — the
cover to upload to Bilibili, which crops one image to both 4:3 and 16:9 (house style §Cover).

## 7. Captions (bilingual)

```bash
(cd $P && node $SKILL_DIR/scripts/captions.mjs build --storyboard ./STORYBOARD.md --audio-meta ./audio_meta.json \
   --hyperframes . --out ./caption_groups.json)
```

Translate **every** group's `text` into `$P/caption-translations.json` (`{"<中文 group text>": "<English>"}`),
then rebuild; the build warns `no translation for N group(s)` until all are covered.

## 8. Frames (≤ 3 parallel workers)

```bash
(cd $P && for b in <code-* blocks named in STORYBOARD>; do hf add "$b"; done)
node $SKILL_DIR/scripts/frame-packets.mjs --project "$P" --storyboard "$P/STORYBOARD.md"
```

Dispatch at most three background workers (Agent tool) at a time, each with the role file path, its packet paths and
the dispatch context from `references/worker-dispatch.md` **verbatim** (it carries the hard rules that broke
earlier runs). A packet over 48 KB means the blueprint + cited rules are too big: drop one cited rule or use
`blueprint: compose`. Mark frames `animated` as workers return. A long film (many frames) goes in waves: give each worker
≤ 5 packets per dispatch, keep frames that share a stage on the same worker, and start the next wave as workers
return. Then:

```bash
node $SEV_SKILL/scripts/check_frames.mjs $P          # frame contract + no system CJK fonts
$SEV_PY -I $SEV_SKILL/scripts/subset_fonts.py --project $P   # re-run after ANY text change
(cd $P && node $SKILL_DIR/scripts/captions.mjs build --storyboard ./STORYBOARD.md --audio-meta ./audio_meta.json --hyperframes . --out ./caption_groups.json \
 && node $SKILL_DIR/scripts/assemble-index.mjs --storyboard ./STORYBOARD.md --hyperframes . \
 && node $SKILL_DIR/scripts/transitions.mjs inject --storyboard ./STORYBOARD.md --hyperframes . \
 && node $SKILL_DIR/scripts/transitions.mjs verify --storyboard ./STORYBOARD.md --index ./index.html \
 && hf lint && hf check)
```

A failed frame goes back to its worker once with the exact finding (SendMessage). `lint`/`check` errors you
fix yourself with the smallest edit. `id_requires_css_escape` warnings and transition-time `content_overlap`
infos are expected.

## 9. Render, QA, deliver

```bash
(cd $P && hf render --skill=pr-to-video --quality high --output renders/video.mp4)
bash $SEV_SKILL/scripts/qa_sheet.sh $P/renders/video.mp4 $P/qa 2.5
```

Read **every** sheet of the rendered MP4 (not Studio snapshots): 乱码, wrong characters, untranslated or
mistranslated terms, captions out of sync, content under the caption band, stray status lines. Fix, re-render,
re-check. Then copy to the Desktop:

```bash
cp $P/renders/video.mp4 ~/Desktop/SGLang_PR<N>_讲解_中英字幕.mp4
cp $P/cover.png ~/Desktop/SGLang_PR<N>_讲解_封面.png
cp $P/cover_bili.png ~/Desktop/SGLang_PR<N>_讲解_封面_B站.png
```

## 10. Publish (only when the user asks)

Follow `references/publishing.md`: the user picks each file in the native dialog, the agent fills the rest.
Bilibili: 含AI生成内容 · 分区 人工智能 · ~8 tags · 1440×1080 B站 cover. 小红书: **always 标记 SGLang** via
添加组件 → 标记地点或标记朋友 → 用户 → SGLang (official, orange SGL avatar) — never an @ in the body — plus topics
#人工智能 #大模型 #SGLang and 笔记含AI合成内容.

## Changing narration after frames exist

Edit `SCRIPT.md` (and the storyboard `voiceover`), re-run Step 4, then — instead of rebuilding frames —

```bash
$SEV_PY -I $SEV_SKILL/scripts/retime_frames.py --project $P --ref-meta $P/audio_meta.built.json
```

re-maps each frame's timeline onto the new take (piecewise-linear, anchors = matched clause starts). Re-make
the cover (`make_cover.py`) with the new frame-1 cues, retranslate changed caption groups, re-subset fonts,
re-assemble, re-render.

## References

- `references/house-style.md` — narration, term glossary, captions, cover, author page, brief template.
- `references/pronunciation.md` — English terms read right: spoken-form rules, lexicon, `pron_check.py`.
- `references/worker-dispatch.md` — the dispatch context to paste into every frame worker.
- `references/troubleshooting.md` — every failure seen so far and its fix.
- `references/publishing.md` — publishing to Bilibili and 小红书 (always 标记 SGLang on 小红书, never @).
