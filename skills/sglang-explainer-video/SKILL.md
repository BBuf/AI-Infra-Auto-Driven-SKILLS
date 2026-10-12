---
name: sglang-explainer-video
description: Turn an SGLang PR (number or URL) or a technical article (Markdown path) into a 1–3 minute Chinese explainer video — natural edge-tts narration, word-synced bilingual 中文/English captions, the unified orange "SGLang PR 讲解 / SGLang 技术解读" cover, SGLang-brand visuals — built on HyperFrames' pr-to-video pipeline and delivered to ~/Desktop. Use when the user sends an sgl-project/sglang PR number or link, or an article file, and asks for a video (做个视频, 讲解视频, PR 视频, 文章转视频).
---

# SGLang explainer video

Input: `sglang PR <N>` (or a github.com/sgl-project/sglang/pull/N URL), or a Markdown article path.
Output: `~/Desktop/SGLang_PR<N>_讲解_中英字幕.mp4` (article: `SGLang_<slug>_技术解读_中英字幕.mp4`) plus a still `…_封面.png`, rendered 1920×1080 with a 中文 narration and bilingual captions.

This skill drives HyperFrames' **pr-to-video** workflow (heygen-com/hyperframes, pinned) with a fixed house
style and several fixes that make it work for Chinese. Read `references/house-style.md` before writing any
script — it holds the rules the user already corrected once (terms, captions, cover, credits, PR status).

## 0. One-time setup (per machine)

```bash
bash <this-skill>/scripts/setup_env.sh      # ~2 min; idempotent
source ~/.cache/sglang-explainer-video/env.sh
```

It needs Node ≥ 22 (`brew install node@22`), `ffmpeg`/`ffprobe`, git, curl and Chrome. It creates a venv
(edge-tts, fonttools), a pinned HyperFrames checkout with `patches/hyperframes-captions-cjk-bilingual.patch`
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
(voice change, length) — otherwise follow the house style.

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

Then put the SGLang brand on the project (frame.md remix, CJK/bilingual caption skin, logos):

```bash
$SEV_PY -I $SEV_SKILL/scripts/apply_brand.py --project $P
```

## 3. Story + script

Follow `$SKILL_DIR/SKILL.md` Step 3 and its `references/story-design.md` (PR: fix-explainer / feature-reveal /
refactor-walkthrough; article: a value-first explainer — hook → landscape → method → 2–4 evidence beats →
lessons → summary). Then apply the house style:

- **Frame 1 is always the cover** `compositions/frames/00-cover.html` (built by `make_cover.py`, not a worker),
  `status: animated`. Frame N's file is `NN-<slug>.html` with `NN = N-1`.
- PR videos end on the **author page** (name/@login/avatar on screen, never spoken; the PR link; no status).
- Chinese narration, English technical terms, ≲ 60 汉字 per frame (~11 s); whole film ≤ 3 min.
- `SCRIPT.md` voice line: `zh-CN-YunyangNeural (edge-tts)`. Numbers only from the PR/article.

## 4. Narration (edge-tts, word timings)

```bash
(cd $P && node $SKILL_DIR/scripts/audio.mjs --script ./SCRIPT.md --storyboard ./STORYBOARD.md --hyperframes . \
   --out ./audio_meta.json --voice zh-CN-YunyangNeural \
 && node $SKILL_DIR/scripts/audio.mjs sync-durations --audio-meta ./audio_meta.json --storyboard ./STORYBOARD.md \
 && cp audio_meta.json audio_meta.built.json)
```

`HF_MEDIA_ENGINE` swaps HyperFrames' TTS for `scripts/edge_tts_engine.py`: per-word timestamps from WordBoundary,
punctuation re-attached, split terms re-joined (FP32, FLUX.2, torch.equal), 2 s of silence after the last line.
Too long? Trim the script or `EDGE_TTS_RATE=+6%`. Print the timings for Step 5:

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
  --duration <frame-1 voice s> --cue-title <s> --cue-meta <s> --png $P/cover.png
```

Article: `--series "SGLang 技术解读" --badge "<one keyword>"`. The design is fixed (see house style §Cover).

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

Dispatch at most three background workers (Agent tool), each with the role file path, its packet paths and
the dispatch context from `references/worker-dispatch.md` **verbatim** (it carries the hard rules that broke
earlier runs). A packet over 48 KB means the blueprint + cited rules are too big: drop one cited rule or use
`blueprint: compose`. Mark frames `animated` as workers return. Then:

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
```

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
- `references/worker-dispatch.md` — the dispatch context to paste into every frame worker.
- `references/troubleshooting.md` — every failure seen so far and its fix.
