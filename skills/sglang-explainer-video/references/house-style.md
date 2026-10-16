# House style (user-confirmed — do not relitigate)

Every rule here came from a correction on a delivered video. Apply all of them by default.

## Narration

- Language: Chinese narration, plain technical register (像给同事讲 PR), no hype.
- **Technical nouns stay English** — never translate them:
  kernel (never 内核), batch / mixed batch / mixed chunk (never 混批), prefill, decode, decode tail, prefix,
  token, attention, metadata, output buffer, fallback, bit-exact, micro benchmark, profile, launch, step, CFG,
  GEMM, quant, epilogue, branch, review, PR. Everyday words stay Chinese: 吞吐, 耗时, 并发, 端到端.
- Write numbers as digits (15%, 3.79 ms); the TTS reads them correctly and captions show them verbatim.
- **Completeness over brevity.** No time limit: the film is as long as it takes to explain every algorithm
  and code detail of the PR / article so a viewer can follow it. Pace per frame, not per film: one idea per
  frame, ≲ 60 汉字 (~11 s at Xiaoxiao +0%); split a frame that runs long instead of trimming content.
- Numbers and claims only from the PR / article. Keep the source's own caveats when they matter
  (e.g. "未在新 branch 上重测", "数据来自不同模型和硬件，不能相加").
- **Never state PR status** — no "还在 review 中", "OPEN", "未合并", "in review" in narration or on screen.
  PRs usually get merged; the video should not date itself.
- Closing line for PR videos: `以上就是这个 PR 的全部改动。`

## Voice

- `zh-CN-XiaoxiaoNeural` (edge-tts, warm female) at `+0%` — every video from 2026-10-16 on. Chosen by the
  user from a 16-voice female audition (Qwen3-TTS VoiceDesign/CustomVoice on sglang-omni, edge
  Xiaoyi/Seraphina/Emma/Ava) after a viewer asked for an "SGL girl" voice; it replaced Yunyang. Only change
  it if the user asks.
- **English terms are read right and fluently** — the user's explicit bar. Follow `pronunciation.md`; run
  `pron_check.py --asr` before TTS and on the recorded take.
- **The pronunciation lexicon keeps learning** (user, 2026-10-16). Each new video that hits a misread word
  updates `pronunciation.json` and `pronunciation.md`, and ships them to the skill repo, so the skill keeps
  working on its own over time.

## Captions

- Bilingual: Chinese karaoke line (word-synced highlight) + English line under it, every group translated.
- Groups are clauses (split at ，。；：！？, max ~26 Han-width); never split after 、 or inside a term.
- Translation: natural English for an engineer, same facts, keep the English terms as-is.

## Cover (fixed: design "A · brand orange")

`scripts/make_cover.py` renders it; never redesign it per video. Reference renders:
`assets/brand/cover-example-pr.png` (PR) and `assets/brand/cover-example-article.png` (article).
In the video frame the meta line and badge sit above the caption band and the text dissolves
just before the crossfade; the still PNG keeps the full layout.

- Full-bleed SGLang orange `#D55816`; SGL wordmark (`assets/brand/sglang-logo.png`) on a peach `#FADDCD`
  card with a 5 px rust `#A5300F` border, top-left.
- Series label, white, heavy: **`SGLang PR 讲解`** for PRs, **`SGLang 技术解读`** for articles; a peach
  underline bar.
- Bold white Chinese title (balanced wrap), peach EB Garamond italic English title, mono meta line.
- The logo's node-graph motif (two nodes + `</>` box) oversized at right, peach, 22 % opacity.
- Badge bottom-right in the logo-"L" style (peach fill, rust outline, trailing rust stripes): `#<PR number>`
  for PRs, one short keyword for articles (e.g. `Diffusion`).
- The cover is frame 1 of the video (with narration introducing the topic) and also exported as a still PNG.
- **Bilibili cover** (`make_cover.py --png-bili`, reference `assets/brand/cover-example-bili.png`): Bilibili
  keeps ONE cover image and crops it to 4:3 (home feed) and a centred 1440×810 16:9 band (personal space);
  a separate image per ratio does not stick. The 1920×1080 still loses its right-hand badge in the 4:3
  crop, so upload the 1440×1080 `_B站` cover, which keeps logo, title and badge inside y 135–945. Leave
  「双比例同步改动」 on and check both saved crops.

## Visuals (body frames)

- frame.md = code-editorial remixed onto the SGLang brand by `apply_brand.py`: cream `#FBF6F1` ground,
  peach cards `#F9E6DB` / `#F2CFBE`, ink `#1F1410`, accent SGLang orange `#D55816` at most once per frame,
  warm near-black code surface.
- Fonts: EB Garamond / Inter / JetBrains Mono + **bundled** Noto Serif SC / Noto Sans SC subsets only.
  Never `local()`, PingFang, Songti, Hiragino — they garble in HyperFrames Studio.
- Prefer real material: code hunks from the diff, the article's own figures (contain, never cropped).
- Bottom ~17 % is the caption band.

## Author page (PR videos)

- Only the PR author (name + @login + avatar) and the PR link. No reviewers, approvers, commit counts or status.
- The name is on screen only — the narration does not read it.

## Follow strip (last frame of every video)

Every SGLang video ends with links that invite viewers to follow the project, as asked by the user.

- The last frame shows the fixed strip from `assets/brand/follow-strip.html` (reference render:
  `assets/brand/follow-strip-example.png`):
  - 关注 SGLang
  - GitHub `github.com/sgl-project/sglang`
  - 博客 `sglang.io/blog`
  - 文档 `docs.sglang.io`
- Where it goes: on the author page in PR videos; on the summary / links frame in article videos.
- Placement: a peach bar at y 792–872, just above the caption band. The frame's own content ends above y≈770.
- Timing: the strip rises in by the start of the frame's last clause and stays until the end. The 2 s hold
  after the last line keeps it readable.
- Copy it verbatim:
  - Do not restyle, translate or shorten the URLs.
  - Do not narrate it.
  - Do not add other channels (X, Discord, …) unless the user asks.
- `check_frames.mjs` fails when the last frame lacks the strip or any of its three URLs.

## Delivery

- Final MP4 and the still cover go straight to `~/Desktop`
  (`SGLang_PR<N>_讲解_中英字幕.mp4`, `SGLang_PR<N>_讲解_封面.png`, `SGLang_PR<N>_讲解_封面_B站.png`).
- Before delivering, review contact sheets of the **rendered MP4** frame by frame.
- Publishing (only when asked) follows `publishing.md`; every 小红书 note must 标记 the official SGLang account
  (添加组件 → 标记地点或标记朋友), never @ it in the body.

## Brief template (`BRIEF.md`)

```markdown
---
workflow: pr-to-video
flow: automation
storyboard: no
source: pr            # or: article
pr: sgl-project/sglang#<N>
series: SGLang PR 讲解  # article: SGLang 技术解读
message: "<one sentence the viewer must take away>"
destination: desktop
aspect: 1920x1080
language: zh
audience: developers
length: <estimate = frames in understanding.md coverage × ~10 s; informational only, never a cap>
angle: <fix-explainer | feature-reveal | refactor-walkthrough | changelog | article-explainer>
narration: yes
---

## Intent
<what the video explains, for whom>

## Notes
- House style: references/house-style.md (English terms read right, bilingual captions, Xiaoxiao voice, cover A, no PR status).
```
