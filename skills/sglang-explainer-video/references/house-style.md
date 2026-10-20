# House style (user-confirmed — do not relitigate)

Every rule here came from a correction on a delivered video. Apply all of them by default.

## Narration

- Language: Chinese narration, plain technical register (像给同事讲 PR), no hype.
- **Technical nouns stay English** — never translate them:
  kernel (never 内核), batch / mixed batch / mixed chunk (never 混批), prefill, decode, decode tail, prefix,
  token, attention, metadata, output buffer, fallback, bit-exact, micro benchmark, profile, launch, step, CFG,
  GEMM, quant, epilogue, branch, review, PR. Everyday words stay Chinese: 吞吐, 耗时, 并发, 端到端.
  Grep the narration **and the on-screen Scene strings** for Chinese renderings before TTS — 预填 (→ prefill),
  前缀 (→ prefix), 每批 / 批次 (→ batch), 内核, 混批 — the user flagged 预填 on a title (DeepSeek V4.1 video, 2026-10-18).
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

## Cover (theme sglang-orange: design "A · brand orange")

`scripts/make_cover.py` renders it; never redesign it per video (pick a theme instead, §Themes). Reference renders:
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

- Theme sglang-orange (default): frame.md = code-editorial remixed onto the SGLang brand by `apply_brand.py`: cream `#FBF6F1` ground,
  peach cards `#F9E6DB` / `#F2CFBE`, ink `#1F1410`, accent SGLang orange `#D55816` at most once per frame,
  warm near-black code surface.
- Fonts: EB Garamond / Inter / JetBrains Mono + **bundled** Noto Serif SC / Noto Sans SC subsets only.
  Never `local()`, PingFang, Songti, Hiragino — they garble in HyperFrames Studio.
- Prefer real material: code hunks from the diff, the article's own figures (contain, never cropped).
- Bottom ~17 % is the caption band.

## Themes

Every video uses one theme, picked once at Step 2 (`apply_brand.py --theme`) and read again by
`make_cover.py` from `.hyperframes/theme`. The default stays **sglang-orange** (cover A + the code-editorial
remix above).

**Subject is not SGLang's own work → no SGLang branding at all** (user, 2026-10-20, DeepSeek DeepInfer video:
「这个视频和 sglang 没有任何关系，不要出现 sglang 和对应风格」). Decide this at Step 2, before any narration or cover:
- no "SGLang 技术解读" in the opening line, no SGLang logo / series label / orange / follow strip, and no SGLang
  mentions in narration or on screen;
- build a project-local theme from the subject's own site instead: read its CSS variables in the browser, write
  `frame.md` and `.hyperframes/caption-skin.html` by hand (as kimi-dark does), and render the cover with a project
  script (the DeepInfer video's `work/make_cover_ds.py`: deepseek.com's sky-blue wash, glass cards, DM Sans);
- never draw the subject's official logo; label the cover 「技术解读」 so it cannot pass as an official video;
- `check_frames.mjs` then reports the missing follow strip on the last frame — expected, ignore it.

**kimi-dark** (added 2026-10 for the Kimi K3 video, after the user said the 19 orange covers had become 审美疲劳):
- When to use it: a video about a Moonshot / Kimi model, or when the user asks for a different look. Don't
  switch silently for an ordinary SGLang PR video.
- Look: aligned with kimi.com and Kimi's research blog. Flat `#121212` ground, `#1B1B1B` panels with white-alpha
  hairlines, Noto Sans SC at 330 / 450 / 300, JetBrains Mono for code, one Kimi-blue `#1A88FF` accent per frame.
- Diagram palette: a muted semantic palette sampled from Moonshot's official architecture figure (blue, green,
  slate, brick, grey, rose). The Video direction fixes what each colour means for the video.
- Files: `assets/themes/kimi-dark/{frame.md, caption-skin.html, follow-strip.html}`.
- Cover: `make_cover.py --theme kimi-dark --title-zh <short title> --subtitle <one line> --title-en <mono line>
  --stats "2.8T=参数;16 / 896=每个 token 激活的专家" [--visual right.svg]`.
  - Layout: SGL logo chip plus the series pill, a 150 px title, light stat numbers, and an optional right-hand
    SVG. The K3 video drew its own redraw of the official block diagram.
  - Never put the subject's logo on the cover. It is an SGLang video, not an official Kimi one. The SGLang logo
    chip and series pill stay.
  - The 1440×1080 `_B站` still keeps every element inside y 135–945, the same rule as the orange cover.
- An official figure that matches the theme (dark background) can be shown as-is. Credit it with 「图源：…」.

## Author page (PR videos)

- Only the PR author (name + @login + avatar) and the PR link. No reviewers, approvers, commit counts or status.
- The name is on screen only — the narration does not read it.

## Follow strip (last frame of every video)

Every SGLang video ends with links that invite viewers to follow the project, as asked by the user.

- The last frame shows the fixed strip that `apply_brand.py` copies to the project's `assets/brand/follow-strip.html`
  (the theme's colours; reference render of the orange one: `assets/brand/follow-strip-example.png`):
  - 关注 SGLang
  - GitHub `github.com/sgl-project/sglang`
  - 博客 `sglang.io/blog`
  - 文档 `docs.sglang.io`
- Where it goes: on the author page in PR videos; on the summary / links frame in article videos.
- Placement: a bar at y 792–872 (peach in sglang-orange, #1B1B1B in kimi-dark), just above the caption band. The frame's own content ends above y≈770.
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
