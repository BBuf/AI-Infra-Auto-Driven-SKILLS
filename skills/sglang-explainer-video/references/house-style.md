# House style (user-confirmed — do not relitigate)

Every rule here came from a correction on a delivered video. Apply all of them by default.

## Narration

- Language: Chinese narration, plain technical register (像给同事讲 PR), no hype.
- **Technical nouns stay English** — never translate them:
  kernel (never 内核), batch / mixed batch / mixed chunk (never 混批), prefill, decode, decode tail, prefix,
  token, attention, metadata, output buffer, fallback, bit-exact, micro benchmark, profile, launch, step, CFG,
  GEMM, quant, epilogue, branch, review, PR. Everyday words stay Chinese: 吞吐, 耗时, 并发, 端到端.
- Write numbers as digits (15%, 3.79 ms); the TTS reads them correctly and captions show them verbatim.
- Budget: ≲ 60 汉字 per frame (~11 s at Yunyang +0%), whole film ≤ 3 min. Measure after TTS and trim.
- Numbers and claims only from the PR / article. Keep the source's own caveats when they matter
  (e.g. "未在新 branch 上重测", "数据来自不同模型和硬件，不能相加").
- **Never state PR status** — no "还在 review 中", "OPEN", "未合并", "in review" in narration or on screen.
  PRs usually get merged; the video should not date itself.
- Closing line for PR videos: `以上就是这个 PR 的全部改动。`

## Voice

- `zh-CN-YunyangNeural` (edge-tts, news-anchor male) at `+0%` — chosen by the user over Yunxi/Xiaoxiao/
  Yunjian/Xiaoyi/Andrew/Ava after A/B samples. Only change it if the user asks.

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

## Delivery

- Final MP4 and the still cover go straight to `~/Desktop`
  (`SGLang_PR<N>_讲解_中英字幕.mp4`, `SGLang_PR<N>_讲解_封面.png`).
- Before delivering, review contact sheets of the **rendered MP4** frame by frame.

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
length: <40–70s for a focused PR, ≤ 180s for an article>
angle: <fix-explainer | feature-reveal | refactor-walkthrough | changelog | article-explainer>
narration: yes
---

## Intent
<what the video explains, for whom>

## Notes
- House style: references/house-style.md (English terms, bilingual captions, Yunyang, cover A, no PR status).
```
