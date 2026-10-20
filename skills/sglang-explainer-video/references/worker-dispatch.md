# Frame-worker dispatch context

Paste this into every worker prompt (Agent tool, background), after the line
`ROLE FILE: $P/.hyperframes/frame-packets/_role.md` and an instruction to read the role first.
Fill the `<…>` slots. At most three workers; balance packets by count and size; keep frames that
share a stage (same diagram before/after, two code frames) on the same worker.

```text
## Dispatch context

- PROJECT_DIR: <absolute $P>
- Assigned packets (build in order; one output each at compositions/frames/<frame_id>.html):
  1. .hyperframes/frame-packets/<id>.md
  …
- <PR video | ARTICLE video>. <Article only: frames <ids> use the author's real figures
  assets/article/figN.png — allowed: <img> inside a hairline card, object-fit: contain, never cropped or filtered.>
- <Code frames: the <code-*> block is installed at compositions/<block>.html; fill it from the packet's
  ### Source excerpt and fit its timing to the Scene window (you may inline an adapted copy).>
- <Worker holding the LAST frame only: frame <id> is the video's last frame. Paste the SGLang follow strip from
  <PROJECT_DIR>/assets/brand/follow-strip.html verbatim (apply_brand.py put the theme's variant there): its CSS goes
  into your <style>, its <div> inside #root.
  It sits at y 792–872, so keep that frame's own content above y≈770. Reveal it by the start of the last
  clause and keep it to the end. Never edit its text or URLs, and never narrate it.>
- Read only: the role file, your packets, frame.md (<theme line, see below>) and the assets named above (the
  last-frame worker also reads follow-strip.html). Never open STORYBOARD.md, SCRIPT.md, capture/*, or other
  workers' frames.
  <kimi-dark only: THEME OVERRIDE for the role file — _role.md was written for the code-editorial preset; wherever
  it says cream ground, coral marker, navy Code Surface, EB Garamond, Inter or hairline-ink on cream, read it as the
  kimi-dark equivalent from frame.md. frame.md + this dispatch win over _role.md on every visual point.>
- HARD RULES (each one broke an earlier run):
  1. Frame contract: the FIRST element inside <template> is the composition root
     <div id="root" data-composition-id="<frame_id>" data-start="0" data-duration="<packet duration>"
     data-width="1920" data-height="1080">, and EVERY <style> and <script> (including the gsap CDN load)
     lives INSIDE that root div.
  2. Chinese fonts: only the bundled faces —
     @font-face{font-family:"Noto Sans SC";font-weight:100 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSansSC-Subset.woff2") format("woff2");}
     @font-face{font-family:"Noto Serif SC";font-weight:200 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSerifSC-Subset.woff2") format("woff2");}
     plus the faces frame.md names (sglang-orange: EB Garamond / Inter / JetBrains Mono; kimi-dark: JetBrains Mono
     only). NEVER local(), PingFang, Songti, Hiragino or any system CJK font (乱码 in Studio). Stacks per theme —
     sglang-orange: display "EB Garamond","Noto Serif SC",serif; body "Inter","Noto Sans SC",sans-serif;
     kimi-dark: everything "Noto Sans SC",sans-serif (330 body / 450 headings / 300 big numbers);
     mono (both) "JetBrains Mono","Noto Sans SC",monospace. No negative letter-spacing on Chinese text.
  3. Every element id and class starts with a letter (prefix f<NN>-…), never a digit.
  4. Copy every on-screen string and number exactly from the packet; technical terms stay English.
     Never invent figures; never show PR status (OPEN / in review).
  5. Primary content stays above y≈896 (bottom ~17% = bilingual caption band).
  6. Timeline length = the packet's duration; reveal each element at the second its Scene line names.
- Self-check before finishing (allowed; not the hyperframes CLI):
  source ~/.cache/sglang-explainer-video/env.sh && node $SEV_SKILL/scripts/check_frames.mjs <PROJECT_DIR>
  — your files must print OK.
- Do not run the hyperframes CLI; edit only your output files (and an installed block you were told to fill).
  Reply with one line per file: path + one-sentence summary.
```

Theme lines for the `frame.md (<theme line>)` slot (`.hyperframes/theme` records the project's theme):
- sglang-orange: `SGLang brand remix: cream #FBF6F1 ground, peach cards #F9E6DB / #F2CFBE, ink #1F1410, accent
  #D55816 at most once per frame, navy = warm near-black code surface`
- kimi-dark: `Kimi dark: ground #121212, panels #1B1B1B/#232323 with 1px rgba(255,255,255,.10) hairlines, text
  #EDEDED/#D6D6D6/white@62%/white@42%, Kimi blue #1A88FF at most once per frame (kicker dot excluded), diagram
  palette blue #345074 / green #4A6552 / slate #444864 / brick #6E403C / grey #38373D / rose #C4908F with the
  per-video meanings from the Video direction; code surface #161616 with a #1E1E1E title bar`

Append these lessons to every dispatch, after the template above. Each one cost a fix and a re-render in an
earlier film. The project's `dispatch.py` keeps the full list.

- No whole-frame camera drift: never tween scale/x/y on a frame-wide wrapper (no slow push-in). Workers kept
  adding `fromTo('#fNN-cam', {scale:1}, {scale:1.01x})`; grep for `-cam` after a wave.
- On SVG elements, give `transformOrigin` on **both** sides of a `fromTo`. Set only on the "to" side, it trips
  GSAP's smoothOrigin and shifted an axis by 480px (24GB video).
- Real video clips: `<video class="clip" muted playsinline data-media-start="0" data-track-index="N">`. When the
  clip is shorter than the frame, chain copies back to back (each `data-duration` ≤ the clip length, the last one
  trimmed to the frame end). `data-start` inside a frame is frame-local; `hf lint`'s
  `nested_media_start_basis_ambiguous` warning on chained copies is expected.
- `check_frames.mjs` treats the highest-numbered frame on disk as the last frame, so mid-run it reports "missing
  follow strip" on whatever frame is newest. Tell workers to ignore it unless they hold the real last frame.
- Excerpt line ranges come from the orchestrator. If a Scene line names a code line that is not in the packet's
  excerpt, the worker should report it, not paraphrase. The fix is upstream: correct the range in `frames.py`
  and regenerate the packet. In the 24GB video, a range was off by one and dropped the `copy_` line the narration
  was about.

- Missing glyphs: Noto Sans SC has no Unicode superscript/subscript characters (ᵀ ₁ ₂ ₗ ⁻ ⁰ ⁵–⁹ ⁺ ᵢ ⱼ), no ⋮ and
  no ✗. Workers must write super/subscripts as `<sup>`/`<sub>` (k<sup>T</sup>, e<sup>−80</sup>, β<sub>1</sub>) even
  where the packet writes the Unicode form, and draw ✓ ✗ ⋮ as inline SVG; ¹ ² ³ ⁴ are fine. `subset_fonts.py` now
  warns about any used character no bundled font covers (K3 video: caught after two waves; one caption had ᵀ).
- Ligatures: JetBrains Mono turns `==` `!=` `->` `>=` into ligature glyphs. Every mono element (code, kickers,
  paths, commands) sets `font-variant-ligatures:none; font-feature-settings:"liga" 0,"calt" 0`.
- kimi-dark: tell workers the theme override for `_role.md` (template above); without it they follow the role
  file's cream/coral wording.

Retry (once per frame): SendMessage to the same worker with the exact validator/lint finding and the
check command; ask it to change only what the finding names.
